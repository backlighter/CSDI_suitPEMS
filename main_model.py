import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config,device):
        super().__init__()
        self.target_dim = target_dim
        self.device =device
        self.emb_time_dim = int(config["model"]["timeemb"])
        self.emb_feature_dim = int(config["model"]["featureemb"])
        self.is_unconditional = False
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = str(self.emb_total_dim)

        input_dim = 1 if self.is_unconditional == True else 2  # 有状态的扩散input_dim=2,无状态的扩散input_dim=1 
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models #生成扩散模型的β参数的分布
        self.num_steps = int(config_diff["num_steps"])
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                float(config_diff["beta_start"]) ** 0.5, float(config_diff["beta_end"]) ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                float(config_diff["beta_start"]), float(config_diff["beta_end"]), self.num_steps
            )

        self.alpha_hat = 1 - self.beta #alpha_hat 序列为[0.9,0.8,0.7]
        self.alpha = np.cumprod(self.alpha_hat)# α序列为[0.9,0.9*0.8,0.9*0.8*0.7] shape(50)
        self.alpha_torch = torch.tensor(self.alpha).float().to(device).unsqueeze(1).unsqueeze(1) #shape(50,1,1)

    def time_embedding(self, pos, d_model=128):  #pos [B,L] = [32,12]  positional Encoding  # 这个位置编码 暂时不纠结
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device) # [32,12,128]  d_model.shape = emb_time_dim
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe  # Transformer的位置编码

    # def get_randmask(self, observed_mask):
    #     rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    #     rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
    #     for i in range(len(observed_mask)):
    #         sample_ratio = np.random.rand()  # missing ratio
    #         num_observed = observed_mask[i].sum().item()
    #         num_masked = round(num_observed * sample_ratio)
    #         rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
    #     cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    #     return cond_mask

    # def get_hist_mask(self, observed_mask, for_pattern_mask=None):
    #     if for_pattern_mask is None:
    #         for_pattern_mask = observed_mask
    #     if self.target_strategy == "mix":
    #         rand_mask = self.get_randmask(observed_mask)

    #     cond_mask = observed_mask.clone()
    #     for i in range(len(cond_mask)):
    #         mask_choice = np.random.rand()
    #         if self.target_strategy == "mix" and mask_choice > 0.5:
    #             cond_mask[i] = rand_mask[i]
    #         else:  # draw another sample for histmask (i-1 corresponds to another sample)
    #             cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
    #     return cond_mask

    def get_side_info(self, observed_tp, cond_mask):  # 这里的K代表站点个数N
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)  observed_tp shape[B,L]
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(    # 把307个站点当做特征嵌入   嵌入的输出维度为16维
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)  # 这里把mask矩阵输进去是什么意思
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid( #这个函数用于在验证过程中计算所有时间步的平均损失。
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(        #用于计算扩散模型在训练和验证过程中单个时间步的损失
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:   #for train
            t = torch.randint(0, self.num_steps, [B]).to(self.device)   #duffusion随机步长
        # 获取当前时间步的alpha系数
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        # 产生噪声
        noise = torch.randn_like(observed_data)
        
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L) #训练时调用diffmodel

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:  # cond_mask 这个地方 用于表示已经构造缺失的mask
            cond_obs = (cond_mask * observed_data).unsqueeze(1) # cond_obs==miss_data
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1) #1 - cond_mask 表示缺失处，(1 - cond_mask) * noisy_data 在缺失处加噪
            # noisy_target  在缺失处加噪后的结果
            # 将 miss_data和 在缺失处加噪后的miss_data输入进去
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        
        ## 初始化 imputed_samples 用于存储填充后的样本，形状为 (B, n_samples, K, L)
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        # 开始生成n_samples 个样本
        for i in range(n_samples):
            # print(i)
            # generate noisy observation for unconditional model
            if self.is_unconditional == True: # 这一段是产生无条件模型的
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)
            # 初始化当前样本为标准正态分布
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                # 如果模型是无条件模型，使用带噪声的观测数据和当前样本计算 diff_input
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    # 如果模型是有条件模型，使用条件观测数据和当前样本计算 diff_input
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                # # 使用扩散模型预测下一步  inference（推理时）调用diffmodel
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                # 计算逆扩散步骤的系数
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                ## 更新当前样本
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                ## 如果还没有到最后一步，添加噪声
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            # 将生成的样本保存到 imputed_samples 中  (B, n_samples, K, L)  [:, i]  表示第一维全取，第二维取第i处
            imputed_samples[:, i] = current_sample.detach() 
            #current_sample是在当前循环迭代中生成的样本张量
            # 方法用于从当前计算图中分离张量。这意味着 current_sample.detach() 返回的张量不再具有梯度信息，不会参与反向传播
            # 使用 detach() 的原因是防止在后续操作中计算图变得过于复杂或者意外地计算梯度。
            
            
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch, is_train)   # 你看给你牛的，你真会写python
        
        cond_mask = gt_mask
        
        
        # 这个地方是不是应该注释掉   cond_mask 就是ob_mask??
        # if self.target_strategy != "random":
        #     cond_mask = self.get_hist_mask(
        #         observed_mask, for_pattern_mask=for_pattern_mask
        #     )
        # else:
        #     cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch,0)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation 交通数据集不需要
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

class CSDI_Traffic(CSDI_base):
    def __init__(self, config,  target_dim,device):
        super(CSDI_Traffic, self).__init__(target_dim, config,device)
        self.device=device

    def process_data(self, batch,is_train=1):
        observed_data = batch["observed_data"].to(self.device).float() #此处为[B,L,N]
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        # 交换2 3维
        observed_data = observed_data.permute(0, 2, 1) #交换之后为[B,N,L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )