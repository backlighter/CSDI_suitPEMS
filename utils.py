import numpy as np
import torch
from torch.optim import Adam
import nni
import time
from tqdm import tqdm
import logging
def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
):
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=1e-6)
    ## 计算第一个学习率下降的epoch，当到达总epochs的百分之75时，学习率乘以gamma
    ## 计算第二个学习率下降的epoch，当到达总epochs的百分之90时，学习率乘以gamma
    p1 = int(0.75 * int(config["epochs"]))
    p2 = int(0.9 * int(config["epochs"]))
    # 学习率调度器，当训练到milestones指定的epoch时，学习率乘以gamma
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    train_start = time.time()
    print("start...")
    for epoch_no in range(int(config["epochs"])):
        avg_loss = 0
        model.train()
        for batch_no, train_batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss = model(train_batch)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()

            if batch_no % 20 == 0:
                print(
                    "Train Epoch: ",
                    epoch_no,
                    "Batch: ",
                    batch_no,
                    "Loss: ",
                    loss.item(),
                )

        lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no, valid_batch in enumerate(valid_loader):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss.item()
                    if (batch_no % 20 == 0):
                        print(
                            "Valid Epoch: ",
                            epoch_no,
                            "Batch: ",
                            batch_no,
                            "Loss: ",
                            loss.item(),
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / (batch_no+1),
                    "at",
                    epoch_no,
                )
    train_end_time = time.time()
    print("Training time:", train_end_time - train_start)

def evaluate(model, test_loader, _std,_mean,use_nni, nsample=10):

    test_start = time.time()
    with torch.no_grad():
        model.eval()
        mse_total = 0 #初始化总均方误差
        mae_total = 0 #初始化平均绝对误差
        mape_total = 0  #初始化总平均绝对百分百误差
        evalpoints_total = 0 #初始化总评估点数
        
        
        all_generated_samples = []
        all_target = []
        all_evalpoint=[]
        all_observed_point=[]
        all_observed_time=[]
        print("START TEST...")
        scaler=_std
        mean_scaler=_mean
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it: #根据batch-size计算
            for batch_no, test_batch in enumerate(it, start=1):  #
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1).long()#(B,L,K)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler
                mape_current = torch.divide(torch.abs((samples_median.values - c_target)*scaler)
                                            ,(c_target*scaler+mean_scaler)*((c_target*scaler+mean_scaler)>(1e-4)))\
                                    .nan_to_num(posinf=0,neginf=0,nan=0)*eval_points
                                     
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mape_total += mape_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "mape_total": mape_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
                logging.info("rmse_total={}".format(np.sqrt(mse_total / evalpoints_total)))
                logging.info("mae_total={}".format(mae_total / evalpoints_total))
                logging.info("mape_total={}".format(mape_total / evalpoints_total))
                logging.info("batch_no={}".format(batch_no))
                
            
        print("RMSE:", np.sqrt(mse_total / evalpoints_total))
        print("MAE:", mae_total / evalpoints_total)
        print("MAPE:",mape_total / evalpoints_total)
            
        #     all_gt.append(c_target[eval_points].view(-1).cpu())
        #     all_imputed.append(samples_median[eval_points].view(-1).cpu())

        #     if batch_no %50 == 0:
        #         print("\ntest batch:",batch_no,"/",len(test_loader),"completed", 
        #               "\ntarget:",c_target[eval_points].view(-1)[:10]*_std+_mean,
        #               "\nimputation:",samples_median[eval_points].view(-1)[:10]*_std+_mean)
        
        # imputed = torch.cat(all_imputed, dim=0).view(-1)*_std+_mean
        # imputed[imputed<0] = 0
        # truth = torch.cat(all_gt, dim=0).view(-1)*_std+_mean

        # print(_std,_mean)
        # print(imputed,truth)

        # MAE = torch.abs(truth - imputed).mean()
        # RMSE = torch.sqrt(((truth - imputed)**2).mean())
        # MAPE = torch.divide(torch.abs(truth - imputed),truth).nan_to_num(posinf=0).mean()

        
        # if use_nni:
        #     nni.report_final_result(MAE.cpu().numpy().item())
        # print(f"MAE:{MAE.__format__('.6f')}      RMSE:{RMSE.__format__('.6f')}      MAPE:{MAPE.__format__('.6f')}")
    test_end_time = time.time()
    print("Testing time:", test_end_time - test_start)
