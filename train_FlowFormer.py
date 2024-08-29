from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core import optimizer
import core.datasets as datasets
from core.loss import sequence_loss  # 導入損失函數
from core.loss import sequence_loss_smooth  # 導入平滑L1損失函數
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer  # 導入FlowFormer模型構建函數

# 嘗試導入自動混合精度（AMP）的GradScaler，如果PyTorch版本低於1.6則定義一個空的GradScaler類
try:
    from torch.cuda.amp import GradScaler
except:
    # 為 PyTorch < 1.6 定義一個假的 GradScaler 類
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# 跳過加載極大的位移參數
def on_load_checkpoint(state_dict, model_state_dict):
    is_changed = False
    # 遍歷檢查點的狀態字典
    for k in state_dict:
        if k in model_state_dict:
            # 如果模型中的參數形狀和檢查點不一致，則跳過加載
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]
                is_changed = True
    return state_dict

# 計算模型中可訓練參數的總數
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 訓練函數
def train(cfg):

    loss_func = sequence_loss  # 設置默認的損失函數
    if cfg.use_smoothl1:
        print("[Using smooth L1 loss]")  # 如果使用平滑L1損失，則設置為平滑L1損失
        loss_func = sequence_loss_smooth

    model = nn.DataParallel(build_flowformer(cfg))  # 使用DataParallel進行多GPU訓練
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))  # 打印模型參數數量

    # 如果配置中提供了檢查點，則加載模型狀態
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        #checkpoint = torch.load(cfg.restore_ckpt)
        #checkpoint = on_load_checkpoint(checkpoint, model.state_dict())
        #model.load_state_dict(checkpoint, strict=False)
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()  # 將模型移動到GPU
    model.train()  # 設置模型為訓練模式

    #if args.stage != 'chairs':
    #    model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(cfg)  # 加載訓練數據
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)  # 獲取優化器和學習率調度器

    total_steps = 0  # 訓練的總步數
    scaler = GradScaler(enabled=cfg.mixed_precision)  # 使用自動混合精度
    logger = Logger(model, scheduler, cfg)  # 初始化日誌記錄器

    #add_noise = True

    should_keep_training = True  # 訓練標誌位
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度歸零
            image1, image2, flow, valid = [x.cuda() for x in data_blob]  # 將數據移動到GPU

            # 如果配置中設置了添加噪聲，則對圖像添加噪聲
            if cfg.add_noise:
                #print("[Adding noise]")
                stdv = np.random.uniform(0.0, 5.0)  # 隨機噪聲強度
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            output = {}  # 初始化模型輸出字典
            flow_predictions = model(image1, image2, output)  # 前向傳播
            loss, metrics = loss_func(flow_predictions, flow, valid, cfg)  # 計算損失
            scaler.scale(loss).backward()  # 使用自動混合精度進行反向傳播
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)  # 梯度裁剪
            
            scaler.step(optimizer)  # 更新參數
            scheduler.step()  # 更新學習率
            scaler.update()  # 更新自動混合精度縮放器

            metrics.update(output)  # 更新度量指標
            logger.push(metrics)  # 推送日誌
            
            total_steps += 1  # 增加總步數

            # 如果達到設定的最大步數，結束訓練
            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()  # 關閉日誌記錄器
    PATH = cfg.log_dir + '/final'  # 保存模型路徑
    torch.save(model.state_dict(), PATH)  # 保存模型狀態字典

    return PATH

# 主函數，解析命令行參數並調用訓練函數
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowformer', help="name your experiment")  # 添加實驗名稱參數
    parser.add_argument('--stage', help="determines which dataset to use for training")  # 設置訓練數據集階段參數
    parser.add_argument('--validation', type=str, nargs='+')  # 設置驗證數據集參數
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')  # 是否使用混合精度

    args = parser.parse_args()

    # 根據不同的數據集階段導入配置
    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg 

    # 加載配置，更新配置並處理
    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")  # 添加日誌文件
    loguru_logger.info(cfg)

    # 設置隨機種子
    torch.manual_seed(1234)
    np.random.seed(1234)

    # 如果檢查點文件夾不存在，則創建
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # 開始訓練
    train(cfg)
