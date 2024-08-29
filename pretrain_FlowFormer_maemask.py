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
import core.pretrain_maemask_datasets as datasets
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer

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

# 計算模型中可訓練參數的總數
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 定義訓練函數
def train(cfg):
    # 構建模型並使用DataParallel進行多GPU訓練
    model = nn.DataParallel(build_flowformer(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    # 如果配置中提供了檢查點，則加載模型狀態
    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    # 加載訓練數據
    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)  # 使用自動混合精度
    logger = Logger(model, scheduler, cfg)

    # 設置訓練循環
    should_keep_training = True
    while should_keep_training:

        # 遍歷訓練數據集
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            # 將圖像數據移動到GPU
            image1, image2, mask = [x.cuda() for x in data_blob]

            # 如果配置中設置了添加噪聲，則對圖像添加噪聲
            if cfg.add_noise:
                #print("[Adding noise]")
                stdv = np.random.uniform(0.0, 5.0)  # 隨機噪聲強度
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            output = {}  # 初始化模型輸出字典
            loss = model(image1, image2, mask=mask, output=output)  # 前向傳播計算損失
            loss = loss.mean()
            metrics = {"loss": loss.item()}  # 儲存損失值

            # 使用自動混合精度進行反向傳播和優化
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # 更新度量指標並記錄到日誌
            metrics.update(output)
            logger.push(metrics)

            total_steps += 1  # 增加總步數
            
            # 如果達到設定的最大步數，結束訓練
            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    # 關閉記錄器並保存模型
    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    return PATH

# 主函數，解析命令行參數並調用訓練函數
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowformer', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    args = parser.parse_args()

    # 根據不同的數據集階段導入配置
    if args.stage == 'youtube':
        from configs.pretrain_config import get_cfg

    # 加載配置，更新配置並處理
    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    # 設置隨機種子
    torch.manual_seed(1234)
    np.random.seed(1234)

    # 開始訓練
    train(cfg)
