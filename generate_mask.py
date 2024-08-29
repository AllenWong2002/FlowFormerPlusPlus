import torch
import random
import numpy as np

# 設定原始圖像的高度和寬度
H1, W1 = 46, 62
# 設定降取樣後的圖像塊的高度和寬度
H2, W2 = 6, 8

# 定義網格大小的範圍（最小和最大）
grid_low = 4
grid_high = 15

# 進行100,000次的循環
for idx in range(100000):
    # 每1000次循環打印一次當前的索引
    if idx % 1000 == 0:
        print(idx)
    
    # 隨機生成網格的高度和寬度
    gh = random.randint(grid_low, grid_high)
    gw = random.randint(grid_low, grid_high)

    # 生成降取樣的隨機噪聲數據，並調整其形狀
    down_sampled_noise = torch.rand(H1//gh+2, W1//gw+2, 1, H2*W2)
    down_sampled_noise = down_sampled_noise.repeat(1, 1, gh*gw, 1)
    
    # 調整噪聲的形狀以進行上取樣
    up_sampled_noise = down_sampled_noise.reshape(H1//gh+2,  W1//gw+2, gh, gw, H2*W2).permute(0,2,1,3,4).reshape((H1//gh+2)*gh, (W1//gw+2)*gw, H2*W2)

    # 隨機選擇裁剪區域的起始位置
    start_h = random.randint(0, (H1//gh+2)*gh-H1-1)
    start_w = random.randint(0, (W1//gw+2)*gw-W1-1)

    # 從上取樣的噪聲中裁剪出所需大小的區域
    croped = up_sampled_noise[start_h:start_h+H1, start_w:start_w+W1, :]

    # 將裁剪後的數據保存為.npy文件
    np.save('mae_mask/mask_46_62_48_{:06d}.npy'.format(idx), croped.numpy())
