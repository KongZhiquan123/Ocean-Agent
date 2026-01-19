#!/usr/bin/env python3
"""
FNO超分辨率结果可视化脚本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ==================== FNO模型定义 ====================
class SpectralConv2d(nn.Module):
    """2D傅里叶卷积层"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """傅里叶神经算子 - 2D超分辨率"""
    def __init__(self, modes1=12, modes2=12, width=32, in_channels=1, out_channels=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9

        self.p = nn.Linear(in_channels + 2, self.width)  # +2 for grid coordinates
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = nn.Linear(self.width, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        
        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
        gridx = gridx.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        gridy = gridy.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        return torch.cat((gridx, gridy), dim=1)


# ==================== 可视化函数 ====================
def visualize_results(model_path, test_file, output_dir='./visualizations', num_samples=10):
    """
    可视化FNO超分辨率结果
    
    参数:
        model_path: 训练好的模型路径
        test_file: 测试数据H5文件
        output_dir: 输出目录
        num_samples: 可视化样本数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = FNO2d(modes1=12, modes2=12, width=32, in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    print(f"加载测试数据: {test_file}")
    with h5py.File(test_file, 'r') as f:
        lr_data = f['lr'][:]
        hr_data = f['hr'][:]
        mean_val = f.attrs['mean']
        std_val = f.attrs['std']
    
    print(f"测试集大小: {len(lr_data)}")
    
    # 随机选择样本
    indices = np.random.choice(len(lr_data), min(num_samples, len(lr_data)), replace=False)
    
    # 计算指标
    mse_list = []
    psnr_list = []
    
    print(f"\n生成预测结果...")
    with torch.no_grad():
        for idx in indices:
            lr = torch.from_numpy(lr_data[idx]).unsqueeze(0).unsqueeze(0).to(device)
            hr = hr_data[idx]
            
            # 预测
            pred = model(lr).cpu().numpy()[0, 0]
            
            # 反归一化
            lr_denorm = lr_data[idx] * std_val + mean_val
            hr_denorm = hr * std_val + mean_val
            pred_denorm = pred * std_val + mean_val
            
            # 计算指标
            mse = np.mean((pred - hr) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            mse_list.append(mse)
            psnr_list.append(psnr)
            
            # 可视化对比
            fig = plt.figure(figsize=(18, 5))
            gs = GridSpec(1, 5, figure=fig, wspace=0.3)
            
            # 低分辨率输入
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(lr_denorm, cmap='RdBu_r', aspect='auto')
            ax1.set_title(f'低分辨率输入\n({lr_denorm.shape[0]}×{lr_denorm.shape[1]})', fontsize=12)
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # 高分辨率真值
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(hr_denorm, cmap='RdBu_r', aspect='auto')
            ax2.set_title(f'高分辨率真值\n({hr_denorm.shape[0]}×{hr_denorm.shape[1]})', fontsize=12)
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # FNO超分结果
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(pred_denorm, cmap='RdBu_r', aspect='auto')
            ax3.set_title(f'FNO超分结果\nPSNR: {psnr:.2f} dB', fontsize=12)
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            
            # 双线性插值对比
            lr_bilinear = F.interpolate(
                torch.from_numpy(lr_denorm).unsqueeze(0).unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=True
            ).numpy()[0, 0]
            
            ax4 = fig.add_subplot(gs[0, 3])
            im4 = ax4.imshow(lr_bilinear, cmap='RdBu_r', aspect='auto')
            mse_bi = np.mean((lr_bilinear - hr_denorm) ** 2)
            psnr_bi = 10 * np.log10(np.max(hr_denorm)**2 / (mse_bi + 1e-10))
            ax4.set_title(f'双线性插值\nPSNR: {psnr_bi:.2f} dB', fontsize=12)
            ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046)
            
            # 误差图
            error = np.abs(pred_denorm - hr_denorm)
            ax5 = fig.add_subplot(gs[0, 4])
            im5 = ax5.imshow(error, cmap='hot', aspect='auto')
            ax5.set_title(f'绝对误差\nMAE: {np.mean(error):.4f}', fontsize=12)
            ax5.axis('off')
            plt.colorbar(im5, ax=ax5, fraction=0.046)
            
            plt.suptitle(f'NavierStokes FNO超分辨率对比 - 样本 {idx}', fontsize=14, fontweight='bold')
            
            save_path = os.path.join(output_dir, f'comparison_{idx:04d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  样本 {idx}: PSNR={psnr:.2f} dB, 保存至 {save_path}")
    
    # 统计结果
    print(f"\n{'='*60}")
    print(f"统计结果:")
    print(f"{'='*60}")
    print(f"平均MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
    print(f"平均PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"\n✅ 可视化完成！结果保存在: {output_dir}")
    
    # 绘制指标分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(mse_list, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('频数')
    axes[0].set_title('MSE分布')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(psnr_list, bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('PSNR (dB)')
    axes[1].set_ylabel('频数')
    axes[1].set_title('PSNR分布')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'metrics_distribution.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"指标分布图保存至: {metrics_path}")
    
    return np.mean(mse_list), np.mean(psnr_list)


if __name__ == "__main__":
    model_path = "./fno_output/best_fno_model.pth"
    test_file = "./ns_dataset/ns_test.h5"
    output_dir = "./visualizations"
    
    visualize_results(
        model_path=model_path,
        test_file=test_file,
        output_dir=output_dir,
        num_samples=10
    )
