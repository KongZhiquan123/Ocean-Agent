#!/usr/bin/env python3
"""
FNO模型训练脚本 - NavierStokes超分辨率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        """复数乘法"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 傅里叶变换
        x_ft = torch.fft.rfft2(x)
        
        # 乘以傅里叶系数
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # 逆傅里叶变换
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """傅里叶神经算子 - 2D超分辨率"""
    def __init__(self, modes1=12, modes2=12, width=32, in_channels=1, out_channels=1):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # 填充以匹配输出尺寸

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
        # x: (batch, 1, 32, 32)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)  # (batch, 3, 32, 32)
        
        x = x.permute(0, 2, 3, 1)  # (batch, 32, 32, 3)
        x = self.p(x)  # (batch, 32, 32, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, 32, 32)
        
        # 上采样到64x64
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        
        # 傅里叶层
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

        x = x.permute(0, 2, 3, 1)  # (batch, 64, 64, width)
        x = self.q(x)  # (batch, 64, 64, 1)
        x = x.permute(0, 3, 1, 2)  # (batch, 1, 64, 64)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy, indexing='ij')
        gridx = gridx.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        gridy = gridy.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        return torch.cat((gridx, gridy), dim=1)


# ==================== 数据集定义 ====================
class NSDataset(Dataset):
    """NavierStokes数据集"""
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.lr = f['lr'][:]
            self.hr = f['hr'][:]
            self.mean = f.attrs['mean']
            self.std = f.attrs['std']
    
    def __len__(self):
        return len(self.lr)
    
    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr[idx]).unsqueeze(0)  # (1, 32, 32)
        hr = torch.from_numpy(self.hr[idx]).unsqueeze(0)  # (1, 64, 64)
        return lr, hr


# ==================== 训练函数 ====================
def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for lr, hr in pbar:
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        pred = model(lr)
        loss = criterion(pred, hr)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            loss = criterion(pred, hr)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_fno_model(train_file, test_file, output_dir='./fno_output', 
                    epochs=15, batch_size=32, lr=0.001):
    """
    训练FNO模型
    
    参数:
        train_file: 训练数据H5文件
        test_file: 测试数据H5文件
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print(f"\n加载数据集...")
    train_dataset = NSDataset(train_file)
    test_dataset = NSDataset(test_file)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print(f"\n创建FNO模型...")
    model = FNO2d(modes1=12, modes2=12, width=32, in_channels=1, out_channels=1).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练循环
    print(f"\n开始训练 ({epochs} 轮)...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, test_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"\n训练损失: {train_loss:.6f}")
        print(f"验证损失: {val_loss:.6f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, 'best_fno_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"✅ 保存最佳模型: {best_model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_fno_model.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, final_model_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FNO Training Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    print(f"\n✅ 训练曲线已保存: {os.path.join(output_dir, 'training_curve.png')}")
    
    print(f"\n{'='*60}")
    print(f"✅ 训练完成！")
    print(f"{'='*60}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳模型: {best_model_path}")
    print(f"最终模型: {final_model_path}")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    # 训练参数
    train_file = "./ns_dataset/ns_train.h5"
    test_file = "./ns_dataset/ns_test.h5"
    output_dir = "./fno_output"
    
    # 开始训练
    model, train_losses, val_losses = train_fno_model(
        train_file=train_file,
        test_file=test_file,
        output_dir=output_dir,
        epochs=15,
        batch_size=32,
        lr=0.001
    )
