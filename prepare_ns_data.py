#!/usr/bin/env python3
"""
NavierStokes数据准备脚本
生成低分辨率和高分辨率训练数据对
"""
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
import h5py
import os

def downsample_2d(data, scale_factor=2):
    """下采样2D数据"""
    if scale_factor == 1:
        return data
    
    # 使用平均池化进行下采样
    h, w = data.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    data_reshaped = data[:new_h*scale_factor, :new_w*scale_factor]
    data_reshaped = data_reshaped.reshape(new_h, scale_factor, new_w, scale_factor)
    return data_reshaped.mean(axis=(1, 3))

def prepare_ns_dataset(mat_file, output_dir, scale_factor=2, train_ratio=0.8):
    """
    准备NavierStokes超分辨率数据集
    
    参数:
        mat_file: 输入MAT文件路径
        output_dir: 输出目录
        scale_factor: 下采样因子 (2表示64->32)
        train_ratio: 训练集比例
    """
    print(f"加载数据: {mat_file}")
    data = sio.loadmat(mat_file)
    
    # 提取数据
    u = data['u']  # (1200, 64, 64, 20)
    print(f"原始数据形状: {u.shape}")
    print(f"数据范围: [{u.min():.4f}, {u.max()}:.4f]")
    
    n_samples, h, w, n_time = u.shape
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备高分辨率和低分辨率数据
    hr_data = []
    lr_data = []
    
    print(f"\n处理数据 (下采样因子: {scale_factor})...")
    for i in range(n_samples):
        for t in range(n_time):
            frame_hr = u[i, :, :, t]  # 高分辨率 (64, 64)
            frame_lr = downsample_2d(frame_hr, scale_factor)  # 低分辨率 (32, 32)
            
            hr_data.append(frame_hr)
            lr_data.append(frame_lr)
        
        if (i + 1) % 100 == 0:
            print(f"  已处理: {i + 1}/{n_samples} 样本")
    
    hr_data = np.array(hr_data, dtype=np.float32)  # (24000, 64, 64)
    lr_data = np.array(lr_data, dtype=np.float32)  # (24000, 32, 32)
    
    print(f"\n生成的数据形状:")
    print(f"  高分辨率: {hr_data.shape}")
    print(f"  低分辨率: {lr_data.shape}")
    
    # 数据归一化
    mean_val = hr_data.mean()
    std_val = hr_data.std()
    print(f"\n数据统计:")
    print(f"  均值: {mean_val:.6f}")
    print(f"  标准差: {std_val:.6f}")
    
    hr_data = (hr_data - mean_val) / std_val
    lr_data = (lr_data - mean_val) / std_val
    
    # 划分训练集和测试集
    n_total = len(hr_data)
    n_train = int(n_total * train_ratio)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {n_train} 样本")
    print(f"  测试集: {n_total - n_train} 样本")
    
    # 保存为HDF5格式
    train_file = os.path.join(output_dir, 'ns_train.h5')
    test_file = os.path.join(output_dir, 'ns_test.h5')
    
    print(f"\n保存训练集: {train_file}")
    with h5py.File(train_file, 'w') as f:
        f.create_dataset('hr', data=hr_data[train_indices])
        f.create_dataset('lr', data=lr_data[train_indices])
        f.attrs['mean'] = mean_val
        f.attrs['std'] = std_val
        f.attrs['scale_factor'] = scale_factor
    
    print(f"保存测试集: {test_file}")
    with h5py.File(test_file, 'w') as f:
        f.create_dataset('hr', data=hr_data[test_indices])
        f.create_dataset('lr', data=lr_data[test_indices])
        f.attrs['mean'] = mean_val
        f.attrs['std'] = std_val
        f.attrs['scale_factor'] = scale_factor
    
    print(f"\n✅ 数据准备完成！")
    print(f"输出目录: {output_dir}")
    
    return mean_val, std_val

if __name__ == "__main__":
    mat_file = "/home/marlon/OceanAgent/workspaces/marlon/NavierStokes/datasets/NavierStokes_V1e-5_N1200_T20.mat"
    output_dir = "./ns_dataset"
    
    prepare_ns_dataset(
        mat_file=mat_file,
        output_dir=output_dir,
        scale_factor=2,
        train_ratio=0.8
    )
