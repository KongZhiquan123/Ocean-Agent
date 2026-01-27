#!/usr/bin/env python3
"""
GPU 显存计算器
用于验证 DiffSR 训练配置是否会导致显存溢出
"""

import argparse
import yaml
import subprocess
import re
from pathlib import Path


def get_gpu_memory():
    """获取 GPU 显存信息 (GB)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        # 获取第一个 GPU 的显存 (MB)
        memory_mb = int(result.stdout.strip().split('\n')[0])
        memory_gb = memory_mb / 1024
        return memory_gb
    except Exception as e:
        print(f"⚠️  无法获取 GPU 信息: {e}")
        print("💡 请手动输入 GPU 显存大小 (GB):")
        return float(input())


def calculate_activation_memory(batch_size, seq_len, height, width, hidden_dim, num_layers):
    """
    计算激活值显存占用

    Args:
        batch_size: 批次大小
        seq_len: 序列长度 (通常为 1)
        height: 输入高度
        width: 输入宽度
        hidden_dim: 隐藏层维度
        num_layers: 网络层数

    Returns:
        per_layer_gb: 单层激活显存 (GB)
        total_gb: 总激活显存 (GB)
    """
    # 每个浮点数占 4 字节 (FP32)
    bytes_per_float = 4

    # 单层激活
    per_layer_bytes = batch_size * seq_len * height * width * hidden_dim * bytes_per_float
    per_layer_gb = per_layer_bytes / (1024 ** 3)

    # 总激活 (所有层)
    total_gb = per_layer_gb * num_layers

    return per_layer_gb, total_gb


def read_model_config(config_path):
    """从 YAML 配置文件读取模型参数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 提取关键参数
    model_config = config.get('model', {})
    data_config = config.get('data', {})

    hidden_dim = model_config.get('width', model_config.get('hidden_dim', 256))

    # 尝试推断层数 (不同模型不同)
    model_name = model_config.get('name', '').lower()
    if 'fno' in model_name:
        num_layers = model_config.get('n_layers', 4)
    elif 'unet' in model_name or 'ddpm' in model_name:
        num_layers = model_config.get('num_res_blocks', 2) * 4  # 假设4个阶段
    else:
        num_layers = model_config.get('num_layers', 10)

    # 读取数据配置
    train_batch = data_config.get('train_batchsize', 8)
    eval_batch = data_config.get('eval_batchsize', train_batch)
    shape = data_config.get('shape', [128, 128])

    if isinstance(shape, list) and len(shape) >= 2:
        height, width = shape[0], shape[1]
    else:
        height = width = 128

    return {
        'train_batchsize': train_batch,
        'eval_batchsize': eval_batch,
        'height': height,
        'width': width,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'model_name': model_name,
    }


def verify_memory(config_path=None, batch_size=None, height=None, width=None,
                  hidden_dim=None, num_layers=None, gpu_memory=None):
    """验证配置是否会导致显存溢出"""

    print("=" * 60)
    print("🧮 GPU 显存计算器")
    print("=" * 60)
    print()

    # 如果提供了配置文件，从中读取参数
    if config_path:
        print(f"📂 读取配置文件: {config_path}")
        params = read_model_config(config_path)
        batch_size = batch_size or params['train_batchsize']
        height = height or params['height']
        width = width or params['width']
        hidden_dim = hidden_dim or params['hidden_dim']
        num_layers = num_layers or params['num_layers']
        print(f"✓ 模型: {params['model_name']}")
        print()

    # 获取 GPU 显存
    if gpu_memory is None:
        print("🔍 检测 GPU 显存...")
        gpu_memory = get_gpu_memory()

    print(f"✓ GPU 总显存: {gpu_memory:.2f} GB")
    print()

    # 计算可用显存 (70% 安全阈值)
    available_memory = gpu_memory * 0.7
    print(f"💡 可用显存 (70%): {available_memory:.2f} GB")
    print(f"   (预留 30% 给模型权重、梯度、优化器)")
    print()

    # 计算激活显存
    seq_len = 1  # 超分任务通常为 1
    per_layer, total = calculate_activation_memory(
        batch_size, seq_len, height, width, hidden_dim, num_layers
    )

    print("📊 配置参数:")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Spatial Resolution: {height} × {width}")
    print(f"   - Hidden Dim: {hidden_dim}")
    print(f"   - Num Layers: {num_layers}")
    print(f"   - Seq Length: {seq_len}")
    print()

    print("🔢 显存占用:")
    print(f"   - 单层激活: {per_layer:.4f} GB")
    print(f"   - 总激活显存: {total:.2f} GB")
    print()

    # 判断是否安全
    if total < available_memory:
        status = "✅ 安全"
        color_code = "\033[92m"  # 绿色
        margin = available_memory - total
        print(f"{color_code}{status}\033[0m")
        print(f"   剩余显存: {margin:.2f} GB ({margin/available_memory*100:.1f}%)")
        print()
        print("💚 配置安全，可以开始训练！")
        return True
    else:
        status = "❌ 超限"
        color_code = "\033[91m"  # 红色
        excess = total - available_memory
        print(f"{color_code}{status}\033[0m")
        print(f"   超出显存: {excess:.2f} GB")
        print()
        print("⚠️  警告: 配置可能导致 OOM (Out of Memory)")
        print()
        print("🔧 建议调整:")

        # 计算推荐的 batch size
        recommended_batch = int(batch_size * available_memory / total)
        if recommended_batch < 1:
            recommended_batch = 1
        print(f"   1. 降低 batch size: {batch_size} → {recommended_batch}")

        # 计算推荐的分辨率
        scale_factor = (available_memory / total) ** 0.5
        recommended_h = int(height * scale_factor)
        recommended_w = int(width * scale_factor)
        print(f"   2. 降低分辨率: [{height}, {width}] → [{recommended_h}, {recommended_w}]")

        print(f"   3. 使用梯度累积模拟大 batch")
        print()
        return False


def main():
    parser = argparse.ArgumentParser(description='GPU 显存计算器')
    parser.add_argument('--config', type=str, help='YAML 配置文件路径')
    parser.add_argument('--batch', type=int, help='批次大小')
    parser.add_argument('--height', type=int, help='输入高度')
    parser.add_argument('--width', type=int, help='输入宽度')
    parser.add_argument('--hidden_dim', type=int, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, help='网络层数')
    parser.add_argument('--gpu_memory', type=float, help='GPU 显存 (GB)')

    args = parser.parse_args()

    if not args.config and not all([args.batch, args.height, args.width,
                                     args.hidden_dim, args.num_layers]):
        print("❌ 错误: 请提供配置文件 (--config) 或所有参数")
        print()
        print("用法 1: 从配置文件读取")
        print("  python gpu_memory_calculator.py --config configs/fno.yaml")
        print()
        print("用法 2: 手动指定参数")
        print("  python gpu_memory_calculator.py --batch 8 --height 128 --width 128 \\")
        print("         --hidden_dim 256 --num_layers 10 --gpu_memory 24")
        return

    success = verify_memory(
        config_path=args.config,
        batch_size=args.batch,
        height=args.height,
        width=args.width,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gpu_memory=args.gpu_memory
    )

    print("=" * 60)

    # 返回退出码 (0: 成功, 1: 失败)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
