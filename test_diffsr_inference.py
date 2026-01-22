#!/usr/bin/env python3
"""
DiffSR Inference Test Script
测试推理功能和报告生成
"""

import os
import sys
import yaml
import json
import shutil
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add DiffSR to path to import normalizer
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'src' / 'services' / 'diffsr'))

# 简单的归一化器（用于备用）
class SimpleNormalizer:
    """简单的归一化器，可被 pickle 序列化"""
    def __init__(self):
        self.mean = torch.tensor(0.0)
        self.std = torch.tensor(1.0)
        self.eps = 0.00001

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, sample_idx=None):
        return (x * (self.std + self.eps)) + self.mean

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def create_test_data(output_dir, num_samples=20, lr_size=32, hr_size=64):
    """创建简单的测试数据"""
    print_header("Step 1: 生成测试数据")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成训练、验证、测试数据
    train_samples = int(num_samples * 0.7)
    valid_samples = int(num_samples * 0.15)
    test_samples = num_samples - train_samples - valid_samples

    print_info(f"生成数据: train={train_samples}, valid={valid_samples}, test={test_samples}")
    print_info(f"分辨率: LR={lr_size}x{lr_size}, HR={hr_size}x{hr_size}")

    # 生成随机数据（模拟海洋温度场）
    train_x = torch.randn(train_samples, lr_size, lr_size, 1)  # 低分辨率
    train_y = torch.randn(train_samples, hr_size, hr_size, 1)  # 高分辨率

    valid_x = torch.randn(valid_samples, lr_size, lr_size, 1)
    valid_y = torch.randn(valid_samples, hr_size, hr_size, 1)

    test_x = torch.randn(test_samples, lr_size, lr_size, 1)
    test_y = torch.randn(test_samples, hr_size, hr_size, 1)

    # 使用 DiffSR 的 GaussianNormalizer
    try:
        from utils.normalizer import GaussianNormalizer
        # 计算所有数据的均值和标准差
        all_data = torch.cat([train_y, valid_y, test_y], dim=0)
        normalizer = GaussianNormalizer(all_data)
        print_success("使用 DiffSR GaussianNormalizer")
    except ImportError:
        print_warning("无法导入 GaussianNormalizer，使用简单归一化")
        # 使用顶层定义的 SimpleNormalizer
        normalizer = SimpleNormalizer()

    # 保存数据
    # Ocean dataset expects file named: {base}_uo_data_sf2_sr.pt
    # So we need to save with this exact naming pattern
    base_name = 'test_ocean_data'
    data_key = 'uo_data'
    sample_factor = 2
    data_filename = f'{base_name}_{data_key}_sf{sample_factor}_sr.pt'
    data_path = output_dir / data_filename
    torch.save((train_x, train_y, valid_x, valid_y, test_x, test_y, normalizer), data_path)

    print_success(f"测试数据已生成: {data_path}")
    # Return base path without the suffix (Ocean dataset will add it)
    return str(output_dir / base_name)


def create_test_config(data_path, output_dir, model_type='FNO2d', diffsr_path=None):
    """创建测试配置文件"""
    print_header("Step 2: 创建测试配置")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志目录（在 diffsr/logs 下）
    if diffsr_path is None:
        diffsr_path = Path(__file__).parent / 'src' / 'services' / 'diffsr'
    log_dir = str(Path(diffsr_path) / 'logs')

    config = {
        'model': {
            'name': model_type,  # 使用正确的模型名称，如 'FNO2d'
            'modes1': [8, 8, 8, 8],  # Required: Fourier modes in first dimension
            'modes2': [8, 8, 8, 8],  # Required: Fourier modes in second dimension
            'width': 32,
            'fc_dim': 64,
            'layers': [8, 16, 16, 16],
            'in_dim': 1,
            'out_dim': 1,
            'act': 'gelu',
            'upsample_factor': [2, 2],  # For super-resolution
        },
        'data': {
            'name': 'ocean',
            'data_path': str(data_path),  # Changed from 'path' to 'data_path'
            'data_key': 'uo_data',  # Added: required by Ocean dataset
            'shape': [64, 64],
            'sample_factor': 2,
            'train_batchsize': 4,
            'eval_batchsize': 4,
            'num_workers': 0,
            'train_ratio': 0.7,
            'valid_ratio': 0.15,
            'test_ratio': 0.15,
        },
        'train': {
            'epochs': 2,  # 快速测试，只训练2个epoch
            'eval_freq': 1,
            'patience': -1,
            'saving_best': True,
            'saving_ckpt': False,
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
        'optimize': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'weight_decay': 0.0001,
        },
        'schedule': {
            'scheduler': 'StepLR',
            'step_size': 100,
            'gamma': 0.5,
        },
        'log': {
            'log': True,
            'log_dir': log_dir,  # 添加缺失的 log_dir
            'wandb': False,
        }
    }

    config_path = output_dir / 'test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print_success(f"配置文件已创建: {config_path}")
    print_info(f"模型类型: {model_type}")
    print_info(f"训练轮数: {config['train']['epochs']} (快速测试)")
    print_info(f"设备: {config['train']['device']}")

    return str(config_path)


def run_training(config_path, diffsr_path, python_path):
    """运行快速训练"""
    print_header("Step 3: 运行快速训练")

    print_info("开始训练（这可能需要几分钟）...")
    print_info("命令: python main.py --config <config_path>")

    # 切换到 DiffSR 目录并运行训练
    original_dir = os.getcwd()
    try:
        os.chdir(diffsr_path)

        import subprocess
        result = subprocess.run(
            [python_path, 'main.py', '--config', config_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print_success("训练完成")

            # 显示训练摘要
            output_lines = result.stdout.split('\n')
            for line in output_lines[-20:]:  # 显示最后20行
                if 'Test metrics' in line or 'Valid metrics' in line:
                    print_info(line.strip())

            return True
        else:
            print_error(f"训练失败: {result.stderr}")
            return False

    finally:
        os.chdir(original_dir)


def run_inference(model_dir, output_dir, diffsr_path, python_path, forecaster_type='base'):
    """运行推理"""
    print_header("Step 4: 运行推理")

    print_info(f"模型目录: {model_dir}")
    print_info(f"Forecaster 类型: {forecaster_type}")
    print_info(f"输出目录: {output_dir}")

    # 验证模型文件存在
    model_path = Path(model_dir) / 'best_model.pth'
    config_path = Path(model_dir) / 'config.yaml'

    if not model_path.exists():
        print_error(f"模型文件不存在: {model_path}")
        return False

    if not config_path.exists():
        print_error(f"配置文件不存在: {config_path}")
        return False

    print_success("模型文件检查通过")

    # 运行推理
    print_info("开始推理...")

    original_dir = os.getcwd()
    try:
        os.chdir(diffsr_path)

        import subprocess
        result = subprocess.run(
            [python_path, 'inference.py',
             '--model_dir', str(model_dir),
             '--forecastor_type', forecaster_type,
             '--output_dir', str(output_dir),
             '--split', 'test'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print_success("推理完成")
            print(result.stdout)
            return True
        else:
            print_error(f"推理失败: {result.stderr}")
            print(result.stdout)
            return False

    finally:
        os.chdir(original_dir)


def verify_results(output_dir):
    """验证推理结果"""
    print_header("Step 5: 验证结果")

    output_dir = Path(output_dir)

    # 检查 metrics.json
    metrics_file = output_dir / 'metrics.json'
    if metrics_file.exists():
        print_success(f"找到 metrics.json: {metrics_file}")

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        print_info("指标内容:")
        print(json.dumps(metrics, indent=2))

        # 验证关键指标
        has_psnr = 'best_psnr' in metrics
        has_ssim = 'best_ssim' in metrics

        if has_psnr:
            psnr_value = metrics['best_psnr']
            is_number = isinstance(psnr_value, (int, float))

            if is_number:
                print_success(f"✓ best_psnr 是数值类型: {psnr_value:.4f}")
            else:
                print_error(f"✗ best_psnr 是 {type(psnr_value).__name__} 类型，应该是数值！")
                return False
        else:
            print_error("✗ metrics.json 中缺少 best_psnr")
            return False

        if has_ssim:
            print_success(f"✓ best_ssim: {metrics['best_ssim']:.4f}")

    else:
        print_error(f"未找到 metrics.json: {metrics_file}")
        return False

    # 检查报告
    report_file = output_dir / 'inference_report.md'
    if report_file.exists():
        print_success(f"找到推理报告: {report_file}")

        # 读取报告内容，检查是否有格式化错误
        with open(report_file, 'r', encoding='utf-8') as f:
            report_content = f.read()

        # 检查是否有 "N/A" 占位符（不应该出现在数值字段）
        if 'PSNR N/A' in report_content or 'PSNR: N/A' in report_content:
            print_warning("报告中 PSNR 显示为 N/A，可能有问题")
        else:
            print_success("✓ 报告生成正常，PSNR 已正确填充")

        # 显示报告摘要
        print_info("报告前20行:")
        lines = report_content.split('\n')
        for line in lines[:20]:
            if line.strip():
                print(f"  {line}")

    else:
        print_warning(f"未找到推理报告: {report_file}")

    return True


def main():
    print_header("DiffSR 推理功能测试")
    print_info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 配置
    script_dir = Path(__file__).parent
    diffsr_path = script_dir / 'src' / 'services' / 'diffsr'
    test_dir = script_dir / 'test_diffsr_output'

    # 获取 Python 路径
    python_path = sys.executable

    print_info(f"Python: {python_path}")
    print_info(f"DiffSR 路径: {diffsr_path}")
    print_info(f"测试输出目录: {test_dir}")

    if not diffsr_path.exists():
        print_error(f"DiffSR 目录不存在: {diffsr_path}")
        return 1

    # 清理旧的测试数据
    if test_dir.exists():
        print_warning(f"清理旧测试数据: {test_dir}")
        shutil.rmtree(test_dir)

    try:
        # Step 1: 生成测试数据
        data_path = create_test_data(test_dir / 'data', num_samples=20)

        # Step 2: 创建配置
        config_path = create_test_config(
            data_path,
            test_dir / 'config',
            model_type='FNO2d',  # 使用正确的模型名称
            diffsr_path=diffsr_path  # 传递 diffsr_path
        )

        # Step 3: 训练模型
        success = run_training(config_path, str(diffsr_path), python_path)
        if not success:
            print_error("训练失败，测试中止")
            return 1

        # 找到训练输出目录
        # 训练会在 diffsr/logs 下创建输出目录
        logs_dir = diffsr_path / 'logs'
        if not logs_dir.exists():
            print_error("未找到训练日志目录")
            return 1

        # 找到最新的模型目录
        model_dirs = sorted([d for d in logs_dir.rglob('*') if (d / 'best_model.pth').exists()],
                           key=lambda x: x.stat().st_mtime,
                           reverse=True)

        if not model_dirs:
            print_error("未找到训练好的模型")
            return 1

        model_dir = model_dirs[0]
        print_success(f"找到模型目录: {model_dir}")

        # Step 4: 运行推理
        inference_output_dir = test_dir / 'inference_results'
        success = run_inference(
            str(model_dir),
            str(inference_output_dir),
            str(diffsr_path),
            python_path,
            forecaster_type='base'
        )

        if not success:
            print_error("推理失败")
            return 1

        # Step 5: 验证结果
        success = verify_results(inference_output_dir)

        if success:
            print_header("测试成功！")
            print_success("所有测试通过")
            print_info(f"\n查看结果:")
            print_info(f"  - 指标文件: {inference_output_dir / 'metrics.json'}")
            print_info(f"  - 推理报告: {inference_output_dir / 'inference_report.md'}")
            return 0
        else:
            print_header("测试失败")
            print_error("部分检查未通过")
            return 1

    except Exception as e:
        print_error(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
