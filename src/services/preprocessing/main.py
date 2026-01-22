import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 环境变量
INPUT_DIR = os.environ.get('PREPROCESS_INPUT_DIR')
OUTPUT_DIR = os.environ.get('PREPROCESS_OUTPUT_DIR')
FILE_PATTERN = os.environ.get('PREPROCESS_FILE_PATTERN', '*.nc')
VAR_NAME = os.environ.get('PREPROCESS_VARIABLE', 'sst')

def preproces():
    """使用 services/preprocessing 模块运行完整的预处理流程"""
    from pipeline import run_preprocessing_pipeline
    if not INPUT_DIR:
        print("❌ 错误：未设置环境变量 PREPROCESS_INPUT_DIR")
        exit(1)

    if not OUTPUT_DIR:
        print("❌ 错误：未设置环境变量 PREPROCESS_OUTPUT_DIR")
        exit(1)

    # 构建配置
    config = {
        'input_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
        'file_pattern': FILE_PATTERN,
        'variable_name': VAR_NAME,
    }

    print(f"📋 配置信息:")
    print(f"   输入目录: {INPUT_DIR}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   文件模式: {FILE_PATTERN}")
    print(f"   变量名称: {VAR_NAME}")
    print()

    # 调用完整的预处理流程
    try:
        success = run_preprocessing_pipeline(config)
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

def generate_report(stats, output_path):
    """生成简化版报告（无 CNN 验证）"""
    import json
    report = f"""# 数据预处理验证报告 (基础版)

## 1. 基本信息
- **变量名**: {VAR_NAME}
- **成功处理文件数**: {stats['files_processed']}
- **失败文件数**: {stats['files_failed']}
- **输出维度**: {stats.get('output_shape', 'N/A')}

## 2. 处理结果
- **总帧数**: {stats['total_frames']}
- **输出文件**: {stats.get('output_file', 'N/A')}

## 3. 说明
本次处理使用基础预处理模式（无 CNN 验证）。
如需验证数据收敛性，请使用带 CNN 验证的完整流程。
"""
    with open(os.path.join(output_path, 'validation_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)

    with open(os.path.join(output_path, 'validation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def preprocess_simple():
    """使用 NCPreprocessor 运行简化的预处理流程（不含 CNN 验证）"""
    from nc_preprocessor import NCPreprocessor

    if not INPUT_DIR:
        print("❌ 错误：未设置环境变量 PREPROCESS_INPUT_DIR")
        exit(1)

    if not OUTPUT_DIR:
        print("❌ 错误：未设置环境变量 PREPROCESS_OUTPUT_DIR")
        exit(1)

    # 构建配置
    config = {
        'input_dir': INPUT_DIR,
        'output_dir': OUTPUT_DIR,
        'file_pattern': FILE_PATTERN,
        'variable_name': VAR_NAME,
    }

    print(f"📋 配置信息 (简化模式，无 CNN 验证):")
    print(f"   输入目录: {INPUT_DIR}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   文件模式: {FILE_PATTERN}")
    print(f"   变量名称: {VAR_NAME}")
    print()

    try:
        # 使用 NCPreprocessor 进行预处理
        preprocessor = NCPreprocessor(config)
        stats = preprocessor.run()

        if stats['files_processed'] == 0:
            print("\n❌ 预处理失败：未成功处理任何文件")
            exit(1)

        # 生成报告
        generate_report(stats, OUTPUT_DIR)

        print("\n✅ 处理完成！")
        print(f"   输出文件: {stats.get('output_file', 'N/A')}")
        print(f"   报告文件: {os.path.join(OUTPUT_DIR, 'validation_report.md')}")
        exit(0)

    except Exception as e:
        print(f"\n❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    # 添加命令行接口以选择完整或简化预处理
    import argparse
    parser = argparse.ArgumentParser(description="数据预处理服务")
    parser.add_argument('--simple', action='store_true', help="使用简化预处理流程（无 CNN 验证）")
    args = parser.parse_args()
    if args.simple:
        preprocess_simple()
    else:
        preproces()