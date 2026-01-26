import os
import sys

# 添加 src 目录到 Python 路径，以便导入 services.preprocessing
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

from services.preprocessing.pipeline import run_preprocessing_pipeline

# 环境变量
INPUT_DIR = os.environ.get('PREPROCESS_INPUT_DIR')
OUTPUT_DIR = os.environ.get('PREPROCESS_OUTPUT_DIR')
FILE_PATTERN = os.environ.get('PREPROCESS_FILE_PATTERN', '*.nc')
VAR_NAME = os.environ.get('PREPROCESS_VARIABLE', 'sst')

def main():
    """使用 services/preprocessing 模块运行完整的预处理流程"""

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

if __name__ == "__main__":
    main()