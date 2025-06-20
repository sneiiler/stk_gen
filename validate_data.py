import os
from utils.misc_utils import validate_jsonl_file

def main():
    """
    Main function to validate a JSONL file.
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建文件的绝对路径
    file_to_validate = os.path.join(current_dir, 'data', 'distilled_training_data_v20250618_think.jsonl')

    print(f"开始校验文件: {file_to_validate}")

    if not os.path.exists(file_to_validate):
        print(f"错误: 文件不存在 -> {file_to_validate}")
        return

    is_valid, error_lines = validate_jsonl_file(file_to_validate)

    if is_valid:
        print("文件格式完全正确。")
    else:
        print(f"文件格式有误，以下是出错的行号: {error_lines}")
        print(f"总共有 {len(error_lines)} 行错误。")

if __name__ == "__main__":
    main() 