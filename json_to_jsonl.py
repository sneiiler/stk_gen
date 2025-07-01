#!/usr/bin/env python3
import json
import os
from pathlib import Path

def convert_json_to_jsonl(input_file, output_file=None):
    """
    将JSON文件转换为JSONL格式
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str, optional): 输出JSONL文件路径，默认为None时自动生成
    
    Returns:
        str: 输出文件路径
    """
    # 如果未指定输出文件，则自动生成
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}.jsonl")
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 写入JSONL文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return output_file

if __name__ == "__main__":
    # 指定输入文件
    input_file = "/Users/yinkaifeng/Desktop/zhejianglab/stk_gen/data/distilled_training_data_v20250618_sharegpt_format_v1.json"
    
    # 转换文件
    output_file = convert_json_to_jsonl(input_file)
    
    # 输出结果
    print(f"转换完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
