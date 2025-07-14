#!/usr/bin/env python3
"""
将OmniThought格式的JSONL文件转换为ShareGPT格式
支持大文件的高效处理，避免内存溢出
"""

import json
import argparse
import os
from typing import Dict, Any, Iterator
import sys


def read_jsonl_efficiently(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    高效读取JSONL文件，逐行处理避免内存溢出
    
    Args:
        file_path: JSONL文件路径
        
    Yields:
        Dict: 每行的JSON对象
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {e}", file=sys.stderr)
                continue


def convert_omnithought_to_sharegpt(omnithought_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个OmniThought记录转换为ShareGPT格式

    Args:
        omnithought_record: OmniThought格式的记录

    Returns:
        Dict: ShareGPT格式的记录
    """
    # 提取question作为user输入
    question = omnithought_record.get('question', '').strip()

    # 提取full_response作为assistant输出，如果没有则使用solution字段
    full_response = omnithought_record['reasoning'][0].get('full_response', '').strip()
    if not full_response:
        full_response = omnithought_record.get('solution', '').strip()

    # 构建ShareGPT格式的消息列表
    messages = [
        {"role":"system","content":""},
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": full_response
        }
    ]

    return {"messages": messages}


def convert_file(input_path: str, output_path: str) -> None:
    """
    转换整个文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    print(f"开始转换文件: {input_path}")
    print(f"输出文件: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    converted_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for record in read_jsonl_efficiently(input_path):
            # 检查必要字段
            if 'question' not in record:
                print(f"警告: 记录缺少question字段，跳过", file=sys.stderr)
                continue

            if 'reasoning' not in record and 'full_response' not in record['reasoning'][0]['full_response']:
                print(f"警告: 记录缺少full_response和solution字段，跳过", file=sys.stderr)
                continue
            
            # 转换格式
            sharegpt_record = convert_omnithought_to_sharegpt(record)
            
            # 写入输出文件
            json.dump(sharegpt_record, out_f, ensure_ascii=False, separators=(',', ':'))
            out_f.write('\n')
            
            converted_count += 1
            
            # 每处理1000条记录打印进度
            if converted_count % 1000 == 0:
                print(f"已处理 {converted_count} 条记录...")
    
    print(f"转换完成! 共处理 {converted_count} 条记录")


def main():
    """主函数"""
    input_file = "./data/OmniThought-0528-sample.jsonl"
    output_path = "./data/OmniThought-0528-sample_sharegpt.jsonl"
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}", file=sys.stderr)
        sys.exit(1)
    

    try:
        convert_file(input_file, output_path)
    except Exception as e:
        print(f"转换过程中发生错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
