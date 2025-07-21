#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性脚本：处理JSONL文件中assistant的content内容
- 删除"思考过程总结:"前面的英文内容
- 将<think>替换为<think>\n
- 将</think>后面的内容用<answer></answer>包裹
"""

import json
import re
import sys
from pathlib import Path

def process_assistant_content(content):
    """
    处理assistant的content内容
    """
    # 1. 找到<think>标签和"思考过程总结:"的位置
    think_pattern = r'<think>'
    summary_pattern = r'思考过程总结:'

    think_match = re.search(think_pattern, content)
    summary_match = re.search(summary_pattern, content)

    if think_match and summary_match:
        # 保留<think>标签，但删除<think>和"思考过程总结:"之间的英文内容
        before_think = content[:think_match.start()]
        think_tag = '<think>\n'
        from_summary = content[summary_match.start():]
        content = before_think + think_tag + from_summary
    elif summary_match:
        # 如果没有<think>标签，但有"思考过程总结:"，则添加<think>标签
        before_summary = content[:summary_match.start()]
        from_summary = content[summary_match.start():]
        content = before_summary + '<think>\n' + from_summary
    else:
        # 如果都没有，只处理<think>标签
        content = re.sub(r'<think>', '<think>\n', content)

    # 2. 处理</think>后面的内容，用<answer></answer>包裹
    think_end_pattern = r'</think>(.*?)$'
    match = re.search(think_end_pattern, content, re.DOTALL)

    if match:
        after_think = match.group(1).strip()
        if after_think:
            # 替换</think>后面的内容
            content = re.sub(think_end_pattern, f'</think>\n\n<answer>\n{after_think}\n</answer>', content, flags=re.DOTALL)

    return content

def process_jsonl_file(input_file, output_file):
    """
    处理JSONL文件
    """
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 检查是否有messages字段
                if 'messages' not in data:
                    outfile.write(line)
                    continue
                
                # 处理每个message
                modified = False
                for message in data['messages']:
                    if message.get('role') == 'assistant' and 'content' in message:
                        original_content = message['content']
                        processed_content = process_assistant_content(original_content)
                        
                        if processed_content != original_content:
                            message['content'] = processed_content
                            modified = True
                            processed_count += 1
                
                # 写入处理后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                if modified:
                    print(f"处理第 {line_num} 行")
                    
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                outfile.write(line)  # 保持原样
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")
                outfile.write(line)  # 保持原样
    
    return processed_count

def main():
    input_file = "data/training_data/training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.1_simplified.jsonl"
    output_file = "data/training_data/training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.1_simplified.jsonl"
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误：输入文件 {input_file} 不存在")
        sys.exit(1)
    
    print(f"开始处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    try:
        processed_count = process_jsonl_file(input_file, output_file)
        print(f"\n处理完成！")
        print(f"共处理了 {processed_count} 条assistant消息")
        print(f"输出文件: {output_file}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
