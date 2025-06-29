#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将卫星观测数据转换为prompt格式的脚本
Convert satellite observation data to prompt format
"""

import json
import os
from datetime import datetime
from utils.misc_utils import get_data_dir, get_current_timestamp  # 导入获取data目录和时间戳的工具函数
from utils.prompt_template import get_prompt_template

def create_prompt_template(data_sample):
    """
    根据数据样本创建prompt模板
    Create prompt template based on data sample
    """
    
    prompt_template = f"""{get_prompt_template()}
```json
{json.dumps(data_sample, ensure_ascii=False, indent=2)}
```
"""

    return prompt_template

def process_data_file(input_file, output_dir):
    """
    处理数据文件并生成prompt文件
    Process data file and generate prompt files
    """
    
    # 读取JSON数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成单个输出文件名
    timestamp = get_current_timestamp()
    output_filename = f"prompts_{timestamp}.md"
    output_file = os.path.join(output_dir, output_filename)
    
    # 将所有prompt写入一个文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, data_sample in enumerate(data):
            # 生成prompt
            prompt = create_prompt_template(data_sample)
            
            # 添加带编号的分隔符
            if i > 0:
                f.write("\n" + "="*80 + "\n")
                f.write(f"=== Prompt #{i+1} ===\n")
                f.write("="*80 + "\n\n")
            
            # 写入prompt
            f.write(prompt)
            
            print(f"已处理样本 {i+1}/{len(data)}")
    
    print(f"\n总共生成了 {len(data)} 个prompt到文件: {output_file}")

def main():
    """主函数 Main function"""
    
    # 输入文件
    input_file =  get_data_dir() / "mock_satellite_observation_data_20250625_225448.json"
    
    # 输出目录
    output_dir = get_data_dir() / "prompts"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 处理数据文件
    process_data_file(input_file, output_dir)

if __name__ == "__main__":
    main() 