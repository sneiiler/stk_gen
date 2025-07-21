#!/usr/bin/env python3
"""
一次性代码：转换assistant回复中的数据结构
为ClusterInfo添加timestamp字段，使其符合新的数据模型定义
"""

import json
import re
from typing import Dict, Any, List

def extract_timestamp_from_user_content(user_content: str) -> str:
    """从user content中提取timestamp"""
    try:
        data = json.loads(user_content)
        return data.get('timestamp', '')
    except json.JSONDecodeError:
        return ''

def convert_cluster_info(cluster_list: List[Dict], timestamp: str) -> List[Dict]:
    """为cluster info添加timestamp字段"""
    converted_clusters = []
    for cluster in cluster_list:
        # 创建新的cluster对象，添加timestamp字段
        new_cluster = {
            'timestamp': timestamp,
            'cluster_id': cluster.get('cluster_id'),
            'master': cluster.get('master'),
            'sats': cluster.get('sats', []),
            'targets': cluster.get('targets', [])
        }
        converted_clusters.append(new_cluster)
    return converted_clusters

def extract_and_convert_assistant_response(assistant_content: str, timestamp: str) -> str:
    """提取并转换assistant回复中的JSON数据"""
    # 查找<think>标签
    think_match = re.search(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        # 提取think标签后的JSON数据
        json_part = assistant_content[think_match.end():].strip()
    else:
        # 如果没有think标签，整个内容就是JSON
        think_content = ""
        json_part = assistant_content.strip()
    
    try:
        # 解析JSON数据
        cluster_data = json.loads(json_part)
        
        # 如果是列表格式，直接转换
        if isinstance(cluster_data, list):
            converted_clusters = convert_cluster_info(cluster_data, timestamp)
        # 如果是对象格式且包含clusters字段
        elif isinstance(cluster_data, dict) and 'clusters' in cluster_data:
            converted_clusters = convert_cluster_info(cluster_data['clusters'], timestamp)
        else:
            # 无法识别的格式，返回原内容
            return assistant_content
        
        # 重新构建assistant回复
        if think_content:
            new_content = f"<think>{think_content}</think>" + json.dumps(converted_clusters, ensure_ascii=False, separators=(',', ':'))
        else:
            new_content = json.dumps(converted_clusters, ensure_ascii=False, separators=(',', ':'))
        
        return new_content
        
    except json.JSONDecodeError:
        # JSON解析失败，返回原内容
        return assistant_content

def process_jsonl_file(input_file: str, output_file: str):
    """处理jsonl文件，转换assistant回复"""
    processed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            total_count += 1
            
            try:
                # 解析每一行的JSON
                record = json.loads(line)
                
                # 检查是否有messages字段
                if 'messages' in record:
                    # 提取user消息中的timestamp
                    timestamp = ""
                    user_content = ""
                    
                    for message in record['messages']:
                        if message.get('role') == 'user':
                            user_content = message['content']
                            timestamp = extract_timestamp_from_user_content(user_content)
                            break
                    
                    # 转换assistant回复
                    for message in record['messages']:
                        if message.get('role') == 'assistant':
                            original_content = message['content']
                            converted_content = extract_and_convert_assistant_response(original_content, timestamp)
                            
                            # 如果内容发生了变化，更新消息
                            if converted_content != original_content:
                                message['content'] = converted_content
                                processed_count += 1
                                print(f"已处理第 {total_count} 行的assistant回复")
                
                # 写入转换后的记录
                outfile.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"第 {total_count} 行JSON解析错误: {e}")
                # 写入原始行
                outfile.write(line + '\n')
    
    print(f"\n转换完成!")
    print(f"总共处理了 {total_count} 行")
    print(f"成功转换了 {processed_count} 个assistant回复")

def main():
    input_file = "data/training_data/training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.1_simplified.jsonl"
    output_file = "data/training_data/training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.1_simplified.jsonl"
    
    print("开始转换assistant回复中的数据结构...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    process_jsonl_file(input_file, output_file)

if __name__ == "__main__":
    main()
