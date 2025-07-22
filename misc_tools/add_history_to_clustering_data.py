"""
为分簇数据添加历史分簇结果的脚本

该脚本读取分簇数据文件，为每个时间切片添加历史分簇数据，
使得验证器能够进行分簇稳定性验证。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from copy import deepcopy

# 添加项目根目录到路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.misc_utils import get_current_timestamp, get_data_dir


def add_history_to_clustering_data(input_file: str, output_file: Optional[str] = None):
    """
    为分簇数据添加历史分簇结果
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则自动生成
    """
    
    # 读取原始数据
    input_path = get_data_dir() / input_file
    if not input_path.exists():
        print(f"错误：输入文件 {input_path} 不存在")
        return
    
    print(f"📖 读取原始数据: {input_path}")
    conversations = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    
    print(f"✅ 共读取 {len(conversations)} 个对话")
    
    # 处理历史数据
    modified_conversations = []
    
    for i, conversation in enumerate(conversations):
        # 深拷贝对话数据
        new_conversation = deepcopy(conversation)
        
        # 解析输入数据
        user_message = None
        for msg in conversation["messages"]:
            if msg["role"] == "user":
                user_message = json.loads(msg["content"])
                break
        
        if not user_message:
            print(f"⚠️  警告：对话 {i} 没有找到用户消息")
            modified_conversations.append(new_conversation)
            continue
        
        # 如果是第一个时间切片，没有历史数据
        if i == 0:
            print(f"⏰ 处理第 {i+1} 个时间切片 ({user_message.get('timestamp', 'unknown')}): 首个切片，无历史数据")
            modified_conversations.append(new_conversation)
            continue
        
        # 获取前一个时间切片的分簇结果作为历史数据
        prev_conversation = conversations[i-1]
        prev_assistant_message = None
        for msg in prev_conversation["messages"]:
            if msg["role"] == "assistant":
                prev_assistant_message = msg["content"]
                break
        
        if not prev_assistant_message:
            print(f"⚠️  警告：对话 {i-1} 没有找到助手消息")
            modified_conversations.append(new_conversation)
            continue
        
        # 解析前一个时间切片的分簇结果
        try:
            # 移除 <think></think> 标签
            clean_content = prev_assistant_message
            if "<think>" in clean_content and "</think>" in clean_content:
                start_idx = clean_content.find("</think>") + len("</think>")
                clean_content = clean_content[start_idx:].strip()
            
            # 解析JSON
            prev_clusters = json.loads(clean_content)
            
            # 将历史分簇数据添加到当前时间切片的输入中
            for msg in new_conversation["messages"]:
                if msg["role"] == "user":
                    user_data = json.loads(msg["content"])
                    user_data["history_cluster_result"] = [prev_clusters]  # 包装成列表，支持多个历史记录
                    msg["content"] = json.dumps(user_data, ensure_ascii=False)
                    break
            
            print(f"⏰ 处理第 {i+1} 个时间切片 ({user_message.get('timestamp', 'unknown')}): 添加了 {len(prev_clusters)} 个历史分簇")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  警告：解析对话 {i-1} 的分簇结果失败: {e}")
            modified_conversations.append(new_conversation)
            continue
        
        modified_conversations.append(new_conversation)
    
    # 生成输出文件名
    if output_file is None:
        input_stem = Path(input_file).stem
        timestamp = get_current_timestamp()
        output_file = f"{input_stem}_with_history_{timestamp}.jsonl"
    
    output_path = get_data_dir() / output_file
    
    # 保存修改后的数据
    print(f"💾 保存修改后的数据: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for conversation in modified_conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    
    print(f"✅ 处理完成！")
    print(f"   输入文件: {input_path}")
    print(f"   输出文件: {output_path}")
    print(f"   处理对话数: {len(modified_conversations)}")
    
    # 统计添加历史数据的数量
    history_added_count = 0
    for i, conversation in enumerate(modified_conversations):
        if i == 0:
            continue  # 跳过第一个
        
        for msg in conversation["messages"]:
            if msg["role"] == "user":
                user_data = json.loads(msg["content"])
                if user_data.get("history_cluster_result") is not None:
                    history_added_count += 1
                break
    
    print(f"   添加历史数据的切片数: {history_added_count}")
    
    return str(output_path)


def print_sample_data(file_path: str, sample_count: int = 3):
    """
    打印样本数据，用于验证处理结果
    
    Args:
        file_path: 文件路径
        sample_count: 样本数量
    """
    print(f"\n📋 样本数据预览 (前 {sample_count} 个对话):")
    print("=" * 80)
    
    path = get_data_dir() / file_path
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_count:
                break
            
            conversation = json.loads(line)
            
            # 提取时间戳和历史数据信息
            user_message = None
            for msg in conversation["messages"]:
                if msg["role"] == "user":
                    user_data = json.loads(msg["content"])
                    timestamp = user_data.get("timestamp", "unknown")
                    history_result = user_data.get("history_cluster_result")
                    
                    print(f"对话 {i+1}:")
                    print(f"  时间戳: {timestamp}")
                    if history_result:
                        print(f"  历史分簇数: {len(history_result[0])} 个簇")
                        for j, cluster in enumerate(history_result[0]):
                            sats = cluster.get("sats", [])
                            targets = cluster.get("targets", [])
                            master = cluster.get("master", "unknown")
                            print(f"    簇{j}: master={master}, sats={sats}, targets={targets}")
                    else:
                        print(f"  历史分簇数: 无")
                    print()
                    break


if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) < 2:
        print("用法: python add_history_to_clustering_data.py <input_file> [output_file]")
        print("示例: python add_history_to_clustering_data.py clustering_results_cmax_200011.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 处理数据
    result_file = add_history_to_clustering_data(input_file, output_file)
    
    # 显示样本数据
    if result_file:
        relative_path = Path(result_file).name
        print_sample_data(relative_path, sample_count=3)
