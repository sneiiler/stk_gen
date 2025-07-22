"""
数据格式转换器

该模块用于将旧格式的JSONL文件转换为符合新数据模型的格式。
主要修复的问题：
1. target_edges中的字段名映射：'from' -> 'sat_id', 'to' -> 'target_id', 'q' -> 'quality'
2. 添加缺失的 history_clustor_result 字段
"""

import json
import copy
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.misc_utils import get_current_timestamp, get_data_dir
from tqdm import tqdm


def convert_target_edge_format(old_edge: Dict[str, Any]) -> Dict[str, Any]:
    """转换target_edge的格式
    
    Args:
        old_edge: 旧格式的边，包含 'from', 'to', 'q' 字段
        
    Returns:
        新格式的边，包含 'sat_id', 'target_id', 'quality' 字段
    """
    return {
        "sat_id": old_edge["from"],
        "target_id": old_edge["to"], 
        "quality": old_edge["q"]
    }


def convert_sat_edge_format(old_edge: Dict[str, Any]) -> Dict[str, Any]:
    """转换sat_edge的格式（如果需要的话）
    
    Args:
        old_edge: 旧格式的边
        
    Returns:
        新格式的边
    """
    # 检查是否需要转换sat_edges格式
    if "from_sat" in old_edge and "to_sat" in old_edge:
        # 已经是正确格式
        return old_edge
    elif "from" in old_edge and "to" in old_edge:
        # 需要转换格式
        return {
            "from_sat": old_edge["from"],
            "to_sat": old_edge["to"],
            "distance": old_edge.get("w", old_edge.get("distance", 0.0))  # 'w' 映射为 'distance'
        }
    else:
        return old_edge


    def convert_conversation_input(self, input_data):
        """转换会话输入数据"""
        converted = copy.deepcopy(input_data)
        
        # 转换 target_edges
        if 'target_edges' in converted:
            converted['target_edges'] = [
                self.convert_target_edge_format(edge) 
                for edge in converted['target_edges']
            ]
        
        # 转换 sat_edges 
        if 'sat_edges' in converted:
            converted['sat_edges'] = [
                self.convert_sat_edge_format(edge) 
                for edge in converted['sat_edges']
            ]
        
        # 添加 history_clustor_result 字段如果不存在
        if 'history_clustor_result' not in converted:
            converted['history_clustor_result'] = []
            
        return converted
    
    def convert_conversation_output(self, output_data):
        """转换会话输出数据 - 添加timestamp字段到clusters"""
        converted = copy.deepcopy(output_data)
        
        # 如果有clusters字段，为每个cluster添加timestamp
        if 'clusters' in converted:
            for cluster in converted['clusters']:
                if 'timestamp' not in cluster:
                    # 使用默认时间戳
                    cluster['timestamp'] = "2025-06-29T02:29:00Z"
        
        return converted


def convert_conversation_input(input_data):
    """转换会话输入数据"""
    converted = copy.deepcopy(input_data)
    
    # 转换 target_edges
    if 'target_edges' in converted:
        converted['target_edges'] = [
            convert_target_edge_format(edge) 
            for edge in converted['target_edges']
        ]
    
    # 转换 sat_edges 
    if 'sat_edges' in converted:
        converted['sat_edges'] = [
            convert_sat_edge_format(edge) 
            for edge in converted['sat_edges']
        ]
    
    # 添加 history_clustor_result 字段如果不存在
    if 'history_clustor_result' not in converted:
        converted['history_clustor_result'] = []
        
    return converted

def convert_conversation_output(output_data):
    """转换会话输出数据 - 添加timestamp字段到clusters"""
    converted = copy.deepcopy(output_data)
    
    # 如果有clusters字段，为每个cluster添加timestamp
    if 'clusters' in converted:
        for cluster in converted['clusters']:
            if 'timestamp' not in cluster:
                # 使用默认时间戳
                cluster['timestamp'] = "2025-06-29T02:29:00Z"
    
    return converted

def convert_assistant_message(content):
    """转换assistant消息格式，为clusters添加timestamp"""
    # 检查是否包含 <think> 格式
    think_start = content.find('<think>')
    think_end = content.find('</think>')
    
    if think_start != -1 and think_end != -1:
        # 提取思考部分和JSON部分
        think_part = content[think_start:think_end + 8]
        json_part = content[think_end + 8:].strip()
        
        try:
            # 解析clusters数据
            clusters_data = json.loads(json_part)
            if isinstance(clusters_data, list):
                # 为每个cluster添加timestamp
                for cluster in clusters_data:
                    if 'timestamp' not in cluster:
                        cluster['timestamp'] = "2025-06-29T02:29:00Z"
                
                # 重新组装消息
                updated_json = json.dumps(clusters_data, ensure_ascii=False, separators=(',', ':'))
                return think_part + updated_json
            else:
                print(f"    警告: assistant消息中的JSON不是数组格式")
                return content
        except json.JSONDecodeError as e:
            print(f"    警告: 无法解析assistant消息中的JSON: {e}")
            return content
    else:
        # 如果不是think格式，尝试直接解析为SatelliteClusterOutput格式
        try:
            output_data = json.loads(content)
            converted = convert_conversation_output(output_data)
            return json.dumps(converted, ensure_ascii=False)
        except json.JSONDecodeError:
            print(f"    assistant消息不是有效的JSON格式")
            return content


def convert_jsonl_file(input_file: Path, output_file: Path) -> None:
    """转换整个JSONL文件的格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"🔄 开始转换文件: {input_file}")
    print(f"📁 输出文件: {output_file}")
    
    converted_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        # 读取所有行来显示进度
        lines = infile.readlines()
        
        for line_num, line in enumerate(tqdm(lines, desc="转换数据"), 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 转换输入数据格式
                if "input" in data:
                    print(f"  处理第 {line_num} 行: 直接input格式")
                    data["input"] = convert_conversation_input(data["input"])
                elif "messages" in data:
                    # ShareGPT格式，需要从user message中提取input
                    print(f"  处理第 {line_num} 行: ShareGPT格式")
                    for message in data["messages"]:
                        if message["role"] == "user":
                            try:
                                # 解析user message中的JSON内容
                                user_input = json.loads(message["content"])
                                # 转换格式
                                converted_input = convert_conversation_input(user_input)
                                # 更新message内容
                                message["content"] = json.dumps(converted_input, ensure_ascii=False)
                                print(f"    已转换user message内容")
                            except json.JSONDecodeError:
                                print(f"    无法解析user message为JSON")
                                pass
                        elif message["role"] == "assistant":
                            # 转换assistant message（添加timestamp到clusters）
                            print(f"    处理assistant message，长度: {len(message['content'])}")
                            converted_content = convert_assistant_message(message["content"])
                            message["content"] = converted_content
                            print(f"    已转换assistant message内容，新长度: {len(converted_content)}")
                
                # 写入转换后的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except Exception as e:
                print(f"❌ 第 {line_num} 行转换失败: {e}")
                error_count += 1
                # 可以选择跳过错误的行，或者写入原始数据
                continue
    
    print(f"✅ 转换完成!")
    print(f"   成功转换: {converted_count} 条记录")
    print(f"   转换失败: {error_count} 条记录")


def validate_converted_data(file_path: Path, sample_size: int = 5) -> None:
    """验证转换后的数据格式是否正确
    
    Args:
        file_path: 转换后的文件路径
        sample_size: 验证的样本数量
    """
    print(f"🔍 验证转换后的数据格式...")
    
    try:
        # 直接检查转换后的数据结构
        print(f"📋 检查前 {sample_size} 条数据的结构...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                    
                data = json.loads(line.strip())
                print(f"第 {i+1} 条数据:")
                
                if "input" in data:
                    input_data = data["input"]
                    
                    # 检查target_edges
                    if "target_edges" in input_data and input_data["target_edges"]:
                        first_target_edge = input_data["target_edges"][0]
                        print(f"  target_edges[0]: {first_target_edge}")
                        
                    # 检查sat_edges
                    if "sat_edges" in input_data and input_data["sat_edges"]:
                        first_sat_edge = input_data["sat_edges"][0]
                        print(f"  sat_edges[0]: {first_sat_edge}")
                        
                    # 检查history_clustor_result
                    print(f"  history_clustor_result存在: {'history_clustor_result' in input_data}")
                
                if i == 0:  # 只显示第一条的详细信息
                    break
        
        print("✅ 数据结构检查完成")
        
        # 尝试使用 Pydantic 模型验证
        print("🔍 尝试 Pydantic 模型验证...")
        try:
            from misc_tools.sharegpt_utils import load_sharegpt_data
            from data_classes.sft_data_models import LLMConversationMessage
            
            # 创建临时文件进行验证
            temp_file = file_path.parent / f"temp_validation_{get_current_timestamp()}.jsonl"
            
            # 只取第一条数据进行验证
            with open(file_path, 'r', encoding='utf-8') as infile, \
                 open(temp_file, 'w', encoding='utf-8') as outfile:
                first_line = infile.readline()
                outfile.write(first_line)
            
            # 尝试加载数据
            loaded_data: List[LLMConversationMessage] = load_sharegpt_data(temp_file)
            print(f"✅ Pydantic 验证成功! 成功加载 {len(loaded_data)} 条样本数据")
            
            # 清理临时文件
            if temp_file.exists():
                temp_file.unlink()
                
        except Exception as e:
            print(f"❌ Pydantic 验证失败: {e}")
            # 清理临时文件
            temp_file = file_path.parent / f"temp_validation_{get_current_timestamp()}.jsonl"
            if temp_file.exists():
                temp_file.unlink()
        
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")


if __name__ == "__main__":
    timestamp = get_current_timestamp()
    
    # 输入文件路径
    input_file = get_data_dir() / "training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3.jsonl"
    
    # 输出文件路径（添加时间戳和格式转换标记）
    output_file = get_data_dir() / f"training_data_sharegpt_gemini-2.5-pro_20250629_103625_30_v3_converted_{timestamp}.jsonl"
    
    # 检查输入文件是否存在
    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        sys.exit(1)
    
    print("🚀 开始数据格式转换...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("=" * 80)
    
    # 执行转换
    convert_jsonl_file(input_file, output_file)
    
    print("=" * 80)
    
    # 验证转换结果
    validate_converted_data(output_file)
    
    print(f"✅ 数据格式转换完成！")
    print(f"📁 转换后的文件: {output_file}")
    print(f"💡 现在可以使用转换后的文件运行验证器了:")
    print(f"   python distill/_3_ClusterDataValidator.py")
