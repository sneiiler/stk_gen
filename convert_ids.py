#!/usr/bin/env python3
"""
将STK访问结果数据中的ID格式进行转换：
- 卫星ID：数字 -> Satellitexx
- 目标ID：数字 -> mxx
"""

import json
import os
from typing import Dict, Any

def convert_satellite_id(sat_id: int) -> str:
    """将卫星ID从数字转换为Satellitexx格式"""
    return f"Satellite{sat_id:02d}"

def convert_target_id(target_id: int) -> str:
    """将目标ID从数字转换为mxx格式"""
    return f"m{target_id:02d}"

def convert_sat_attrs(sat_attrs: list) -> list:
    """转换sat_attrs中的卫星ID"""
    converted = []
    for attr in sat_attrs:
        new_attr = attr.copy()
        new_attr["id"] = convert_satellite_id(attr["id"])
        converted.append(new_attr)
    return converted

def convert_sat_edges(sat_edges: list) -> list:
    """转换sat_edges中的卫星ID"""
    converted = []
    for edge in sat_edges:
        new_edge = edge.copy()
        new_edge["from_sat"] = convert_satellite_id(edge["from_sat"])
        new_edge["to_sat"] = convert_satellite_id(edge["to_sat"])
        converted.append(new_edge)
    return converted

def convert_target_edges(target_edges: list) -> list:
    """转换target_edges中的卫星ID和目标ID"""
    converted = []
    for edge in target_edges:
        new_edge = edge.copy()
        new_edge["sat_id"] = convert_satellite_id(edge["sat_id"])
        new_edge["target_id"] = convert_target_id(edge["target_id"])
        converted.append(new_edge)
    return converted

def convert_jsonl_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """转换单条JSONL记录"""
    converted_record = {
        "timestamp": record["timestamp"]
    }
    
    # 转换sat_attrs
    if "sat_attrs" in record:
        converted_record["sat_attrs"] = convert_sat_attrs(record["sat_attrs"])
    
    # 转换sat_edges
    if "sat_edges" in record:
        converted_record["sat_edges"] = convert_sat_edges(record["sat_edges"])
    
    # 转换target_edges
    if "target_edges" in record:
        converted_record["target_edges"] = convert_target_edges(record["target_edges"])
    
    return converted_record

def convert_file(input_file: str, output_file: str):
    """转换整个JSONL文件"""
    print(f"开始转换文件: {input_file}")
    
    converted_records = []
    record_count = 0
    
    # 读取并转换每一行
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                converted_record = convert_jsonl_record(record)
                converted_records.append(converted_record)
                record_count += 1
                
                if record_count % 100 == 0:
                    print(f"已处理 {record_count} 条记录...")
                    
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")
                continue
    
    # 写入转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in converted_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"转换完成！共处理 {record_count} 条记录")
    print(f"输出文件: {output_file}")

def main():
    input_file = "/Users/yinkaifeng/Desktop/zhejianglab/stk_gen/data/stk_access_result_data/raw_constellation_data_scenario_1.jsonl"
    output_file = "/Users/yinkaifeng/Desktop/zhejianglab/stk_gen/data/stk_access_result_data/raw_constellation_data_scenario_1_converted.jsonl"
    
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 执行转换
    convert_file(input_file, output_file)
    
    # 显示转换前后的对比示例
    print("\n转换示例对比：")
    print("原始格式:")
    print('  卫星ID: 143 -> Satellite143')
    print('  目标ID: 1 -> m01')
    print("  距离等数值保持不变")

if __name__ == "__main__":
    main()
