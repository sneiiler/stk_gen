#!/usr/bin/env python3
"""
测试JSON解析功能的脚本
"""

import json
from data_models.sft_data_models import SatelliteClusterOutput, ClusterInfo
from langchain.output_parsers import PydanticOutputParser

def test_parser():
    """测试不同的输出格式"""
    
    # 创建输出解析器
    output_parser = PydanticOutputParser(pydantic_object=SatelliteClusterOutput)
    
    # 测试用例1：纯JSON格式
    test_case_1 = '''{
  "chain_of_thought": "根据输入数据和balanced策略，首先确定每个目标的核心观测者（观测质量q最高的卫星）。例如，目标21的核心观测者是卫星161（q=0.84），目标40的核心观测者是卫星126（q=0.8）。随后，以每个核心观测者为基础构建簇，优先选择健康度高的卫星作为主节点。例如，卫星161（health=1.0）和126（health=0.71）均满足健康度要求。接着检查星间链路，确保簇内连通性。例如，卫星126与163之间无直接链路，但卫星163与126的链路强度w=0.41较弱，因此不纳入簇。最终每个核心观测者独立成簇，因链路强度不足或健康度较低无法合并。同时，所有目标均被覆盖，无卫星重复分配。",
  "clusters": [
    {
      "cluster_id": 1,
      "master": 161,
      "sats": [161],
      "targets": [21]
    },
    {
      "cluster_id": 2,
      "master": 126,
      "sats": [126],
      "targets": [40]
    }
  ]
}'''

    # 测试用例2：被代码块包裹的JSON
    test_case_2 = '''思考过程总结为：```json
{
  "chain_of_thought": "根据输入数据和balanced策略，首先确定每个目标的核心观测者（观测质量q最高的卫星）。例如，目标21的核心观测者是卫星161（q=0.84），目标40的核心观测者是卫星126（q=0.8）。随后，以每个核心观测者为基础构建簇，优先选择健康度高的卫星作为主节点。例如，卫星161（health=1.0）和126（health=0.71）均满足健康度要求。接着检查星间链路，确保簇内连通性。例如，卫星126与163之间无直接链路，但卫星163与126的链路强度w=0.41较弱，因此不纳入簇。最终每个核心观测者独立成簇，因链路强度不足或健康度较低无法合并。同时，所有目标均被覆盖，无卫星重复分配。",
  "clusters": [
    {
      "cluster_id": 1,
      "master": 161,
      "sats": [161],
      "targets": [21]
    },
    {
      "cluster_id": 2,
      "master": 126,
      "sats": [126],
      "targets": [40]
    }
  ]
}
```'''

    # 测试用例3：带reasoning_content的情况
    test_case_3 = '''思考过程总结为：```json
{
  "chain_of_thought": "根据输入数据和balanced策略，首先确定每个目标的核心观测者（观测质量q最高的卫星）。例如，目标21的核心观测者是卫星161（q=0.84），目标40的核心观测者是卫星126（q=0.8）。随后，以每个核心观测者为基础构建簇，优先选择健康度高的卫星作为主节点。例如，卫星161（health=1.0）和126（health=0.71）均满足健康度要求。接着检查星间链路，确保簇内连通性。例如，卫星126与163之间无直接链路，但卫星163与126的链路强度w=0.41较弱，因此不纳入簇。最终每个核心观测者独立成簇，因链路强度不足或健康度较低无法合并。同时，所有目标均被覆盖，无卫星重复分配。",
  "clusters": [
    {
      "cluster_id": 1,
      "master": 161,
      "sats": [161],
      "targets": [21]
    },
    {
      "cluster_id": 2,
      "master": 126,
      "sats": [126],
      "targets": [40]
    }
  ]
}
```'''
    
    reasoning_content = "首先分析卫星和目标的关系，然后进行分簇..."

    test_cases = [
        ("纯JSON格式", test_case_1, ""),
        ("代码块包裹的JSON", test_case_2, ""),
        ("带reasoning_content的代码块JSON", test_case_3, reasoning_content)
    ]
    
    for i, (name, test_input, reasoning) in enumerate(test_cases, 1):
        print(f"\n=== 测试用例 {i}: {name} ===")
        print(f"输入: {test_input[:100]}...")
        
        try:
            # 使用output_parser解析
            parsed_output = output_parser.parse(test_input)
            
            # 提取思维链
            cot = parsed_output.chain_of_thought
            if reasoning:
                cot = reasoning.strip() + "\n思考过程总结为:" + cot
            
            print(f"✅ 解析成功!")
            print(f"思维链长度: {len(cot)} 字符")
            print(f"思维链前50字符: {cot[:50]}...")
            print(f"簇数量: {len(parsed_output.clusters)}")
            
            # 显示簇信息
            for cluster in parsed_output.clusters:
                print(f"  簇 {cluster.cluster_id}: 主节点={cluster.master}, 卫星={cluster.sats}, 目标={cluster.targets}")
                
        except Exception as e:
            print(f"❌ 解析失败: {str(e)}")
            print(f"错误类型: {type(e).__name__}")

if __name__ == "__main__":
    test_parser() 