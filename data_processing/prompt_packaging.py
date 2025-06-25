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


def create_prompt_template(data_sample):
    """
    根据数据样本创建prompt模板
    Create prompt template based on data sample
    """
    
    prompt_template = f"""
忽略之前所有的对话记忆，开始新的深度推理过程。不要调用python工具进行计算，依靠纯粹逻辑推理。你是一位专攻复杂优化与图论的大模型专家，负责动态卫星集群的划分。在给定卫星属性、星间链路与对目标的观测数据后，你需将卫星划分为若干簇（Clusters），以高效完成对目标（Targets）的观测任务。

一、优化目标（Objective）

1. 最大化簇内链路强度：增大同一簇内所有 sat_edges.w 之和。
2. 最大化观测质量：增大簇所覆盖目标的 target_edges.q 之和。
3. 优先使用健康卫星：优先选取 sat_attrs.health 较高的卫星。

二、决策逻辑与约束（Constraints & Logic）

1. 主节点（Master）选择

    - 必须在簇内优先挑选健康度高且与其它成员连通度（边权之和）高的卫星。
    - 每簇仅一个主节点。
2. 分簇策略（strategy）

    - **"balanced"**：

      1. 识别所有可被观测的目标。
      2. 对每个目标，从能够观测它的卫星中（根据`target_edges`），选择观测质量`q`最高的卫星作为该目标的核心观测者。
      3. 以该核心观测者为主节点（Master）或簇成员，形成一个基础簇。此策略下，不要求所有卫星都参与组簇，保证资源有效利用。
    - **"quality"**：

      1. 按上述"balanced"生成初始簇；
      2. 进行**簇扩展**：寻找那些当前未被分配、健康度良好、且与簇内已有成员（尤其是主节点）具有高链路强度`w`的卫星，将它们吸纳进簇作为成员星。这旨在为未来的观测任务或链路中继做好准备。
    - 分簇校验：

      **尽可能少的分簇**，判断如果不同的簇之间的卫星如果存在链接关系，并且评估之后链接质量可以的情况下，应该合并不同的簇
3. 校验

    - 输出的 targets 集合必须与输入 target_edges 完全一致，无新增、无遗漏。 禁止幻觉。
    - "balanced" 模式下，每簇卫星数 ≤ 目标数；"quality" 模式下，每簇卫星数 ≤ 2×目标数。

三、输入格式
```json
{{
  "timestamp": "ISO8601 字符串",
  "strategy": "balanced" 或 "quality",
  "sat_attrs": [
    {{ "id": 整数, "health": 0–10, "pos": [浮点, 浮点, 浮点] }}
  ],
  "sat_edges": [
    {{ "from": 卫星ID, "to": 卫星ID, "w": 0–1 }}
  ],
  "target_edges": [
    {{ "from": 卫星ID, "to": 目标ID, "q": 0–1 }}
  ]
}}
````

四、输出格式

你的输出必须严格遵守以下JSON结构，并包含一个详细的思考过程。

- 思维链: 你需要完整的把你的思考过程放在标签<think>完整的思考过程在这里...</think>。尽可能详尽的描述，并确保正确，不要幻觉，不要遗漏。
- **结果**:紧跟在思维链后面，在这个部分，提供最终的JSON结果数组，你不要有额外的字符了。
- <|EOF|>: 输出截止符号

以上的输出格式必须严格遵守，否则视为无效输出，输出格式这个要求不要出现在思维链中

```json
<think>
// 详尽的思考链，逐步说明如何选主节点、分簇与校验……
</think>
[
  {{
    "cluster_id": 整数,
    "master": 卫星ID,
    "sats": [卫星ID, …],
    "targets": [目标ID, …]
  }}
]
<|EOF|>
```

请基于此模板，针对如下输入数据生成输出：

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