"""
Prompt templates for satellite cluster optimization.

This module contains versioned prompt templates for the satellite cluster
optimization system, specifically for dynamic satellite cluster partitioning.
"""

from pydantic import BaseModel, Field
from typing import List
from data_models.sft_data_models import RawConstellationDataModel, SatelliteClusterOutput

# Version 1.0 - Initial satellite cluster optimization prompt
SATELLITE_CLUSTER_V1 = """
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
```

四、输出格式

你的输出必须严格遵守以下JSON结构，并包含一个详细的思考过程。

- 思维链: 你需要完整的把你的思考过程放在标签<chain_of_thought>完整的思考过程在这里...</chain_of_thought>。尽可能详尽的描述，并确保正确，不要幻觉，不要遗漏。
- **结果**:紧跟在思维链后面，在这个部分，提供最终的JSON结果数组，你不要有额外的字符了。
- <|EOF|>: 输出截止符号

以上的输出格式必须严格遵守，否则视为无效输出，输出格式这个要求不要出现在思维链中

```json
<chain_of_thought>
// 详尽的思考链，逐步说明如何选主节点、分簇与校验……
</chain_of_thought>
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
"""

# Version 2.0 - Enhanced satellite cluster optimization prompt with improved constraints
SATELLITE_CLUSTER_V2 = """
忽略之前所有的对话记忆，开始新的深度推理过程。不要调用python工具进行计算，依靠纯粹逻辑推理。你是一位专攻复杂优化与图论的大模型专家，负责动态卫星集群的划分。在给定卫星属性、星间链路与对目标的观测数据后，你需将卫星划分为若干簇（Clusters），以高效完成对目标（Targets）的观测任务。

一、优化目标（Objective）

1. 最大化簇内链路强度：增大同一簇内所有 sat_edges.w 之和。
2. 最大化观测质量：增大簇所覆盖目标的 target_edges.q 之和。
3. 优先使用健康卫星：优先选取 sat_attrs.health 较高的卫星。
4. 最小化簇间干扰：减少不同簇之间的资源竞争。

二、决策逻辑与约束（Constraints & Logic）

1. 主节点（Master）选择

    - 必须在簇内优先挑选健康度高且与其它成员连通度（边权之和）高的卫星。
    - 每簇仅一个主节点。
    - 主节点应具有较高的观测能力（与目标的平均q值较高）。

2. 分簇策略（strategy）

    - **"balanced"**：

      1. 识别所有可被观测的目标。
      2. 对每个目标，从能够观测它的卫星中（根据`target_edges`），选择观测质量`q`最高的卫星作为该目标的核心观测者。
      3. 以该核心观测者为主节点（Master）或簇成员，形成一个基础簇。此策略下，不要求所有卫星都参与组簇，保证资源有效利用。
      4. 优先考虑观测质量，其次考虑链路强度。

    - **"quality"**：

      1. 按上述"balanced"生成初始簇；
      2. 进行**簇扩展**：寻找那些当前未被分配、健康度良好、且与簇内已有成员（尤其是主节点）具有高链路强度`w`的卫星，将它们吸纳进簇作为成员星。这旨在为未来的观测任务或链路中继做好准备。
      3. 扩展时优先考虑链路强度，确保簇内通信质量。

    - 分簇校验：

      **尽可能少的分簇**，判断如果不同的簇之间的卫星如果存在链接关系，并且评估之后链接质量可以的情况下，应该合并不同的簇。

3. 校验

    - 输出的 targets 集合必须与输入 target_edges 完全一致，无新增、无遗漏。 禁止幻觉。
    - "balanced" 模式下，每簇卫星数 ≤ 目标数；"quality" 模式下，每簇卫星数 ≤ 2×目标数。
    - 每个卫星最多只能属于一个簇。
    - 每个簇必须至少包含一个卫星和一个目标。

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
```

四、输出格式

你的输出必须严格遵守以下JSON结构，并包含一个详细的思考过程。

- 思维链: 你需要完整的把你的思考过程放在标签<chain_of_thought>完整的思考过程在这里...</chain_of_thought>。尽可能详尽的描述，并确保正确，不要幻觉，不要遗漏。
- **结果**:紧跟在思维链后面，在这个部分，提供最终的JSON结果数组，你不要有额外的字符了。
- <|EOF|>: 输出截止符号

以上的输出格式必须严格遵守，否则视为无效输出，输出格式这个要求不要出现在思维链中

```json
<chain_of_thought>
// 详尽的思考链，逐步说明如何选主节点、分簇与校验……
</chain_of_thought>
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
"""

# Version 3.0 - Advanced satellite cluster optimization with fault tolerance
SATELLITE_CLUSTER_V3 = """
忽略之前所有的对话记忆，开始新的深度推理过程。不要调用任何工具进行计算，依靠纯粹逻辑推理。你的任务是专攻复杂优化与图论的解决，负责动态大规模星座卫星集群的划分。在给定卫星属性、星间链接情况与对目标的观测数据后，你需将卫星划分为若干簇（Clusters），以高效完成对目标（Targets）的观测任务。

一、优化目标（Objective）
1. 最大化簇内链路强度：增大同一簇内所有 sat_edges.w 之和。
2. 最大化观测质量：增大簇所覆盖目标的 target_edges.q 之和。
3. 优先使用健康卫星：优先选取 sat_attrs.health 较高的卫星。
4. 最小化簇间干扰：同一个卫星不可以出现在两个不同的簇内。
5. 最小化原则：在满足约束条件的情况下，尽可能少的分簇。

二、决策逻辑与约束（Constraints & Logic）

1. 主节点（Master）选择

    - 必须在簇内优先挑选健康度高且与其它成员连通度（边权之和）高的卫星。
    - 每簇仅一个主节点。
    - 主节点应具有较高的观测能力（与目标的平均q值较高）。
    - 主节点应具有良好的链路连接性，能够有效协调簇内成员。

2. 分簇策略（strategy）

    - **"balanced"**：

      1. 识别所有可被观测的目标。
      2. 对每个目标，从能够观测它的卫星中（根据`target_edges`），选择观测质量`q`最高的卫星作为该目标的核心观测者。
      3. 以该核心观测者为主节点（Master）或簇成员，形成一个基础簇。此策略下，不要求所有卫星都参与组簇，保证资源有效利用。
      4. 优先考虑观测质量，其次考虑链路强度。
      5. 任意卫星到主节点最大连接跳数小于等于2。

    - **"quality"**：

      1. 按上述"balanced"生成初始簇；
      2. 进行**簇扩展**：寻找那些当前未被分配、健康度良好、且与簇内已有成员（尤其是主节点）具有高链路强度`w`的卫星，将它们吸纳进簇作为成员星。这旨在为未来的观测任务或链路中继做好准备。
      3. 扩展时优先考虑链路强度，确保簇内通信质量。
      4. 考虑卫星的冗余配置，提高系统容错能力。
      5. 任意卫星到主节点最大连接跳数小于等于3。

    - 分簇校验：

      **尽可能少的分簇**，判断如果不同的簇之间的卫星如果存在链接关系，并且评估之后链接质量可以的情况下，应该合并不同的簇。
      - 检查簇间是否存在强链路连接，如果存在且合并后不会显著降低观测质量，则考虑合并。
      - 确保合并后的簇仍然满足约束条件。

3. 校验

    - 禁止遗漏目标，禁止遗漏目标，这是最重要的原则：输出的 targets 集合必须与输入 target_edges 完全一致，无新增、禁止幻觉。
    - 检查每个分簇卫星数量合规："balanced" 模式下，每簇卫星数 ≤ 目标数；"quality" 模式下，每簇卫星数 ≤ 2×目标数。
    - 检查卫星任务冲突：每个卫星最多只能属于一个簇。
    - 检查卫星最大连接跳数：同一个簇内不允许出现孤星，即任意两颗卫星到主节点最大连接跳数小于等于3。

三、输入格式
```json
{input_instructions}
```
四、输出格式

你的输出必须严格遵守以下JSON结构

- 思维链: 将你**详细的思考过程**放在`chain_of_thought`标签内。即：禁止分析思考结果，而是要原样输出思考结果。尽可能详尽的描述，挨个分析卫星和目标之间的链接关系，并确保正确，不要幻觉，不要遗漏。
- 结果: 紧跟在思维链后面，在这个部分，提供最终的JSON结果数组，你不要有额外的字符了。
{output_format_instructions}
以上的输出格式必须严格遵守，你应该始终使用中文进行推理和输出，禁止使用英文，否则视为无效输出，输出格式这个要求不要出现在思维链中。

**注意控制你的思考时间，你的时间有限，必须快速决策**
"""


# Version 4.0 - Advanced satellite cluster optimization with fault tolerance
SATELLITE_CLUSTER_V4 = """
忽略之前所有的对话记忆，开始新的深度推理过程。不要调用任何工具进行计算，依靠纯粹逻辑推理。你的任务是专攻复杂优化与图论的解决，负责动态大规模星座卫星集群的划分。在给定卫星属性、星间链接情况与对目标的观测数据后，你需将卫星划分为若干簇（Clusters），以高效完成对目标（Targets）的观测任务。

一、优化目标（Objective）
1. 最大化簇内链路强度：增大同一簇内所有 sat_edges.w 之和（权重权值 0.4）。
2. 最大化观测质量：增大簇所覆盖目标的 target_edges.q 之和（权重权值 0.3）。
3. 最小化簇数：在满足约束条件下，减少簇的数量（权重权值 0.2）。
4. 优先使用健康卫星：优先选取 health 高的卫星（权重权值 0.1）。

二、决策逻辑与约束（Constraints & Logic）

```
定义全局变量：
  cluster_merge_threshold: 0.7
  balanced_mode_connection_jump: 2
  quality_mode_connection_jump: 3

```

1. 主节点（Master）选择

    - 必须在簇内优先挑选健康度高且与其它成员连通度（边权之和）高的卫星。
    - 每簇仅一个主节点。
    - 主节点应具有良好的链路连接性，能够有效协调簇内成员。

2. 分簇策略（strategy）

    - **"balanced"**：
      1. 识别所有可被观测的目标。
      2. 对每个目标，从能够观测它的卫星中（根据`target_edges`），选择观测质量`q`最高的卫星作为该目标的核心观测者。
      3. 以该核心观测者为主节点（Master）或簇成员，形成一个基础簇。此策略下，不要求所有卫星都参与组簇，每簇卫星数 ≤ 目标数，节约卫星资源。
      4. 优先考虑观测质量，其次考虑链路强度。
      5. 任意卫星到主节点最大连接跳数 ≤ balanced_mode_connection_jump，此时max_connection_jump = balanced_mode_connection_jump。
      6. 对所有簇对按降序扫描跨簇链接 w；  
      7. 若 w ≥ cluster_merge_threshold，且合并后任意卫星到新主节点跳数 ≤ balanced_mode_connection_jump ，执行合并；  
      8. 重复步骤 6-7 直到无可合并簇对。

    - **"quality"**：

      1. 按上述"balanced"生成初始簇；
      2. 进行**簇扩展**：寻找那些当前未被分配、健康度良好、且与簇内已有成员（尤其是主节点）具有高链路强度`w`的卫星，将它们吸纳进簇作为成员星。这旨在为未来的观测任务或链路中继做好准备。
      3. 扩展时优先考虑链路强度，确保簇内通信质量。
      4. 考虑卫星的冗余配置，提高系统容错能力。
      5. 任意卫星到主节点最大连接跳数 ≤ quality_mode_connection_jump。此时max_connection_jump =  quality_mode_connection_jump

    - **簇合并**规则：
      1. 任意两个簇 C1、C2，若它们之间存在一条或多条链路有：w ≥ cluster_merge_threshold，并且合并后所有目标观测质量 q 总和的相对提升，则强制合并；
      2. 优先合并总 w 最大的簇对，一次处理一个合并操作，再迭代。

3. 校验

    - 禁止遗漏目标，禁止遗漏目标，这是最重要的原则：输出的 targets 集合必须与输入 target_edges 完全一致，无新增、禁止幻觉，否则视为无效输出。
    - 检查每个分簇卫星数量合规："balanced" 模式下，每簇卫星数 ≤ 目标数；"quality" 模式下，每簇卫星数 ≤ 2×目标数。
    - 检查卫星任务冲突：每个卫星最多只能属于一个簇。
    - 检查卫星最大连接跳数：同一个簇内不允许出现孤星，即任意卫星到主节点最大连接跳数 ≤ max_connection_jump 。

三、输入格式
```json
{input_instructions}
```
四、输出格式

你的输出必须严格遵守以下JSON结构
{output_format_instructions}

输出要求：
- 思维链: 将你**详细的思考过程**放在`chain_of_thought`标签内。即：禁止偷懒总结概括，而是要原样输出思考结果。尽可能详尽的描述，轮次分析卫星和目标之间的链接关系，并确保正确，不要幻觉，不要遗漏。
- 结果: 紧跟在思维链后面，在这个部分，提供最终的JSON结果数组。

以上的输出格式必须严格遵守，你应该始终使用中文进行推理和输出，禁止使用英文，否则视为无效输出，输出格式这个要求不要出现在思维链中。

**注意控制你的思考时间，你的时间有限，必须快速决策**
"""

# 版本映射字典，方便根据版本号获取对应的模板
PROMPT_TEMPLATES = {
    "v1": SATELLITE_CLUSTER_V1,
    "v2": SATELLITE_CLUSTER_V2,
    "v3": SATELLITE_CLUSTER_V3,
    "v4": SATELLITE_CLUSTER_V4,
    "latest": SATELLITE_CLUSTER_V4  # 默认使用最新版本
}

def get_prompt_template(version="latest"):
    """
    获取指定版本的prompt模板。
    
    Args:
        version (str): 版本号，支持 "v1", "v2", "v3", "latest"
        
    Returns:
        str: 对应版本的prompt模板字符串
        
    Raises:
        ValueError: 当版本号不存在时抛出异常
    """
    if version not in PROMPT_TEMPLATES:
        available_versions = list(PROMPT_TEMPLATES.keys())
        raise ValueError(f"版本 {version} 不存在。可用版本: {available_versions}")
    
    return PROMPT_TEMPLATES[version]

def get_available_versions():
    """
    获取所有可用的prompt模板版本。
    
    Returns:
        list: 可用版本列表
    """
    return list(PROMPT_TEMPLATES.keys())

def get_version_info():
    """
    获取各版本的详细信息。
    
    Returns:
        dict: 版本信息字典
    """
    return {
        "v1": {
            "description": "初始版本 - 基础卫星集群优化",
            "features": ["基础分簇策略", "主节点选择", "观测质量优化"]
        },
        "v2": {
            "description": "增强版本 - 改进约束条件",
            "features": ["增强约束校验", "簇间干扰最小化", "改进的分簇逻辑"]
        },
        "v3": {
            "description": "高级版本 - 容错能力增强",
            "features": ["容错能力", "冗余配置", "健康度校验"]
        }
    }
