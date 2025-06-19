"""
数据蒸馏系统的主要实现。

This module implements a data distillation system using OpenAI API.
It provides functionality for batch processing and optimizing training data.
"""

import os
import json
from typing import List, Dict, Any, TypedDict, Literal, Optional
import datetime
from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm
from icecream import ic
from dotenv import load_dotenv

from utils.misc_utils import get_data_dir

# 加载环境变量
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# 从环境变量获取配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', 'o4-mini')  # 使用默认值

# 验证必要的配置
if not OPENAI_API_KEY or not OPENAI_API_BASE:
    raise ValueError("未找到.env文件或环境变量，请检查配置")

# 设置环境变量
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE


class Message(TypedDict):
    """消息类型定义"""

    role: Literal["system", "user", "assistant"]
    content: str


class AlpacaFormat(TypedDict):
    """Alpaca格式的训练数据定义"""

    instruction: str
    input: str
    output: str


class DataDistiller:
    """数据蒸馏器类。

    用于批量处理和优化训练数据的类。

    Attributes:
        client: OpenAI客户端实例
        system_prompt: 系统提示信息
    """

    def __init__(self, model_name: str = OPENAI_MODEL_NAME):
        """初始化数据蒸馏器。

        Args:
            model_name: 要使用的OpenAI模型名称
        """
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
        self.model_name = model_name
        self.system_prompt = "你是一个专攻解决复杂优化和图论问题的AI专家。你的任务是扮演一个动态卫星集群的求解器。根据给定的卫星状态、星间链路和对地观测数据，解决大规模星座在分组观测动态目标时的动态分簇问题。你需要将一组卫星（Satellites）划分成多个最优的簇（Clusters），以高效地完成对一组目标（Targets）的观测任务。"

        # 基础指令模板

    def generate_prompt(self, data: Dict) -> List[ChatCompletionMessageParam]:
        """生成提示消息。

        Args:
            data: 输入数据

        Returns:
            包含消息的列表
        """
        prompt_content = f"""**目标函数 (Objective Function):**
你的分簇决策需要综合优化以下三个目标：
1. **最大化簇内链路强度**: 簇内卫星之间的 `sat_edges.w` 之和应尽可能大。
2. **最大化对目标观测质量**: 簇所覆盖目标的 `target_edges.q` 之和应尽可能大。
3. **优先使用健康卫星**: `sat_attrs.health` 值高的卫星应被优先考虑。

**约束与决策逻辑 (Constraints and Decision Logic):**
**1. 簇内连通性要求:**
- 簇内任意两颗卫星之间必须能通过不超过2跳的路径实现互联（即最大跳数≤2）
- 连通性通过 `sat_edges.w` 定义，任何 w>0 的边都视为可连通
- 这确保了簇内卫星间的高效通信和数据中继能力

**2. 主节点（Master）选择策略:**
- 每个簇必须指定一个主节点，负责数据中继和任务协调
- 主节点候选优先级由以下因素决定：
  1. 健康度（`health`）高
  2. 总连通度高（所有相连 `sat_edges.w` 之和）
  3. 观测能力强（相关 `target_edges.q` 之和）

**3. 分簇策略 (`strategy`):**
- 当`strategy`为 "balance": 首先确保每个目标都被观测到。为每个被观测的目标分配一个核心观测卫星，核心观测卫星尽可能不重复
  - **逻辑**:
    1. 识别所有可被观测的目标
    2. 对每个目标，从能观测它的卫星中选择观测质量`q`最高的作为核心观测者，如果当前观测者已经被占用，则选择次优观测者
    3. 如果多个目标的最优观测卫星之间能通过≤2跳互联，将它们组成一个簇
    4. 星簇的卫星数量应尽可能接近目标的数量

- 当`strategy`为 "quality": 尽可能多地利用可用卫星，形成高质量、高韧性的观测簇
  - **逻辑**:
    1. 首先按 "balance" 策略形成初始簇
    2. 然后进行**簇扩展**：
       - 寻找未分配、健康度良好的卫星
       - 确保它们与簇内已有成员的连通跳数为1
       - 优先选择与主节点直接相连的卫星
       - 将它们加入到合适的簇中
    3. 如果某个簇内卫星间的连通性超过2跳，考虑将其拆分为多个簇

**关键原则：**
1. 保持簇的紧凑性：簇内主节点的卫星到任意一颗成员卫星的跳数应该≤3，否则重新分簇
2. 避免过度分簇：只在确实需要时才创建新的簇，但通常来说一个星簇内的卫星不应该超过10个
3. 确保资源利用效率：不强制要求所有卫星都必须参与分簇，但是确保每个目标都被观测到。
4. 非必要不要单独把≤2颗的卫星划分为一个簇，这个重要程度更高，可以把跳数约束调整为≤4
5. 最重要的原则：【最后务必检查是否所有的目标都被观测到，如果没有则重新规划】。

**Output Data Schema:**
你的输出必须严格遵守以下JSON结构，并包含一个详细的思考过程。重要：使用中文回答。

- **`<|chain_of_thought|>`**: 你需要用清晰的、分步数字编号表明步骤，解释如何得到最终分簇结果，长度不应当超过500字。应当考虑当前的策略，进行观测关系、连通关系的数据洞察、主节点评估候选以及分析簇的形成过程，以及其他合理的分析过程，并最后按簇总结最后的决策为什么是最优的。
- **`<|result|>`**: 在这个部分，提供最终的JSON结果数组。
- `<|EOF|>`: 输出截止符号

```json
<|chain_of_thought|>
// 在这里逐步展示你的推理过程...
<|result|>
[
  {{
    "cluster_id": "integer",  // 分簇的ID
    "master": "integer",      // 主节点的卫星ID
    "sats": ["integer", "integer", ...],  // 包含主节点在内的所有成员卫星ID列表
    "targets": ["integer", "integer", ...] // 该簇负责观测的目标ID列表
  }}
]
<|EOF|>
```
**现在给你的输入数据为：**
{json.dumps(data, ensure_ascii=False, separators=(",", ":"))}
请输出：
"""

        # 删除回车符和多余空格以节省token
        def clean_text(text: str) -> str:
            """清理文本中的多余空白字符"""
            # 1. 替换回车符和换行符为空格
            text = text.replace('\n', ' ').replace('\r', ' ')
            # 2. 替换制表符为空格
            text = text.replace('\t', ' ')
            # 3. 合并多个连续空格为单个空格
            import re
            text = re.sub(r'\s+', ' ', text)
            # 4. 删除首尾空白
            text = text.strip()
            return text
        
        system_prompt_clean = clean_text(self.system_prompt)
        prompt_content_clean = clean_text(prompt_content)
        
        resp: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt_clean},
            {"role": "user", "content": prompt_content_clean},
        ]
        # ic(resp)
        return resp

    def extract_parts(self, output: str) -> tuple[Optional[str], Optional[str]]:
        """从输出中提取思考过程和结果。

        Args:
            output: API返回的输出字符串

        Returns:
            思考过程和结果的元组
        """
        thought = None
        result = None

        # 清理输出文本，移除多余的空白字符
        output = "".join(line.strip() for line in output.splitlines())

        # 提取思考过程
        if "<|chain_of_thought|>" in output:
            parts = output.split("<|chain_of_thought|>")
            if len(parts) > 1:
                thought_parts = parts[1].split("<|result|>")
                thought = thought_parts[0].strip()

        # 提取结果
        if "<|result|>" in output:
            parts = output.split("<|result|>")
            if len(parts) > 1:
                result_parts = parts[1].split("<|EOF|>")
                result = result_parts[0].strip()

        return thought, result

    def process_batch(self, batch_data: List[Dict[str, Any]], output_file: Path) -> None:
        """批量处理数据并实时保存。

        Args:
            batch_data: 包含任务描述和输入数据的列表
            output_file: 输出文件路径
        """
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "a", encoding="utf-8") as f:
            for data in tqdm(batch_data, desc="处理数据批次"):
                try:
                    messages = self.generate_prompt(data)
                    completion = self.client.chat.completions.create(model=self.model_name, messages=messages)
                    output = completion.choices[0].message.content
                    if output is None:
                        raise ValueError("API返回的输出为空")

                    # ic(output)

                    # 提取思考过程和结果
                    thought, result = self.extract_parts(output)

                    # 构造格式化的输出
                    formatted_output = {
                        "chain_of_thought": thought.strip() if thought else "",
                        "result": json.loads(result.strip()) if result else None,
                    }

                    # 构造Alpaca格式的训练数据
                    # 消息内容已在generate_prompt时清理过回车符
                    system_content = str(messages[0].get("content", ""))
                    user_content = str(messages[1].get("content", ""))
                    
                    alpaca_data = {
                        "instruction": system_content + user_content,
                        "input": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                        "output": json.dumps(formatted_output, ensure_ascii=False, separators=(",", ":")),
                    }

                    # 实时写入文件
                    f.write(json.dumps(alpaca_data, ensure_ascii=False) + "\n")
                    f.flush()  # 确保立即写入磁盘

                except Exception as e:
                    print(f"处理数据时出错: {str(e)}")
                    # 记录错误数据，方便后续重试
                    error_data = {
                        "error": str(e),
                        "data": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                    }
                    f.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                    f.flush()

    def generate_prompt_text(self, data: Dict) -> str:
        """生成可直接用于web界面的prompt文本。

        Args:
            data: 输入数据

        Returns:
            prompt字符串
        """
        prompt_content = f"""**目标函数 (Objective Function):**\n你的分簇决策需要综合优化以下三个目标：\n1. **最大化簇内链路强度**: 簇内卫星之间的 `sat_edges.w` 之和应尽可能大。\n2. **最大化对目标观测质量**: 簇所覆盖目标的 `target_edges.q` 之和应尽可能大。\n3. **优先使用健康卫星**: `sat_attrs.health` 值高的卫星应被优先考虑。\n\n**约束与决策逻辑 (Constraints and Decision Logic):**\n**1. 簇内连通性要求:**\n- 簇内任意两颗卫星之间必须能通过不超过2跳的路径实现互联（即最大跳数≤2）\n- 连通性通过 `sat_edges.w` 定义，任何 w>0 的边都视为可连通\n- 这确保了簇内卫星间的高效通信和数据中继能力\n\n**2. 主节点（Master）选择策略:**\n- 每个簇必须指定一个主节点，负责数据中继和任务协调\n- 主节点候选优先级由以下因素决定：\n  1. 健康度（`health`）高\n  2. 总连通度高（所有相连 `sat_edges.w` 之和）\n  3. 观测能力强（相关 `target_edges.q` 之和）\n\n**3. 分簇策略 (`strategy`):**\n- 当`strategy`为 \"balance\": 首先确保每个目标都被观测到。为每个被观测的目标分配一个核心观测卫星，核心观测卫星尽可能不重复\n  - **逻辑**:\n    1. 识别所有可被观测的目标\n    2. 对每个目标，从能观测它的卫星中选择观测质量`q`最高的作为核心观测者，如果当前观测者已经被占用，则选择次优观测者\n    3. 如果多个目标的最优观测卫星之间能通过≤2跳互联，将它们组成一个簇\n    4. 星簇的卫星数量应尽可能接近目标的数量\n\n- 当`strategy`为 \"quality\": 尽可能多地利用可用卫星，形成高质量、高韧性的观测簇\n  - **逻辑**:\n    1. 首先按 \"balance\" 策略形成初始簇\n    2. 然后进行**簇扩展**：\n       - 寻找未分配、健康度良好的卫星\n       - 确保它们与簇内已有成员的连通跳数为1\n       - 优先选择与主节点直接相连的卫星\n       - 将它们加入到合适的簇中\n    3. 如果某个簇内卫星间的连通性超过2跳，考虑将其拆分为多个簇\n\n**关键原则：**\n1. 保持簇的紧凑性：簇内主节点的卫星到任意一颗成员卫星的跳数应该≤3，否则重新分簇\n2. 避免过度分簇：只在确实需要时才创建新的簇，但通常来说一个星簇内的卫星不应该超过10个\n3. 确保资源利用效率：不强制要求所有卫星都必须参与分簇，但是确保每个目标都被观测到。\n4. 不要遗漏对任何一个可见目标的观测\n\n  【最后务必检查是否所有的目标都被观测到，如果没有则重新规划】。\n**Output Data Schema:**\n你的输出必须严格遵守以下JSON结构，并包含一个详细的思考过程。重要：使用中文回答。\n\n- **`<|chain_of_thought|>`**: 你需要用清晰的、分步数字编号表明步骤，解释如何得到最终分簇结果，长度不应当超过500字。应当考虑当前的策略，进行观测关系、连通关系的数据洞察、主节点评估候选以及分析簇的形成过程，以及其他合理的分析过程，并最后按簇总结最后的决策为什么是最优的。\n- **`<|result|>`**: 在这个部分，提供最终的JSON结果数组。\n- `<|EOF|>`: 输出截止符号\n\n```json\n<|chain_of_thought|>\n// 在这里逐步展示你的推理过程...\n<|result|>\n[\n  {{\n    \"cluster_id\": \"integer\",  // 分簇的ID\n    \"master\": \"integer\",      // 主节点的卫星ID\n    \"sats\": [\"integer\", \"integer\", ...],  // 包含主节点在内的所有成员卫星ID列表\n    \"targets\": [\"integer\", \"integer\", ...] // 该簇负责观测的目标ID列表\n  }}\n]\n<|EOF|>\n```\n**现在给你的输入数据为：**\n{json.dumps(data, ensure_ascii=False, separators=(",", ":"))}\n请输出：\n"""
        return prompt_content


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """从JSON文件加载数据。

    Args:
        file_path: JSON文件路径

    Returns:
        加载的数据列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def main():
    """主函数。"""
    # 从JSON文件加载数据
    input_file = get_data_dir() / "training_data_raw_scen1_full.json"
    batch_data = load_json_data(input_file)

    # 初始化蒸馏器
    distiller = DataDistiller()

    # 生成输出文件路径
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = get_data_dir() / f"training_data_distilled_{now}.jsonl"

    # 处理数据并实时保存
    # distiller.process_batch(batch_data, output_file)
    # print(f"处理完成，结果已保存到 {output_file}")

    # 新增：批量生成prompt文本，便于手动复制
    prompt_txt_file = get_data_dir() / f"prompts_{now}.txt"
    with open(prompt_txt_file, "w", encoding="utf-8") as f:
        for idx, data in enumerate(batch_data):
            prompt = distiller.generate_prompt_text(data)
            f.write(f"================ PROMPT {idx+1} ================\n")
            f.write(prompt)
            f.write("\n\n")
            f.write("Answer:")
            f.write("\n\n")
    print(f"已批量生成prompt文本，保存在 {prompt_txt_file}")


if __name__ == "__main__":
    main()
