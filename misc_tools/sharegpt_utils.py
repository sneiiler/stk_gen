"""
ShareGPT格式数据处理工具函数

提供创建和处理ShareGPT格式训练数据的通用函数
"""

import json
from pathlib import Path
import re
from typing import List
from data_class.sft_data_models import RawConstellationDataModel, SatelliteClusterOutput, LLMConversationMessage


def create_sharegpt_format(instruction: str, input_data: RawConstellationDataModel, output_data: SatelliteClusterOutput) -> LLMConversationMessage:
    """创建ShareGPT格式的训练数据。

    Args:
        instruction: 指令内容
        input_data: 输入的卫星星座数据
        output_data: 卫星分簇输出数据

    Returns:
        LLMConversationMessage格式的数据
    """
    return LLMConversationMessage(
        instruction=instruction,
        input=input_data,
        response=output_data
    )


def load_sharegpt_data(file_path: str | Path) -> List[LLMConversationMessage]:
    """加载ShareGPT格式的数据文件

    Args:
        file_path: 数据文件路径，支持 .json 和 .jsonl 格式

    Returns:
        List[LLMConversationMessage]: 包含解析后的卫星分簇对话数据模型列表
    """
    # ensure file_path is string for suffix checks
    file_path_str = str(file_path)
    raw_data = []
    validation_inputs = []

    if file_path_str.endswith(".json"):
        # 如果是JSON文件，直接加载
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    elif file_path_str.endswith(".jsonl"):
        # 如果是JSONL文件，逐行加载
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # 移除行尾的换行符
                cleaned_line = line.strip()
                if cleaned_line:  # 确保非空行
                    # 解析JSON
                    data = json.loads(cleaned_line)
                    raw_data.append(data)

    for index,line_data in enumerate(raw_data):
        instruction = ""
        input_data = None
        response_data = None

        for message in line_data["messages"]:
            if message["role"] == "system":
                instruction = message["content"]
            elif message["role"] == "user":
                # 解析用户输入的卫星星座数据
                input_data = RawConstellationDataModel(**json.loads(message["content"]))
            elif message["role"] == "assistant":
                # 使用正则表达式匹配 <think>...</think> 和后续的分簇结果
                pattern = r"<think>(.*?)</think>(.*?)\[(.*)\]$$"
                match = re.search(pattern, message["content"], re.DOTALL)
                if match:
                    try:
                        reasoning = match.group(1).strip()
                        # 先去除所有反斜杠，避免解析失败
                        group3_cleaned = match.group(3).replace("\\", "")
                        clusters_data = json.loads("[" + group3_cleaned + "]")

                        # 创建 SatelliteClusterOutput 对象
                        response_data = SatelliteClusterOutput(
                            chain_of_thought=reasoning,
                            clusters=clusters_data
                        )
                    except json.JSONDecodeError as e:
                        print("JSON 解析失败:", e)
                        raise ValueError(f"JSON 解析失败: {e}")

        # 如果所有必要数据都解析成功，创建 SatelliteClusteringConversation 对象
        if instruction and input_data and response_data:
            validation_inputs.append(LLMConversationMessage(
                instruction=instruction,
                input=input_data,
                response=response_data
            ))

    return validation_inputs


