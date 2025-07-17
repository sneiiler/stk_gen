"""
ShareGPT格式数据处理工具函数

提供创建和处理ShareGPT格式训练数据的通用函数
"""

import json
from pathlib import Path
import re
from typing import List
from pydantic import BaseModel
from data_classes.sft_data_models import ShareGPTFormat, ShareGPTMessage


class ValidationInput(BaseModel):
    """验证输入数据模型"""
    input_user_data: List[dict]
    output_reasoning_data: List[str]
    output_result_data: List[list]


def create_sharegpt_format(instruction: str, input_data: str, output_data: str) -> ShareGPTFormat:
    """创建ShareGPT格式的训练数据。

    Args:
        instruction: 指令内容
        input_data: 输入数据
        output_data: 输出数据

    Returns:
        ShareGPT格式的数据
    """
    return ShareGPTFormat(
        messages=[
            ShareGPTMessage(role="system", content=instruction),
            ShareGPTMessage(role="user", content=input_data),
            ShareGPTMessage(role="assistant", content=output_data),
        ]
    )


def load_sharegpt_data(file_path: str | Path) -> ValidationInput:
    """加载ShareGPT格式的数据文件

    Args:
        file_path: 数据文件路径，支持 .json 和 .jsonl 格式

    Returns:
        ValidationInput: 包含解析后的用户输入数据、推理数据和结果数据
    """
    # ensure file_path is string for suffix checks
    file_path_str = str(file_path)
    raw_data = []
    input_user_data = []
    output_resoning_data = []
    output_result_data = []

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

    for line_index, line_data in enumerate(raw_data):
        for message in line_data["messages"]:
            if message["role"] == "user":
                input_user_data.append(json.loads(message["content"]))
            if message["role"] == "assistant":
                # 使用正则表达式匹配 </think> 后面到下一个 ``` 之间的内容
                pattern = r"<think>(.*?)</think>(.*?)\[(.*)\]$$"
                match = re.search(pattern, message["content"], re.DOTALL)
                if match:
                    try:
                        output_resoning_data.append(match.group(1))
                        # 先去除所有反斜杠，避免解析失败
                        group3_cleaned = match.group(3).replace("\\", "")
                        output_result_data.append(
                            json.loads("[" + group3_cleaned + "]")
                        )
                    except json.JSONDecodeError as e:
                        print("JSON 解析失败:", e)
                        raise ValueError(f"JSON 解析失败: {e}")
    return ValidationInput(
        input_user_data=input_user_data,
        output_reasoning_data=output_resoning_data,
        output_result_data=output_result_data,
    )


