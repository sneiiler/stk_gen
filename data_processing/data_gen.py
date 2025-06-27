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

from utils.misc_utils import get_data_dir, get_project_root
from utils.prompt_template import get_prompt_template

# 加载环境变量
env_path = get_project_root() / ".env"
load_dotenv(env_path)

# 从环境变量获取配置
OPENAI_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_COMPATIBLE_API_BASE")
OPENAI_MODEL_NAME = os.getenv("OPENAI_COMPATIBLE_MODEL_NAME", "qwen-max")  # 使用默认值
ic(os.getenv("OPENAI_COMPATIBLE_API_KEY"))
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
        self.system_prompt = get_prompt_template()

        # 基础指令模板

    def generate_prompt(self, data: Dict) -> List[ChatCompletionMessageParam]:
        """生成提示消息。

        Args:
            data: 输入数据

        Returns:
            包含消息的列表
        """
        prompt_content = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

        # 删除回车符和多余空格以节省token
        def clean_text(text: str) -> str:
            """清理文本中的多余空白字符"""
            # 1. 替换回车符和换行符为空格
            text = text.replace("\n", " ").replace("\r", " ")
            # 2. 替换制表符为空格
            text = text.replace("\t", " ")
            # 3. 合并多个连续空格为单个空格
            import re

            text = re.sub(r"\s+", " ", text)
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

    def process_batch(
        self, batch_data: List[Dict[str, Any]], output_file: Path
    ) -> None:
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
                    ic(messages)
                    completion = self.client.chat.completions.create(
                        model=self.model_name, messages=messages
                    )
                    output = completion.choices[0].message.content
                    if output is None:
                        raise ValueError("API返回的输出为空")

                    ic(output)
                    exit()

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
                        "input": json.dumps(
                            data, ensure_ascii=False, separators=(",", ":")
                        ),
                        "output": json.dumps(
                            formatted_output, ensure_ascii=False, separators=(",", ":")
                        ),
                    }

                    # 实时写入文件
                    f.write(json.dumps(alpaca_data, ensure_ascii=False) + "\n")
                    f.flush()  # 确保立即写入磁盘

                except Exception as e:
                    print(f"处理数据时出错: {str(e)}")
                    # 记录错误数据，方便后续重试
                    error_data = {
                        "error": str(e),
                        "data": json.dumps(
                            data, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                    f.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                    f.flush()


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
    input_file = get_data_dir() / "mock_satellite_observation_data_20250625_225448.json"
    batch_data = load_json_data(input_file)

    # 初始化蒸馏器
    distiller = DataDistiller()

    # 生成输出文件路径
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = get_data_dir() / f"training_data_distilled_{now}.json"

    # 处理数据并实时保存
    distiller.process_batch(batch_data, output_file)
    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
