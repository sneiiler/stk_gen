"""
基于Gemini模型的多线程数据蒸馏系统实现。

This module implements a multi-threaded data distillation system using Gemini models
through OpenAI-compatible interface for better universality and performance.
"""

import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
print(root_dir)
sys.path.append(str(root_dir))

import json
import re
import threading
import time
from typing import List, Dict, Any, Tuple, Optional
import datetime
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from tqdm import tqdm
from icecream import ic, install

install()

from utils.misc_utils import get_data_dir, get_project_root
from utils.prompt_template import get_prompt_template
from data_classes.sft_data_models import (
    ClusterInfo,
    RawConstellationDataModel,
    SatelliteClusterOutput,
    ShareGPTFormat,
    ShareGPTMessage,
)
from dotenv import load_dotenv

env_path = get_project_root() / ".env"
print(env_path)
load_dotenv(env_path)
# 获取Gemini API配置
api_base_gemini = os.getenv("QWEN_API_BASE")
api_key_gemini = os.getenv("QWEN_API_KEY")

print(f"API配置: {'✓' if api_key_gemini and api_base_gemini else '✗'}")

if not api_key_gemini or not api_base_gemini:
    raise ValueError(
        "未找到API配置，请检查环境变量 DASHSCOPE_API_KEY 和 DASHSCOPE_API_BASE"
    )


class ThreadSafeWriter:
    """线程安全的文件写入器"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.lock = threading.Lock()
        # 确保输出目录存在
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def write_line(self, data: str):
        """线程安全地写入一行数据"""
        with self.lock:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()


class RejectionSampler:
    """基于Gemini模型的多线程数据蒸馏器类。

    主要功能：
    - 多线程并发处理提升效率
    - 支持流式响应获取思考过程
    - 智能速率限制避免API限制
    - 线程安全的文件写入
    - 专门针对Gemini模型优化

    Attributes:
        client: OpenAI兼容客户端
        prompt_template: 提示模板
        output_parser: 输出解析器
        : 速率限制器
    """

    def __init__(
        self,
        model_name: str = "",
        temperature: float = 0.1,
        requests_per_minute: int = 60,
    ):
        """初始化数据蒸馏器。

        Args:
            model_name: Gemini模型名称
            temperature: 生成温度，控制输出的随机性
            proxy: 代理设置
            requests_per_minute: 每分钟最大请求数
            max_workers: 最大并发线程数
        """

        self.model_name = model_name
        self.temperature = temperature

        self.client = OpenAI(api_key=api_key_gemini, base_url=api_base_gemini)

        # 获取prompt模板
        self.prompt_template = ChatPromptTemplate.from_template(
            template=get_prompt_template("latest")
        )

        # 创建输出解析器
        self.output_parser = PydanticOutputParser(
            pydantic_object=SatelliteClusterOutput
        )

        print(f"Gemini数据蒸馏器初始化完成:")
        print(f"- 模型: {self.model_name}")
        print(f"- 请求频率: {requests_per_minute}/分钟")

    def _extract_reasoning_and_content(self, response_stream) -> Tuple[str, str]:
        """从流式响应中提取思考过程和内容"""
        reasoning_content = ""
        content = ""

        try:
            for chunk in response_stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 获取思考过程（reasoning）- Gemini 2.5不支持
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    # print(f"获取到新的reasoning，长度: {len(delta.reasoning_content)}")

                # 获取正常内容
                if hasattr(delta, "content") and delta.content:
                    content += delta.content
                    # print(f"获取到新的content，长度: {len(delta.content)}")
                # 检查是否是思维链内容（包含在<think>标签中）
                thought_match = re.search(
                    r"<think>(.*?)</think>", content, re.DOTALL
                )
                if thought_match:
                    thought_content = thought_match.group(1)
                    reasoning_content += thought_content
                    # 从content中移除thought部分，保持内容干净
                    content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    )

        except Exception as e:
            print(f"处理流式响应时出错: {str(e)}")

        return reasoning_content.strip(), content.strip()

    def generate_distill_result(
        self, data: Dict[str, Any], sample_index: int = 0
    ) -> Tuple[Optional[SatelliteClusterOutput], str, str]:
        """生成蒸馏结果。

        Args:
            data: 输入数据
            sample_index: 样本索引，用于错误追踪

        Returns:
            (结果对象, 系统提示, 错误信息)的元组
        """
        try:

            # 格式化输入数据
            user_content = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            user_content = (
                user_content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            )
            user_content = re.sub(r"\s+", " ", user_content)

            # 获取格式说明
            input_instructions = RawConstellationDataModel.model_json_schema()
            format_instructions = self.output_parser.get_format_instructions()

            # 生成完整提示
            system_prompt = self.prompt_template.format(
                input_instructions=input_instructions,
                output_format_instructions=format_instructions,
            )
            system_prompt = (
                system_prompt.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            )
            system_prompt = re.sub(r"\s+", " ", system_prompt)

            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content.strip()},
            ]

            # 创建流式请求 - 针对Gemini优化
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                stream=True,
                temperature=self.temperature
            )

            # 提取思考过程和内容
            reasoning_content, content = self._extract_reasoning_and_content(response)

            print(f"样本 {sample_index} - 思考过程长度: {len(reasoning_content)}")
            print(f"样本 {sample_index} - 内容长度: {len(content)}")

            if not content.strip():
                return None, system_prompt, f"API返回内容为空 (样本 {sample_index})"

            # 解析JSON输出
            try:
                # 直接使用output_parser解析
                parsed_output = self.output_parser.parse(content)

                # 提取思维链
                cot = parsed_output.chain_of_thought
                # 如果有思考过程，将其添加到思维链前面
                if reasoning_content:
                    cot = f"{reasoning_content}\n思考过程总结:\n{cot}"

                result = SatelliteClusterOutput(
                    chain_of_thought=cot,
                    clusters=[
                        ClusterInfo(
                            timestamp="",
                            cluster_id=cluster.cluster_id,
                            master=cluster.master,
                            sats=cluster.sats,
                            targets=cluster.targets,
                        )
                        for cluster in parsed_output.clusters
                    ],
                )
                return result, system_prompt, ""

            except Exception as e:
                if len(reasoning_content) == 0 and "Invalid json output" in str(e):
                    error_msg = f"输出内容长度为:{len(content)}，思考内容解析失败，输出Json解析失败。原始错误：{str(e)}"
                else:
                    error_msg = f"解析输出失败 (样本 {sample_index}): {str(e)}"
                print(error_msg[:200])
                return None, system_prompt, error_msg

        except Exception as e:
            error_msg = f"API调用失败 (样本 {sample_index}): {str(e)}"
            print(error_msg)
            return None, "", error_msg

    def create_sharegpt_format(
        self, instruction: str, input_data: str, output_data: str
    ) -> ShareGPTFormat:
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

    def process_single_item(
        self,
        list_data: List,
        out_file_path: Path | str ,
        stats_queue: Queue,
    ) -> None:
        history_result:List[SatelliteClusterOutput]
        """处理单个数据项（在线程中调用）"""
        for index,item in tqdm(enumerate(list_data), desc="处理数据批次"):
            try:
                # 处理数据
                result, system_prompt, error_msg = self.generate_distill_result(item, index)

                if result is None:
                    # 记录错误
                    error_data = {
                        "error": error_msg,
                        "data": json.dumps(item, ensure_ascii=False, separators=(",", ":")),
                        "sample_index": index,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model": self.model_name,
                    }
                    
                    with open(out_file_path, "a", encoding="utf-8") as f:
                        f"ERROR: {json.dumps(error_data, ensure_ascii=False)}"
                        f.flush()
                    stats_queue.put("failed")
                    continue
                # history_result.append(result)

                # 构造ShareGPT格式的训练数据
                input_str = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                output_str = result.to_think_json()

                sharegpt_data = self.create_sharegpt_format(
                    instruction=system_prompt,
                    input_data=input_str,
                    output_data=output_str,
                )

                # 写入成功结果
                with open(out_file_path, "a", encoding="utf-8") as f:
                    f.write(sharegpt_data.model_dump_json() + "\n")
                    f.flush()
                stats_queue.put("success")

            except Exception as e:
                error_data = {
                    "error": f"处理异常: {str(e)}",
                    "sample_index": index,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": self.model_name,
                }
                # writer.write_line(f"ERROR: {json.dumps(error_data, ensure_ascii=False)}")
                stats_queue.put("failed")


def load_json_data(file_path: Path) -> List[Dict[str, Any]]:
    """从JSON文件加载数据。

    Args:
        file_path: JSON文件路径

    Returns:
        加载的数据列表
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    """主函数。"""
    # 从JSON文件加载数据
    input_file = (
        get_data_dir() / "training_data_raw_scenario_3_20250723_104802.json"
    )
    batch_data = load_json_data(input_file)

    print(f"加载了 {len(batch_data)} 个数据样本")
    model_name = "qwen3-8b"  # 或 "gemini-2.5-pro" 如需更高质量
    stats_queue = Queue()

    # 初始化Gemini蒸馏器
    distiller = RejectionSampler(
        model_name=model_name,  # 或 "gemini-2.5-pro" 如需更高质量
        temperature=0.6,
        requests_per_minute=60,  # 根据你的API限制调整
    )

    # 生成输出文件路径
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        get_data_dir()
        / f"rejection_sampling_training_data_sharegpt_{model_name}_{now}_{str(input_file)[-10:-5]}.jsonl"
    )

    # 多线程处理数据并实时保存
    start_time = time.time()
    distiller.process_single_item(batch_data, output_file,stats_queue=stats_queue)
    end_time = time.time()

    print(f"\n✅ 处理完成!")
    print(f"📁 结果已保存到: {output_file}")
    print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 平均每个样本耗时: {(end_time - start_time) / len(batch_data):.2f} 秒")


if __name__ == "__main__":
    main()
