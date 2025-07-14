"""
基于Gemini模型的多线程数据蒸馏系统实现。

This module implements a multi-threaded data distillation system using Gemini models
through OpenAI-compatible interface for better universality and performance.
"""

import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
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
load_dotenv(env_path)

# 获取Gemini API配置
api_base_gemini = os.getenv("GEMINI_API_BASE")
api_key_gemini = os.getenv("GEMINI_API_KEY_4")

print(f"Gemini API配置: {'✓' if api_key_gemini and api_base_gemini else '✗'}")

if not api_key_gemini or not api_base_gemini:
    raise ValueError(
        "未找到Gemini API配置，请检查环境变量 GEMINI_API_KEY 和 GEMINI_API_BASE"
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


class RateLimiter:
    """简单的速率限制器"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """如果需要的话等待以满足速率限制"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_request_time = time.time()


class DataDistiller:
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
        rate_limiter: 速率限制器
    """

    def __init__(
        self,
        model_name: str = "",
        temperature: float = 0.1,
        proxy: Optional[str] = None,
        requests_per_minute: int = 60,
        max_workers: int = 5,
        reasoning_effort: str = "high",  # low, medium, high
    ):
        """初始化数据蒸馏器。

        Args:
            model_name: Gemini模型名称
            temperature: 生成温度，控制输出的随机性
            proxy: 代理设置
            requests_per_minute: 每分钟最大请求数
            max_workers: 最大并发线程数
            reasoning_effort: 推理强度 (low, medium, high)
        """

        self.model_name = model_name
        self.temperature = temperature
        self.max_workers = max_workers
        self.reasoning_effort = reasoning_effort
        self.rate_limiter = RateLimiter(requests_per_minute)

        # 初始化OpenAI兼容客户端
        if proxy is not None:
            import httpx

            httpx_client = httpx.Client(proxy=proxy)
            print(f"使用代理: {proxy}")
            self.client = OpenAI(
                api_key=api_key_gemini,
                base_url=api_base_gemini,
                http_client=httpx_client,
            )
        else:
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
        print(f"- 推理强度: {self.reasoning_effort}")
        print(f"- 最大并发: {self.max_workers}")
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

                # 获取正常内容
                if hasattr(delta, "content") and delta.content:
                    content += delta.content
                    print(f"获取到新的content，长度: {len(delta.content)}")



                # 检查是否是Gemini的思维链内容（包含在<thought>标签中）
                thought_match = re.search(
                    r"<thought>(.*?)</thought>", content, re.DOTALL
                )
                if thought_match:
                    thought_content = thought_match.group(1)
                    reasoning_content += thought_content
                    # 从content中移除thought部分，保持内容干净
                    content = re.sub(
                        r"<thought>.*?</thought>", "", content, flags=re.DOTALL
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
            # 应用速率限制
            self.rate_limiter.wait_if_needed()

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
                # reasoning_effort=self.reasoning_effort,  # Gemini 2.5支持 # type: ignore
                messages=messages,  # type: ignore
                stream=True,
                temperature=self.temperature,
                extra_body={
                    "extra_body": {
                        "google": {
                            "thinking_config": {
                                "thinking_budget": 12000,
                                "include_thoughts": True,
                            }
                        }
                    }
                },
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
                error_msg = f"解析输出失败 (样本 {sample_index}): {str(e)}"
                print(error_msg)
                print(f"原始内容: {content[:500]}...")
                print(f"思考过程: {reasoning_content[:200]}...")
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
        item_data: Tuple[int, Dict[str, Any]],
        writer: ThreadSafeWriter,
        stats_queue: Queue,
    ) -> None:
        """处理单个数据项（在线程中调用）"""
        index, data = item_data

        try:
            # 处理数据
            result, system_prompt, error_msg = self.generate_distill_result(data, index)

            if result is None:
                # 记录错误
                error_data = {
                    "error": error_msg,
                    "data": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                    "sample_index": index,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": self.model_name,
                }
                writer.write_line(
                    f"ERROR: {json.dumps(error_data, ensure_ascii=False)}"
                )
                stats_queue.put("failed")
                return

            # 构造ShareGPT格式的训练数据
            input_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            output_str = result.to_think_json()

            sharegpt_data = self.create_sharegpt_format(
                instruction=system_prompt,
                input_data=input_str,
                output_data=output_str,
            )

            # 写入成功结果
            writer.write_line(sharegpt_data.model_dump_json())
            stats_queue.put("success")

        except Exception as e:
            error_data = {
                "error": f"处理异常: {str(e)}",
                "sample_index": index,
                "timestamp": datetime.datetime.now().isoformat(),
                "model": self.model_name,
            }
            writer.write_line(f"ERROR: {json.dumps(error_data, ensure_ascii=False)}")
            stats_queue.put("failed")

    def process_batch_multithread(
        self,
        batch_data: List[Dict[str, Any]],
        output_file: Path,
    ) -> None:
        """多线程批量处理数据并实时保存。

        Args:
            batch_data: 包含任务描述和输入数据的列表
            output_file: 输出文件路径
        """
        # 初始化线程安全的写入器
        writer = ThreadSafeWriter(output_file)

        # 统计信息队列
        stats_queue = Queue()

        # 创建带索引的数据列表
        indexed_data = list(enumerate(batch_data))

        # 使用线程池处理
        print(f"开始多线程处理，使用 {self.max_workers} 个线程")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(
                    self.process_single_item, item, writer, stats_queue
                ): item[0]
                for item in indexed_data
            }

            # 使用tqdm显示进度
            with tqdm(total=len(batch_data), desc="处理数据批次") as pbar:
                for future in as_completed(future_to_index):
                    try:
                        future.result()  # 获取结果，如果有异常会抛出
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"线程处理样本 {index} 时出现异常: {str(e)}")
                    finally:
                        pbar.update(1)

        # 收集统计信息
        stats = {"total": len(batch_data), "success": 0, "failed": 0}
        while not stats_queue.empty():
            result = stats_queue.get()
            if result == "success":
                stats["success"] += 1
            elif result == "failed":
                stats["failed"] += 1

        # 打印统计信息
        print(f"多线程处理完成统计: {stats}")
        print(f"成功率: {stats['success']/stats['total']*100:.2f}%")


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
        get_data_dir() / "mock_satellite_observation_data_20250630_090749_v7.json"
    )
    batch_data = load_json_data(input_file)

    print(f"加载了 {len(batch_data)} 个数据样本")

    proxy = "socks5://127.0.0.1:1089"
    model_name = "gemini-2.5-pro"  # 或 "gemini-2.5-pro" 如需更高质量

    # 初始化Gemini蒸馏器
    distiller = DataDistiller(
        model_name=model_name,  # 或 "gemini-2.5-pro" 如需更高质量
        temperature=0.1,
        proxy=proxy,  # "socks5://127.0.0.1:1089" 如果需要代理
        requests_per_minute=60,  # 根据你的API限制调整
        max_workers=6,  # 建议从4开始，成功后可以增加到6-8
        reasoning_effort="high",  # low/medium/high, 控制思考深度
    )

    # 生成输出文件路径
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        get_data_dir()
        / f"training_data_sharegpt_{model_name}_{now}_{str(input_file)[-10:-5]}.jsonl"
    )

    print(f"\n📋 处理配置:")
    print(f"- 模型: {distiller.model_name}")
    print(f"- 推理强度: {distiller.reasoning_effort}")
    print(f"- 并发线程: {distiller.max_workers}")
    print(f"- 请求频率: {distiller.rate_limiter.requests_per_minute}/分钟")
    print(f"- 输出文件: {output_file}")
    print(f"- 数据量: {len(batch_data)} 个样本")

    # 多线程处理数据并实时保存
    start_time = time.time()
    distiller.process_batch_multithread(batch_data, output_file)
    end_time = time.time()

    print(f"\n✅ 处理完成!")
    print(f"📁 结果已保存到: {output_file}")
    print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 平均每个样本耗时: {(end_time - start_time) / len(batch_data):.2f} 秒")

    # 提供优化建议
    print(f"\n💡 优化建议:")
    print(f"1. 如果成功率较高，可以将 max_workers 增加到 6-8")
    print(f"2. 如果遇到429错误，降低 requests_per_minute 或 max_workers")
    print(f"3. 如果需要更高质量的思考，将 reasoning_effort 改为 'high'")
    print(f"4. 如果需要更高整体质量，可以尝试 'gemini-2.5-pro' 模型")


if __name__ == "__main__":
    main()
