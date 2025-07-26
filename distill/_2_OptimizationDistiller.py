"""
卫星分簇优化蒸馏系统

基于验证反馈对分簇结果进行优化，生成高质量的训练数据
"""

# 标准库导入
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import List, Optional, Tuple

# 第三方库导入
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from tqdm import tqdm

# 本地模块导入
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from data_classes.sft_data_models import (
    ClusterOptimizationOutput,
    ShareGPTFormat,
    ShareGPTMessage,
)
from data_classes.data_validation_models import ValidationItem
from utils.misc_utils import get_current_timestamp, get_data_dir, get_project_root
from utils.prompt_template import get_cluster_optimization_prompt

env_path = get_project_root() / ".env"
load_dotenv(env_path, override=True)

# 获取API配置
api_base_openai = os.getenv("DASHSCOPE_API_BASE")
api_key_openai = os.getenv("DASHSCOPE_API_KEY")

print(f"API配置: {'✓' if api_key_openai and api_base_openai else '✗'}")

if not api_key_openai or not api_base_openai:
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


class OptimizationDistiller:
    """卫星分簇优化蒸馏器

    核心功能：
    1. 加载验证结果数据
    2. 基于验证反馈进行分簇优化
    3. 生成ShareGPT格式的训练数据
    4. 多线程并发处理提升效率
    """

    def __init__(
        self, 
        model_name: str, 
        temperature: float,
        requests_per_minute: int = 60,
        max_workers: int = 3
    ):
        """初始化优化蒸馏器

        Args:
            model_name: 使用的模型名称
            temperature: 生成温度，控制输出的随机性（优化任务建议使用较低温度）
            requests_per_minute: 每分钟最大请求数
            max_workers: 最大并发线程数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(requests_per_minute)

        # 初始化OpenAI客户端和解析器
        self.client = OpenAI(api_key=api_key_openai, base_url=api_base_openai)
        self.output_parser = PydanticOutputParser(
            pydantic_object=ClusterOptimizationOutput
        )

        # 构建聊天提示模板，包含输入和输出数据结构描述
        self.prompt_template = ChatPromptTemplate.from_template(
            template=get_cluster_optimization_prompt()
        )

        print(f"优化蒸馏器初始化完成:")
        print(f"- 模型: {self.model_name}")
        print(f"- 最大并发: {self.max_workers}")
        print(f"- 请求频率: {requests_per_minute}/分钟")

    def _extract_reasoning_and_content(self, response_stream) -> tuple[str, str]:
        """从流式响应中提取思考过程和内容"""
        reasoning_content = ""
        content = ""

        try:
            for chunk in response_stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # 获取思考过程（reasoning）
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content

                # 获取正常内容
                if hasattr(delta, "content") and delta.content:
                    content += delta.content

        except Exception as e:
            print(f"处理流式响应时出错: {str(e)}")

        return reasoning_content.strip(), content.strip()

    def optimize_single_validation_item(
        self, validation_item: ValidationItem, sample_index: int = 0
    ) -> Optional[tuple[ClusterOptimizationOutput, str, dict]]:
        """优化单个验证项

        Args:
            validation_item: 包含输入数据、当前分簇结果和验证反馈的ValidationItem
            sample_index: 样本索引，用于错误追踪

        Returns:
            (优化结果, 格式化系统提示, 原始输出信息) 的元组，如果优化失败则返回None
        """
        try:
            # 应用速率限制
            self.rate_limiter.wait_if_needed()
            
            # 直接使用ValidationItem作为用户输入
            user_input = validation_item.model_dump_json()

            # 格式化输入数据
            user_content = user_input
            user_content = (
                user_content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            )
            user_content = re.sub(r"\s+", " ", user_content)

            # 获取格式说明
            input_instructions = ValidationItem.model_json_schema()
            format_instructions = self.output_parser.get_format_instructions()

            # 生成完整提示
            system_prompt = self.prompt_template.format(
                input_instructions=input_instructions,
                output_format_instructions=format_instructions,
            )
            # system_prompt = (
            #     system_prompt.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            # )
            system_prompt = re.sub(r"\s+", " ", system_prompt)

            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content.strip()},
            ]
            
            print(f"🔧 开始优化样本 {sample_index}")

            # 调用API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                stream=True,
                temperature=self.temperature,
            )

            # 提取思考过程和内容
            reasoning_content, content = self._extract_reasoning_and_content(response)

            if not content.strip():
                print(f"❌ 样本 {sample_index} 优化失败: API返回内容为空")
                return None

            # 解析优化结果
            try:
                optimization_result = self.output_parser.parse(content)

                # 确保时间戳一致
                original_timestamp = validation_item.input.timestamp
                for cluster in optimization_result.clusters:
                    if not cluster.timestamp:
                        cluster.timestamp = original_timestamp

                # 构建完整的思维链（用<think>标签包裹）
                if reasoning_content:
                    optimization_result.chain_of_thought = (
                        reasoning_content
                        + "思考过程总结为："
                        + optimization_result.chain_of_thought
                        if optimization_result.chain_of_thought
                        else reasoning_content
                    )
                else:
                    # 如果没有思考内容，直接使用优化结果
                    optimization_result.chain_of_thought = "<think></think>"

                # 保存原始输出信息
                raw_output_info = {
                    "reasoning_content": reasoning_content,
                    "raw_content": content,
                    "parsed_successfully": True,
                    "timestamp": get_current_timestamp(),
                }

                print(
                    f"✅ 样本 {sample_index} 优化成功 - 是否优化: {optimization_result.is_optimized}"
                )
                return optimization_result, system_prompt, raw_output_info

            except Exception as e:
                print(f"❌ 样本 {sample_index} 优化失败 - 解析输出失败: {str(e)}")
                return None

        except Exception as e:
            print(f"❌ 样本 {sample_index} 优化失败 - API调用失败: {str(e)}")
            return None

    def create_sharegpt_format(
        self,
        validation_item: ValidationItem,
        optimization_result: ClusterOptimizationOutput,
        formatted_system_prompt: str,
    ) -> ShareGPTFormat:
        """创建ShareGPT格式的训练数据

        Args:
            validation_item: 原始验证项
            chain_of_thought: 思维链（已包含<think>标签）
            optimization_result: 优化结果
            formatted_system_prompt: 已格式化的系统提示（包含输入输出格式说明）

        Returns:
            ShareGPT格式的对话数据
        """
        # 用户输入：验证项的完整信息
        user_content = validation_item.model_dump_json()

        return ShareGPTFormat(
            messages=[
                ShareGPTMessage(role="system", content=formatted_system_prompt),
                ShareGPTMessage(role="user", content=user_content),
                ShareGPTMessage(role="assistant", content=optimization_result.model_dump_json()),
            ]
        )

    def process_single_item(
        self,
        item_data: Tuple[int, ValidationItem],
        writer: ThreadSafeWriter,
        stats_queue: Queue,
    ) -> None:
        """处理单个数据项（在线程中调用）"""
        index, validation_item = item_data

        try:
            # 处理数据
            result = self.optimize_single_validation_item(validation_item, index)

            if result is None:
                # 记录错误
                error_data = {
                    "error": "优化失败",
                    "sample_index": index,
                    "timestamp": get_current_timestamp(),
                    "model": self.model_name,
                }
                writer.write_line(
                    f"ERROR: {json.dumps(error_data, ensure_ascii=False)}"
                )
                stats_queue.put("failed")
                return

            optimization_result, formatted_system_prompt, raw_output_info = result

            # 创建ShareGPT格式数据
            sharegpt_data = self.create_sharegpt_format(
                validation_item,
                optimization_result,
                formatted_system_prompt,
            )

            # 保存完整的处理结果（包含原始输出）
            full_result = {
                "sample_index": index,
                "timestamp": get_current_timestamp(),
                "model": self.model_name,
                "input_validation_item": validation_item.model_dump(),
                "raw_optimization_output": optimization_result.model_dump(),
                "formatted_system_prompt": formatted_system_prompt,
                "raw_llm_output": raw_output_info,  # 添加原始LLM输出
                "sharegpt_format": sharegpt_data.model_dump(),
            }

            # 写入完整结果
            writer.write_line(json.dumps(full_result, ensure_ascii=False))
            stats_queue.put("success")

        except Exception as e:
            error_data = {
                "error": f"处理异常: {str(e)}",
                "sample_index": index,
                "timestamp": get_current_timestamp(),
                "model": self.model_name,
            }
            writer.write_line(f"ERROR: {json.dumps(error_data, ensure_ascii=False)}")
            stats_queue.put("failed")

    def process_validation_data_batch_multithread(
        self, validation_data: List[ValidationItem], output_file: Path
    ) -> Tuple[int, int]:
        """多线程批量处理验证数据并实时保存。

        Args:
            validation_data: 验证数据列表
            output_file: 输出文件路径
            
        Returns:
            (成功数, 失败数)
        """
        # 初始化线程安全的写入器
        writer = ThreadSafeWriter(output_file)
        
        # 创建ShareGPT专用输出文件
        sharegpt_output_file = output_file.parent / f"sharegpt_{output_file.name}"
        sharegpt_writer = ThreadSafeWriter(sharegpt_output_file)

        # 统计信息队列
        stats_queue = Queue()

        # 创建带索引的数据列表
        indexed_data = list(enumerate(validation_data))

        # 使用线程池处理
        print(f"开始多线程处理，使用 {self.max_workers} 个线程")
        print(f"📁 完整输出: {output_file}")
        print(f"📁 ShareGPT格式: {sharegpt_output_file}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(
                    self.process_single_item_with_dual_output, 
                    item, writer, sharegpt_writer, stats_queue
                ): item[0]
                for item in indexed_data
            }

            # 使用tqdm显示进度
            with tqdm(total=len(validation_data), desc="优化进度") as pbar:
                for future in as_completed(future_to_index):
                    try:
                        future.result()  # 获取结果，如果有异常会抛出
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"线程处理样本 {index} 时出现异常: {str(e)}")
                    finally:
                        pbar.update(1)

        # 收集统计信息
        stats = {"total": len(validation_data), "success": 0, "failed": 0}
        while not stats_queue.empty():
            result = stats_queue.get()
            if result == "success":
                stats["success"] += 1
            elif result == "failed":
                stats["failed"] += 1

        # 打印统计信息
        print(f"多线程处理完成统计: {stats}")
        print(f"成功率: {stats['success'] / stats['total'] * 100:.2f}%")
        
        return stats["success"], stats["failed"]

    def process_single_item_with_dual_output(
        self,
        item_data: Tuple[int, ValidationItem],
        full_writer: ThreadSafeWriter,
        sharegpt_writer: ThreadSafeWriter,
        stats_queue: Queue,
    ) -> None:
        """处理单个数据项并写入两个输出文件"""
        index, validation_item = item_data

        try:
            # 处理数据
            result = self.optimize_single_validation_item(validation_item, index)

            if result is None:
                # 记录错误
                error_data = {
                    "error": "优化失败",
                    "sample_index": index,
                    "timestamp": get_current_timestamp(),
                    "model": self.model_name,
                }
                error_json = f"ERROR: {json.dumps(error_data, ensure_ascii=False)}"
                full_writer.write_line(error_json)
                sharegpt_writer.write_line(error_json)
                stats_queue.put("failed")
                return

            optimization_result, formatted_system_prompt, raw_output_info = result

            # 创建ShareGPT格式数据
            sharegpt_data = self.create_sharegpt_format(
                validation_item,
                optimization_result,
                formatted_system_prompt,
            )

            # 保存完整的处理结果（用于分析和调试）
            full_result = {
                "sample_index": index,
                "timestamp": get_current_timestamp(),
                "model": self.model_name,
                "input_validation_item": validation_item.model_dump(),
                "raw_optimization_output": optimization_result.model_dump(),
                "formatted_system_prompt": formatted_system_prompt,
                "raw_llm_output": raw_output_info,
                "sharegpt_format": sharegpt_data.model_dump(),
            }

            # 分别写入两个文件
            full_writer.write_line(json.dumps(full_result, ensure_ascii=False))
            sharegpt_writer.write_line(sharegpt_data.model_dump_json())
            stats_queue.put("success")

        except Exception as e:
            error_data = {
                "error": f"处理异常: {str(e)}",
                "sample_index": index,
                "timestamp": get_current_timestamp(),
                "model": self.model_name,
            }
            error_json = f"ERROR: {json.dumps(error_data, ensure_ascii=False)}"
            full_writer.write_line(error_json)
            sharegpt_writer.write_line(error_json)
            stats_queue.put("failed")

    def process_validation_data_batch(
        self, validation_data: List[ValidationItem], output_file: Path
    ) -> tuple[int, int]:
        """批量处理验证数据（兼容性方法，调用多线程版本）

        Args:
            validation_data: 验证数据列表
            output_file: 输出文件路径

        Returns:
            (成功数, 失败数)
        """
        return self.process_validation_data_batch_multithread(validation_data, output_file)


def main():
    """主函数：加载验证数据并进行优化蒸馏"""

    # 输入文件路径
    input_file = (
        get_data_dir()
        / "cluster_results_sharegpt_training_data"
        / "max_overlap_alg_for_raw_constellation_data_scenario_3_with_history_validation_result_20250726_165538.jsonl"
    )

    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return

    # 加载验证数据
    print(f"📂 加载验证数据: {input_file}")
    validation_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item_data = json.loads(line.strip())
                validation_item = ValidationItem(**item_data)
                validation_data.append(validation_item)
            except Exception as e:
                print(f"⚠️  跳过第{line_num}行，解析失败: {str(e)}")

    print(f"📊 成功加载 {len(validation_data)} 个验证项")

    if not validation_data:
        print("❌ 没有有效的验证数据")
        return
    validation_data = validation_data[:5]

    # 初始化优化蒸馏器
    distiller = OptimizationDistiller(
        model_name="qwen3-235b-a22b-thinking-2507",  # 可以根据需要修改
        temperature=0.3,
        requests_per_minute=60,  # 根据API限制调整
        max_workers=3,  # 建议从3开始，成功后可以增加到5-6
    )

    # 生成输出文件路径
    timestamp = get_current_timestamp()
    output_file = (
        get_data_dir()
        / "optimization_training_data"
        / f"optimization_distilled_sharegpt_{distiller.model_name}_{timestamp}.jsonl"
    )

    # 开始处理
    start_time = time.time()
    success_count, failed_count = distiller.process_validation_data_batch(
        validation_data, output_file
    )
    end_time = time.time()

    print(f"\n✅ 优化蒸馏完成!")
    print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
    print(
        f"📊 平均每个样本耗时: {(end_time - start_time) / len(validation_data):.2f} 秒"
    )
    print(f"📈 成功率: {success_count / len(validation_data) * 100:.1f}%")
    
    # 输出文件信息
    sharegpt_output_file = output_file.parent / f"sharegpt_{output_file.name}"
    print(f"\n📁 输出文件:")
    print(f"- 完整数据（含原始输出）: {output_file}")
    print(f"- ShareGPT训练格式: {sharegpt_output_file}")
    
    # 提供优化建议
    print(f"\n💡 优化建议:")
    print(f"1. 如果成功率较高，可以将 max_workers 增加到 5-6")
    print(f"2. 如果遇到429错误，降低 requests_per_minute 或 max_workers")
    print(f"3. 如果需要更高质量的优化，可以降低 temperature")
    print(f"4. 当前使用模型: {distiller.model_name}")
    print(f"5. 完整数据可用于分析优化效果和调试")
    print(f"6. ShareGPT格式可直接用于模型训练")


if __name__ == "__main__":
    main()
