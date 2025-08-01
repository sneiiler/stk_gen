"""
基于多线程数据蒸馏系统实现。

This module implements a multi-threaded data distillation system using  models
through OpenAI-compatible interface for better universality and performance.
"""

# 标准库导入
import datetime
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 第三方库导入
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from tqdm import tqdm
from icecream import ic
from pydantic import BaseModel, Field

# 本地模块导入
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from data_class.sft_data_models import (
    ClusterInfo,
    RawConstellationDataModel,
    SatelliteClusterOutput,
    ShareGPTFormat,
    ShareGPTMessage,
    LLMConversationMessage,
    ClusterOptimizationResult,
    ClusterOptimizationResult,
)
from data_class.data_validation_models import ValidationItem
from utils.misc_utils import get_data_dir, get_project_root
from utils.prompt_template import get_prompt_template, CLUSTER_OPTIMIZATION_PROMPT_COMPACT

env_path = get_project_root() / ".env"
load_dotenv(env_path)
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

class RejectionSampler:
    """基于拒绝采样的多线程数据蒸馏器

    核心流程：
    1. 拿到原始数据，注入历史上下文
    2. 构建prompt和格式化指令
    3. 多线程并行获取多个采样结果
    4. 使用 ClusterDataValidator 对所有采样结果进行评估
    5. 选择评分最高的结果作为正样本
    6. 其余的采样结果（评分较低的成功结果 + 失败结果）作为负样本

    主要功能：
    - 多线程并发采样提升效率
    - 专业的4维度评价体系（正确性、稳定性、通信代价、观测效能）
    - 拒绝采样机制保证输出质量
    - 自动分类正负样本用于后续训练
    """

    def __init__(
        self,
        model_name: str = "",
        temperature: float = 0.6,
        previous_results: Optional[List[List[ClusterInfo]]] = None,
        max_history_length: int = 5,
        sample_times: int = 6,
        max_parallel_requests: int = 6,
    ):
        """初始化拒绝采样器

        Args:
            model_name: 使用的模型名称
            temperature: 生成温度，控制输出的随机性
            previous_results: 历史分簇结果，用于上下文注入
            max_history_length: 保留的历史记录最大长度
            sample_times: 每次拒绝采样的次数
            max_parallel_requests: 最大并行请求数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.previous_results = previous_results or []
        self.max_history_length = max_history_length
        self.sample_times = sample_times
        self.max_parallel_requests = max_parallel_requests

        # 初始化OpenAI客户端和解析器
        self.client = OpenAI(api_key=api_key_openai, base_url=api_base_openai)
        self.prompt_template = ChatPromptTemplate.from_template(
            template=get_prompt_template("latest")
        )
        self.output_parser = PydanticOutputParser(
            pydantic_object=SatelliteClusterOutput
        )

        print(f"拒绝采样器初始化完成: {self.model_name}, 每次拒绝采样的次数: {self.sample_times}")

    def _inject_historical_context(
        self, data: RawConstellationDataModel
    ) -> RawConstellationDataModel:
        """步骤1：将历史上下文注入到原始数据中

        Args:
            data: 原始的卫星星座数据

        Returns:
            注入历史数据后的增强数据
        """
        enhanced_data = data.model_copy()

        if self.previous_results:
            # 只取最近一条历史记录，避免上下文过长
            latest_result = self.previous_results[-1]
            enhanced_data.history_cluster_result = [latest_result]
        else:
            enhanced_data.history_cluster_result = []

        return enhanced_data

    def update_history(self, new_result: List[ClusterInfo]):
        """更新历史记录"""
        self.previous_results.append(new_result)
        # 保持历史记录长度不超过限制
        if len(self.previous_results) > self.max_history_length:
            self.previous_results = self.previous_results[-self.max_history_length :]

    def _extract_reasoning_and_content(self, response_stream) -> Tuple[str, str]:
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
                    # print(f"获取到新的reasoning，长度: {len(delta.reasoning_content)}")

                # 获取正常内容
                if hasattr(delta, "content") and delta.content:
                    content += delta.content
                    # print(f"获取到新的content，长度: {len(delta.content)}")
                # 检查是否是思维链内容（包含在<think>标签中）
                thought_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if thought_match:
                    thought_content = thought_match.group(1)
                    reasoning_content += thought_content
                    content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    )

        except Exception as e:
            print(f"处理流式响应时出错: {str(e)}")

        return reasoning_content.strip(), content.strip()

    def _single_sample_request(
        self,
        enhanced_data: RawConstellationDataModel,
        sample_index: int,
        attempt_id: int,
    ) -> Optional[LLMConversationMessage]:
        """步骤2：执行单次采样请求（构建prompt并调用模型）

        Args:
            enhanced_data: 已注入历史数据的输入数据
            sample_index: 样本索引，用于错误追踪
            attempt_id: 采样尝试ID（0到sample_times-1）

        Returns:
            LLMConversationMessage对象或None
            如果成功则返回完整的对话消息，失败则返回None（错误信息直接打印）
        """
        try:
            # 格式化输入数据为JSON字符串
            user_content = json.dumps(
                enhanced_data.model_dump(), ensure_ascii=False, separators=(",", ":")
            )
            user_content = re.sub(
                r"\s+",
                " ",
                user_content.replace("\n", " ").replace("\r", " ").replace("\t", " "),
            )

            # 获取输入数据的JSON schema和输出格式说明
            input_instructions = RawConstellationDataModel.model_json_schema()
            format_instructions = self.output_parser.get_format_instructions()

            # 构建完整的系统提示词
            system_prompt = self.prompt_template.format(
                input_instructions=input_instructions,
                output_format_instructions=format_instructions,
            )
            system_prompt = re.sub(
                r"\s+",
                " ",
                system_prompt.replace("\n", " ").replace("\r", " ").replace("\t", " "),
            )

            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content.strip()},
            ]

            # 创建流式请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                stream=True,
                temperature=self.temperature,
            )

            # 提取思考过程和内容
            reasoning_content, content = self._extract_reasoning_and_content(response)

            if not content.strip():
                print(f"样本 {sample_index} 采样 {attempt_id} 失败: API返回内容为空")
                return None

            # 解析JSON输出为结构化对象
            try:
                parsed_output = self.output_parser.parse(content)

                # 合并思考过程和思维链
                cot = parsed_output.chain_of_thought
                if reasoning_content:
                    cot = f"{reasoning_content}\n思考过程总结:\n{cot}"

                # 确保时间戳一致
                original_timestamp = enhanced_data.timestamp

                result = SatelliteClusterOutput(
                    chain_of_thought=cot,
                    clusters=[
                        ClusterInfo(
                            timestamp=(
                                cluster.timestamp
                                if cluster.timestamp
                                else original_timestamp
                            ),
                            cluster_id=cluster.cluster_id,
                            master=cluster.master,
                            sats=cluster.sats,
                            targets=cluster.targets,
                        )
                        for cluster in parsed_output.clusters
                    ],
                )
                
                # 创建LLMConversationMessage对象
                conversation_message = LLMConversationMessage(
                    instruction=system_prompt,
                    input=enhanced_data,
                    response=result
                )
                
                print(f"样本 {sample_index} 采样 {attempt_id} 成功")
                return conversation_message

            except Exception as e:
                if len(reasoning_content) == 0 and "Invalid json output" in str(e):
                    error_msg = f"输出内容长度为:{len(content)}，思考内容解析失败，输出Json解析失败。原始错误：{str(e)}"
                else:
                    error_msg = f"解析输出失败 (样本 {sample_index}, 尝试 {attempt_id}): {str(e)}"
                print(f"样本 {sample_index} 采样 {attempt_id} 失败: {error_msg}")
                return None

        except Exception as e:
            error_msg = f"API调用失败 (样本 {sample_index}, 尝试 {attempt_id}): {str(e)}"
            print(f"样本 {sample_index} 采样 {attempt_id} 失败: {error_msg}")
            return None

    def generate_rejection_sampling_results(
        self, enhanced_data: RawConstellationDataModel, sample_index: int = 0
    ) -> List[LLMConversationMessage]:
        """步骤3：多线程并行获取多个采样结果

        Args:
            enhanced_data: 已注入历史数据的输入数据
            sample_index: 样本索引，用于错误追踪

        Returns:
            成功的LLMConversationMessage列表，包含完整的对话信息
            错误信息直接打印，不再传递
        """
        successful_conversations = []

        # 使用线程池并行执行多次采样
        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            # 提交所有采样任务
            future_to_attempt = {
                executor.submit(
                    self._single_sample_request, enhanced_data, sample_index, i
                ): i
                for i in range(self.sample_times)
            }

            # 收集结果
            for future in as_completed(future_to_attempt):
                attempt_id = future_to_attempt[future]
                try:
                    conversation_message = future.result()

                    if conversation_message is not None:
                        successful_conversations.append(conversation_message)

                except Exception as e:
                    print(f"样本 {sample_index} 采样 {attempt_id} 异常: {str(e)}")
                    continue

        print(
            f"样本 {sample_index} 拒绝采样完成: {len(successful_conversations)}/{self.sample_times} 成功"
        )
        return successful_conversations

    def multiple_sampling_and_evaluation(
        self, enhanced_data: RawConstellationDataModel, sample_index: int = 0
    ) -> Tuple[
        LLMConversationMessage,
        ValidationItem,
        List[LLMConversationMessage],
        List[ValidationItem],
    ]:
        """多次采样并评价选择最佳结果的核心流程

        Args:
            enhanced_data: 注入历史数据后的输入数据
            sample_index: 样本索引，用于错误追踪

        Returns:
            (最佳对话消息, 最佳验证结果, 所有成功的对话消息, 所有验证结果)的元组
            错误信息直接打印，不再传递
        """
        # 1. 采样：多线程并行获取多个采样结果
        successful_conversations = self.generate_rejection_sampling_results(enhanced_data, sample_index)

        if not successful_conversations:
            raise ValueError(
                f"所有 {self.sample_times} 次采样都失败"
            )

        # 2. 评价：一次性评价所有采样结果
        print(f"样本 {sample_index} 开始评价 {len(successful_conversations)} 个采样结果...")
        
        from distill._3_ClusterDataValidator import ClusterDataValidator
        validator = ClusterDataValidator()
        all_validations = validator.validate_output(successful_conversations)
        
        # 找到评分最高的结果
        best_conversation = None
        best_validation = None
        best_score = -1
        
        for i, (conversation, validation) in enumerate(zip(successful_conversations, all_validations)):
            current_score = validation.score
            print(f"  采样 {i}: 总分 {current_score:.1f}")
            
            if current_score > best_score:
                best_conversation = conversation
                best_validation = validation
                best_score = current_score
        
        if best_conversation is None or best_validation is None:
            raise ValueError(f"样本 {sample_index} 评价失败，无法选择最佳结果")
            
        print(f"样本 {sample_index} 选择最佳结果: 总分 {best_score:.1f}")

        # 3. 记录：返回结果，由调用方决定如何记录正负样本
        return (
            best_conversation,
            best_validation,
            successful_conversations,
            all_validations,
        )

    def _save_samples(
        self,
        enhanced_data: RawConstellationDataModel,
        results: List[SatelliteClusterOutput],
        system_prompts: List[str],
        output_file: Path,
        sample_index: int,
        validations: Optional[List[ValidationItem]] = None,
        status: str = "positive",  # "positive" 或 "negative"
        reason: str = "best_result",  # "best_result" 或 "low_score_result"
    ) -> None:
        """保存样本（统一处理正样本和负样本，包含验证评分信息）"""
        try:
            input_str = json.dumps(
                enhanced_data.model_dump(),
                ensure_ascii=False,
                separators=(",", ":"),
            )

            for i, (result, system_prompt) in enumerate(zip(results, system_prompts)):
                output_str = result.to_think_json()

                sample_data = {
                    "sample_index": sample_index,
                    "attempt_id": i,
                    "status": status,
                    "reason": reason,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": self.model_name,
                    "input_data": input_str,
                    "output_data": output_str,
                    "system_prompt": system_prompt,
                    "sharegpt_format": self.create_sharegpt_format(
                        instruction=system_prompt,
                        input_data=input_str,
                        output_data=output_str,
                    ).model_dump(),
                }

                # 添加验证评分信息
                if validations and i < len(validations):
                    validation = validations[i]
                    sample_data["validation"] = {
                        "total_score": validation.score,
                        "validation_details": [
                            {
                                "validation_type": detail.validation_type,
                                "score": detail.score,
                                "info": detail.info,
                            }
                            for detail in validation.validation_details
                        ],
                    }

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                    f.flush()

        except Exception as e:
            print(f"保存{status}样本时出错 (样本 {sample_index}): {str(e)}")

    def create_sharegpt_format(
        self, instruction: str, input_data: str, output_data: str
    ) -> ShareGPTFormat:
        """创建ShareGPT格式的训练数据"""
        return ShareGPTFormat(
            messages=[
                ShareGPTMessage(role="system", content=instruction),
                ShareGPTMessage(role="user", content=input_data),
                ShareGPTMessage(role="assistant", content=output_data),
            ]
        )

    def process_data_batch(
        self,
        list_data: List,
        out_file_path: Path | str,
        stats_queue: Queue,
    ) -> None:
        """批量处理数据

        这是整个拒绝采样系统的主要入口，对每个数据项执行完整的处理流程：
        1. 拿到原始的数据：从list_data中获取原始星座数据
        2. 构建prompt：注入历史分簇信息，构建系统提示词
        3. 多线程获取采样：并行调用模型API，获取多个候选结果
        4. 对采样结果进行评估：使用ClusterDataValidator评分，找到最好的一个结果
        5. 放到正样本，其余的放在负样本：保存训练数据，更新历史记录

        Args:
            list_data: 待处理的原始星座数据列表
            out_file_path: 主要输出文件路径（ShareGPT格式）
            stats_queue: 统计队列，用于多进程间的进度跟踪
        """
        # 如果需要保存所有样本，创建对应的文件路径
        base_path = Path(out_file_path)
        positive_samples_file = (
            base_path.parent / f"{base_path.stem}_positive_samples{base_path.suffix}"
        )
        negative_samples_file = (
            base_path.parent / f"{base_path.stem}_negative_samples{base_path.suffix}"
        )

        for index, item in tqdm(
            enumerate(list_data), total=len(list_data), desc="处理数据批次"
        ):
            # 步骤1：拿到原始的数据
            raw_data = RawConstellationDataModel(**item)

            # 步骤2：构建prompt（注入历史分簇信息）
            enhanced_data = self._inject_historical_context(raw_data)

            # API调用重试机制
            max_retries = 3
            retry_count = 0

            # 重试机制 - 只重试API调用部分
            while retry_count < max_retries:
                try:
                    # 步骤3-4：采样 → 评价 → 记录的核心流程
                    (
                        best_conversation,
                        best_validation,
                        all_conversations,
                        all_validations,
                    ) = self.multiple_sampling_and_evaluation(enhanced_data, index)

                    # 步骤5：结果记录 - 最好的放到正样本，其余的放在负样本
                    # 保存正样本：分数最高的最佳结果
                    best_index = all_conversations.index(best_conversation)
                    self._save_samples(
                        enhanced_data,
                        [best_conversation.response],
                        [best_conversation.instruction],
                        positive_samples_file,
                        index,
                        [all_validations[best_index]],
                        "positive",
                        "best_result",
                    )

                    # 保存负样本：仅保存评分较低的成功结果，错误信息不记录
                    other_conversations = [c for c in all_conversations if c != best_conversation]

                    if other_conversations:
                        # 从其他对话中提取response和instruction
                        other_results = [c.response for c in other_conversations]
                        other_prompts = [c.instruction for c in other_conversations]
                        # 获取对应的验证结果
                        other_validations = [
                            all_validations[i] for i, c in enumerate(all_conversations) 
                            if c != best_conversation
                        ]
                        self._save_samples(  # 评分较低的成功结果作为负样本
                            enhanced_data,
                            other_results,
                            other_prompts,
                            negative_samples_file,
                            index,
                            other_validations,
                            "negative",
                            "low_score_result",
                        )

                    # 错误信息已经在采样过程中直接打印，这里不需要额外处理

                    print(f"样本 {index} 完成，最佳评分: {best_validation.score:.1f}")
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_count
                        print(f"样本 {index} 失败，{wait_time}秒后重试: {str(e)[:100]}")
                        time.sleep(wait_time)
                    else:
                        print(f"样本 {index} 重试 {max_retries} 次后仍失败: {str(e)}")
                        stats_queue.put("failed")
                        break
                    continue

            # 如果处理失败
            if retry_count >= max_retries:
                continue

            # 处理成功后，保存最终的ShareGPT格式训练数据
            try:
                # 将最佳结果保存为标准的ShareGPT训练格式
                input_str = json.dumps(
                    enhanced_data.model_dump(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                output_str = best_conversation.response.to_think_json()

                sharegpt_data = self.create_sharegpt_format(
                    instruction=best_conversation.instruction,
                    input_data=input_str,
                    output_data=output_str,
                )

                # 写入主要的训练数据文件（ShareGPT格式）
                with open(out_file_path, "a", encoding="utf-8") as f:
                    f.write(sharegpt_data.model_dump_json() + "\n")
                    f.flush()

                # 更新历史记录，为下一个样本提供分簇历史信息
                self.update_history(best_conversation.response.clusters)

                stats_queue.put("success")

            except Exception as e:
                print(f"保存结果失败 (样本 {index}): {str(e)}")
                stats_queue.put("failed")

    def optimize_clusters_with_validation(
        self,
        validation_item: ValidationItem,
        sample_index: int = 0
    ) -> Optional[ClusterOptimizationResult]:
        """基于验证反馈优化分簇结果
        
        Args:
            validation_item: 包含输入数据、当前分簇结果和验证反馈的ValidationItem
            sample_index: 样本索引，用于错误追踪
            
        Returns:
            优化后的分簇结果，如果优化失败则返回None
        """
        try:
            # 构建优化专用的输出解析器
            optimization_parser = PydanticOutputParser(pydantic_object=ClusterOptimizationResult)
            
            # 构建优化输入数据
            optimization_input = {
                "timestamp": validation_item.input.timestamp,
                "current_clusters": [cluster.model_dump() for cluster in validation_item.response],
                "validation_details": {
                    detail.validation_type: {
                        "score": detail.score,
                        "info": detail.info
                    }
                    for detail in validation_item.validation_details
                },
                "sat_attrs": [attr.model_dump() for attr in validation_item.input.sat_attrs],
                "sat_edges": [edge.model_dump() for edge in validation_item.input.sat_edges],
                "target_edges": [edge.model_dump() for edge in validation_item.input.target_edges],
                "history_cluster_result": (
                    [cluster.model_dump() for cluster in validation_item.input.history_cluster_result[-1]]
                    if validation_item.input.history_cluster_result 
                    else None
                )
            }
            
            # 准备用户输入
            user_input = json.dumps(optimization_input, ensure_ascii=False)
            
            # 构建消息
            messages = [
                {"role": "system", "content": CLUSTER_OPTIMIZATION_PROMPT_COMPACT},
                {"role": "user", "content": user_input}
            ]
            
            print(f"🔧 开始优化样本 {sample_index} - 基于验证反馈进行精细化调整")
            
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
                optimization_result = optimization_parser.parse(content)
                
                # 确保时间戳一致
                original_timestamp = validation_item.input.timestamp
                for cluster in optimization_result.clusters:
                    if not cluster.timestamp:
                        cluster.timestamp = original_timestamp
                
                print(f"✅ 样本 {sample_index} 优化成功")
                return optimization_result
                
            except Exception as e:
                print(f"❌ 样本 {sample_index} 优化失败 - 解析输出失败: {str(e)}")
                return None
                
        except Exception as e:
            print(f"❌ 样本 {sample_index} 优化失败 - API调用失败: {str(e)}")
            return None
    
    def batch_optimize_with_validation(
        self,
        validation_items: List[ValidationItem],
        output_file: str,
        max_workers: int = 3
    ) -> Tuple[int, int]:
        """批量优化分簇结果
        
        Args:
            validation_items: 验证项列表，每项包含输入数据、当前分簇结果和验证反馈
            output_file: 输出文件路径
            max_workers: 最大并行工作线程数
            
        Returns:
            (成功数, 失败数)
        """
        success_count = 0
        failed_count = 0
        
        print(f"🚀 开始批量优化 {len(validation_items)} 个分簇结果")
        print(f"📝 输出文件: {output_file}")
        
        # 创建线程安全的写入器
        writer = ThreadSafeWriter(Path(output_file))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有优化任务
            future_to_index = {
                executor.submit(self.optimize_clusters_with_validation, validation_item, i): i
                for i, validation_item in enumerate(validation_items)
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(validation_items), desc="优化进度") as pbar:
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    
                    try:
                        result = future.result()
                        
                        if result is not None:
                            # 保存优化结果
                            optimization_data = {
                                "index": index,
                                "input": validation_items[index].model_dump(),
                                "output": result.model_dump(),
                                "timestamp": validation_items[index].input.timestamp
                            }
                            
                            writer.write_line(json.dumps(optimization_data, ensure_ascii=False))
                            success_count += 1
                            
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        print(f"❌ 样本 {index} 处理异常: {str(e)}")
                        failed_count += 1
                    
                    pbar.update(1)
        
        print(f"✅ 批量优化完成!")
        print(f"📊 成功: {success_count}, 失败: {failed_count}")
        print(f"📁 优化结果已保存到: {output_file}")
        
        return success_count, failed_count


def main():
    """拒绝采样系统的主函数示例

    演示如何使用RejectionSampler完成从原始数据到训练数据的完整流程：
    1. 加载原始的星座观测数据
    2. 配置采样参数（模型、温度、并行数等）
    3. 批量处理数据，对每个样本执行"采样→评价→记录"
    4. 保存ShareGPT格式的训练数据和正负样本分析文件
    """
    # 从JSON文件加载数据
    input_file = get_data_dir() / "raw_constellation_data_scenario_3.json"
    with open(input_file, "r", encoding="utf-8") as f:
        batch_data = json.load(f)

    print(f"加载了 {len(batch_data)} 个数据样本")
    model_name = "qwen3-1.7b"
    stats_queue = Queue()

    # 初始化蒸馏器（带历史记录和拒绝采样）
    distiller = RejectionSampler(
        model_name=model_name,
        temperature=0.6,
        previous_results=None,  # 如果有历史结果，在这里传入
        max_history_length=5,
        sample_times=10,  # 拒绝采样次数
        max_parallel_requests=5,  # 最大并行请求数
    )

    # 生成输出文件路径
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        get_data_dir()
        / f"rejection_sampling_training_data_sharegpt_{model_name}_{now}_{str(input_file)[-10:-5]}.jsonl"
    )

    # 处理数据批次并实时保存
    start_time = time.time()
    distiller.process_data_batch(
        batch_data,
        output_file,
        stats_queue=stats_queue,
    )
    end_time = time.time()

    print(f"\n✅ 处理完成!")
    print(f"📁 结果已保存到: {output_file}")

    base_path = Path(output_file)
    positive_file = (
        base_path.parent / f"{base_path.stem}_positive_samples{base_path.suffix}"
    )
    negative_file = (
        base_path.parent / f"{base_path.stem}_negative_samples{base_path.suffix}"
    )
    print(f"📁 正样本文件: {positive_file}")
    print(f"📁 负样本文件: {negative_file}")

    print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 平均每个样本耗时: {(end_time - start_time) / len(batch_data):.2f} 秒")
    print(f"\n💡 系统现在使用 ClusterDataValidator 进行专业评价")
    print(f"💡 流程: 采样 → 评价 → 记录，简单直接")


if __name__ == "__main__":
    main()
