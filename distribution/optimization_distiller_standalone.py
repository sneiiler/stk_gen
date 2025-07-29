#!/usr/bin/env python3
"""
卫星分簇优化蒸馏系统 - 独立分发版本

基于验证反馈对分簇结果进行优化，生成高质量的训练数据
支持单文件分发，包含所有必要的依赖定义

Version: 2.0.0 (Standalone Distribution)
Author: YinKaifeng
"""

# =============================================================================
# 标准库导入
# =============================================================================
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import List, Optional, Tuple, Dict, Any, Union, Literal
from collections import defaultdict
import datetime

# =============================================================================
# 第三方库导入
# =============================================================================
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel, Field

# =============================================================================
# 内置数据模型定义 (代替外部依赖)
# =============================================================================

# --- 基础数据模型 ---
class SatelliteAttributes(BaseModel):
    """卫星属性模型"""
    id: Union[str, int] = Field(..., description="卫星ID")
    health: float = Field(..., ge=0, le=10, description="卫星健康状态 (0-10)")
    pos: List[float] = Field(..., description="卫星ECEF位置坐标 [x, y, z] km")

class SatelliteEdge(BaseModel):
    """卫星间连接关系模型"""
    from_sat: str = Field(..., description="起始卫星ID")
    to_sat: str = Field(..., description="目标卫星ID")
    distance: float = Field(..., description="卫星距离，单位km")

class TargetEdge(BaseModel):
    """卫星到目标的连接关系模型"""
    sat_id: str = Field(..., description="起始卫星ID")
    target_id: str = Field(..., description="目标ID")
    quality: float = Field(..., description="连接质量 (0-1)")

class ClusterInfo(BaseModel):
    """分簇信息模型"""
    timestamp: Optional[str] = Field(..., description="ISO8601格式的时间戳字符串")
    cluster_id: int = Field(description="分簇ID")
    master: str = Field(description="主节点卫星ID")
    sats: List[str] = Field(description="分簇中的卫星ID列表")
    targets: List[str] = Field(description="分簇观测的目标ID列表")

class RawConstellationDataModel(BaseModel):
    """原始卫星分簇数据模型"""
    timestamp: str = Field(..., description="ISO8601格式的时间戳字符串")
    sat_attrs: List[SatelliteAttributes] = Field(..., description="卫星属性列表")
    sat_edges: List[SatelliteEdge] = Field(..., description="卫星间连接关系列表")
    target_edges: List[TargetEdge] = Field(..., description="卫星到目标的连接关系列表")
    history_cluster_result: Optional[List[List[ClusterInfo]]] = Field(
        ..., description="上n次分簇结果"
    )

class SatelliteClusterOutput(BaseModel):
    """卫星分簇划分输出模型"""
    chain_of_thought: Optional[str] = Field(
        description="推理过程，大模型生成阶段不需要填写，后期封装"
    )
    clusters: List[ClusterInfo] = Field(description="划分的卫星分簇列表")

    class Config:
        extra = "forbid"

class LLMConversationMessage(BaseModel):
    """卫星分簇算法的完整模型问答对话数据"""
    instruction: str = Field(..., description="给模型的指令，通常是系统提示词")
    input: RawConstellationDataModel = Field(..., description="输入的卫星星座数据")
    response: SatelliteClusterOutput = Field(
        ..., description="包含推理过程和分簇结果的完整响应"
    )

# --- 验证数据模型 ---
class ValidationDetail(BaseModel):
    """验证详情模型"""
    validation_type: Literal[
        "correctness_validation",
        "stability_validation",
        "communication_cost_validation",
        "observation_efficiency_validation",
    ] = Field(..., description="验证类型")
    score: int = Field(..., description="分项得分")
    info: str = Field(..., description="警告信息")

class ValidationItem(BaseModel):
    """验证项模型"""
    input: RawConstellationDataModel = Field(..., description="输入数据")
    response: List[ClusterInfo] = Field(..., description="模型响应提取的数据")
    validation_details: List[ValidationDetail] = Field(..., description="验证详情")

    @property
    def score(self) -> float:
        """计算总分数，基于加权求和"""
        weights = {
            "correctness_validation": 0.4,
            "stability_validation": 0.3,
            "communication_cost_validation": 0.15,
            "observation_efficiency_validation": 0.15,
        }
        weighted_score = 0.0
        for detail in self.validation_details:
            weight = weights.get(detail.validation_type, 0)
            weighted_score += detail.score * weight
        return weighted_score

# --- 优化输出模型 ---
class ClusterOptimizationOutput(BaseModel):
    """卫星分簇优化输出模型"""
    is_optimized: bool = Field(description="是否进行了优化")
    optimization_reason: str = Field(description="优化原因")
    chain_of_thought: Optional[str] = Field(
        description="思维链，包含优化决策过程", default=""
    )
    clusters: List[ClusterInfo] = Field(description="优化后的分簇结果")

    class Config:
        extra = "forbid"

# --- ShareGPT格式模型 ---
class ShareGPTMessage(BaseModel):
    """ShareGPT消息格式"""
    role: Literal["system", "user", "assistant"]
    content: str

class ShareGPTFormat(BaseModel):
    """ShareGPT格式的训练数据定义"""
    messages: List[ShareGPTMessage]

# =============================================================================
# 内置工具函数 (代替外部依赖)
# =============================================================================

def get_current_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_project_root() -> Path:
    """获取项目根目录路径"""
    # 对于分发版本，直接使用当前脚本所在目录作为根目录
    return Path(__file__).parent

def get_data_dir() -> Path:
    """获取数据目录路径"""
    data_dir = get_project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

# =============================================================================
# 内置提示模板 (代替外部依赖)
# =============================================================================

CLUSTER_OPTIMIZATION_PROMPT_COMPACT = """
你是卫星动态观测分簇任务优化专家。基于规则分簇结果和验证反馈进行**精细化优化**注意：你的任务是优化现有结果，而不是重新生成全新的分簇方案，如果没有优化的空间可以不进行优化。

## 核心任务
**在保持历史稳定性和现有分簇结构基础上，最小化调整，最大化四维验证评分，保持历史稳定性**

## 评分体系（100分）
1. **正确性验证**(40分): 禁止虚构ID，100%目标覆盖，无重复分配，避免跨簇连接
2. **稳定性验证**(30分): 保持历史伙伴关系，Jaccard≥0.8，合理环境变化豁免
3. **通信代价**(15分): 主节点属于其簇，无孤星，代价<10%
4. **观测效能**(15分): 目标簇有观测卫星，平均观测重数≥2.5

## 优化策略
### 诊断分析
1. 解析validation_details中的具体失分点
2. 优先级: 致命错误(0分)→高权重维度→其他维度

### 最小化调整原则
1. 保持现有分簇结构基础
2. 针对validation_details的具体问题进行修复
3. 如有历史数据，优先恢复仍有效的伙伴关系：
   - 目标伙伴：保持仍可见的历史观测卫星
   - 卫星伙伴：保持仍连通的历史协作关系
4. 每次调整控制在≤20%成员变化

### 质量检查
- ✅ 所有target_edges中目标都被分配且无重复
- ✅ 所有ID都真实存在
- ✅ 目标簇有能观测它的卫星
- ✅ 主节点属于其簇且连通
- ✅ 保持历史连续性

## 输入数据格式概要：
ValidationItem: input (原始输入数据), response (模型分簇结果), validation_details (各项验证详情)
ClusterInfo: timestamp (时间戳), cluster_id (分簇ID), master (主节点), sats (簇内卫星列表), targets (簇观测目标列表)
RawConstellationDataModel: timestamp (时间戳), sat_attrs (卫星属性列表), sat_edges (卫星间连接关系), target_edges (卫星到目标观测关系), history_cluster_result (历史分簇结果)
SatelliteAttributes: id (卫星ID), health (健康状态), pos (位置坐标)
SatelliteEdge: from_sat (起始卫星ID), to_sat (目标卫星ID), distance (距离)
TargetEdge: sat_id (观测卫星ID), target_id (目标ID), quality (连接质量)
ValidationDetail: validation_type (验证类型), score (分项得分), info (详细信息/警告)

## IMPORTANT:
{output_format_instructions}
"""

def get_cluster_optimization_prompt():
    """获取卫星分簇优化专用prompt"""
    return CLUSTER_OPTIMIZATION_PROMPT_COMPACT

# =============================================================================
# 主要功能类
# =============================================================================

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
        api_base_url: str,
        api_key: str,
        proxy: Optional[str] = None,
        requests_per_minute: int = 60,
        max_workers: int = 3,
    ):
        """初始化优化蒸馏器

        Args:
            model_name: 使用的模型名称
            temperature: 生成温度，控制输出的随机性
            api_base_url: API基础URL
            api_key: API密钥
            proxy: 代理设置
            requests_per_minute: 每分钟最大请求数
            max_workers: 最大并发线程数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_workers = max_workers
        self.rate_limiter = RateLimiter(requests_per_minute)

        # 初始化OpenAI兼容客户端
        if proxy is not None:
            import httpx
            httpx_client = httpx.Client(proxy=proxy)
            print(f"使用代理: {proxy}")
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
                http_client=httpx_client,
            )
        else:
            self.client = OpenAI(api_key=api_key, base_url=api_base_url)

        self.output_parser = PydanticOutputParser(
            pydantic_object=ClusterOptimizationOutput
        )

        # 构建聊天提示模板
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

        # 针对Gemini模型的特殊处理
        if "gemini" in self.model_name.lower():
            # 检查是否是Gemini的思维链内容（包含在<thought>标签中）
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought_content = thought_match.group(1).strip()

                if not reasoning_content.strip() or len(thought_content) > len(
                    reasoning_content
                ):
                    reasoning_content = thought_content

                # 从content中移除thought部分
                content = re.sub(
                    r"<thought>.*?</thought>", "", content, flags=re.DOTALL
                ).strip()

        return reasoning_content.strip(), content.strip()

    def optimize_single_validation_item(
        self, validation_item: ValidationItem, sample_index: int = 0
    ) -> Optional[tuple[ClusterOptimizationOutput, str, dict]]:
        """优化单个验证项"""
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
            system_prompt = re.sub(r"\s+", " ", system_prompt)

            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content.strip()},
            ]

            print(f"🔧 开始优化样本 {sample_index}")

            # 调用API
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "temperature": self.temperature,
            }

            # 如果是Gemini模型，添加thinking配置
            if "gemini" in self.model_name.lower():
                api_params["extra_body"] = {
                    "extra_body": {
                        "google": {
                            "thinking_config": {
                                "thinking_budget": 12000,
                                "include_thoughts": True,
                            }
                        }
                    }
                }

            response = self.client.chat.completions.create(**api_params)

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
        """创建ShareGPT格式的训练数据"""
        # 用户输入：验证项的完整信息
        user_content = validation_item.model_dump_json()

        return ShareGPTFormat(
            messages=[
                ShareGPTMessage(role="system", content=formatted_system_prompt),
                ShareGPTMessage(role="user", content=user_content),
                ShareGPTMessage(
                    role="assistant", content=optimization_result.model_dump_json()
                ),
            ]
        )

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

    def process_validation_data_batch_multithread(
        self, validation_data: List[ValidationItem], output_file: Path
    ) -> Tuple[int, int]:
        """多线程批量处理验证数据并实时保存"""
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
                    item,
                    writer,
                    sharegpt_writer,
                    stats_queue,
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

    def process_validation_data_batch(
        self, validation_data: List[ValidationItem], output_file: Path
    ) -> tuple[int, int]:
        """批量处理验证数据（兼容性方法，调用多线程版本）"""
        return self.process_validation_data_batch_multithread(
            validation_data, output_file
        )


# =============================================================================
# 配置和主函数
# =============================================================================

# =============================================================================
# 配置和主函数
# =============================================================================

def get_api_config():
    """获取API配置 - 直接在代码中配置"""
    
    # ========================================
    # 在这里直接配置你的API信息
    # ========================================
    API_CONFIG = {
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai/",  # 修改为你的API基础URL
        "api_key": "dOmKUb0X49w",              # 修改为你的API密钥
        "proxy": "socks5://127.0.0.1:1089",                               # 代理设置，例如: "socks5://127.0.0.1:1089"
    }
    
    return API_CONFIG["api_base"], API_CONFIG["api_key"], API_CONFIG["proxy"]


def load_environment_config():
    """加载环境配置 - 已废弃，使用 get_api_config() 代替"""
    return get_api_config()


def main():
    """主函数：加载验证数据并进行优化蒸馏"""
    
    # 加载环境配置
    api_base, api_key, proxy = get_api_config()
    if not api_base or not api_key:
        return

    # 输入文件路径配置
    # 请根据实际情况修改输入文件路径
    input_file = (
        get_data_dir()
        / "validation_result.jsonl"  # 请将此处修改为你的实际输入文件名
    )

    if not input_file.exists():
        print(f"❌ 输入文件不存在: {input_file}")
        print(f"请确保将验证数据文件放置在 {get_data_dir()} 目录下")
        print(f"或者修改脚本中的 input_file 路径配置")
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

    # 初始化优化蒸馏器
    distiller = OptimizationDistiller(
        model_name="gemini-2.5-pro",  # 可以根据需要修改
        temperature=0.3,  # 优化任务建议使用较低温度
        api_base_url=api_base,
        api_key=api_key,
        proxy=proxy,
        requests_per_minute=100,  # 根据API限制调整
        max_workers=4,  # 建议从3开始，成功后可以增加到5-6
    )

    # 生成输出文件路径
    timestamp = get_current_timestamp()
    output_file = (
        get_data_dir()
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


if __name__ == "__main__":
    print("=" * 60)
    print("卫星分簇优化蒸馏系统 - 独立分发版本")
    print("Version: 2.0.0 (Standalone Distribution)")
    print("=" * 60)
    main()
