"""
基于LangChain的数据蒸馏系统实现。

This module implements a data distillation system using LangChain and OpenAI API.
It provides functionality for batch processing and optimizing training data with
better output parsing capabilities.
"""

import asyncio
from http import HTTPStatus
import os
import json
import re
from typing import List, Dict, Any, TypedDict, Literal, Optional, Tuple
import datetime
from pathlib import Path
import dashscope
try:
    from langchain_core.runnables import RunnableConfig
    from langchain.callbacks.base import AsyncCallbackHandler
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from langchain.schema import BaseOutputParser
except ImportError:
    print("警告: LangChain依赖未安装，请运行: pip install langchain langchain-openai")
    raise

from pydantic import BaseModel, Field
from tqdm import tqdm
from icecream import ic,install
install()
from utils.misc_utils import get_data_dir, get_project_root
from utils.prompt_template import get_prompt_template
from data_models.sft_data_models import ClusterInfo, RawConstellationDataModel, SatelliteClusterOutput, ShareGPTFormat, ShareGPTMessage
from dotenv import load_dotenv
env_path = get_project_root() / ".env"
load_dotenv(env_path)

# 获取配置
api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
api_base = os.getenv("OPENAI_COMPATIBLE_API_BASE")

if not api_key or not api_base:
    raise ValueError("未找到.env文件或环境变量，请检查配置")


class CaptureTokenHandler(AsyncCallbackHandler):
    def __init__(self):
        self.tokens = []

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        if not token:
            return
        self.tokens.append(token)
        print(token, end="", flush=True)

    def get_full_output(self) -> str:
        return "".join(self.tokens)


class LangChainDataDistiller:
    """基于LangChain的数据蒸馏器类。

    用于批量处理和优化训练数据的类，使用LangChain的输出解析功能。

    Attributes:
        llm: LangChain的LLM实例
        prompt_template: LangChain的提示模板
        output_parser: 输出解析器
    """

    def __init__(self, model_name: str = "qwen3-235b-a22b", temperature: float = 0.1):
        """初始化数据蒸馏器。

        Args:
            model_name: 要使用的模型名称
            temperature: 生成温度，控制输出的随机性
        """
        
        # 初始化LangChain组件
        # self.llm = ChatOpenAI(
        #     model=model_name,
        #     temperature=temperature,
        #     api_key=api_key,  # type: ignore
        #     base_url=api_base,
        #     streaming=True
        # )

        self.model_name = model_name
        self.temperature = temperature
        
        # 获取prompt模板
        self.prompt_template_str = get_prompt_template("latest")  # 使用最新版本
        
        # 创建输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=SatelliteClusterOutput)
        
        # 创建提示模板
        self.prompt_template = ChatPromptTemplate.from_template(
            template=self.prompt_template_str
        )

    def generate_prompt(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """生成提示消息。

        Args:
            data: 输入数据

        Returns:
            格式化的提示字符串和输入数据的元组
        """
        # 清理数据，移除多余空白字符
        def clean_text(text: str) -> str:
            """清理文本中的多余空白字符"""
            import re
            text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        
        # 格式化输入数据
        data_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        data_str = clean_text(data_str)
        
        # 获取格式说明
        input_instructions = RawConstellationDataModel.model_json_schema()
        format_instructions = self.output_parser.get_format_instructions()
        
        # 生成完整提示
        prompt = self.prompt_template.format(
            input_instructions=input_instructions,
            format_instructions=format_instructions,
        )
        return prompt, data_str


    def process_single_data_dashscope(self, data: Dict[str, Any]):
        """使用dashscope处理单条数据。

        Args:
            data: 输入数据

        Returns:
            处理结果，包含chain_of_thought和clusters
        """
        prompt, data_str = self.generate_prompt(data)
        messages = [{"role":"system","content":prompt},
                    {"role":"user","content":data_str}]

        # 确保api_key不为None
        if not api_key:
            raise ValueError("API key不能为空")

        responses = dashscope.Generation.call(
            model=self.model_name,
            api_key=api_key,
            messages=messages, # type: ignore
            stream=True,
            result_format='message',  # 将返回结果格式设置为 message
            top_p=0.8,
            temperature=0.1,
            enable_search=False,
            enable_thinking=True,
            thinking_budget=2000
        )

        full_output = ""
        reasoning_content = ""
        
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                # 检查是否有reasoning_content（思维链）
                if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                    for choice in response.output.choices:
                        if hasattr(choice, 'message'):
                            # 获取reasoning_content（思维链）
                            if hasattr(choice.message, 'reasoning_content'):
                                content = choice.message.reasoning_content
                                if content and isinstance(content, str):
                                    reasoning_content += content
                                    print(content, end="", flush=True)
                            
                            # 获取content（最终结果）
                            if hasattr(choice.message, 'content'):
                                content = choice.message.content
                                if content and isinstance(content, str):
                                    full_output += content
                                    print(content, end="", flush=True)
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
                raise Exception(f"API调用失败: {response.message}")

        # 解析JSON输出
        try:
            # 直接使用output_parser解析，它会自动处理各种格式
            parsed_output = self.output_parser.parse(full_output)
            
            # 提取思维链
            cot = parsed_output.chain_of_thought
            # 如果有reasoning_content，将其添加到思维链前面
            if reasoning_content:
                cot = reasoning_content.strip() + "\n思考过程总结为:" + cot
            
            result = SatelliteClusterOutput(
                chain_of_thought=cot,
                clusters=[
                    ClusterInfo(
                        cluster_id=cluster.cluster_id,
                        master=cluster.master,
                        sats=cluster.sats,
                        targets=cluster.targets
                    )
                    for cluster in parsed_output.clusters
                ]
            )
            return result

        except Exception as e:
            ic(f"解析输出失败: {str(e)}")
            ic(f"原始输出: {full_output}")
            ic(f"思维链内容: {reasoning_content}")
            return None


    def validate_output(self, output: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证输出结果。

        Args:
            output: LLM输出结果
            input_data: 原始输入数据

        Returns:
            验证结果，包含验证状态和错误信息
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # 1. 检查输出格式
            if not output or "clusters" not in output:
                validation_result["is_valid"] = False
                validation_result["errors"].append("输出格式错误：缺少clusters字段")
                return validation_result
            
            # 2. 检查目标覆盖
            input_targets = set()
            for edge in input_data.get("target_edges", []):
                input_targets.add(edge["to"])
            
            output_targets = set()
            for cluster in output["clusters"]:
                output_targets.update(cluster.get("targets", []))
            
            if input_targets != output_targets:
                validation_result["is_valid"] = False
                validation_result["errors"].append(
                    f"目标覆盖不匹配：输入目标 {input_targets}，输出目标 {output_targets}"
                )
            
            # 3. 检查卫星分配
            input_satellites = set()
            for attr in input_data.get("sat_attrs", []):
                input_satellites.add(attr["id"])
            
            output_satellites = set()
            for cluster in output["clusters"]:
                output_satellites.update(cluster.get("sats", []))
            
            # 检查是否有卫星被重复分配
            all_output_sats = []
            for cluster in output["clusters"]:
                all_output_sats.extend(cluster.get("sats", []))
            
            if len(all_output_sats) != len(set(all_output_sats)):
                validation_result["is_valid"] = False
                validation_result["errors"].append("存在重复分配的卫星")
            
            # 4. 检查主节点
            for cluster in output["clusters"]:
                master = cluster.get("master")
                sats = cluster.get("sats", [])
                if master not in sats:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(
                        f"集群 {cluster.get('cluster_id')} 的主节点不在卫星列表中"
                    )
            
            # 5. 检查策略约束
            strategy = input_data.get("strategy", "balanced")
            for cluster in output["clusters"]:
                sat_count = len(cluster.get("sats", []))
                target_count = len(cluster.get("targets", []))
                
                if strategy == "balanced" and sat_count > target_count:
                    validation_result["warnings"].append(
                        f"集群 {cluster.get('cluster_id')} 卫星数超过目标数"
                    )
                elif strategy == "quality" and sat_count > 2 * target_count:
                    validation_result["warnings"].append(
                        f"集群 {cluster.get('cluster_id')} 卫星数超过2倍目标数"
                    )
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"验证过程出错: {str(e)}")
        
        return validation_result

    def create_sharegpt_format(self, instruction: str, input_data: str, output_data: str) -> ShareGPTFormat:
        """创建ShareGPT格式的训练数据。

        Args:
            instruction: 指令内容
            input_data: 输入数据
            output_data: 输出数据

        Returns:
            ShareGPT格式的数据
        """
        # 构建助手消息（包含输出）
        assistant_message = output_data
        
        return ShareGPTFormat(
            messages=[
                ShareGPTMessage(role="system", content=instruction),
                ShareGPTMessage(role="user", content=input_data),
                ShareGPTMessage(role="assistant", content=assistant_message)
            ]
        )

    def process_batch(
        self, 
        batch_data: List[Dict[str, Any]], 
        output_file: Path,
        validation_enabled: bool = True
    ) -> None:
        """批量处理数据并实时保存。

        Args:
            batch_data: 包含任务描述和输入数据的列表
            output_file: 输出文件路径
            validation_enabled: 是否启用输出验证
        """
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        stats = {
            "total": len(batch_data),
            "success": 0,
            "failed": 0,
            "validation_errors": 0
        }

        with open(output_file, "a", encoding="utf-8") as f:
            for i, data in enumerate(tqdm(batch_data, desc="处理数据批次")):
                try:
                    # 处理数据
                    result = self.process_single_data_dashscope(data)
                    
                    if result is None:
                        stats["failed"] += 1
                        continue
                    
                    # 验证输出（如果启用）
                    validation_result = None
                    if validation_enabled:
                        validation_result = self.validate_output(result.model_dump(), data)
                        if not validation_result["is_valid"]:
                            stats["validation_errors"] += 1
                            ic(f"验证失败 (样本 {i}): {validation_result['errors']}")
                            # 可以选择是否继续保存验证失败的结果
                            # continue
                    
                    # 构造ShareGPT格式的训练数据
                    input_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                    output_str = result.to_think_json()
                    
                    sharegpt_data = self.create_sharegpt_format(
                        instruction=self.prompt_template_str,
                        input_data=input_str,
                        output_data=output_str
                    )
                    # 实时写入文件
                    f.write(sharegpt_data.model_dump_json() + "\n")  
                    f.flush()  # 确保立即写入磁盘

                    stats["success"] += 1
                except Exception as e:
                    stats["failed"] += 1
                    ic(f"处理数据时出错 (样本 {i}): {str(e)}")
                    # 记录错误数据
                    error_data = {
                        "error": str(e),
                        "data": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                        "sample_index": i
                    }
                    f.write(json.dumps(error_data, ensure_ascii=False) + "\n")
                    f.flush()
        
        # 打印统计信息
        ic(f"处理完成统计: {stats}")


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
    input_file = get_data_dir() / "mock_satellite_observation_data_20250627_093341_v2.json"
    batch_data = load_json_data(input_file)
    
    ic(f"加载了 {len(batch_data)} 个数据样本")

    # 初始化蒸馏器
    distiller = LangChainDataDistiller()

    # 生成输出文件路径
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = get_data_dir() / f"training_data_sharegpt_{now}_v2.json"

    # 处理数据并实时保存
    distiller.process_batch(batch_data, output_file, validation_enabled=False)
    print(f"处理完成，结果已保存到 {output_file}")

    # # 测试单条数据处理
    # result = distiller.process_single_data_dashscope(batch_data[0])
    # if result:
    #     print("处理结果:")
    #     print(f"思维链 (chain_of_thought): {result['chain_of_thought']}")
    #     print(f"分簇结果 (clusters): {json.dumps(result['clusters'], ensure_ascii=False, indent=2)}")
    # else:
    #     print("处理失败")


if __name__ == "__main__":
    main() 