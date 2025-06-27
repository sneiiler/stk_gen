"""
基于LangChain的数据蒸馏系统实现。

This module implements a data distillation system using LangChain and OpenAI API.
It provides functionality for batch processing and optimizing training data with
better output parsing capabilities.
"""
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
print(root_dir)
sys.path.append(str(root_dir))

import json
import re
from typing import List, Dict, Any, Tuple
import datetime
from pathlib import Path
from openai import OpenAI

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
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
api_base_deepseek = os.getenv("DEEPSEEK_API_BASE")
api_key_deepseek = os.getenv("DEEPSEEK_API_KEY")


if not api_key or not api_base:
    raise ValueError("未找到.env文件或环境变量，请检查配置")


class DataDistiller:
    """基于OpenAI API的数据蒸馏器类。

    用于批量处理和优化训练数据的类，使用OpenAI API的输出解析功能。

    Attributes:
        llm: OpenAI API的LLM实例
        prompt_template: OpenAI API的提示模板
        output_parser: OpenAI API的输出解析器
    """

    def __init__(self, model_name: str = "deepseek-reasoner", temperature: float = 0.1):
        """初始化数据蒸馏器。

        Args:
            model_name: 要使用的模型名称
            temperature: 生成温度，控制输出的随机性
        """

        self.model_name = model_name
        self.temperature = temperature

        self.client = OpenAI(api_key=api_key_deepseek, base_url=api_base_deepseek)
        
        # 获取prompt模板
        self.prompt_template = ChatPromptTemplate.from_template(
            template=get_prompt_template("latest")
        )
        
        # 创建输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=SatelliteClusterOutput)

    def generate_distill_result(self, data: Dict[str, Any]) -> Tuple[SatelliteClusterOutput | None, str]:
        """生成提示消息。

        Args:
            data: 输入数据

        Returns:
            格式化的提示字符串和输入数据的元组
        """
        # 格式化输入数据
        user_content = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        user_content = user_content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        user_content = re.sub(r"\s+", " ", user_content)
        
        # 获取格式说明
        input_instructions = RawConstellationDataModel.model_json_schema()
        format_instructions = self.output_parser.get_format_instructions()
        
        # 生成完整提示
        system_prompt = self.prompt_template.format(
            input_instructions=input_instructions,
            output_format_instructions=format_instructions,
        )
        system_prompt = system_prompt.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        system_prompt = re.sub(r"\s+", " ", system_prompt)

        messages = [
            {"role":"system","content":system_prompt.strip()},
            {"role":"user","content":user_content.strip()}
        ]

        print(messages)
        exit()


        response = self.client.chat.completions.create( # type: ignore
            model=self.model_name,
            messages=messages, # type: ignore
            stream=True,
            max_tokens=5000
        )

        reasoning_content = ""
        content = ""

        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                # reasoning_content += chunk.choices[0].delta.reasoning_content
                print(chunk.choices[0].delta.reasoning_content,end="",flush=True)
            else:
                # content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content,end="",flush=True)

        # ic(reasoning_content)
        # ic(content)
        # exit()
        # 解析JSON输出
        try:
            # 直接使用output_parser解析，它会自动处理各种格式
            parsed_output = self.output_parser.parse(content)
            
            # 提取思维链
            cot = parsed_output.chain_of_thought
            # 如果有reasoning_content，将其添加到思维链前面
            if reasoning_content:
                cot = reasoning_content.strip() + "\n 思考过程总结为:" + cot
            
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
            return result,system_prompt

        except Exception as e:
            ic(f"解析输出失败: {str(e)}")
            ic(f"原始输出: {content}")
            ic(f"思维链内容: {reasoning_content}")
            return None,system_prompt

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
                    result,system_prompt = self.generate_distill_result(data)
                    
                    if result is None:
                        stats["failed"] += 1
                        continue
                    
                    # 构造ShareGPT格式的训练数据
                    input_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                    output_str = result.to_think_json()
                    
                    sharegpt_data = self.create_sharegpt_format(
                        instruction=system_prompt,
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
                    exit()
        
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
    distiller = DataDistiller()

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