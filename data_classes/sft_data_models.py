from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import json


class SatelliteAttributes(BaseModel):
    """卫星属性模型

    Attributes:
        id: 卫星ID
        health: 卫星健康状态 (0-10)
        pos: 卫星位置坐标 [x, y, z]
    """

    id: str | int = Field(..., description="卫星ID")
    health: float = Field(..., ge=0, le=10, description="卫星健康状态 (0-10)")
    pos: List[float] = Field(..., description="卫星ECEF位置坐标 [x, y, z] km")


class SatelliteEdge(BaseModel):
    """卫星间连接关系模型

    Attributes:
        from_sat: 起始卫星ID
        to_sat: 目标卫星ID
        distance: 卫星距离，单位km
    """

    from_sat: str | int = Field(..., description="起始卫星ID")
    to_sat: str | int = Field(..., description="目标卫星ID")
    distance: float = Field(..., description="卫星距离，单位km")


class TargetEdge(BaseModel):
    """卫星到目标的连接关系模型

    Attributes:
        sat_id: 起始卫星ID
        target_id: 目标ID
        quality: 连接质量 (0-1)
    """

    sat_id: str | int = Field(..., description="起始卫星ID")
    target_id: str | int = Field(..., description="目标ID")
    quality: float = Field(..., description="连接质量 (0-1)")


class ClusterInfo(BaseModel):
    """分簇信息模型"""

    timestamp: Optional[str] = Field(..., description="ISO8601格式的时间戳字符串")
    cluster_id: str | int = Field(description="分簇ID")
    master: str | int = Field(description="主节点卫星ID")
    sats: List[str | int] = Field(description="分簇中的卫星ID列表")
    targets: List[str | int] = Field(description="分簇观测的目标ID列表")


class RawConstellationDataModel(BaseModel):
    """原始卫星分簇数据模型

    Attributes:
        timestamp: ISO8601格式的时间戳字符串
        sat_attrs: 卫星属性列表
        sat_edges: 卫星间连接关系列表
        target_edges: 卫星到目标的连接关系列表
        history_cluster_result: 上一次分簇结果
    """

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

    def to_think_json(self):
        thought_content = self.chain_of_thought or ""
        return (
            "<think>"
            + thought_content
            + "</think>"
            + json.dumps(
                [cluster.model_dump() for cluster in self.clusters],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    class Config:
        """Pydantic配置"""

        extra = "forbid"  # 禁止额外字段


class SatelliteClusterClearOutput(BaseModel):
    """卫星分簇划分输出模型"""

    clusters: List[ClusterInfo] = Field(description="划分的卫星分簇列表")

    class Config:
        """Pydantic配置"""

        extra = "forbid"  # 禁止额外字段


class ShareGPTMessage(BaseModel):
    """ShareGPT消息格式"""

    role: Literal["system", "user", "assistant"]
    content: str


class ShareGPTFormat(BaseModel):
    """ShareGPT格式的训练数据定义"""

    messages: List[ShareGPTMessage]


class LLMConversationMessage(BaseModel):
    """卫星分簇算法的完整模型问答对话数据

    这个模型表示一个完整的模型问答过程，包含指令、输入数据和响应结果。
    统一了整个项目的数据流格式，使验证工具与主要的卫星分簇算法使用相同的数据结构。

    Attributes:
        instruction: 给模型的指令，通常是系统提示词
        input: 输入的卫星星座数据，包含卫星属性、连接关系等信息
        response: 包含推理过程和分簇结果的完整响应
    """

    instruction: str = Field(..., description="给模型的指令，通常是系统提示词")
    input: RawConstellationDataModel = Field(..., description="输入的卫星星座数据")
    response: SatelliteClusterOutput = Field(
        ..., description="包含推理过程和分簇结果的完整响应"
    )

    def to_sharegpt_json(self) -> str:
        """转换为ShareGPT格式的JSON字符串

        Returns:
            符合ShareGPT格式要求的JSON字符串
        """
        sharegpt_data = {
            "messages": [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": self.input.model_dump_json()},
                {"role": "assistant", "content": self.response.to_think_json()},
            ]
        }
        return json.dumps(sharegpt_data, ensure_ascii=False, separators=(",", ":"))
