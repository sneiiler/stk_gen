from typing import List, Literal, Union
from pydantic import BaseModel, Field
import json

class SatelliteAttributes(BaseModel):
    """卫星属性模型
    
    Attributes:
        id: 卫星ID
        health: 卫星健康状态 (0-10)
        pos: 卫星位置坐标 [x, y, z]
    """
    id: int = Field(..., description="卫星ID")
    health: int = Field(..., ge=0, le=10, description="卫星健康状态 (0-10)")
    pos: List[float] = Field(..., description="卫星位置坐标 [x, y, z]")


class SatelliteEdge(BaseModel):
    """卫星间连接关系模型
    
    Attributes:
        from_id: 起始卫星ID
        to_id: 目标卫星ID
        w: 连接权重 (0-1)
    """
    from_id: int = Field(..., alias="from", description="起始卫星ID")
    to_id: int = Field(..., alias="to", description="目标卫星ID")
    w: float = Field(..., ge=0.0, le=1.0, description="连接权重 (0-1)")


class TargetEdge(BaseModel):
    """卫星到目标的连接关系模型
    
    Attributes:
        from_id: 起始卫星ID
        to_id: 目标ID
        q: 连接质量 (0-1)
    """
    from_id: int = Field(..., alias="from", description="起始卫星ID")
    to_id: int = Field(..., alias="to", description="目标ID")
    q: float = Field(..., ge=0.0, le=1.0, description="连接质量 (0-1)")


class RawConstellationDataModel(BaseModel):
    """原始卫星分簇数据模型
    
    Attributes:
        timestamp: ISO8601格式的时间戳字符串
        strategy: 策略类型 ("balanced" 或 "quality")
        sat_attrs: 卫星属性列表
        sat_edges: 卫星间连接关系列表
        target_edges: 卫星到目标的连接关系列表
    """
    timestamp: str = Field(..., description="ISO8601格式的时间戳字符串")
    strategy: Literal["balanced", "quality"] = Field(..., description="策略类型")
    sat_attrs: List[SatelliteAttributes] = Field(..., description="卫星属性列表")
    sat_edges: List[SatelliteEdge] = Field(..., description="卫星间连接关系列表")
    target_edges: List[TargetEdge] = Field(..., description="卫星到目标的连接关系列表")

class ClusterInfo(BaseModel):
    """分簇信息模型"""
    cluster_id: int = Field(description="分簇ID")
    master: int = Field(description="主节点卫星ID")
    sats: List[int] = Field(description="分簇中的卫星ID列表")
    targets: List[int] = Field(description="分簇观测的目标ID列表")


class SatelliteClusterOutput(BaseModel):
    """卫星分簇划分输出模型"""
    chain_of_thought: str = Field(description="思维链和推理过程")
    clusters: List[ClusterInfo] = Field(description="划分的卫星分簇列表")
    
    def to_think_json(self):
        return "<think>" + self.chain_of_thought + "</think>" \
            + json.dumps([cluster.model_dump() for cluster in self.clusters], ensure_ascii=False, separators=(",", ":"))
    
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