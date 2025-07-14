"""分簇相关数据模型

这个模块定义了卫星分簇算法相关的数据模型。
"""

from typing import List
from pydantic import BaseModel, Field


class ClusterInfo(BaseModel):
    """分簇信息模型"""
    cluster_id: int = Field(description="分簇ID")
    master: int = Field(description="主节点卫星ID")
    sats: List[int] = Field(description="分簇中的卫星ID列表")
    targets: List[int] = Field(description="分簇观测的目标ID列表")
