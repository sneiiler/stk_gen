"""可见性数据模型

这个模块定义了卫星可见性相关的数据模型。
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SatelliteInfo(BaseModel):
    """卫星信息模型
    
    Attributes:
        id: 卫星ID
        position: 卫星ECEF位置坐标 [x, y, z] km
        health_status: 卫星健康状态
        full_visibility_time_window_length: 完整可见时间窗口长度（秒）
    """
    id: str = Field(..., description="卫星ID")
    position: List[float] = Field(..., description="卫星ECEF位置坐标 [x, y, z] km")
    health_status: str = Field(..., description="卫星健康状态")
    full_visibility_time_window_length: int = Field(..., description="完整可见时间窗口长度（秒）")


class TargetVisibility(BaseModel):
    """目标可见性模型
    
    Attributes:
        target_id: 目标ID
        target_value: 目标价值
        observation_priority: 观测优先级
        position: 目标ECEF位置坐标 [x, y, z] km
        visibility_time_window: 可见时间窗口 [start_time, end_time]
    """
    target_id: str = Field(..., description="目标ID")
    target_value: int = Field(..., description="目标价值")
    observation_priority: int = Field(..., description="观测优先级")
    position: List[float] = Field(..., description="目标ECEF位置坐标 [x, y, z] km")
    visibility_time_window: List[float] = Field(..., description="可见时间窗口 [start_time, end_time]")


class InterSatelliteConnectivity(BaseModel):
    """卫星间连接性模型
    
    Attributes:
        to_satellite_id: 目标卫星ID
        position: 目标卫星ECEF位置坐标 [x, y, z] km
        connection_quality: 连接质量 (0-100)
        visibility_time_window: 可见时间窗口 [start_time, end_time]
    """
    to_satellite_id: str = Field(..., description="目标卫星ID")
    position: List[float] = Field(..., description="目标卫星ECEF位置坐标 [x, y, z] km")
    connection_quality: int = Field(..., description="连接质量 (0-100)")
    visibility_time_window: List[float] = Field(..., description="可见时间窗口 [start_time, end_time]")


class SatelliteVisibilityData(BaseModel):
    """卫星可见性数据模型
    
    Attributes:
        satellite_info: 卫星信息
        inter_satellite_connectivity: 卫星间连接性列表
        target_visibility: 目标可见性列表
        timestamp: 时间戳字符串
        time_offset_from_scenario_start: 从场景开始的时间偏移（秒）
    """
    satellite_info: SatelliteInfo = Field(..., description="卫星信息")
    inter_satellite_connectivity: List[InterSatelliteConnectivity] = Field(..., description="卫星间连接性列表")
    target_visibility: List[TargetVisibility] = Field(..., description="目标可见性列表")
    timestamp: str = Field(..., description="时间戳字符串")
    time_offset_from_scenario_start: float = Field(..., description="从场景开始的时间偏移（秒）")
