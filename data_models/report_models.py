"""报告相关数据模型

这个模块定义了报告相关的数据模型。
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from data_models.observation_target_models import ObservationTargetInfo
from data_models.satellite_models import (
    SatelliteInfo,
    PayloadInfo,
    ImagingParams,
    UserRequirements,
    OrbitDesignConstraints,
    OrbitalElementsInfo
)

class OrbitRelatedParameters(BaseModel):
    """轨道相关参数模型"""
    elements_info: OrbitalElementsInfo = Field(..., description="轨道根数信息")
    regression_period: int = Field(..., description="轨道周期，单位天")
    regression_period_pass_count: int = Field(..., description="轨道一个回归周期内轨道圈数")
    nodal_period_min: float = Field(..., description="轨道交点周期")
    descending_node_local_time: str = Field(..., description="降交点地方时，单位小时")

    max_orbit_height: float = Field(..., description="轨道最大高度，单位km")
    min_orbit_height_revisit: float = Field(..., description="重访周期约束下的轨道最小高度，单位km")
    min_orbit_height_imaging_swath: float = Field(..., description="成像带宽约束下的轨道最小高度，单位km")
    min_orbit_height: float = Field(..., description="轨道最小高度，单位km")
    average_orbit_height: float = Field(..., description="平均轨道高度，单位km")

    coverage_latitude: Tuple[float, float] = Field(..., description="南北覆盖纬度范围，单位度")
    equator_spacing: float = Field(..., description="相邻轨迹之间赤道间距，单位km")
    two_day_spacing: float = Field(..., description="相邻两天相邻轨迹赤道间距，单位km")
    two_orbit_spacing: float = Field(..., description="相邻两圈轨道赤道间距，单位km")

    coverage_bandwidth: float = Field(..., description="覆盖带宽，单位km")
    number_of_strips: int = Field(..., description="条带数量")
    equator_overlap_bandwidth: float = Field(..., description="赤道重叠带宽，单位km")

class TargetRevisitParams(BaseModel):
    """目标重访统计信息模型。

    Attributes:
        info (ObservationTargetInfo): 观测目标信息
        max_revisit_period (float): 最大重访周期，单位天
        min_revisit_period (float): 最小重访周期，单位天
        avg_revisit_period (float): 平均重访周期，单位天
        total_count (int): 总观测次数

        better_than_xx_count (int): 优于某分辨率次数
        xx_to_yy_count (int): 另一区间分辨率次数
        yy_to_zz_count (int): 另一区间分辨率次数
        worse_than_zz_count (int): 低于分辨率区间次数
        xx_resolution (float): 优于某分辨率
        yy_resolution (float): 另一区间分辨率
        zz_resolution (float): 另一区间分辨率
    """

    info: ObservationTargetInfo = Field(..., description="观测目标信息")
    max_revisit_period: float = Field(..., description="最大重访周期，单位天")
    min_revisit_period: float = Field(..., description="最小重访周期，单位天")
    avg_revisit_period: float = Field(..., description="平均重访周期，单位天")

    total_count: int = Field(..., description="总观测次数")
    better_than_xx_count: int = Field(..., description="优于某分辨率次数")
    xx_to_yy_count: int = Field(..., description="另一区间分辨率次数")
    yy_to_zz_count: int = Field(..., description="另一区间分辨率次数")
    worse_than_zz_count: int = Field(..., description="低于分辨率区间次数")
    xx_resolution: float = Field(..., description="优于某分辨率")
    yy_resolution: float = Field(..., description="另一区间分辨率")
    zz_resolution: float = Field(..., description="另一区间分辨率")

class LightingParams(BaseModel):
    """光照参数模型"""
    min_beta_angle: float = Field(..., description="最小太阳入射角，单位度")
    max_beta_angle: float = Field(..., description="最大太阳入射角，单位度")
    min_shadow_time: float = Field(..., description="最小地影时间，单位分钟")
    max_shadow_time: float = Field(..., description="最大地影时间，单位分钟")
    total_shadow_time: float = Field(..., description="一年累计地影时间，单位天") 