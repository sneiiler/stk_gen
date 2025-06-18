"""约束相关数据模型

这个模块定义了各种约束相关的数据模型。
"""

from typing import List, Dict, Optional, Tuple, Literal
from pydantic import BaseModel, Field
from .observation_target_models import ObservationTargetInfo, RevisitAnalysisConstraints


class UserRequirements(BaseModel):
    """用户需求

    Attributes:
        required_gsd (float): 要求的地面采样距离，单位：米。
        required_revisit_period (float): 要求的重访周期，单位：小时。
        required_coverage_latitude (float): 要求的覆盖纬度，单位：度。
        required_imaging_swath (float): 要求的成像幅宽，单位：千米。
        required_target_revisit (List[ObservationTargetInfo]): 要求的目标重访信息列表。
    """

    required_gsd: float = Field(description="要求的地面采样距离，单位：米")
    required_revisit_period: float = Field(description="要求的重访周期，单位：小时")
    required_coverage_latitude: float = Field(description="要求的覆盖纬度，单位：度")
    required_imaging_swath: float = Field(description="要求的成像幅宽，单位：千米")
    required_target_revisit: List[ObservationTargetInfo] = Field(
        default_factory=list, description="要求的目标重访信息列表"
    )


class OrbitDataConstraints(BaseModel):
    """轨道数据约束

    Attributes:
        coverage_latitude_bounds (Tuple[float, float]): 覆盖纬度范围，单位：度。
        descending_node_local_time_bounds (Tuple[float, float]): 降交点地方时范围，单位：小时。
    """

    coverage_latitude_bounds: Tuple[float, float] = Field(
        default=(0.0, 85.0), description="覆盖纬度范围，单位：度"
    )
    descending_node_local_time_bounds: Tuple[float, float] = Field(
        default=(0.0, 24.0), description="降交点地方时范围，单位：小时"
    )


class LightingTimeDataConstraints(BaseModel):
    """光照时间数据约束

    Attributes:
        umbra_bounds (Tuple[float, float]): 本影的时间数据范围上下界，单位：分钟。
    """

    umbra_bounds: Tuple[float, float] = Field(
        default=(0.0, 24 * 60.0), description="本影的时间数据范围上下界，单位：分钟"
    )


class RegressionAnalysisConstraints(BaseModel):
    """回归分析约束

    Attributes:
        regression_period_bounds (Tuple[float, float]): 回归周期上下界，单位：天。
    """

    regression_period_bounds: Tuple[float, float] = Field(
        default=(0.0, 50.0), description="回归周期上下界，单位：天"
    )


class OrbitDesignConstraints(BaseModel):
    """轨道设计约束

    Attributes:
        orbit_data_constraints (OrbitDataConstraints): 轨道数据约束。
        lighting_time_data_constraints (LightingTimeDataConstraints): 光照时间数据约束。
        regression_analysis_constraints (RegressionAnalysisConstraints): 回归分析约束。
        revisit_analysis_constraints (RevisitAnalysisConstraints): 重访分析约束。
    """

    orbit_data_constraints: OrbitDataConstraints = Field(description="轨道数据约束")
    lighting_time_data_constraints: LightingTimeDataConstraints = Field(description="光照时间数据约束")
    regression_analysis_constraints: RegressionAnalysisConstraints = Field(description="回归分析约束")
    revisit_analysis_constraints: RevisitAnalysisConstraints = Field(description="重访分析约束")
