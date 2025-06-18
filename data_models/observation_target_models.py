"""观测目标相关数据模型

这个模块定义了观测目标相关的数据模型。
"""

from typing import List, Optional, Tuple, Dict, ClassVar
from pydantic import BaseModel, Field


class ObservationTargetInfo(BaseModel):
    """观测目标信息

    Attributes:
        target_name (str): 目标名称（英文）。
        target_local_name (str): 目标本地名称（中文）。
        target_coords (Tuple[float, float, float]): 目标坐标（经度，纬度，高度），单位：（度，度，千米）。
    """

    target_name: str = Field(description="目标名称（英文）")
    target_local_name: str = Field(description="目标本地名称（中文）")
    target_coords: Tuple[float, float, float] = Field(
        description="目标坐标（经度，纬度，高度），单位：（度，度，千米）"
    )


class TargetAccessInfo(BaseModel):
    """目标访问信息

    Attributes:
        target_name (str): 目标名称。
        access_times (List[str]): 访问时间列表，格式为：YYYY-MM-DD HH:MM:SS。
        elevation_angles (List[float]): 仰角列表，单位：度。
        ranges (List[float]): 距离列表，单位：千米。
    """

    target_name: str = Field(description="目标名称")
    access_times: List[str] = Field(description="访问时间列表，格式为：YYYY-MM-DD HH:MM:SS")
    elevation_angles: List[float] = Field(description="仰角列表，单位：度")
    ranges: List[float] = Field(description="距离列表，单位：千米")


class STKAccessEvent(BaseModel):
    """
    单次访问事件数据容器类

    Attributes:
        end_time (str): 访问结束时间（UTC字符串格式）
        end_timestamp (int): 访问结束时间戳（Unix时间）
        start_time (str): 访问开始时间（UTC字符串格式）
        start_timestamp (int): 访问开始时间戳（Unix时间）
        satellite_name (str): 卫星名称
        sensor (str): 传感器名称
        gsd (float): 地面采样距离，单位：米/像素（m/px）
        max_elevation (float): 最大仰角，单位：度（°）
        min_range (float): 最小观测距离，单位：千米（km）
    """

    end_time: str
    end_timestamp: int
    start_time: str
    start_timestamp: int
    satellite_name: str
    sensor: str
    gsd: float
    max_elevation: float
    min_range: float


class RevisitAnalysisConstraints(BaseModel):
    """重访分析约束

    Attributes:
        avg_revisit_bounds (Tuple[float, float]): 平均重访时间上下界，单位：小时。
        max_revisit_bounds (Tuple[float, float]): 最大重访间隔上下界，单位：小时。
        min_revisit_bounds (Tuple[float, float]): 最小重访间隔上下界，单位：小时。
        gsd_bounds (Tuple[float, float]): 星下点地面采样距离上下界，单位：米。
        imaging_swath_bounds (Tuple[float, float]): 载荷视场角对应的地面覆盖幅宽上下界，单位：千米。
    """

    avg_revisit_bounds: Tuple[float, float] = Field(
        default=(0.0, 720.0), description="平均重访时间上下界，单位：小时"
    )
    max_revisit_bounds: Tuple[float, float] = Field(
        default=(0.0, 720.0), description="最大重访间隔上下界，单位：小时"
    )
    min_revisit_bounds: Tuple[float, float] = Field(
        default=(0.0, 720.0), description="最小重访间隔上下界，单位：小时"
    )
    gsd_bounds: Tuple[float, float] = Field(
        default=(0.0, 15.0), description="星下点地面采样距离上下界，单位：米"
    )
    imaging_swath_bounds: Tuple[float, float] = Field(
        default=(0.0, 30.0), description="载荷视场角对应的地面覆盖幅宽上下界，单位：千米"
    )

    bounds: ClassVar[Dict[str, Tuple[float, float]]] = {
        "avg_revisit_bounds": (0.0, 720.0),
        "max_revisit_bounds": (0.0, 720.0),
        "min_revisit_bounds": (0.0, 720.0),
    }


class RevisitAnalysisInfo(BaseModel):
    """
    单个点目标的重访分析输出容器类

    Attributes:
        satellite_name (str): 卫星名称
        target (ObservationTargetInfo): 观测目标信息
        access_events (Optional[List[STKAccessEvent]]): 观测机会事件列表
        revisit_epoch (float): 分析统计周期（天）,默认30天
        gsd (float): 地面采样距离,单位：米（m）
        imaging_swath (float): 地面成像宽度,单位：千米（km）

    Variables:
        avg_revisit (float): 平均重访间隔,单位：小时
        max_revisit (float): 最大重访间隔,单位：小时
        min_revisit (float): 最小重访间隔,单位：小时
        revisit_time_unit (str): 重访时间单位,默认"hours"

    """

    # 属性
    satellite_name: str = Field(description="卫星名称")
    target: ObservationTargetInfo = Field(description="观测点信息")
    access_events: Optional[List[STKAccessEvent]] = Field(description="详细访问数据")
    revisit_epoch: float = Field(description="重访计算周期(天)，默认为30天")
    gsd: float = Field(description="星下点地面采样距离（分辨率）")
    imaging_swath: float = Field(description="载荷视场角对应的地面覆盖幅宽，单位：千米(km)")

    # 变量
    avg_revisit: float = Field(description="平均重访时间（小时）")
    max_revisit: float = Field(description="最大重访间隔（小时）")
    min_revisit: float = Field(description="最小重访间隔（小时）")
    revisit_time_unit: str = Field(description="重访时间单位")

    # 变量约束
    revisit_constraints: RevisitAnalysisConstraints = Field(description="重访分析约束条件")

    def check_revisit_constraints(self) -> bool:
        """检查重访时间是否满足约束条件

        Returns:
            bool: True表示满足所有约束条件,False表示不满足
        """
        # 检查最大重访间隔是否在约束范围内
        if not (
            self.revisit_constraints.max_revisit_bounds[0]
            <= self.max_revisit
            <= self.revisit_constraints.max_revisit_bounds[1]
        ):
            return False

        # 检查最小重访间隔是否在约束范围内
        if not (
            self.revisit_constraints.min_revisit_bounds[0]
            <= self.min_revisit
            <= self.revisit_constraints.min_revisit_bounds[1]
        ):
            return False

        # 检查平均重访时间是否在约束范围内
        if not (
            self.revisit_constraints.avg_revisit_bounds[0]
            <= self.avg_revisit
            <= self.revisit_constraints.avg_revisit_bounds[1]
        ):
            return False
        return True

    @property
    def bounds(self):
        """
        获取重访分析约束的边界值

        Returns:
            dict: 包含平均重访时间、最大重访间隔、最小重访间隔等约束的边界值字典
        """
        return self.revisit_constraints.bounds

    def update(self, updates: dict):
        """
        更新实例的属性。

        Args:
            updates (dict): 包含需要更新的属性和新值的字典。
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"RevisitAnalysisInfo has no attribute '{key}'")

class MissileInfo(BaseModel):
    """导弹信息数据模型

    Attributes:
        name (str): 导弹名称
        trajectory_epoch_second (int): 导弹轨迹时间，相对当前场景开始时间的偏移秒数，范围[0, 300]
        speed (float): 导弹速度，单位：km/s，范围[3.0, 10.0]
        altitude (float): 导弹高度，单位：km，范围[100.0, 1500.0]
        latitude (float): 导弹发射纬度，单位：度
        longitude (float): 导弹发射经度，单位：度
        impact_latitude (float): 导弹撞击点纬度，单位：度
        impact_longitude (float): 导弹撞击点经度，单位：度
    """
    name: str = Field(description="导弹名称")  # 导弹名称
    trajectory_epoch_second: int = Field(
        description="导弹轨迹时间(相对当前场景开始时间的偏移秒数)",
        ge=0,
        le=300
    )  # 导弹轨迹时间
    speed: float = Field(
        description="导弹速度(km/s)",
        ge=3.0,
        le=10.0
    )  # 导弹速度（单位：km/s）
    altitude: float = Field(
        description="导弹高度(km)",
        ge=100,
        le=1500.0
    )  # 导弹高度（单位：km）
    latitude: float = Field(description="导弹纬度(deg)")  # 导弹发射纬度（单位：度）
    longitude: float = Field(description="导弹经度(deg)")  # 导弹发射经度（单位：度）
    impact_latitude: float = Field(description="导弹撞击点纬度(deg)")  # 导弹撞击点纬度（单位：度）
    impact_longitude: float = Field(description="导弹撞击点经度(deg)")  # 导弹撞击点经度（单位：度）