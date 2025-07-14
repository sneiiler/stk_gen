"""卫星相关数据模型

这个模块定义了卫星相关的数据模型。
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

from data_models.payload_models import PayloadInfo
from data_models.constraint_models import OrbitDesignConstraints, UserRequirements


class OrbitalElementsInfo(BaseModel):
    """轨道六根数参数容器类，用于定义轨道要素及其优化边界条件

    Attributes:
        semi_axis (float): 轨道半长轴，单位：千米（km）
        eccentricity (float): 轨道偏心率（无量纲）
        inclination (float): 轨道倾角，单位：度（°）
        raan (float): 升交点赤经，单位：度（°）
        arg_of_perigee (float): 近地点幅角，单位：度（°）
        mean_anomaly (float): 平近点角，单位：度（°）
    """

    semi_axis: float = Field(description="卫星轨道半长轴 (单位：km)")
    eccentricity: float = Field(description="卫星轨道偏心率")
    inclination: float = Field(description="卫星轨道倾角(°)")
    raan: float = Field(description="卫星轨道升交点赤经 (°)")
    arg_of_perigee: float = Field(description="卫星轨道近地点幅角 (°)")
    mean_anomaly: float = Field(description="卫星轨道平近点角 (°)")


class OrbitData(BaseModel):
    """轨道数据类型定义，用于轨道预报。

    轨道预报数据模型，包含卫星名称、位置预报、覆盖范围和降交点信息。
    Attributes:
        satellite_name (str): 卫星名称
        propagation_lla (List[Tuple[str, float, float, float]]): 预报未来的卫星位置列表，每个元素为
                (time, latitude, longitude, altitude)的四元组，经度范围为-180°~180°，纬度范围为-90°~90°
        coverage_latitude (Tuple[float, float]): 覆盖纬度范围，格式为(北纬纬度, 南纬纬度)
        descending_node (List[Tuple[str, float]]): 降交点经度，格式为[(时间, 经度)]
        descending_node_local_time (str): 降交点地方时，形式hh:mm，如 10:30
    """

    satellite_name: str = Field(description="卫星名称")
    propagation_lla: List[Tuple[str, float, float, float]] = Field(
        description="预报未来的卫星位置列表，每个元素为(t, lat, lon, alt)的四元组，经度范围为-180°~180°，纬度范围为-90°~90°"
    )
    coverage_latitude: Tuple[float, float] = Field(description="覆盖纬度范围，格式为(北纬纬度, 南纬纬度)")
    descending_node: List[Tuple[str, float]] = Field(description="降交点经度，格式为[(时间, 经度)]")
    descending_node_local_time: str = Field(description="降交点地方时，形式hh:mm，如 10:30")


class SunBetaAngleInfo(BaseModel):
    """表示卫星太阳高度角beta角信息的模型。

    属性:
        satellite_name (str): 卫星名称。
        time (Optional[List[str]]): 太阳高度角beta角的时间序列。
        beta_angle (Optional[List[float]]): 对应的太阳高度beta角值列表（度）。
        beta_angle_min (float): 最小太阳高度beta角值（度）。
        beta_angle_max (float): 最大太阳高度beta角值（度）。
        beta_angle_avg (float): 平均太阳高度beta角值（度）。
        graph_path (Optional[str]): 太阳高度角beta角图路径。
    """

    satellite_name: str = Field(description="卫星名称")
    beta_angle_time: Optional[List[str]] = Field(description="太阳高度角beta角的时间序列")
    beta_angle_values: Optional[List[float]] = Field(description="对应的太阳高度beta角值列表(度)")
    beta_angle_min: float = Field(description="最小太阳高度beta角值列表(度)")
    beta_angle_max: float = Field(description="最大太阳高度beta角值列表(度)")
    beta_angle_avg: float = Field(description="平均太阳高度beta角值列表(度)")
    graph_path: Optional[str] = Field(description="太阳高度角beta角图路径")


class PropagateParams(BaseModel):
    """
    轨道预报参数容器类

    Attributes:
        step_size (int): 轨道预报时长(秒)
        propagate_duration (int): 轨道递推步长(秒)
    """

    step_size: int = 10  # 轨道预报时长(秒)
    propagate_duration: int = 300  # 轨道递推步长(秒)


class LightingDuration(BaseModel):
    """
    表示光照或阴影持续时间的数据。

    Attributes:
        min (float): 最小持续时间，单位：分钟。
        max (float): 最大持续时间，单位：分钟。
        avg (float): 平均持续时间，单位：分钟。
        total (float): 总持续时间，单位：天/年。
        unit (str): 单位。
    """

    min: float = Field(description="最小持续时间，单位：分钟")
    max: float = Field(description="最大持续时间，单位：分钟")
    avg: float = Field(description="平均持续时间，单位：分钟")
    total: Optional[float] = Field(description="总持续时间，单位：天/年")
    unit: str = Field(default="minutes", description="单位")


class LightingTimeData(BaseModel):
    """
    表示光照时间的数据。

    Attributes:
        satellite_name (str): 卫星名称。
        sunlight (LightingDuration): 直接光照的时间数据。
        penumbra (LightingDuration): 半影的时间数据。
        umbra (LightingDuration): 本影的时间数据。
    """

    satellite_name: str = Field(description="卫星名称")
    sunlight: LightingDuration = Field(..., description="直接光照的时间数据")
    penumbra: LightingDuration = Field(..., description="半影的时间数据")
    umbra: LightingDuration = Field(..., description="本影的时间数据")


class RegressionAnalysisInfo(BaseModel):
    """
    轨道回归周期计算结果。

    该类用于表示卫星轨道回归周期的计算结果，目前支持不超过360天的回归周期计算。

    Attributes:
        satellite_name (str): 卫星名称。
        regression_period (float): 回归周期，单位为天，按照升交点经度的相似重访计算得到的轨道回归周期。
        regression_period_pass_count (int): 一个轨道回归周期内的圈数。
        threshold (float): 相似度阈值，单位为度，用于计算升交点经度相似度的阈值，程序默认值为0.2。
        regression_period (Optional[float]): 回归周期（单位：天）。按照升交点经度的相似重访，计算得到的轨道回归周期。
            - 必须大于0.01且小于360。
        regression_period_pass_count (int): 一个轨道回归周期内的绕地圈数。
        threshold (float): 相似度阈值（单位：度）。用于计算升交点经度相似度的阈值，默认为0.2°。
    """

    satellite_name: str = Field(description="卫星名称")
    regression_period: float = Field(
        ..., title="回归周期", description="单位day，按照升交点经度的相似重访，计算得到的轨道回归周期"
    )
    regression_period_pass_count: int = Field(
        ..., title="一个轨道回归周期内的圈数", description="一个轨道回归周期内的圈数"
    )
    threshold: float = Field(
        ...,
        title="回归周期计算升交点经度的相似度阈值",
        description="单位degree，是用于计算升交点经度相似度的阈值，程序默认0.2",
    )


class LifetimeEstimationInfo(BaseModel):
    """
    卫星寿命估计信息

    Attributes:
        satellite_name (str): 卫星名称
        lifetime_years (float): 寿命年数
        decay_altitude_rate (float): 轨道下降速率 米/天
        fuel_consumption_rate (float): 燃料消耗速率, kg/天
    """

    satellite_name: str = Field(description="卫星名称")
    lifetime_years: float = Field(description="寿命年数")
    decay_altitude_rate: float = Field(description="轨道下降速率 米/天")
    fuel_consumption_rate: float = Field(description="燃料消耗速率, kg/天")


class SatelliteInfo(BaseModel):
    """卫星信息

    Attributes:
        name (str): 卫星名称。
        orbit_epoch_time (str): 轨道历元时间，格式为参考：22 Mar 2025 04:00:00.000。
        orbit_elements (OrbitalElementsInfo): 轨道根数。
        drag_area (float): 卫星迎风面积。
        sun_area (float): 卫星太阳辐射面积。
        mass (float): 卫星质量，单位：千克。
        fuel_consumption_factor (float): 维持轨道高度所需要的燃料效率，与推进系统的能力相关，单位：千克/千米。
    """

    name: Optional[str] = Field(description="卫星名称")
    orbit_epoch_time: Optional[str] = Field(description="轨道历元时间，格式为参考：22 Mar 2025 04:00:00.000")
    orbit_elements: Optional[OrbitalElementsInfo] = Field(description="轨道根数")
    drag_area: Optional[float] = Field(default=10.0, description="卫星迎风面积，单位：平方米")
    sun_area: Optional[float] = Field(default=10.0, description="卫星太阳辐射面积，单位：平方米")
    mass: Optional[float] = Field(default=1000.0, description="卫星质量，单位：千克")
    fuel_consumption_factor: Optional[float] = Field(
        default=1.0, description="维持轨道高度所需要的燃料效率，与推进系统的能力相关，单位：千克/千米"
    )


class ImagingParams(BaseModel):
    """成像参数

    Attributes:
        payload (PayloadInfo): 载荷信息
        pixel_resolution (float): 地面像元分辨率，单位：米
        imaging_bandwidth (float): 成像带宽，单位：千米
        tilt_angle (float): 侧摆角度，单位：度
    """

    payload: PayloadInfo = Field(..., description="载荷信息")
    pixel_resolution: float = Field(..., description="地面像元分辨率，单位：米")
    imaging_bandwidth: float = Field(..., description="成像带宽，单位：千米")
    tilt_angle: float = Field(..., description="侧摆角度，单位：度")


class SatelliteState(BaseModel):
    """卫星状态

    Attributes:
        satellite_info (SatelliteInfo): 卫星信息。
        payload_info (PayloadInfo): 载荷信息。
        imaging_params (ImagingParams): 成像参数。
        user_requirements (UserRequirements): 用户需求。
        orbit_design_constraints (OrbitDesignConstraints): 轨道设计约束。
        message_history (List[dict]): 消息历史。
    """

    satellite_info: SatelliteInfo = Field(description="卫星信息")
    payload_info: PayloadInfo = Field(description="载荷信息")
    imaging_params: ImagingParams = Field(description="成像参数")
    user_requirements: UserRequirements = Field(description="用户需求")
    orbit_design_constraints: OrbitDesignConstraints = Field(description="轨道设计约束")
    message_history: List[dict] = Field(default_factory=list, description="消息历史")
