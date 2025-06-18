"""载荷相关数据模型

这个模块定义了载荷相关的数据模型。
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PayloadInfo(BaseModel):
    """
    载荷信息输入参数容器类，用于定义遥感载荷特性及其优化边界条件

    Attributes:
        sensor_type (str)：载荷名称
        gsd_factor (float): gsd因子，用于计算地面采样距离，单位m/km
        fov_angle (float): 卫星载荷的视场角，通常为1~5度，全视场角而非视场半角，单位：度（°）
        light_condition(int):光照约束, 类别： "eDirectSun" | "eUmbra" | "eUmbraOrDirectSun"
        los_angle (float): Line of Sight，用于计算卫星对目标可见性的约束，单位：度（°）
        sensor_cone_half_angle (float): 卫星机动能力带来的视场半角，通常为15~30度，单位：度（°）

    """

    # 属性参数
    sensor_type: Optional[str] = Field(None, description="载荷名称")
    gsd_factor: Optional[float] = Field(default=1.6e-3, description="gsd因子，用于计算地面采样距离，单位m/km")
    fov_angle: Optional[float] = Field(default=1.0, description="卫星载荷的视场角，通常为1~5度，单位：度（°）")
    light_condition: Optional[Literal["eDirectSun", "eUmbra", "eUmbraOrDirectSun"]] = Field(
        ..., description="光照约束, 类别：'eDirectSun' | 'eUmbra' | 'eUmbraOrDirectSun'"
    )
    los_angle: Optional[float] = Field(
        ..., description="Line of Sight，用于计算卫星对目标可见性的约束，单位：度（°）"
    )
    sensor_cone_half_angle: Optional[float] = Field(
        ..., description="卫星机动能力带来的视场半角，通常为15~30度，单位：度（°）"
    )
