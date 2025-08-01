"""数据验证器类型

这个模块定义了观测目标相关的数据模型。
"""

from typing import List, Literal
from pydantic import BaseModel, Field

from data_classes.sft_data_models import ClusterInfo, RawConstellationDataModel


class ValidationDetail(BaseModel):
    """验证详情模型"""

    validation_type: Literal[
        "correctness_validation",
        "stability_validation",
        "communication_cost_validation",
        "observation_efficiency_validation",
    ] = Field(..., description="验证类型")
    score: int = Field(..., description="分项得分")
    info: str = Field(..., description="警告信息")


class ValidationItem(BaseModel):
    """验证项模型"""

    input: RawConstellationDataModel = Field(..., description="输入数据")
    response: List[ClusterInfo] = Field(..., description="模型响应提取的数据")
    validation_details: List[ValidationDetail] = Field(..., description="验证详情")

    @property
    def score(self) -> float:
        """计算总分数，基于加权求和"""
        # 定义权重
        weights = {
            "correctness_validation": 0.4,  # 正确性和隔离性 40%
            "stability_validation": 0.3,  # 分簇稳定性 30%
            "communication_cost_validation": 0.1,  # 通信代价 10%
            "observation_efficiency_validation": 0.1,  # 观测效能 10%
        }

        weighted_score = 0.0
        for detail in self.validation_details:
            weight = weights.get(detail.validation_type, 0)
            weighted_score += detail.score * weight

        return weighted_score
