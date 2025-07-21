#!/usr/bin/env python3
"""测试加权分数计算"""

import sys
from pathlib import Path

# 添加项目根目录到路径
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

from data_classes.data_validation_models import ValidationDetail, ValidationItem
from data_classes.sft_data_models import RawConstellationDataModel, ClusterInfo

def test_weighted_score():
    """测试加权分数计算"""
    
    # 创建模拟的验证详情
    validation_details = [
        ValidationDetail(
            validation_type="correctness_validation",
            score=80,  # 80分
            info="正确性验证结果"
        ),
        ValidationDetail(
            validation_type="stability_validation", 
            score=90,  # 90分
            info="稳定性验证结果"
        ),
        ValidationDetail(
            validation_type="communication_cost_validation",
            score=70,  # 70分
            info="通信代价验证结果"
        ),
        ValidationDetail(
            validation_type="observation_efficiency_validation",
            score=85,  # 85分
            info="观测效能验证结果"
        ),
        ValidationDetail(
            validation_type="cluster_size_validation",
            score=95,  # 95分
            info="分簇规模验证结果"
        )
    ]
    
    # 创建空的输入数据和响应（仅用于测试）
    mock_input = RawConstellationDataModel(
        target_edges=[], 
        sat_edges=[], 
        sat_attrs=[], 
        history_cluster_result=[],
        timestamp="test_timestamp"
    )
    mock_response = []
    
    # 创建验证项
    validation_item = ValidationItem(
        input=mock_input,
        response=mock_response,
        validation_details=validation_details
    )
    
    # 计算期望的加权分数
    expected_score = (
        80 * 0.4 +   # 正确性 80 × 40% = 32
        90 * 0.3 +   # 稳定性 90 × 30% = 27  
        70 * 0.1 +   # 通信代价 70 × 10% = 7
        85 * 0.1 +   # 观测效能 85 × 10% = 8.5
        95 * 0.1     # 分簇规模 95 × 10% = 9.5
    )  # 总计: 32 + 27 + 7 + 8.5 + 9.5 = 84.0
    
    # 获取实际分数
    actual_score = validation_item.score
    
    print(f"📊 权重分数计算测试")
    print(f"=" * 50)
    print(f"各项分数:")
    print(f"  正确性验证: 80分 × 40% = {80 * 0.4}")
    print(f"  稳定性验证: 90分 × 30% = {90 * 0.3}")
    print(f"  通信代价: 70分 × 10% = {70 * 0.1}")
    print(f"  观测效能: 85分 × 10% = {85 * 0.1}")
    print(f"  分簇规模: 95分 × 10% = {95 * 0.1}")
    print(f"=" * 50)
    print(f"期望总分: {expected_score}")
    print(f"实际总分: {actual_score}")
    print(f"计算结果: {'✅ 正确' if abs(expected_score - actual_score) < 0.001 else '❌ 错误'}")

if __name__ == "__main__":
    test_weighted_score()
