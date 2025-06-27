#!/usr/bin/env python3
"""
数据验证器测试脚本

演示如何使用SatelliteClusterValidator验证大模型生成的卫星分簇结果。
"""

import json
import logging
from data_processing.SatelliteClusterValidator import (
    SatelliteClusterValidator, 
    validate_satellite_clustering, 
    generate_report
)


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_test_data():
    """加载测试数据"""
    # 示例输入数据
    input_data = {
        "timestamp": "2025-06-27T01:41:21Z",
        "strategy": "quality",
        "sat_attrs": [
            {"id": 132, "health": 0.87, "pos": [5231.847, -1289.387, -2519.124]},
            {"id": 133, "health": 0.94, "pos": [-7693.171, -7198.445, -4887.208]},
            {"id": 134, "health": 0.52, "pos": [-1867.829, -1481.63, 6607.229]},
            {"id": 135, "health": 0.6, "pos": [-403.845, 4230.973, 6247.4]},
            {"id": 136, "health": 0.66, "pos": [1680.431, 1628.839, 7823.812]},
            {"id": 143, "health": 0.84, "pos": [-5400.805, 4999.156, -7618.133]},
            {"id": 144, "health": 0.81, "pos": [-5660.731, 7871.925, -5448.375]},
            {"id": 145, "health": 0.97, "pos": [7740.468, -1228.166, 2964.448]},
            {"id": 146, "health": 0.8, "pos": [2015.698, 6321.6, -6458.866]},
            {"id": 151, "health": 0.6, "pos": [-836.917, -6103.703, -6547.617]},
            {"id": 153, "health": 0.63, "pos": [-1384.164, 2633.782, -1945.832]},
            {"id": 154, "health": 0.63, "pos": [-2615.123, 7887.198, 5633.477]},
            {"id": 155, "health": 1.0, "pos": [7802.729, 5117.391, 2828.367]},
            {"id": 161, "health": 0.77, "pos": [4648.125, 3681.344, 7259.989]},
            {"id": 162, "health": 0.86, "pos": [4548.898, -4132.845, 4160.725]},
            {"id": 163, "health": 0.64, "pos": [-3854.781, 945.113, -7.173]},
            {"id": 165, "health": 0.76, "pos": [1269.62, -6641.84, -2687.341]},
            {"id": 166, "health": 0.55, "pos": [1502.668, -2076.357, -6947.712]},
            {"id": 111, "health": 0.85, "pos": [2413.383, 1236.673, -2967.662]},
            {"id": 113, "health": 0.55, "pos": [-7567.903, 2783.617, -3289.413]},
            {"id": 114, "health": 0.95, "pos": [2225.978, -3370.548, -4403.578]},
            {"id": 115, "health": 0.7, "pos": [-2603.784, -2124.826, -1279.521]},
            {"id": 116, "health": 0.83, "pos": [2497.692, 6023.306, 5118.791]},
            {"id": 121, "health": 0.77, "pos": [-7962.326, 1233.002, -7528.661]},
            {"id": 122, "health": 0.75, "pos": [2445.126, 5115.013, -7937.369]},
            {"id": 124, "health": 0.6, "pos": [-2010.079, 1236.28, 1588.676]},
            {"id": 125, "health": 0.78, "pos": [2912.689, 6321.827, 2679.13]}
        ],
        "sat_edges": [
            {"from": 125, "to": 166, "w": 0.94},
            {"from": 155, "to": 143, "w": 0.83},
            {"from": 115, "to": 145, "w": 0.36},
            {"from": 111, "to": 162, "w": 0.95},
            {"from": 116, "to": 135, "w": 0.23},
            {"from": 124, "to": 121, "w": 0.42},
            {"from": 125, "to": 111, "w": 0.56},
            {"from": 163, "to": 132, "w": 0.25},
            {"from": 122, "to": 136, "w": 0.26},
            {"from": 124, "to": 136, "w": 0.75},
            {"from": 143, "to": 144, "w": 0.44},
            {"from": 153, "to": 136, "w": 0.23},
            {"from": 132, "to": 136, "w": 0.72},
            {"from": 124, "to": 132, "w": 0.21},
            {"from": 111, "to": 165, "w": 0.58},
            {"from": 133, "to": 166, "w": 0.88},
            {"from": 154, "to": 136, "w": 0.89},
            {"from": 154, "to": 151, "w": 0.76},
            {"from": 125, "to": 163, "w": 0.75},
            {"from": 114, "to": 146, "w": 0.33},
            {"from": 114, "to": 161, "w": 0.61},
            {"from": 115, "to": 163, "w": 0.58},
            {"from": 113, "to": 163, "w": 0.93},
            {"from": 151, "to": 111, "w": 0.83},
            {"from": 154, "to": 135, "w": 0.77},
            {"from": 163, "to": 132, "w": 0.75},
            {"from": 162, "to": 134, "w": 0.76}
        ],
        "target_edges": [
            {"from": 116, "to": 5, "q": 0.86},
            {"from": 166, "to": 18, "q": 0.9},
            {"from": 114, "to": 15, "q": 0.65},
            {"from": 154, "to": 6, "q": 0.35},
            {"from": 135, "to": 25, "q": 0.48},
            {"from": 162, "to": 5, "q": 0.83},
            {"from": 135, "to": 31, "q": 0.43},
            {"from": 116, "to": 48, "q": 0.53},
            {"from": 136, "to": 39, "q": 0.62},
            {"from": 162, "to": 31, "q": 0.43},
            {"from": 162, "to": 14, "q": 0.82},
            {"from": 146, "to": 8, "q": 0.5},
            {"from": 146, "to": 22, "q": 0.52},
            {"from": 114, "to": 25, "q": 0.99},
            {"from": 113, "to": 12, "q": 0.84},
            {"from": 133, "to": 19, "q": 0.61},
            {"from": 132, "to": 34, "q": 0.4},
            {"from": 143, "to": 10, "q": 0.72},
            {"from": 161, "to": 47, "q": 0.88},
            {"from": 132, "to": 23, "q": 0.43},
            {"from": 124, "to": 16, "q": 0.35},
            {"from": 162, "to": 42, "q": 0.52},
            {"from": 145, "to": 3, "q": 0.76},
            {"from": 136, "to": 8, "q": 0.98},
            {"from": 153, "to": 7, "q": 0.88},
            {"from": 146, "to": 8, "q": 0.29},
            {"from": 116, "to": 12, "q": 0.93},
            {"from": 125, "to": 2, "q": 0.77},
            {"from": 151, "to": 33, "q": 0.83},
            {"from": 146, "to": 39, "q": 0.82},
            {"from": 146, "to": 25, "q": 0.37},
            {"from": 122, "to": 2, "q": 0.47},
            {"from": 161, "to": 15, "q": 0.58},
            {"from": 114, "to": 27, "q": 0.91},
            {"from": 144, "to": 31, "q": 1.0},
            {"from": 115, "to": 15, "q": 0.82},
            {"from": 133, "to": 48, "q": 0.5},
            {"from": 144, "to": 7, "q": 0.57},
            {"from": 155, "to": 10, "q": 0.76},
            {"from": 166, "to": 37, "q": 0.88}
        ]
    }
    
    return input_data


def test_valid_output():
    """测试有效的输出"""
    logger = setup_logging()
    input_data = load_test_data()
    
    # 有效的输出数据
    valid_output = {
        "clusters": [
            {
                "cluster_id": 1,
                "master": 125,
                "sats": [125, 166, 165],
                "targets": [2]
            },
            {
                "cluster_id": 2,
                "master": 145,
                "sats": [145, 115, 155, 143],
                "targets": [3, 10]
            },
            {
                "cluster_id": 3,
                "master": 116,
                "sats": [116, 162, 125, 136, 111, 151],
                "targets": [5, 12, 48, 14, 33]
            },
            {
                "cluster_id": 4,
                "master": 153,
                "sats": [153, 144, 135, 163],
                "targets": [7, 31, 6]
            },
            {
                "cluster_id": 5,
                "master": 114,
                "sats": [114, 146, 161, 124],
                "targets": [25, 27, 8, 39, 47]
            },
            {
                "cluster_id": 6,
                "master": 133,
                "sats": [133, 132, 121],
                "targets": [19, 18, 37]
            },
            {
                "cluster_id": 7,
                "master": 122,
                "sats": [122],
                "targets": [2]
            }
        ]
    }
    
    print("=" * 60)
    print("测试1: 验证有效的输出")
    print("=" * 60)
    
    # 使用类方法验证
    validator = SatelliteClusterValidator(logger)
    result = validator.validate_output(valid_output, input_data)
    
    # 生成报告
    report = validator.generate_validation_report(result)
    print(report)
    
    return result


def test_invalid_output():
    """测试无效的输出"""
    logger = setup_logging()
    input_data = load_test_data()
    
    # 无效的输出数据（包含错误）
    invalid_output = {
        "clusters": [
            {
                "cluster_id": 1,
                "master": 999,  # 不存在的卫星
                "sats": [125, 166, 165],
                "targets": [2]
            },
            {
                "cluster_id": 2,
                "master": 145,
                "sats": [145, 115, 155, 143],
                "targets": [3, 10, 999]  # 不存在的目标
            },
            {
                "cluster_id": 3,
                "master": 116,
                "sats": [116, 162, 125, 136, 111, 151, 125],  # 重复的卫星
                "targets": [5, 12, 48, 14, 33]
            }
        ]
    }
    
    print("\n" + "=" * 60)
    print("测试2: 验证无效的输出")
    print("=" * 60)
    
    # 使用便捷函数验证
    result = validate_satellite_clustering(invalid_output, input_data, logger)
    
    # 生成报告
    report = generate_report(result)
    print(report)
    
    return result


def test_malformed_output():
    """测试格式错误的输出"""
    logger = setup_logging()
    input_data = load_test_data()
    
    # 格式错误的输出数据
    malformed_output = {
        "clusters": "not_a_list"  # 应该是列表
    }
    
    print("\n" + "=" * 60)
    print("测试3: 验证格式错误的输出")
    print("=" * 60)
    
    result = validate_satellite_clustering(malformed_output, input_data, logger)
    report = generate_report(result)
    print(report)
    
    return result


def main():
    """主函数"""
    print("卫星分簇结果验证器测试")
    print("=" * 60)
    
    # 运行所有测试
    test_valid_output()
    test_invalid_output()
    test_malformed_output()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main() 