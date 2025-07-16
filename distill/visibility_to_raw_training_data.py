import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

from data_classes.sft_data_models import TargetEdge, SatelliteEdge, RawConstellationDataModel
from stk_server.Packages.Tools import ecef_distance
from utils.misc_utils import get_data_dir


def convert_timestamp(timestamp_str: str) -> str:
    """将时间戳转换为ISO格式
    
    Args:
        timestamp_str: 原始时间戳字符串 (例如: "06 Jun 2025 04:01:50.000")
        
    Returns:
        ISO格式的时间戳字符串
    """
    dt = datetime.strptime(timestamp_str, "%d %b %Y %H:%M:%S.%f")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
    """计算两个卫星之间的距离
    
    Args:
        pos1: 第一个卫星的位置 [x, y, z]
        pos2: 第二个卫星的位置 [x, y, z]
        
    Returns:
        两个卫星之间的ECEF距离（单位：千米）
    """
    p1 = {"x": pos1[0], "y": pos1[1], "z": pos1[2]}
    p2 = {"x": pos2[0], "y": pos2[1], "z": pos2[2]}
    return ecef_distance(p1, p2)


def group_by_time_offset(input_data: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """按时间偏移对数据进行分组
    
    Args:
        input_data: 原始卫星数据列表
        
    Returns:
        按时间偏移分组的数据字典
    """
    grouped_data = defaultdict(list)
    for entry in input_data:
        time_offset = entry["time_offset_from_scenario_start"]
        grouped_data[time_offset].append(entry)
    return dict(grouped_data)


def gaussian_sample_health() -> float:
    """使用高斯分布生成卫星健康状态。
    
    使用均值为0.8的高斯分布生成健康状态，并将结果限制在0-1范围内。
    标准差设置为0.2，这样大约95%的值会落在0.4-1.2的范围内。
    结果保留两位小数。
    
    Returns:
        float: 0到1之间的健康状态值，保留两位小数
    """
    # 使用高斯分布生成值，均值0.8，标准差0.2
    value = np.random.normal(0.8, 0.2)
    # 将值限制在0-1范围内并保留两位小数
    return round(max(0.0, min(1.0, value)), 2)


def convert_satellite_data(input_data: List[Dict[str, Any]]) -> List[RawConstellationDataModel]:
    """转换卫星数据格式
    
    Args:
        input_data: 原始卫星数据列表
        
    Returns:
        转换后的数据列表
    """
    # 按时间偏移分组
    grouped_data = group_by_time_offset(input_data)

    converted_data = []
    for time_offset, entries in sorted(grouped_data.items()):
        # 收集该时间切片的所有卫星信息
        sat_attrs = []
        sat_edges = []
        target_edges = []
        timestamp = None

        # 用于去重的集合
        processed_satellites = set()
        processed_connections = set()
        processed_targets = set()

        # 用于存储原始距离
        sat_distances = []
        target_distances = []

        for entry in entries:
            if timestamp is None:
                timestamp = convert_timestamp(entry["timestamp"])

            # 处理卫星属性
            sat_id = int(entry["satellite_info"]["id"].replace("Satellite", ""))
            if sat_id not in processed_satellites:
                sat_attrs.append({
                    "id": sat_id,
                    "health": gaussian_sample_health(),
                    "pos": entry["satellite_info"]["position"]
                })
                processed_satellites.add(sat_id)

            # 处理卫星间连接
            for conn in entry["inter_satellite_connectivity"]:
                from_id = sat_id
                to_id = int(conn["to_satellite_id"].replace("Satellite", ""))
                conn_key = (from_id, to_id)

                if conn_key not in processed_connections:
                    # 计算距离
                    distance = calculate_distance(
                        entry["satellite_info"]["position"],
                        conn["position"]
                    )
                    sat_distances.append(distance)
                    processed_connections.add(conn_key)

            # 处理目标可见性
            for target in entry["target_visibility"]:
                from_id = sat_id
                to_id = int(target["target_id"].replace("m", ""))
                target_key = (from_id, to_id)

                if target_key not in processed_targets:
                    # 计算距离
                    distance = calculate_distance(
                        entry["satellite_info"]["position"],
                        target["position"]
                    )
                    target_distances.append(distance)
                    processed_targets.add(target_key)

        # 重置集合以重新处理连接
        processed_connections.clear()
        processed_targets.clear()

        # 使用归一化后的距离创建边
        sat_distance_idx = 0
        target_distance_idx = 0

        for entry in entries:
            # 处理卫星间连接
            for conn in entry["inter_satellite_connectivity"]:
                from_id = int(entry["satellite_info"]["id"].replace("Satellite", ""))
                to_id = int(conn["to_satellite_id"].replace("Satellite", ""))
                conn_key = (from_id, to_id)

                if conn_key not in processed_connections:
                    sat_edges.append(SatelliteEdge(
                        from_id=from_id,
                        to_id=to_id,
                        distance=round(sat_distances[sat_distance_idx],2)
                    ))
                    sat_distance_idx += 1
                    processed_connections.add(conn_key)

            # 处理目标可见性
            for target in entry["target_visibility"]:
                from_id = int(entry["satellite_info"]["id"].replace("Satellite", ""))
                to_id = int(target["target_id"].replace("m", ""))
                target_key = (from_id, to_id)

                if target_key not in processed_targets:
                    target_edges.append(TargetEdge(
                        sat_id=from_id,
                        target_id=to_id,
                        quality=round(target_distances[target_distance_idx],2)
                    ))
                    target_distance_idx += 1
                    processed_targets.add(target_key)

        converted_data.append(RawConstellationDataModel(
            timestamp=timestamp,
            sat_attrs=sat_attrs,
            sat_edges=sat_edges,
            target_edges=target_edges
        ))

    return converted_data


def main():
    # 读取输入文件
    with open(get_data_dir() / "satellite_target_visibility_data_sc1.json", "r") as f:
        input_data = json.load(f)

    # 转换数据
    converted_data = convert_satellite_data(input_data)

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = get_data_dir() / f"training_data_raw_{timestamp}.json"

    target_distances=[]
    for timeslice in converted_data:
        print(timeslice.sat_edges)
        for target_edge in timeslice.target_edges:
            target_distances.append(target_edge.quality)

    target_distances = []
    for timeslice in converted_data:
        print(timeslice.sat_edges)
        for target_edge in timeslice.target_edges:
            target_distances.append(target_edge.quality)

    # 分析target_distances的分布情况
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    if target_distances:
        distances_array = np.array(target_distances)

        print(f"\n=== Target Distance 分布分析 ===")
        print(f"总数据点: {len(target_distances)}")
        print(f"最小值: {np.min(distances_array):.2f}")
        print(f"最大值: {np.max(distances_array):.2f}")
        print(f"平均值: {np.mean(distances_array):.2f}")
        print(f"中位数: {np.median(distances_array):.2f}")
        print(f"标准差: {np.std(distances_array):.2f}")
        print(f"25%分位数: {np.percentile(distances_array, 25):.2f}")
        print(f"75%分位数: {np.percentile(distances_array, 75):.2f}")
        print(f"95%分位数: {np.percentile(distances_array, 95):.2f}")

        # 分段统计
        ranges = [
            (0, 1000, "0-1000"),
            (1000, 5000, "1000-5000"),
            (5000, 10000, "5000-10000"),
            (10000, 20000, "10000-20000"),
            (20000, float('inf'), "20000+")
        ]

        print(f"\n=== 距离分段统计 ===")
        for min_val, max_val, label in ranges:
            count = sum(1 for d in target_distances if min_val <= d < max_val)
            percentage = (count / len(target_distances)) * 100
            print(f"{label}: {count} ({percentage:.1f}%)")

        # 绘制分布图
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(target_distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Target Distance')
        plt.ylabel('Frequency')
        plt.title('Target Distance Distribution')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.boxplot(target_distances)
        plt.ylabel('Target Distance')
        plt.title('Target Distance Box Plot')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sc_1_target_distance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 查找异常值 (IQR方法)
        Q1 = np.percentile(distances_array, 25)
        Q3 = np.percentile(distances_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = [d for d in target_distances if d < lower_bound or d > upper_bound]
        print(f"\n=== 异常值检测 (IQR方法) ===")
        print(f"下界: {lower_bound:.2f}")
        print(f"上界: {upper_bound:.2f}")
        print(f"异常值数量: {len(outliers)}")
        if outliers:
            print(f"异常值范围: {min(outliers):.2f} - {max(outliers):.2f}")
    else:
        print("❌ 未找到 target_distances 数据")
    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(converted_data):
            f.write(result.model_dump_json()+ "\n")


if __name__ == "__main__":
    main()
