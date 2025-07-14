from typing import List, Dict, Tuple, Set
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from pulp import *
from pydantic import BaseModel, Field
import math

from utils.misc_utils import get_data_dir


class ClusterInfo(BaseModel):
    """分簇信息模型"""
    cluster_id: int = Field(description="分簇ID")
    master: int = Field(description="主节点卫星ID")
    sats: List[int] = Field(description="分簇中的卫星ID列表")
    targets: List[int] = Field(description="分簇观测的目标ID列表")


class SatelliteClusteringAlgorithm:
    def __init__(self, c_max: float = 100.0):
        """
        初始化分簇算法

        Args:
            c_max: 容许的最大总通信代价
        """
        self.c_max = c_max

    def calculate_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """计算两个位置之间的欧氏距离"""
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))

    def calculate_overlap(self, window1: List[float], window2: List[float]) -> float:
        """
        计算两个时间窗口的重叠度

        Args:
            window1: [start, end] 时间窗口1
            window2: [start, end] 时间窗口2

        Returns:
            重叠时长
        """
        if not window1 or not window2:
            return 0.0

        start = max(window1[0], window2[0])
        end = min(window1[1], window2[1])
        overlap = max(0, end - start)
        return overlap

    def build_matrices(self, time_slice: dict) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        构建算法所需的矩阵

        Returns:
            insight_matrix: 卫星间可见性矩阵
            cost_matrix: 通信代价矩阵
            overlap_dict: 卫星对目标的重叠度字典 {(sat_i, sat_j, target_k): overlap}
            satellite_target_visibility: 卫星对目标的可见性字典 {(sat_id, target_id): window}
        """
        satellites = time_slice.get("satellites", [])
        connectivity = time_slice.get("inter_satellite_connectivity", [])
        target_visibility = time_slice.get("target_visibility", [])

        # 创建卫星ID到索引的映射
        sat_ids = [sat["id"] for sat in satellites]
        sat_id_to_idx = {sat_id: idx for idx, sat_id in enumerate(sat_ids)}
        n_sats = len(sat_ids)

        # 初始化矩阵
        insight_matrix = np.zeros((n_sats, n_sats))
        cost_matrix = np.zeros((n_sats, n_sats))

        # 构建卫星间可见性和代价矩阵
        for conn in connectivity:
            from_id = conn["from_satellite"]["id"]
            to_id = conn["to_satellite"]["id"]

            if from_id in sat_id_to_idx and to_id in sat_id_to_idx:
                i = sat_id_to_idx[from_id]
                j = sat_id_to_idx[to_id]

                # 设置可见性（基于连接质量）
                if conn.get("connection_quality", 0) > 0:
                    insight_matrix[i, j] = 1
                    insight_matrix[j, i] = 1

                # 计算通信代价（基于距离和连接质量）
                pos1 = conn["from_satellite"]["position"]
                pos2 = conn["to_satellite"]["position"]
                distance = self.calculate_distance(pos1, pos2)
                quality = conn.get("connection_quality", 1)

                # 代价 = 距离 / 连接质量（质量越高，代价越低）
                cost = distance / max(quality, 0.01)
                cost_matrix[i, j] = cost
                cost_matrix[j, i] = cost

        # 构建卫星对目标的可见性窗口
        satellite_target_visibility = {}
        for vis in target_visibility:
            sat_id = vis["from_satellite"]["id"]
            target_id = vis["to_target"]["id"]
            window = vis.get("visibility_time_window", [])
            if sat_id in sat_id_to_idx and window:
                satellite_target_visibility[(sat_id, target_id)] = window

        # 计算卫星对之间对同一目标的观测重叠度
        overlap_dict = {}
        target_ids = list(set(vis["to_target"]["id"] for vis in target_visibility))

        for target_id in target_ids:
            for i, sat_i in enumerate(sat_ids):
                for j, sat_j in enumerate(sat_ids):
                    if i < j:  # 避免重复计算
                        window_i = satellite_target_visibility.get((sat_i, target_id), [])
                        window_j = satellite_target_visibility.get((sat_j, target_id), [])

                        if window_i and window_j:
                            overlap = self.calculate_overlap(window_i, window_j)
                            if overlap > 0:
                                overlap_dict[(i, j, target_id)] = overlap

        return insight_matrix, cost_matrix, overlap_dict, satellite_target_visibility

    def solve_clustering(self, time_slice: dict) -> List[ClusterInfo]:
        """
        求解分簇问题

        Args:
            time_slice: 时间切片数据

        Returns:
            分簇结果列表
        """
        satellites = time_slice.get("satellites", [])
        if not satellites:
            return []

        sat_ids = [sat["id"] for sat in satellites]
        n_sats = len(sat_ids)

        # 构建矩阵
        insight_matrix, cost_matrix, overlap_dict, sat_target_vis = self.build_matrices(time_slice)

        # 创建优化问题
        prob = LpProblem("Satellite_Clustering", LpMaximize)

        # 决策变量：a_ij 表示卫星i和j是否在同一个簇中
        a_vars = {}
        for i in range(n_sats):
            for j in range(n_sats):
                if i != j:
                    a_vars[(i, j)] = LpVariable(f"a_{i}_{j}", cat='Binary')

        # 目标函数：最大化观测重叠度
        objective = 0
        for (i, j, target_id), overlap in overlap_dict.items():
            objective += overlap * a_vars[(i, j)]
        prob += objective

        # 约束1：通信代价约束
        total_cost = 0
        for i in range(n_sats):
            for j in range(i + 1, n_sats):
                if (i, j) in a_vars:
                    total_cost += cost_matrix[i, j] * a_vars[(i, j)]
        prob += total_cost <= self.c_max

        # 约束2：可见性约束
        for i in range(n_sats):
            for j in range(n_sats):
                if i != j and (i, j) in a_vars:
                    prob += a_vars[(i, j)] <= insight_matrix[i, j]

        # 约束3：对称性约束
        for i in range(n_sats):
            for j in range(n_sats):
                if i < j and (i, j) in a_vars and (j, i) in a_vars:
                    prob += a_vars[(i, j)] == a_vars[(j, i)]

        # 约束4：传递性约束（如果i和j在同一簇，j和k在同一簇，则i和k也在同一簇）
        for i in range(n_sats):
            for j in range(n_sats):
                for k in range(n_sats):
                    if i != j and j != k and i != k:
                        if (i, j) in a_vars and (j, k) in a_vars and (i, k) in a_vars:
                            prob += a_vars[(i, k)] >= a_vars[(i, j)] + a_vars[(j, k)] - 1

        # 求解
        prob.solve(PULP_CBC_CMD(msg=0))

        # 提取分簇结果
        clusters = self.extract_clusters(a_vars, sat_ids, sat_target_vis, n_sats)

        return clusters

    def extract_clusters(self, a_vars: Dict, sat_ids: List[str],
                         sat_target_vis: Dict, n_sats: int) -> List[ClusterInfo]:
        """
        从求解结果中提取分簇信息
        """
        # 构建邻接矩阵
        adj_matrix = np.zeros((n_sats, n_sats))
        for (i, j), var in a_vars.items():
            if value(var) == 1:
                adj_matrix[i, j] = 1

        # 使用并查集找出连通分量（簇）
        parent = list(range(n_sats))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 合并连通的节点
        for i in range(n_sats):
            for j in range(n_sats):
                if adj_matrix[i, j] == 1:
                    union(i, j)

        # 收集簇
        cluster_dict = defaultdict(list)
        for i in range(n_sats):
            root = find(i)
            cluster_dict[root].append(i)

        # 构建ClusterInfo对象
        clusters = []
        cluster_id = 0

        for root, members in cluster_dict.items():
            if len(members) > 0:
                # 选择主节点（选择与其他节点连接最多的节点）
                max_connections = -1
                master_idx = members[0]

                for idx in members:
                    connections = sum(adj_matrix[idx, j] for j in members if j != idx)
                    if connections > max_connections:
                        max_connections = connections
                        master_idx = idx

                # 收集该簇观测的目标
                cluster_targets = set()
                for idx in members:
                    sat_id = sat_ids[idx]
                    for (s_id, t_id), _ in sat_target_vis.items():
                        if s_id == sat_id:
                            cluster_targets.add(t_id)

                # 提取卫星ID中的数字部分
                def extract_sat_id(sat_id_str):
                    # 从 'SatelliteXXX' 格式中提取数字
                    import re
                    match = re.search(r'\d+', sat_id_str)
                    return int(match.group()) if match else 0

                # 提取目标ID中的数字部分
                def extract_target_id(target_id_str):
                    # 从 'TargetXXX' 格式中提取数字
                    import re
                    match = re.search(r'\d+', target_id_str)
                    return int(match.group()) if match else 0

                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    master=extract_sat_id(sat_ids[master_idx]),
                    sats=[extract_sat_id(sat_ids[idx]) for idx in members],
                    targets=[extract_target_id(t_id) for t_id in cluster_targets]
                )
                clusters.append(cluster_info)
                cluster_id += 1

        return clusters


def cluster_satellites(time_slices: List[dict], c_max: float = 100.0) -> Dict[str, List[ClusterInfo]]:
    """
    对所有时间切片进行卫星分簇

    Args:
        time_slices: 时间切片数据列表
        c_max: 容许的最大总通信代价

    Returns:
        按时间戳组织的分簇结果
    """
    algorithm = SatelliteClusteringAlgorithm(c_max=c_max)
    results = {}

    for time_slice in time_slices:
        timestamp = time_slice["timestamp"]
        clusters = algorithm.solve_clustering(time_slice)
        results[timestamp] = clusters

    return results


def load_data(file_path: Path) -> List[dict]:
    """加载时间序列数据"""
    data = json.loads(file_path.read_text())

    timestamp_groups = defaultdict(list)
    for record in data:
        timestamp = record["timestamp"]
        timestamp_groups[timestamp].append(record)

    time_slices = []
    for timestamp, records in sorted(timestamp_groups.items()):
        satellites = []
        all_inter_connectivity = []
        all_target_visibility = []
        time_offset = None

        for record in records:
            satellite_info = record.get("satellite_info", {})
            if satellite_info:
                satellites.append(satellite_info)

            inter_connectivity = record.get("inter_satellite_connectivity", [])
            for conn in inter_connectivity:
                enhanced_conn = {
                    "from_satellite": {
                        "id": satellite_info.get("id"),
                        "position": satellite_info.get("position"),
                    },
                    "to_satellite": {
                        "id": conn.get("to_satellite_id"),
                        "position": conn.get("position"),
                    },
                    "connection_quality": conn.get("connection_quality"),
                    "visibility_time_window": conn.get("visibility_time_window"),
                }
                all_inter_connectivity.append(enhanced_conn)

            target_visibility = record.get("target_visibility", [])
            for target in target_visibility:
                enhanced_target = {
                    "from_satellite": {
                        "id": satellite_info.get("id"),
                        "position": satellite_info.get("position"),
                    },
                    "to_target": {
                        "id": target.get("target_id"),
                        "position": target.get("position"),
                    },
                    "target_value": target.get("target_value"),
                    "observation_priority": target.get("observation_priority"),
                    "visibility_time_window": target.get("visibility_time_window"),
                }
                all_target_visibility.append(enhanced_target)

            if time_offset is None:
                time_offset = record.get("time_offset_from_scenario_start")

        slice_data = {
            "timestamp": timestamp,
            "time_offset_from_scenario_start": time_offset,
            "satellites": satellites,
            "inter_satellite_connectivity": all_inter_connectivity,
            "target_visibility": all_target_visibility,
        }

        time_slices.append(slice_data)

    return time_slices


# 示例使用
if __name__ == "__main__":
    # 加载数据
    data_file = get_data_dir() / "satellite_target_visibility_data.json"

    if not data_file.exists():
        print("数据文件不存在，请检查路径...")
        exit()

    time_slices = load_data(data_file)
    print(f"成功加载 {len(time_slices)} 个时间切片")

    # 执行分簇
    clustering_results = cluster_satellites(time_slices, c_max=200.0)

    # 输出结果
    for timestamp, clusters in clustering_results.items():
        print(f"\n时间戳: {timestamp}")
        for cluster in clusters:
            print(f"  分簇 {cluster.cluster_id}:")
            print(f"    主节点: {cluster.master}")
            print(f"    卫星: {cluster.sats}")
            print(f"    目标: {cluster.targets}")