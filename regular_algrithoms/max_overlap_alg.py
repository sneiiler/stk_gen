from typing import List, Dict, Tuple, Union
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, PULP_CBC_CMD, value
import math

from tqdm import tqdm

from misc_tools.sharegpt_utils import create_sharegpt_format
from utils.misc_utils import get_data_dir
from data_classes.sft_data_models import ClusterInfo, SatelliteClusterOutput, LLMConversationMessage, RawConstellationDataModel, SatelliteAttributes, SatelliteEdge, TargetEdge


def convert_time_slice_to_model(time_slice: dict) -> RawConstellationDataModel:
    """
    将time_slice数据转换为RawConstellationDataModel格式
    
    Args:
        time_slice: 原始时间切片数据，示例格式:
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "time_offset_from_scenario_start": 0,
            "satellites": [
                {
                    "id": "Satellite1",
                    "health": 10.0,
                    "position": [7000.0, 0.0, 0.0]
                }
            ],
            "inter_satellite_connectivity": [
                {
                    "from_satellite": {
                        "id": "Satellite1",
                        "position": [7000.0, 0.0, 0.0]
                    },
                    "to_satellite": {
                        "id": "Satellite2", 
                        "position": [7000.0, 1000.0, 0.0]
                    },
                    "connection_quality": 0.8,
                    "visibility_time_window": [0, 100]
                }
            ],
            "target_visibility": [
                {
                    "from_satellite": {
                        "id": "Satellite1",
                        "position": [7000.0, 0.0, 0.0]
                    },
                    "to_target": {
                        "id": "m1",
                        "position": [6371.0, 100.0, 0.0]
                    },
                    "target_value": 1.0,
                    "observation_priority": 1,
                    "visibility_time_window": [10, 90]
                }
            ]
        }
        
    Returns:
        转换后的RawConstellationDataModel对象
    """
    # 提取卫星属性
    sat_attrs = []
    satellites = time_slice.get("satellites", [])
    for sat in satellites:
        sat_id = int(sat["id"].replace("Satellite", "")) if "Satellite" in sat["id"] else int(sat["id"])
        sat_attrs.append(SatelliteAttributes(
            id=sat_id,
            health=sat.get("health", 10.0),  # 默认健康状态为10
            pos=sat.get("position", [0.0, 0.0, 0.0])
        ))
    
    # 提取卫星间连接关系
    sat_edges = []
    inter_connectivity = time_slice.get("inter_satellite_connectivity", [])
    for conn in inter_connectivity:
        from_sat_id = conn["from_satellite"]["id"]
        to_sat_id = conn["to_satellite"]["id"]
        
        # 计算距离
        from_pos = conn["from_satellite"]["position"]
        to_pos = conn["to_satellite"]["position"]
        distance = round(math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(from_pos, to_pos))),2)
        
        sat_edges.append(SatelliteEdge(
            from_sat=from_sat_id,
            to_sat=to_sat_id,
            distance=distance
        ))
    
    # 提取目标连接关系
    target_edges = []
    target_visibility = time_slice.get("target_visibility", [])
    for vis in target_visibility:
        sat_id = vis["from_satellite"]["id"]
        target_id = vis["to_target"]["id"]
        
        # 计算连接质量（基于可见时间窗口长度）
        window = vis.get("visibility_time_window", [])
        quality = (window[1] - window[0]) / 100.0 if len(window) == 2 else 0.5  # 归一化到0-1，默认0.5
        quality = min(1.0, max(0.0, quality))  # 确保在0-1范围内
        
        target_edges.append(TargetEdge(
            sat_id=sat_id,
            target_id=target_id,
            quality=quality
        ))
    
    return RawConstellationDataModel(
        timestamp=time_slice["timestamp"],
        sat_attrs=sat_attrs,
        sat_edges=sat_edges,
        target_edges=target_edges,
        history_cluster_result=None  # 暂时设为None，可以后续扩展
    )


class SatelliteClusteringAlgorithm:
    def __init__(self, c_max, connectivity_weight=0.4, cost_weight=0.6):
        """
        初始化分簇算法

        Args:
            c_max: 容许的最大总通信代价（硬约束）
            connectivity_weight: 选主节点用的连通性权重（默认0.4）
            cost_weight: 选主节点用的通信代价权重（默认0.6）
        
        核心设计思想：
        - 目标函数：最大化观测重叠度（重叠观测可以提高观测精度）
        - 硬约束：通信代价不超过c_max
        - 连通性：确保簇内卫星之间可以通信
        """
        self.c_max = c_max
        self.master_node_connectivity_weight = connectivity_weight
        self.master_node_cost_weight = cost_weight
        
        # 确保主节点选择权重之和为1
        total_weight = connectivity_weight + cost_weight
        if total_weight != 1.0:
            self.master_node_connectivity_weight = connectivity_weight / total_weight
            self.master_node_cost_weight = cost_weight / total_weight

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

    def _check_connectivity_path(self, node1: int, node2: int, adj_matrix: np.ndarray, max_hops: int = 3) -> bool:
        """
        检查两个节点之间是否存在指定跳数内的连接路径
        
        Args:
            node1: 起始节点索引
            node2: 目标节点索引  
            adj_matrix: 邻接矩阵
            max_hops: 最大跳数
            
        Returns:
            是否存在连接路径
        """
        if node1 == node2:
            return True
            
        # 使用BFS搜索路径
        from collections import deque
        
        queue = deque([(node1, 0)])  # (节点, 跳数)
        visited = set([node1])
        
        while queue:
            current_node, hops = queue.popleft()
            
            if hops >= max_hops:
                continue
                
            # 检查所有邻居节点
            for neighbor in range(adj_matrix.shape[0]):
                if adj_matrix[current_node, neighbor] > 0 and neighbor not in visited:
                    if neighbor == node2:
                        return True
                    
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1))
        
        return False

    def _verify_cluster_connectivity(self, cluster_members: List[int], insight_matrix: np.ndarray) -> bool:
        """
        验证簇内所有节点之间是否连通，确保没有孤星
        
        Args:
            cluster_members: 簇成员索引列表
            insight_matrix: 可见性矩阵
            
        Returns:
            True if 簇内所有节点连通，False if 存在孤星
        """
        if len(cluster_members) <= 1:
            return True
            
        # 构建簇内邻接矩阵
        n_members = len(cluster_members)
        cluster_adj = np.zeros((n_members, n_members))
        
        for i, idx1 in enumerate(cluster_members):
            for j, idx2 in enumerate(cluster_members):
                if i != j and insight_matrix[idx1, idx2] > 0:
                    cluster_adj[i, j] = 1
        
        # 使用BFS检查连通性
        from collections import deque
        visited = [False] * n_members
        queue = deque([0])  # 从第一个节点开始
        visited[0] = True
        visited_count = 1
        
        while queue:
            current = queue.popleft()
            for neighbor in range(n_members):
                if cluster_adj[current, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    visited_count += 1
                    queue.append(neighbor)
        
        # 如果所有节点都被访问到，说明连通
        return visited_count == n_members

    def build_matrices(
        self, time_slice: dict
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        构建算法所需的矩阵

        Returns:
            insight_matrix: 卫星间可见性矩阵
            cost_matrix: 通信代价矩阵
            overlap_dict: 卫星对目标的重叠度字典 {(sat_i, sat_j, target_k): overlap}
            satellite_target_visibility: 卫星对目标的可见性字典 {(sat_id, target_id): window}
        """
        connectivity = time_slice.get("inter_satellite_connectivity", [])
        target_visibility = time_slice.get("target_visibility", [])

        # 从inter_satellite_connectivity和target_visibility中获取所有卫星ID
        all_sat_ids = set()
        for conn in connectivity:
            all_sat_ids.add(conn["from_satellite"]["id"])
            all_sat_ids.add(conn["to_satellite"]["id"])

        # 从target_visibility中也获取卫星ID
        for vis in target_visibility:
            if "from_satellite" in vis:
                all_sat_ids.add(vis["from_satellite"]["id"])

        # 创建卫星ID到索引的映射
        sat_ids = sorted(list(all_sat_ids))  # 排序保证一致性
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
            # 处理两种数据格式
            if "from_satellite" in vis and "to_target" in vis:
                # 新格式：有from_satellite和to_target字段
                sat_id = vis["from_satellite"]["id"]
                target_id = vis["to_target"]["id"]
            else:
                # 原始格式：直接有target_id字段，需要从时间切片中找到对应的卫星
                # 这种情况下，我们需要从satellites列表中找到对应的卫星
                satellites = time_slice.get("satellites", [])
                if satellites:
                    # 假设每个时间切片只有一个卫星的数据
                    sat_id = satellites[0].get("id")
                    target_id = vis.get("target_id")
                else:
                    continue

            window = vis.get("visibility_time_window", [])
            if sat_id and target_id and window and sat_id in sat_id_to_idx:
                satellite_target_visibility[(sat_id, target_id)] = window

        # 计算卫星对之间对同一目标的观测重叠度
        overlap_dict = {}
        # 获取所有目标ID
        target_ids = set()
        for vis in target_visibility:
            if "to_target" in vis:
                target_ids.add(vis["to_target"]["id"])
            elif "target_id" in vis:
                target_ids.add(vis["target_id"])

        target_ids = list(target_ids)

        for target_id in target_ids:
            for i, sat_i in enumerate(sat_ids):
                for j, sat_j in enumerate(sat_ids):
                    if i < j:  # 避免重复计算
                        window_i = satellite_target_visibility.get(
                            (sat_i, target_id), []
                        )
                        window_j = satellite_target_visibility.get(
                            (sat_j, target_id), []
                        )

                        if window_i and window_j:
                            overlap = self.calculate_overlap(window_i, window_j)
                            if overlap > 0:
                                overlap_dict[(i, j, target_id)] = overlap

        return insight_matrix, cost_matrix, overlap_dict, satellite_target_visibility

    def solve_clustering_from_model(self, constellation: RawConstellationDataModel) -> SatelliteClusterOutput:
        """
        从RawConstellationDataModel格式求解分簇问题

        Args:
            constellation: RawConstellationDataModel数据

        Returns:
            分簇结果
        """
        if not constellation.sat_edges:
            return SatelliteClusterOutput(chain_of_thought="无卫星连接数据", clusters=[])

        # 构建矩阵
        insight_matrix, cost_matrix, overlap_dict, sat_target_vis = self.build_matrices_from_model(constellation)

        # 获取卫星数量
        n_sats = insight_matrix.shape[0]

        # 获取所有卫星ID - 直接使用字符串格式，避免双重前缀
        sat_ids = [attr.id for attr in constellation.sat_attrs]

        # 创建优化问题
        prob = LpProblem("Satellite_Clustering", LpMaximize)

        # 决策变量：a_ij 表示卫星i和j是否在同一个簇中
        a_vars = {}
        for i in range(n_sats):
            for j in range(n_sats):
                if i != j:
                    a_vars[(i, j)] = LpVariable(f"a_{i}_{j}", cat="Binary")

        # 目标函数：最大化观测重叠度（通信代价作为硬约束）
        observation_objective = 0
        for (i, j, _), overlap in overlap_dict.items():
            observation_objective += overlap * a_vars[(i, j)]
        
        # 单目标优化：只优化观测重叠度
        prob += observation_objective

        # 约束1：通信代价约束（保留作为上限约束）
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
                            prob += (
                                a_vars[(i, k)] >= a_vars[(i, j)] + a_vars[(j, k)] - 1
                            )

        # 求解
        prob.solve(PULP_CBC_CMD(msg=False))

        # 提取分簇结果
        clusters = self.extract_clusters_from_model(
            a_vars, sat_ids, sat_target_vis, n_sats, constellation.timestamp, cost_matrix, insight_matrix
        )

        return clusters

    def build_matrices_from_model(
        self, constellation: RawConstellationDataModel
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        从RawConstellationDataModel构建算法所需的矩阵

        Returns:
            insight_matrix: 卫星间可见性矩阵
            cost_matrix: 通信代价矩阵
            overlap_dict: 卫星对目标的重叠度字典 {(sat_i, sat_j, target_k): overlap}
            satellite_target_visibility: 卫星对目标的可见性字典 {(sat_id, target_id): quality}
        """
        # 获取所有卫星ID并创建映射 - 直接使用字符串格式，避免双重前缀
        sat_ids = [attr.id for attr in constellation.sat_attrs]
        sat_id_to_idx = {sat_id: idx for idx, sat_id in enumerate(sat_ids)}
        n_sats = len(sat_ids)

        # 创建卫星ID到位置的映射 - 直接使用字符串格式的卫星ID
        sat_positions = {attr.id: attr.pos for attr in constellation.sat_attrs}

        # 初始化矩阵
        insight_matrix = np.zeros((n_sats, n_sats))
        cost_matrix = np.zeros((n_sats, n_sats))

        # 构建卫星间可见性和代价矩阵
        for edge in constellation.sat_edges:
            from_id = edge.from_sat
            to_id = edge.to_sat

            if from_id in sat_id_to_idx and to_id in sat_id_to_idx:
                i = sat_id_to_idx[from_id]
                j = sat_id_to_idx[to_id]

                # 设置可见性（如果有边就认为可见）
                insight_matrix[i, j] = 1
                insight_matrix[j, i] = 1

                # 使用边中的距离作为通信代价
                cost = edge.distance
                cost_matrix[i, j] = cost
                cost_matrix[j, i] = cost

        # 构建卫星对目标的可见性
        satellite_target_visibility = {}
        for edge in constellation.target_edges:
            sat_id = edge.sat_id
            target_id = edge.target_id
            quality = edge.quality
            satellite_target_visibility[(sat_id, target_id)] = quality

        # 计算卫星对之间对同一目标的观测重叠度
        overlap_dict = {}
        # 获取所有目标ID
        target_ids = set(edge.target_id for edge in constellation.target_edges)

        for target_id in target_ids:
            for i, sat_i in enumerate(sat_ids):
                for j, sat_j in enumerate(sat_ids):
                    if i < j:  # 避免重复计算
                        quality_i = satellite_target_visibility.get((sat_i, target_id), 0)
                        quality_j = satellite_target_visibility.get((sat_j, target_id), 0)

                        if quality_i > 0 and quality_j > 0:
                            # 重叠度基于两个卫星的观测质量
                            overlap = min(quality_i, quality_j) * max(quality_i, quality_j)
                            overlap_dict[(i, j, target_id)] = overlap

        return insight_matrix, cost_matrix, overlap_dict, satellite_target_visibility

    def extract_clusters_from_model(
        self,
        a_vars: Dict,
        sat_ids: List[Union[int, str]],
        sat_target_vis: Dict,
        n_sats: int,
        timestamp: str,
        cost_matrix: np.ndarray,
        insight_matrix: np.ndarray,
    ) -> SatelliteClusterOutput:
        """
        从求解结果中提取分簇信息（适用于新的数据模型）
        """
        print(f"\n🔧 extract_clusters_from_model: 开始处理时间戳 {timestamp}")
        print(f"📊 输入数据: {len(sat_ids)} 个卫星, {len(sat_target_vis)} 条卫星-目标可见性记录")
        # 记录矩阵信息用于后续验证
        self._last_insight_matrix = insight_matrix
        self._last_sat_ids = sat_ids
        
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

        # 过滤掉过小的簇，并尝试合并到邻近的大簇中
        MIN_CLUSTER_SIZE = 2  # 最小簇大小
        large_clusters = {}
        small_clusters = {}
        
        for root, members in cluster_dict.items():
            if len(members) >= MIN_CLUSTER_SIZE:
                large_clusters[root] = members
            else:
                small_clusters[root] = members
        
        # 尝试将小簇合并到最近的大簇中
        for small_root, small_members in small_clusters.items():
            best_merge_target = None
            min_merge_cost = float('inf')
            
            for large_root, large_members in large_clusters.items():
                # 计算小簇到大簇的最小连接代价
                min_cost = float('inf')
                for small_idx in small_members:
                    for large_idx in large_members:
                        # 检查是否有直接连接或间接连接
                        if adj_matrix[small_idx, large_idx] > 0:  # 直接连接
                            cost = float(cost_matrix[small_idx, large_idx])
                            min_cost = min(min_cost, cost)
                        else:
                            # 尝试通过其他节点中转（2跳连接）
                            for intermediate in range(n_sats):
                                if (adj_matrix[small_idx, intermediate] > 0 and 
                                    adj_matrix[intermediate, large_idx] > 0):
                                    cost = (float(cost_matrix[small_idx, intermediate]) + 
                                           float(cost_matrix[intermediate, large_idx]))
                                    min_cost = min(min_cost, cost)
                
                if min_cost < min_merge_cost:
                    min_merge_cost = min_cost
                    best_merge_target = large_root
            
            # 如果找到合适的合并目标，就合并
            if best_merge_target is not None and min_merge_cost < float('inf'):
                large_clusters[best_merge_target].extend(small_members)
            else:
                # 如果没有找到任何连接，检查是否真的没有连接
                has_any_connection = False
                for small_idx in small_members:
                    for large_root, large_members in large_clusters.items():
                        for large_idx in large_members:
                            # 检查是否有3跳以内的连接路径
                            if self._check_connectivity_path(small_idx, large_idx, adj_matrix, max_hops=3):
                                has_any_connection = True
                                break
                        if has_any_connection:
                            break
                    if has_any_connection:
                        break
                
                if has_any_connection:
                    # 如果确实有连接路径，找到距离最近的大簇进行合并
                    min_size_target = min(large_clusters.keys(), key=lambda k: len(large_clusters[k]))
                    large_clusters[min_size_target].extend(small_members)
                else:
                    # 强制合并：基于可见性连接找到最佳合并目标
                    best_force_merge = None
                    min_force_cost = float('inf')
                    best_connection_pair = None
                    
                    for small_idx in small_members:
                        for large_root, large_members in large_clusters.items():
                            for large_idx in large_members:
                                # 检查原始连接矩阵中是否有可见性
                                if insight_matrix[small_idx, large_idx] > 0:
                                    cost = float(cost_matrix[small_idx, large_idx])
                                    if cost < min_force_cost:
                                        min_force_cost = cost
                                        best_force_merge = large_root
                                        best_connection_pair = (small_idx, large_idx)
                    
                    if best_force_merge is not None:
                        # 检查强制合并后的簇内连通性
                        merged_members = large_clusters[best_force_merge] + small_members
                        if self._verify_cluster_connectivity(merged_members, insight_matrix):
                            large_clusters[best_force_merge].extend(small_members)
                        else:
                            # 如果会产生孤星，保留为独立小簇
                            large_clusters[small_root] = small_members
                    else:
                        # 真的没有任何可见性连接，保留为独立小簇
                        large_clusters[small_root] = small_members
        
        # 使用合并后的簇
        cluster_dict = large_clusters

        # 首先收集所有目标及其可见卫星
        target_to_satellites = defaultdict(set)
        print(f"🔍 收集目标-卫星可见性关系，共 {len(sat_target_vis)} 条记录")
        
        for (s_id, t_id), quality in sat_target_vis.items():
            # 直接使用字符串格式的卫星ID进行匹配，不进行数字转换
            sat_idx = sat_ids.index(s_id) if s_id in sat_ids else -1
            if sat_idx >= 0:
                # 找到卫星所属的簇（在合并后的簇中）
                cluster_root = None
                for root, members in cluster_dict.items():
                    if sat_idx in members:
                        cluster_root = root
                        break
                if cluster_root is not None:
                    target_to_satellites[t_id].add(cluster_root)
        
        print(f"🎯 发现 {len(target_to_satellites)} 个目标可被观测")

        # 为每个目标分配最佳簇，考虑负载均衡
        target_assignments = {}  # 目标 -> 簇根节点
        cluster_target_counts = defaultdict(int)  # 簇 -> 已分配目标数量

        # 按目标被观测的簇数量排序，优先处理竞争激烈的目标
        sorted_targets = sorted(target_to_satellites.items(),
                               key=lambda x: len(x[1]))

        for t_id, cluster_roots in sorted_targets:
            if len(cluster_roots) == 0:
                continue
            elif len(cluster_roots) == 1:
                # 如果目标只被一个簇观测到，直接分配
                root = next(iter(cluster_roots))
                target_assignments[t_id] = root
                cluster_target_counts[root] += 1
            else:
                # 如果目标被多个簇观测到，考虑多个因素选择最佳簇
                best_root = None
                best_score = -float('inf')

                for root in cluster_roots:
                    # 计算该簇中能观测到此目标的卫星数量
                    observers = sum(1 for idx in cluster_dict[root]
                                   if (sat_ids[idx], t_id) in sat_target_vis)

                    # 计算当前簇的负载（目标数/卫星数）
                    cluster_size = len(cluster_dict[root])
                    current_load = cluster_target_counts[root] / cluster_size
                    
                    # 计算观测质量得分
                    total_quality = sum(sat_target_vis.get((sat_ids[idx], t_id), 0) 
                                      for idx in cluster_dict[root] 
                                      if (sat_ids[idx], t_id) in sat_target_vis)
                    quality_score = total_quality / cluster_size if cluster_size > 0 else 0

                    # 大幅强化负载均衡：使用更严格的指数惩罚函数
                    # 目标是让每个簇的负载趋近于1.0（即每个卫星承担1个目标）
                    if current_load <= 0.8:
                        load_penalty = current_load * 0.1  # 几乎无惩罚，鼓励使用低负载簇
                    elif current_load <= 1.2:
                        load_penalty = 0.08 + (current_load - 0.8) * 0.2  # 轻微惩罚
                    elif current_load <= 1.8:
                        load_penalty = 0.16 + (current_load - 1.2) * 0.8  # 中等惩罚
                    else:
                        load_penalty = 0.64 + (current_load - 1.8) ** 2 * 2.0  # 二次方惩罚，快速增长
                    
                    # 进一步调整权重：极大地重视负载均衡
                    # 观测质量权重0.1，负载均衡权重0.9
                    score = 0.1 * quality_score - 0.9 * load_penalty

                    if score > best_score:
                        best_score = score
                        best_root = root

                if best_root is not None:
                    target_assignments[t_id] = best_root
                    cluster_target_counts[best_root] += 1

        print(f"� 目标分配完成，共分配 {len(target_assignments)} 个目标")

        # 构建ClusterInfo对象
        clusters = []
        cluster_id = 0

        for root, members in cluster_dict.items():
            if len(members) > 0:
                # 优化的主节点选择策略：综合考虑连通性和通信代价 
                # TODO 还应该考虑主节点的健康度情况
                best_score = -float('inf')  
                master_idx = members[0]

                for idx in members:
                    # 计算连通性得分
                    connections = sum(adj_matrix[idx, j] for j in members if j != idx)
                    connectivity_score = connections / (len(members) - 1) if len(members) > 1 else 1.0
                    
                    # 计算通信代价得分（以该节点为主节点时的总代价）
                    total_cost = 0.0
                    for member_idx in members:
                        if member_idx != idx:
                            # 计算从成员节点到候选主节点的最短路径代价
                            if adj_matrix[idx, member_idx] > 0:
                                # 直接连接
                                total_cost += float(cost_matrix[idx, member_idx])
                            else:
                                # 需要通过其他节点中转，使用最短路径
                                min_path_cost = float('inf')
                                for intermediate in members:
                                    if intermediate != idx and intermediate != member_idx:
                                        if (adj_matrix[member_idx, intermediate] > 0 and 
                                            adj_matrix[intermediate, idx] > 0):
                                            path_cost = (float(cost_matrix[member_idx, intermediate]) + 
                                                       float(cost_matrix[intermediate, idx]))
                                            min_path_cost = min(min_path_cost, path_cost)
                                
                                if min_path_cost != float('inf'):
                                    total_cost += min_path_cost
                                else:
                                    # 无法连通，给予惩罚
                                    total_cost += 10000.0
                    
                    # 代价得分（代价越低得分越高）
                    cost_score = 1.0 / (1.0 + total_cost / 1000.0)  # 归一化代价得分
                    
                    # 综合得分：使用配置的权重
                    composite_score = (self.master_node_connectivity_weight * connectivity_score + 
                                     self.master_node_cost_weight * cost_score)
                    
                    if composite_score > best_score:
                        best_score = composite_score
                        master_idx = idx

                # 收集分配给该簇的目标
                cluster_targets = set()
                for t_id, assigned_root in target_assignments.items():
                    if assigned_root == root:
                        cluster_targets.add(t_id)

                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    master=str(sat_ids[master_idx]),
                    sats=[str(sat_ids[idx]) for idx in members],
                    targets=list(cluster_targets),
                    timestamp=timestamp,
                )
                # 只包含有目标的簇（有效簇）
                if cluster_info.targets:
                    clusters.append(cluster_info)
                    cluster_id += 1
                else:
                    print(f"  ⚠️ 跳过无目标的簇：{len(members)}个卫星，主节点: {sat_ids[master_idx]}")

        # Double check: 验证分配结果
        self._validate_clustering_result_from_model(clusters, sat_target_vis)

        # 输出分簇结果摘要
        if clusters:
            print(f"⭐ 时间戳 {timestamp}: 生成 {len(clusters)} 个分簇")
            for i, cluster in enumerate(clusters):
                print(f"  簇{i}: {len(cluster.sats)}个卫星, {len(cluster.targets)}个目标, 主节点: {cluster.master}")

        return SatelliteClusterOutput(chain_of_thought="", clusters=clusters)

    def _validate_clustering_result_from_model(self, clusters: List[ClusterInfo],
                                              sat_target_vis: Dict):
        """
        验证分簇结果的正确性（适用于新的数据模型）
        """
        all_satellites = set()
        all_targets = set()

        for cluster in clusters:
            # 检查卫星重复
            for sat in cluster.sats:
                if sat in all_satellites:
                    raise ValueError(f"卫星 {sat} 被分配到多个簇")
                all_satellites.add(sat)

            # 检查目标重复
            for target in cluster.targets:
                if target in all_targets:
                    raise ValueError(f"目标 {target} 被分配到多个簇")
                all_targets.add(target)

            # 检查目标可见性
            for target in cluster.targets:
                visible_sats = []
                for sat in cluster.sats:
                    # 确保使用正确的格式进行查找
                    # sat_target_vis 的键格式是 (sat_id, target_id)
                    if (sat, target) in sat_target_vis:
                        visible_sats.append(sat)

                if not visible_sats:
                    print(f"⚠️  警告：簇 {cluster.cluster_id} 中目标 {target} 不被任何卫星观测")
                    # 调试信息：显示该目标在 sat_target_vis 中的相关记录
                    target_records = [(k, v) for k, v in sat_target_vis.items() if k[1] == target]
                    print(f"     调试：目标 {target} 在 sat_target_vis 中的记录数: {len(target_records)}")
                    if target_records:
                        print(f"     调试：前5条记录: {target_records[:5]}")
                    
                    # 调试信息：显示簇中卫星的格式
                    print(f"     调试：簇中卫星格式: {cluster.sats[:3]}") # 只显示前3个

        # 计算整体负载均衡度
        load_ratios = [len(c.targets) / len(c.sats) for c in clusters if c.sats]
        if load_ratios:
            avg_load = sum(load_ratios) / len(load_ratios)
            load_variance = sum((r - avg_load) ** 2 for r in load_ratios) / len(load_ratios)
            print(f"📊 分簇结果：{len(clusters)} 个簇，平均负载: {avg_load:.2f}, 负载方差: {load_variance:.4f}")
        
        # 检查是否还有单卫星簇
        single_sat_clusters = [c for c in clusters if len(c.sats) == 1]
        if single_sat_clusters:
            print(f"⚠️  警告：仍有 {len(single_sat_clusters)} 个单卫星簇")
        else:
            print("✅ 所有簇都包含至少2个卫星")
        
        # 验证每个簇内的连通性
        self._check_intra_cluster_connectivity_from_model(clusters)

    def _check_intra_cluster_connectivity_from_model(self, clusters: List[ClusterInfo]):
        """检查每个簇内是否存在孤星（适用于新数据模型）"""
        if not hasattr(self, '_last_insight_matrix') or not hasattr(self, '_last_sat_ids'):
            return
            
        insight_matrix = self._last_insight_matrix
        sat_ids = self._last_sat_ids
        
        for cluster in clusters:
            if len(cluster.sats) <= 1:
                continue
                
            # 获取簇内卫星的索引
            cluster_indices = []
            for sat_id in cluster.sats:
                try:
                    idx = sat_ids.index(sat_id)
                    cluster_indices.append(idx)
                except ValueError:
                    continue
            
            if len(cluster_indices) <= 1:
                continue
                
            # 检查簇内连通性
            if not self._verify_cluster_connectivity(cluster_indices, insight_matrix):
                print(f"❌ 错误：簇 {cluster.cluster_id} 内存在孤星卫星！")

    def solve_clustering(self, time_slice: dict) -> SatelliteClusterOutput:
        """
        求解分簇问题

        Args:
            time_slice: 时间切片数据

        Returns:
            分簇结果列表
        """
        connectivity = time_slice.get("inter_satellite_connectivity", [])
        if not connectivity:
            return SatelliteClusterOutput(chain_of_thought="无卫星连接数据", clusters=[])

        # 构建矩阵
        insight_matrix, cost_matrix, overlap_dict, sat_target_vis = self.build_matrices(
            time_slice
        )

        # 从构建的矩阵中获取卫星信息
        n_sats = insight_matrix.shape[0]

        # 从inter_satellite_connectivity中获取所有卫星ID
        all_sat_ids = set()
        for conn in connectivity:
            all_sat_ids.add(conn["from_satellite"]["id"])
            all_sat_ids.add(conn["to_satellite"]["id"])
        sat_ids = sorted(list(all_sat_ids))

        # 创建优化问题
        prob = LpProblem("Satellite_Clustering", LpMaximize)

        # 决策变量：a_ij 表示卫星i和j是否在同一个簇中
        a_vars = {}
        for i in range(n_sats):
            for j in range(n_sats):
                if i != j:
                    a_vars[(i, j)] = LpVariable(f"a_{i}_{j}", cat="Binary")

        # 目标函数：最大化观测重叠度（通信代价作为硬约束）
        observation_objective = 0
        for (i, j, _), overlap in overlap_dict.items():
            observation_objective += overlap * a_vars[(i, j)]
        
        # 单目标优化：只优化观测重叠度
        prob += observation_objective

        # 约束1：通信代价约束（保留作为上限约束）
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
                            prob += (
                                a_vars[(i, k)] >= a_vars[(i, j)] + a_vars[(j, k)] - 1
                            )

        # 求解
        prob.solve(PULP_CBC_CMD(msg=False))

        # 提取分簇结果
        clusters = self.extract_clusters(
            a_vars, sat_ids, sat_target_vis, n_sats, time_slice["timestamp"], cost_matrix
        )

        return clusters

    def extract_clusters(
        self,
        a_vars: Dict,
        sat_ids: List[str],
        sat_target_vis: Dict,
        n_sats: int,
        timestamp: str,
        cost_matrix: np.ndarray,
    ) -> SatelliteClusterOutput:
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

        # 首先收集所有目标及其可见卫星
        target_to_satellites = defaultdict(set)
        for (s_id, t_id), _ in sat_target_vis.items():
            sat_idx = sat_ids.index(s_id) if s_id in sat_ids else -1
            if sat_idx >= 0:
                # 找到卫星所属的簇
                cluster_root = find(sat_idx)
                target_to_satellites[t_id].add(cluster_root)

        # 为每个目标分配最佳簇，考虑负载均衡
        target_assignments = {}  # 目标 -> 簇根节点
        cluster_target_counts = defaultdict(int)  # 簇 -> 已分配目标数量

        # 按目标被观测的簇数量排序，优先处理竞争激烈的目标
        sorted_targets = sorted(target_to_satellites.items(),
                               key=lambda x: len(x[1]))

        for t_id, cluster_roots in sorted_targets:
            if len(cluster_roots) == 1:
                # 如果目标只被一个簇观测到，直接分配
                root = next(iter(cluster_roots))
                target_assignments[t_id] = root
                cluster_target_counts[root] += 1
            else:
                # 如果目标被多个簇观测到，考虑多个因素选择最佳簇
                best_root = None
                best_score = -float('inf')

                for root in cluster_roots:
                    # 计算该簇中能观测到此目标的卫星数量
                    observers = sum(1 for idx in cluster_dict[root]
                                   if (sat_ids[idx], t_id) in sat_target_vis)

                    # 计算当前簇的负载（目标数/卫星数）
                    cluster_size = len(cluster_dict[root])
                    current_load = cluster_target_counts[root] / cluster_size

                    # 综合评分：观测能力强 + 负载低
                    # 观测能力权重0.6，负载均衡权重0.4
                    observation_score = observers / cluster_size  # 归一化观测能力
                    load_penalty = current_load  # 负载惩罚

                    score = 0.6 * observation_score - 0.4 * load_penalty

                    if score > best_score:
                        best_score = score
                        best_root = root

                if best_root is not None:
                    target_assignments[t_id] = best_root
                    cluster_target_counts[best_root] += 1

        # 构建ClusterInfo对象
        clusters = []
        cluster_id = 0

        for root, members in cluster_dict.items():
            if len(members) > 0:
                # 优化的主节点选择策略：综合考虑连通性和通信代价
                best_score = -float('inf')
                master_idx = members[0]

                for idx in members:
                    # 计算连通性得分
                    connections = sum(adj_matrix[idx, j] for j in members if j != idx)
                    connectivity_score = connections / (len(members) - 1) if len(members) > 1 else 1.0
                    
                    # 计算通信代价得分（以该节点为主节点时的总代价）
                    total_cost = 0.0
                    for member_idx in members:
                        if member_idx != idx:
                            # 计算从成员节点到候选主节点的最短路径代价
                            if adj_matrix[idx, member_idx] > 0:
                                # 直接连接
                                total_cost += float(cost_matrix[idx, member_idx])
                            else:
                                # 需要通过其他节点中转，使用最短路径
                                min_path_cost = float('inf')
                                for intermediate in members:
                                    if intermediate != idx and intermediate != member_idx:
                                        if (adj_matrix[member_idx, intermediate] > 0 and 
                                            adj_matrix[intermediate, idx] > 0):
                                            path_cost = (float(cost_matrix[member_idx, intermediate]) + 
                                                       float(cost_matrix[intermediate, idx]))
                                            min_path_cost = min(min_path_cost, path_cost)
                                
                                if min_path_cost != float('inf'):
                                    total_cost += min_path_cost
                                else:
                                    # 无法连通，给予惩罚
                                    total_cost += 10000.0
                    
                    # 代价得分（代价越低得分越高）
                    cost_score = 1.0 / (1.0 + total_cost / 1000.0)  # 归一化代价得分
                    
                    # 综合得分：使用配置的权重
                    # 通信代价权重更高，因为这是我们要优化的主要目标
                    composite_score = (self.master_node_connectivity_weight * connectivity_score + 
                                     self.master_node_cost_weight * cost_score)
                    
                    # 可选：调试输出（仅在需要时启用）
                    # print(f"候选主节点 {sat_ids[idx]}: 连通性={connectivity_score:.3f}, "
                    #       f"代价得分={cost_score:.3f}, 总代价={total_cost:.1f}, "
                    #       f"综合得分={composite_score:.3f}")
                    
                    if composite_score > best_score:
                        best_score = composite_score
                        master_idx = idx

                # 收集分配给该簇的目标
                cluster_targets = set()
                for t_id, assigned_root in target_assignments.items():
                    if assigned_root == root:
                        cluster_targets.add(t_id)

                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    master=sat_ids[master_idx],
                    sats=[sat_ids[idx] for idx in members],
                    targets=[t_id for t_id in cluster_targets],
                    timestamp=timestamp,
                )
                if cluster_info.targets:
                    clusters.append(cluster_info)
                    cluster_id += 1

        # Double check: 验证分配结果
        self._validate_clustering_result(clusters, sat_target_vis)

        resp = SatelliteClusterOutput(chain_of_thought="",clusters=clusters)

        return resp

    def _validate_clustering_result(self, clusters: List[ClusterInfo],
                                   sat_target_vis: Dict):
        """
        验证分簇结果的正确性
        """
        all_satellites = set()
        all_targets = set()

        for cluster in clusters:
            # 检查卫星重复
            for sat in cluster.sats:
                if sat in all_satellites:
                    raise ValueError(f"卫星 {sat} 被分配到多个簇")
                all_satellites.add(sat)

            # 检查目标重复
            for target in cluster.targets:
                if target in all_targets:
                    raise ValueError(f"目标 {target} 被分配到多个簇")
                all_targets.add(target)

            # 检查目标可见性
            for target in cluster.targets:
                target_str = f"m{target}"  # 转换为原始格式
                visible_sats = []
                for sat in cluster.sats:
                    sat_str = f"Satellite{sat}"  # 转换为原始格式
                    if (sat_str, target_str) in sat_target_vis:
                        visible_sats.append(sat)

                if not visible_sats:
                    print(f"⚠️  警告：簇 {cluster.cluster_id} 中目标 {target} 不被任何卫星观测")

        # 计算整体负载均衡度
        load_ratios = [len(c.targets) / len(c.sats) for c in clusters if c.sats]
        if load_ratios:
            avg_load = sum(load_ratios) / len(load_ratios)
            load_variance = sum((r - avg_load) ** 2 for r in load_ratios) / len(load_ratios)
            print(f"分簇结果：{len(clusters)} 个簇，平均负载: {avg_load:.2f}, 负载方差: {load_variance:.4f}")
        
        # 检查是否还有单卫星簇
        single_sat_clusters = [c for c in clusters if len(c.sats) == 1]
        if single_sat_clusters:
            print(f"⚠️  警告：仍有 {len(single_sat_clusters)} 个单卫星簇")
        else:
            print("✅ 所有簇都包含至少2个卫星")


def cluster_satellites(
    constellation_data: List[RawConstellationDataModel], c_max: float
) -> List[LLMConversationMessage]:
    """
    对所有时间切片进行卫星分簇

    Args:
        constellation_data: RawConstellationDataModel数据列表
        c_max: 容许的最大总通信代价

    Returns:
        按时间戳组织的分簇结果
    """
    algorithm = SatelliteClusteringAlgorithm(c_max=c_max)
    results: List[LLMConversationMessage] = []

    # 构造ShareGPT格式的训练数据
    for constellation in tqdm(constellation_data):
        cluster: SatelliteClusterOutput = algorithm.solve_clustering_from_model(constellation)
        results.append(
            create_sharegpt_format(
                instruction="max_overlap_alg",
                input_data=constellation,
                output_data=cluster,
            )
        )

    return results


def load_data(file_path: Path) -> List[RawConstellationDataModel]:
    """加载时间序列数据 - 新格式直接是RawConstellationDataModel格式"""
    # JSONL格式：每行一个JSON对象
    constellation_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            record = json.loads(line)
            
            # 数据已经是RawConstellationDataModel格式，直接解析
            # 解析satellite attributes
            sat_attrs = []
            for sat in record.get("sat_attrs", []):
                sat_attrs.append(SatelliteAttributes(
                    id=sat["id"],
                    health=sat["health"],
                    pos=sat["pos"]
                ))
            
            # 解析satellite edges  
            sat_edges = []
            for edge in record.get("sat_edges", []):
                sat_edges.append(SatelliteEdge(
                    from_sat=edge["from_sat"],
                    to_sat=edge["to_sat"],
                    distance=edge["distance"]
                ))
            
            # 解析target edges
            target_edges = []
            for edge in record.get("target_edges", []):
                target_edges.append(TargetEdge(
                    sat_id=edge["sat_id"],
                    target_id=edge["target_id"],
                    quality=edge["quality"]
                ))
            
            constellation_data.append(RawConstellationDataModel(
                timestamp=record["timestamp"],
                sat_attrs=sat_attrs,
                sat_edges=sat_edges,
                target_edges=target_edges,
                history_cluster_result=record.get("history_cluster_result", [])
            ))
    
    return constellation_data


# 示例使用
if __name__ == "__main__":
    # 加载数据
    data_file = get_data_dir() / "stk_access_result_data/raw_constellation_data_scenario_1.jsonl"

    if not data_file.exists():
        print("数据文件不存在，请检查路径...")
        exit()

    constellation_data = load_data(data_file)
    print(f"成功加载 {len(constellation_data)} 个时间切片")
    # constellation_data = constellation_data[:3]  # 只测试前3个时间切片
    # print(f"测试处理前 {len(constellation_data)} 个时间切片")
    # 执行分簇
    clustering_results = cluster_satellites(constellation_data, c_max=60000.0)

    # 将结果保存为JSONL文件，每个时间切片对应一行
    output_file = get_data_dir() / f"cluster_results_sharegpt_training_data/max_overlap_alg_for_{data_file.stem}.jsonl"

    # 写入JSONL文件
    with open(output_file, "w", encoding="utf-8") as f:
        for i, result in enumerate(clustering_results):
            f.write(result.to_sharegpt_json()+ "\n")

    print(f"\n分簇结果已保存到: {output_file}")
    print(f"共保存了 {len(clustering_results)} 个时间切片的分簇结果")
