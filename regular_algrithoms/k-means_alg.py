import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Optional
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum
import heapq

root_dir = Path(__file__).parent.parent
print(root_dir)
sys.path.append(str(root_dir))

from utils.misc_utils import get_data_dir, get_documents_dir, get_project_root


class ClusteringStrategy(Enum):
    BALANCED = "balanced"
    QUALITY = "quality"


class ClusterInfo(BaseModel):
    """分簇信息模型"""
    cluster_id: int = Field(description="分簇ID")
    master: int = Field(description="主节点卫星ID")
    sats: List[int] = Field(description="分簇中的卫星ID列表")
    targets: List[int] = Field(description="分簇观测的目标ID列表")


class SatelliteClusteringSystem:
    """
    基于目标导向的动态卫星分簇系统
    支持balanced和quality两种策略
    """

    def __init__(self, strategy: ClusteringStrategy = ClusteringStrategy.BALANCED):
        """
        初始化聚类系统

        Args:
            strategy: 分簇策略 (balanced/quality)
        """
        self.strategy = strategy
        
        # 全局变量
        self.cluster_merge_threshold = 0.7
        self.balanced_mode_connection_jump = 2
        self.quality_mode_connection_jump = 3
        
        # 优化目标权重
        self.weight_link_strength = 0.4  # 簇内链路强度
        self.weight_observation_quality = 0.3  # 观测质量
        self.weight_cluster_count = 0.2  # 簇数量
        self.weight_satellite_health = 0.1  # 卫星健康度
        
        self.cluster_history = []

    def cluster_time_slice(self, time_slice: Dict) -> List[ClusterInfo]:
        """
        对单个时间切片进行动态聚类

        Args:
            time_slice: 时间切片数据

        Returns:
            List[ClusterInfo]: 聚类信息列表
        """
        # 解析数据
        satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
        
        if not satellites or not target_edges:
            return []
        
        # 生成初始簇（基于目标）
        initial_clusters = self._generate_target_based_clusters(satellites, sat_edges, target_edges)
        
        # 簇合并
        merged_clusters = self._merge_clusters(initial_clusters, sat_edges, target_edges)
        
        # 根据策略进行扩展
        if self.strategy == ClusteringStrategy.QUALITY:
            final_clusters = self._extend_clusters_for_quality(merged_clusters, satellites, sat_edges, target_edges)
        else:
            final_clusters = merged_clusters
        
        # 验证约束条件
        validated_clusters = self._validate_clusters(final_clusters, satellites, sat_edges, target_edges)
        
        # 转换为ClusterInfo格式
        return self._convert_to_cluster_info(validated_clusters, target_edges)

    def _parse_time_slice(self, time_slice: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        解析时间切片数据
        
        Returns:
            satellites: {sat_id: {'health': float, 'position': [x,y,z]}}
            sat_edges: {(sat1, sat2): {'w': float}}
            target_edges: {(sat_id, target_id): {'q': float}}
        """
        satellites = {}
        sat_edges = {}
        target_edges = {}
        
        # 解析卫星信息
        for sat in time_slice.get("satellites", []):
            sat_id = self._normalize_id(sat["id"])
            satellites[sat_id] = {
                'health': sat.get('health', 1.0),  # 默认健康度为1.0
                'position': sat.get('position', [0, 0, 0])
            }
        
        # 从连接和观测数据中补充卫星信息
        for conn in time_slice.get("inter_satellite_connectivity", []):
            from_sat = self._normalize_id(conn["from_satellite"]["id"])
            to_sat = self._normalize_id(conn["to_satellite"]["id"])
            
            if from_sat not in satellites:
                satellites[from_sat] = {
                    'health': 1.0,
                    'position': conn["from_satellite"]["position"]
                }
            if to_sat not in satellites:
                satellites[to_sat] = {
                    'health': 1.0,
                    'position': conn["to_satellite"]["position"]
                }
        
        for obs in time_slice.get("target_visibility", []):
            sat_id = self._normalize_id(obs["from_satellite"]["id"])
            if sat_id not in satellites:
                satellites[sat_id] = {
                    'health': 1.0,
                    'position': obs["from_satellite"]["position"]
                }
        
        # 解析卫星间连接
        for conn in time_slice.get("inter_satellite_connectivity", []):
            from_sat = self._normalize_id(conn["from_satellite"]["id"])
            to_sat = self._normalize_id(conn["to_satellite"]["id"])
            weight = conn.get("connection_quality", 0.5)
            
            # 无向图，存储两个方向
            sat_edges[(from_sat, to_sat)] = {'w': weight}
            sat_edges[(to_sat, from_sat)] = {'w': weight}
        
        # 解析目标观测
        for obs in time_slice.get("target_visibility", []):
            sat_id = self._normalize_id(obs["from_satellite"]["id"])
            target_id = self._normalize_id(obs["to_target"]["id"])
            quality = obs.get("observation_priority", 1.0) * obs.get("target_value", 1.0)
            
            target_edges[(sat_id, target_id)] = {'q': quality}
        
        return satellites, sat_edges, target_edges

    def _normalize_id(self, id_value) -> int:
        """将ID标准化为整数"""
        if isinstance(id_value, int):
            return id_value
        if isinstance(id_value, str) and id_value.isdigit():
            return int(id_value)
        return hash(id_value) % 100000

    def _generate_target_based_clusters(self, satellites: Dict, sat_edges: Dict, target_edges: Dict) -> List[Dict]:
        """
        基于目标生成初始簇
        为每个目标找到最佳观测卫星作为核心
        """
        clusters = []
        used_satellites = set()
        
        # 按目标分组观测关系
        targets_observers = defaultdict(list)
        for (sat_id, target_id), edge_info in target_edges.items():
            targets_observers[target_id].append((sat_id, edge_info['q']))
        
        cluster_id = 0
        for target_id, observers in targets_observers.items():
            # 为每个目标选择最佳观测卫星
            observers.sort(key=lambda x: x[1], reverse=True)  # 按观测质量排序
            
            # 选择观测质量最高且未被使用的卫星
            best_satellite = None
            for sat_id, quality in observers:
                if sat_id not in used_satellites:
                    best_satellite = sat_id
                    break
            
            if best_satellite is not None:
                cluster = {
                    'id': cluster_id,
                    'master': best_satellite,
                    'satellites': {best_satellite},
                    'targets': {target_id}
                }
                clusters.append(cluster)
                used_satellites.add(best_satellite)
                cluster_id += 1
        
        return clusters

    def _merge_clusters(self, clusters: List[Dict], sat_edges: Dict, target_edges: Dict) -> List[Dict]:
        """
        执行簇合并操作
        """
        max_jump = (self.balanced_mode_connection_jump if self.strategy == ClusteringStrategy.BALANCED 
                   else self.quality_mode_connection_jump)
        
        merged = True
        while merged:
            merged = False
            best_merge = None
            best_score = 0
            
            # 寻找最佳合并对
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster1, cluster2 = clusters[i], clusters[j]
                    
                    # 检查簇间连接强度
                    inter_cluster_strength = self._calculate_inter_cluster_strength(
                        cluster1, cluster2, sat_edges
                    )
                    
                    if inter_cluster_strength >= self.cluster_merge_threshold:
                        # 检查合并后跳数约束
                        if self._can_merge_clusters(cluster1, cluster2, sat_edges, max_jump):
                            # 计算合并收益
                            merge_score = self._calculate_merge_score(
                                cluster1, cluster2, sat_edges, target_edges
                            )
                            
                            if merge_score > best_score:
                                best_score = merge_score
                                best_merge = (i, j)
            
            # 执行最佳合并
            if best_merge:
                i, j = best_merge
                merged_cluster = self._merge_two_clusters(
                    clusters[i], clusters[j], sat_edges
                )
                
                # 移除原簇，添加合并后的簇
                clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
                clusters.append(merged_cluster)
                merged = True
        
        return clusters

    def _calculate_inter_cluster_strength(self, cluster1: Dict, cluster2: Dict, sat_edges: Dict) -> float:
        """计算两个簇之间的连接强度"""
        total_strength = 0
        connection_count = 0
        
        for sat1 in cluster1['satellites']:
            for sat2 in cluster2['satellites']:
                if (sat1, sat2) in sat_edges:
                    total_strength += sat_edges[(sat1, sat2)]['w']
                    connection_count += 1
        
        return total_strength / connection_count if connection_count > 0 else 0

    def _can_merge_clusters(self, cluster1: Dict, cluster2: Dict, sat_edges: Dict, max_jump: int) -> bool:
        """检查两个簇是否可以合并（跳数约束）"""
        # 选择新的主节点（健康度和连通性最高的）
        all_satellites = cluster1['satellites'] | cluster2['satellites']
        potential_master = self._select_best_master(all_satellites, sat_edges)
        
        # 检查所有卫星到新主节点的跳数
        for sat_id in all_satellites:
            if sat_id != potential_master:
                if self._calculate_hop_distance(sat_id, potential_master, sat_edges) > max_jump:
                    return False
        
        return True

    def _calculate_merge_score(self, cluster1: Dict, cluster2: Dict, sat_edges: Dict, target_edges: Dict) -> float:
        """计算合并收益分数"""
        # 合并后的簇内链路强度提升
        merged_satellites = cluster1['satellites'] | cluster2['satellites']
        merged_targets = cluster1['targets'] | cluster2['targets']
        
        # 计算合并后的链路强度
        link_strength = self._calculate_intra_cluster_link_strength(merged_satellites, sat_edges)
        
        # 计算观测质量
        observation_quality = self._calculate_cluster_observation_quality(merged_satellites, merged_targets, target_edges)
        
        return link_strength * self.weight_link_strength + observation_quality * self.weight_observation_quality

    def _merge_two_clusters(self, cluster1: Dict, cluster2: Dict, sat_edges: Dict) -> Dict:
        """合并两个簇"""
        merged_satellites = cluster1['satellites'] | cluster2['satellites']
        merged_targets = cluster1['targets'] | cluster2['targets']
        
        # 选择新的主节点
        new_master = self._select_best_master(merged_satellites, sat_edges)
        
        return {
            'id': min(cluster1['id'], cluster2['id']),
            'master': new_master,
            'satellites': merged_satellites,
            'targets': merged_targets
        }

    def _extend_clusters_for_quality(self, clusters: List[Dict], satellites: Dict, sat_edges: Dict, target_edges: Dict) -> List[Dict]:
        """
        质量模式下的簇扩展
        吸纳未分配的健康卫星
        """
        max_jump = self.quality_mode_connection_jump
        used_satellites = set()
        for cluster in clusters:
            used_satellites.update(cluster['satellites'])
        
        # 获取未分配的卫星
        unassigned_satellites = set(satellites.keys()) - used_satellites
        
        # 为每个簇尝试扩展
        for cluster in clusters:
            target_count = len(cluster['targets'])
            max_satellites = 2 * target_count  # quality模式下每簇卫星数 ≤ 2×目标数
            
            if len(cluster['satellites']) >= max_satellites:
                continue
            
            # 寻找可扩展的卫星
            candidates = []
            for sat_id in unassigned_satellites:
                # 检查与簇的连接强度
                connection_strength = self._calculate_satellite_cluster_connection(
                    sat_id, cluster, sat_edges
                )
                
                # 检查跳数约束
                hop_distance = self._calculate_hop_distance(sat_id, cluster['master'], sat_edges)
                
                if connection_strength > 0 and hop_distance <= max_jump:
                    candidates.append((sat_id, connection_strength, satellites[sat_id]['health']))
            
            # 按连接强度和健康度排序
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            # 添加最佳候选卫星
            added_count = 0
            for sat_id, _, _ in candidates:
                if len(cluster['satellites']) < max_satellites and sat_id in unassigned_satellites:
                    cluster['satellites'].add(sat_id)
                    unassigned_satellites.remove(sat_id)
                    added_count += 1
        
        return clusters

    def _calculate_satellite_cluster_connection(self, sat_id: int, cluster: Dict, sat_edges: Dict) -> float:
        """计算卫星与簇的连接强度"""
        total_strength = 0
        connection_count = 0
        
        for cluster_sat in cluster['satellites']:
            if (sat_id, cluster_sat) in sat_edges:
                total_strength += sat_edges[(sat_id, cluster_sat)]['w']
                connection_count += 1
        
        return total_strength / connection_count if connection_count > 0 else 0

    def _select_best_master(self, satellites: Set[int], sat_edges: Dict) -> int:
        """选择最佳主节点"""
        best_master = None
        best_score = -1
        
        for sat_id in satellites:
            # 计算与其他卫星的连接强度总和
            connection_strength = 0
            for other_sat in satellites:
                if other_sat != sat_id and (sat_id, other_sat) in sat_edges:
                    connection_strength += sat_edges[(sat_id, other_sat)]['w']
            
            # 这里假设健康度为1.0，实际应该从satellites数据中获取
            health = 1.0  # satellites[sat_id]['health']
            
            score = connection_strength + health * 0.1
            
            if score > best_score:
                best_score = score
                best_master = sat_id
        
        return best_master if best_master is not None else list(satellites)[0]

    def _calculate_hop_distance(self, sat1: int, sat2: int, sat_edges: Dict) -> int:
        """计算两个卫星之间的跳数距离"""
        if sat1 == sat2:
            return 0
        
        # 构建图
        graph = defaultdict(list)
        for (s1, s2), edge_info in sat_edges.items():
            if edge_info['w'] > 0:  # 只考虑有效连接
                graph[s1].append(s2)
        
        # BFS寻找最短路径
        queue = deque([(sat1, 0)])
        visited = {sat1}
        
        while queue:
            current_sat, distance = queue.popleft()
            
            if current_sat == sat2:
                return distance
            
            for neighbor in graph[current_sat]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return float('inf')  # 不可达

    def _validate_clusters(self, clusters: List[Dict], satellites: Dict, sat_edges: Dict, target_edges: Dict) -> List[Dict]:
        """验证约束条件"""
        valid_clusters = []
        
        # 检查目标覆盖完整性
        all_targets = set()
        for (sat_id, target_id), _ in target_edges.items():
            all_targets.add(target_id)
        
        covered_targets = set()
        for cluster in clusters:
            covered_targets.update(cluster['targets'])
        
        if covered_targets != all_targets:
            print(f"警告：目标覆盖不完整，遗漏目标：{all_targets - covered_targets}")
        
        # 检查每个簇的约束
        for cluster in clusters:
            # 检查卫星数量约束
            target_count = len(cluster['targets'])
            max_satellites = (target_count if self.strategy == ClusteringStrategy.BALANCED 
                            else 2 * target_count)
            
            if len(cluster['satellites']) > max_satellites:
                print(f"警告：簇 {cluster['id']} 卫星数量超限")
                continue
            
            # 检查跳数约束
            max_jump = (self.balanced_mode_connection_jump if self.strategy == ClusteringStrategy.BALANCED 
                       else self.quality_mode_connection_jump)
            
            valid_cluster = True
            for sat_id in cluster['satellites']:
                if sat_id != cluster['master']:
                    hop_distance = self._calculate_hop_distance(sat_id, cluster['master'], sat_edges)
                    if hop_distance > max_jump:
                        print(f"警告：簇 {cluster['id']} 中卫星 {sat_id} 跳数超限")
                        valid_cluster = False
                        break
            
            if valid_cluster:
                valid_clusters.append(cluster)
        
        return valid_clusters

    def _calculate_intra_cluster_link_strength(self, satellites: Set[int], sat_edges: Dict) -> float:
        """计算簇内链路强度"""
        total_strength = 0
        for sat1 in satellites:
            for sat2 in satellites:
                if sat1 != sat2 and (sat1, sat2) in sat_edges:
                    total_strength += sat_edges[(sat1, sat2)]['w']
        return total_strength

    def _calculate_cluster_observation_quality(self, satellites: Set[int], targets: Set[int], target_edges: Dict) -> float:
        """计算簇的观测质量"""
        total_quality = 0
        for sat_id in satellites:
            for target_id in targets:
                if (sat_id, target_id) in target_edges:
                    total_quality += target_edges[(sat_id, target_id)]['q']
        return total_quality

    def _convert_to_cluster_info(self, clusters: List[Dict], target_edges: Dict) -> List[ClusterInfo]:
        """转换为ClusterInfo格式"""
        cluster_infos = []
        
        for cluster in clusters:
            # 确保目标列表包含该簇实际能观测的目标
            actual_targets = set()
            for sat_id in cluster['satellites']:
                for (s_id, t_id), _ in target_edges.items():
                    if s_id == sat_id:
                        actual_targets.add(t_id)
            
            cluster_info = ClusterInfo(
                cluster_id=cluster['id'],
                master=cluster['master'],
                sats=sorted(list(cluster['satellites'])),
                targets=sorted(list(actual_targets))
            )
            
            cluster_infos.append(cluster_info)
        
        return sorted(cluster_infos, key=lambda x: x.cluster_id)

    def calculate_overall_score(self, cluster_infos: List[ClusterInfo], satellites: Dict, sat_edges: Dict, target_edges: Dict) -> float:
        """计算整体优化分数"""
        total_link_strength = 0
        total_observation_quality = 0
        total_clusters = len(cluster_infos)
        total_health = 0
        
        for cluster_info in cluster_infos:
            # 簇内链路强度
            cluster_sats = set(cluster_info.sats)
            total_link_strength += self._calculate_intra_cluster_link_strength(cluster_sats, sat_edges)
            
            # 观测质量
            cluster_targets = set(cluster_info.targets)
            total_observation_quality += self._calculate_cluster_observation_quality(cluster_sats, cluster_targets, target_edges)
            
            # 健康度
            for sat_id in cluster_info.sats:
                total_health += satellites.get(sat_id, {}).get('health', 1.0)
        
        # 归一化分数
        normalized_link = total_link_strength / max(len(sat_edges), 1)
        normalized_quality = total_observation_quality / max(len(target_edges), 1)
        normalized_clusters = 1.0 / max(total_clusters, 1)  # 簇数越少越好
        normalized_health = total_health / max(len(satellites), 1)
        
        # 综合分数
        score = (self.weight_link_strength * normalized_link +
                self.weight_observation_quality * normalized_quality +
                self.weight_cluster_count * normalized_clusters +
                self.weight_satellite_health * normalized_health)
        
        return score

    def visualize_clusters(self, time_slice: Dict, cluster_infos: List[ClusterInfo]):
        """可视化聚类结果"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection="3d")

        satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
        
        # 颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_infos)))
        
        # 绘制卫星
        for i, cluster_info in enumerate(cluster_infos):
            cluster_positions = []
            master_position = None
            
            for sat_id in cluster_info.sats:
                if sat_id in satellites:
                    pos = satellites[sat_id]['position']
                    cluster_positions.append(pos)
                    
                    if sat_id == cluster_info.master:
                        master_position = pos
            
            if cluster_positions:
                cluster_positions = np.array(cluster_positions)
                
                # 绘制簇成员
                ax.scatter(
                    cluster_positions[:, 0],
                    cluster_positions[:, 1], 
                    cluster_positions[:, 2],
                    c=[colors[i]], s=60, alpha=0.7,
                    label=f'Cluster {cluster_info.cluster_id}'
                )
                
                # 标记主节点
                if master_position is not None:
                    ax.scatter(
                        master_position[0], master_position[1], master_position[2],
                        c='red', marker='^', s=200, linewidth=3
                    )
        
        # 绘制连接
        for (sat1, sat2), edge_info in sat_edges.items():
            if sat1 in satellites and sat2 in satellites:
                pos1 = satellites[sat1]['position']
                pos2 = satellites[sat2]['position']
                
                ax.plot(
                    [pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                    'gray', alpha=0.3, linewidth=edge_info['w']
                )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title(f"Dynamic Satellite Clustering ({self.strategy.value} mode)")
        ax.legend()

        plt.show()

    def export_results(
        self,
        clustering_results: List[List[ClusterInfo]],
        time_series_data: List[Dict],
        filename: str = "dynamic_clustering_results.json",
    ):
        """导出聚类结果"""
        export_data = []

        for i, (cluster_infos, time_slice) in enumerate(
            zip(clustering_results, time_series_data)
        ):
            timestamp = time_slice.get("timestamp", i)
            
            # 计算质量评分
            satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
            overall_score = self.calculate_overall_score(cluster_infos, satellites, sat_edges, target_edges)
            
            time_slice_data = {
                "timestamp": timestamp,
                "strategy": self.strategy.value,
                "overall_score": overall_score,
                "cluster_count": len(cluster_infos),
                "clusters": [cluster_info.dict() for cluster_info in cluster_infos]
            }
            
            export_data.append(time_slice_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {filename}")


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
    data_file = get_data_dir() / "satellite_target_visibility_data_sc1.json"

    if not data_file.exists():
        print("数据文件不存在，请检查路径...")
        exit()

    time_slices = load_data(data_file)
    print(f"成功加载 {len(time_slices)} 个时间切片")

    # 测试两种策略
    for strategy in [ClusteringStrategy.BALANCED, ClusteringStrategy.QUALITY]:
        print(f"\n=== 测试 {strategy.value} 策略 ===")
        
        # 创建聚类系统
        clustering_system = SatelliteClusteringSystem(strategy=strategy)
        
        # 对单个时间切片进行聚类
        cluster_infos = clustering_system.cluster_time_slice(time_slices[0])
        
        # 打印结果
        print(f"生成 {len(cluster_infos)} 个簇：")
        for cluster_info in cluster_infos:
            print(f"  簇 {cluster_info.cluster_id}: 主节点={cluster_info.master}, "
                  f"卫星数={len(cluster_info.sats)}, 目标数={len(cluster_info.targets)}")
        
        # 计算质量分数
        satellites, sat_edges, target_edges = clustering_system._parse_time_slice(time_slices[0])
        score = clustering_system.calculate_overall_score(cluster_infos, satellites, sat_edges, target_edges)
        print(f"整体质量分数: {score:.4f}")
        
        # 可视化
        clustering_system.visualize_clusters(time_slices[0], cluster_infos)
        
        # 导出结果
        clustering_system.export_results(
            [cluster_infos], [time_slices[0]], 
            f"clustering_results_{strategy.value}.json"
        )

    print("\n动态分簇系统测试完成！")