import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum
import heapq
from itertools import combinations

# 假设这些工具函数存在
# from utils.misc_utils import get_data_dir, get_documents_dir, get_project_root


class ClusteringStrategy(Enum):
    BALANCED = "balanced"
    QUALITY = "quality"


class ClusterInfo(BaseModel):
    """分簇信息模型"""
    cluster_id: int = Field(description="分簇ID")
    master: int = Field(description="主节点卫星ID")
    sats: List[int] = Field(description="分簇中的卫星ID列表")
    targets: List[int] = Field(description="分簇观测的目标ID列表")
    
    # 新增字段用于评估
    avg_intra_link_strength: float = Field(default=0.0, description="簇内平均链路强度")
    total_observation_quality: float = Field(default=0.0, description="总观测质量")
    avg_health: float = Field(default=1.0, description="平均健康度")


class SatelliteClusteringSystem:
    """
    基于目标导向的动态卫星分簇系统 - 改进版
    """

    def __init__(self, strategy: ClusteringStrategy = ClusteringStrategy.QUALITY):
        """
        初始化聚类系统

        Args:
            strategy: 分簇策略 (balanced/quality)
        """
        self.strategy = strategy
        
        # 全局参数
        self.cluster_merge_threshold = 0.7
        self.balanced_mode_connection_jump = 2
        self.quality_mode_connection_jump = 3
        
        # 优化目标权重
        self.weight_link_strength = 0.4
        self.weight_observation_quality = 0.3
        self.weight_cluster_count = 0.2
        self.weight_satellite_health = 0.1
        
        # 新增参数
        self.min_cluster_link_density = 0.3  # 最小簇内链路密度
        self.redundancy_factor = 1.5  # quality模式的冗余系数
        
        self.debug_mode = True  # 调试模式
        self.clustering_history = []

    def cluster_time_slice(self, time_slice: Dict) -> List[ClusterInfo]:
        """
        对单个时间切片进行动态聚类 - 改进版

        Args:
            time_slice: 时间切片数据

        Returns:
            List[ClusterInfo]: 聚类信息列表
        """
        # 解析数据
        satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
        
        if not satellites or not target_edges:
            return []
        
        if self.debug_mode:
            print(f"\n=== 开始聚类 ===")
            print(f"卫星数量: {len(satellites)}")
            print(f"目标数量: {len(set(t for _, t in target_edges.keys()))}")
            print(f"策略: {self.strategy.value}")
        
        # 第一步：基于目标生成初始簇（改进版）
        initial_clusters = self._generate_smart_initial_clusters(
            satellites, sat_edges, target_edges
        )
        
        if self.debug_mode:
            print(f"\n初始簇数量: {len(initial_clusters)}")
            self._print_clusters_summary(initial_clusters)
        
        # 第二步：智能簇合并
        merged_clusters = self._smart_merge_clusters(
            initial_clusters, satellites, sat_edges, target_edges
        )
        
        if self.debug_mode:
            print(f"\n合并后簇数量: {len(merged_clusters)}")
            self._print_clusters_summary(merged_clusters)
        
        # 第三步：根据策略进行优化
        if self.strategy == ClusteringStrategy.QUALITY:
            final_clusters = self._optimize_clusters_for_quality(
                merged_clusters, satellites, sat_edges, target_edges
            )
        else:
            final_clusters = self._optimize_clusters_for_balanced(
                merged_clusters, satellites, sat_edges, target_edges
            )
        
        if self.debug_mode:
            print(f"\n优化后簇数量: {len(final_clusters)}")
            self._print_clusters_summary(final_clusters)
        
        # 第四步：验证和修复
        validated_clusters = self._validate_and_fix_clusters(
            final_clusters, satellites, sat_edges, target_edges
        )
        
        # 转换为ClusterInfo格式
        cluster_infos = self._convert_to_cluster_info(
            validated_clusters, satellites, sat_edges, target_edges
        )
        
        if self.debug_mode:
            print(f"\n=== 聚类完成 ===")
            print(f"最终簇数量: {len(cluster_infos)}")
            for info in cluster_infos:
                print(f"簇 {info.cluster_id}: 主节点={info.master}, "
                      f"卫星数={len(info.sats)}, 目标数={len(info.targets)}, "
                      f"链路强度={info.avg_intra_link_strength:.3f}")
        
        return cluster_infos

    def _generate_smart_initial_clusters(
        self, satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> List[Dict]:
        """
        智能生成初始簇 - 考虑多目标协同
        """
        clusters = []
        used_satellites = set()
        covered_targets = set()
        
        # 构建目标-卫星观测图
        target_observers = defaultdict(list)
        for (sat_id, target_id), edge_info in target_edges.items():
            target_observers[target_id].append((sat_id, edge_info['q']))
        
        # 构建卫星-目标覆盖图
        satellite_targets = defaultdict(list)
        for (sat_id, target_id), edge_info in target_edges.items():
            satellite_targets[sat_id].append((target_id, edge_info['q']))
        
        # 计算每个卫星的"价值"（能观测的目标数量和质量）
        satellite_values = {}
        for sat_id in satellites:
            targets = satellite_targets.get(sat_id, [])
            # 价值 = 目标数量 + 平均观测质量
            value = len(targets) + (sum(q for _, q in targets) / len(targets) if targets else 0)
            satellite_values[sat_id] = value
        
        cluster_id = 0
        
        # 优先处理高价值卫星
        sorted_satellites = sorted(satellite_values.items(), key=lambda x: x[1], reverse=True)
        
        for sat_id, _ in sorted_satellites:
            if sat_id in used_satellites:
                continue
                
            # 获取该卫星能观测的未覆盖目标
            uncovered_targets = [
                (t_id, q) for t_id, q in satellite_targets[sat_id]
                if t_id not in covered_targets
            ]
            
            if not uncovered_targets:
                continue
            
            # 创建以该卫星为核心的簇
            cluster_satellites = {sat_id}
            cluster_targets = {t_id for t_id, _ in uncovered_targets}
            
            # 尝试吸纳能协同观测这些目标的邻近卫星
            neighbors = self._find_connected_satellites(sat_id, sat_edges, max_hops=2)
            
            for neighbor_id in neighbors:
                if neighbor_id in used_satellites:
                    continue
                    
                # 检查邻居是否能观测簇内目标
                neighbor_targets = set(t_id for t_id, _ in satellite_targets.get(neighbor_id, []))
                if neighbor_targets & cluster_targets:  # 有交集
                    cluster_satellites.add(neighbor_id)
                    # 限制初始簇大小
                    if len(cluster_satellites) >= len(cluster_targets):
                        break
            
            # 创建簇
            cluster = {
                'id': cluster_id,
                'master': self._select_cluster_master(cluster_satellites, sat_edges, satellites),
                'satellites': cluster_satellites,
                'targets': cluster_targets
            }
            
            clusters.append(cluster)
            used_satellites.update(cluster_satellites)
            covered_targets.update(cluster_targets)
            cluster_id += 1
        
        # 处理剩余未覆盖的目标
        uncovered_targets = set(target_observers.keys()) - covered_targets
        for target_id in uncovered_targets:
            observers = target_observers[target_id]
            if not observers:
                continue
                
            # 找到最佳观测者
            observers.sort(key=lambda x: x[1], reverse=True)
            for sat_id, quality in observers:
                if sat_id not in used_satellites:
                    cluster = {
                        'id': cluster_id,
                        'master': sat_id,
                        'satellites': {sat_id},
                        'targets': {target_id}
                    }
                    clusters.append(cluster)
                    used_satellites.add(sat_id)
                    covered_targets.add(target_id)
                    cluster_id += 1
                    break
        
        return clusters

    def _smart_merge_clusters(
        self, clusters: List[Dict], satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> List[Dict]:
        """
        智能簇合并 - 基于多维度评分
        """
        max_jump = (self.balanced_mode_connection_jump if self.strategy == ClusteringStrategy.BALANCED 
                   else self.quality_mode_connection_jump)
        
        merged = True
        iteration = 0
        
        while merged and iteration < 10:  # 防止无限循环
            merged = False
            iteration += 1
            
            # 计算所有可能的合并候选
            merge_candidates = []
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster1, cluster2 = clusters[i], clusters[j]
                    
                    # 计算合并评分
                    merge_score = self._calculate_comprehensive_merge_score(
                        cluster1, cluster2, satellites, sat_edges, target_edges, max_jump
                    )
                    
                    if merge_score > 0:
                        merge_candidates.append((merge_score, i, j))
            
            # 按评分排序，选择最佳合并
            if merge_candidates:
                merge_candidates.sort(reverse=True)
                best_score, i, j = merge_candidates[0]
                
                if self.debug_mode:
                    print(f"\n合并簇 {clusters[i]['id']} 和 {clusters[j]['id']}, 评分: {best_score:.3f}")
                
                # 执行合并
                merged_cluster = self._perform_cluster_merge(
                    clusters[i], clusters[j], satellites, sat_edges
                )
                
                # 更新簇列表
                new_clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
                new_clusters.append(merged_cluster)
                clusters = new_clusters
                merged = True
        
        return clusters

    def _calculate_comprehensive_merge_score(
        self, cluster1: Dict, cluster2: Dict, satellites: Dict, 
        sat_edges: Dict, target_edges: Dict, max_jump: int
    ) -> float:
        """
        计算综合合并评分
        """
        # 1. 检查基本合并条件
        inter_strength = self._calculate_inter_cluster_strength(cluster1, cluster2, sat_edges)
        if inter_strength < self.cluster_merge_threshold:
            return 0
        
        # 2. 检查跳数约束
        merged_sats = cluster1['satellites'] | cluster2['satellites']
        potential_master = self._select_cluster_master(merged_sats, sat_edges, satellites)
        
        for sat_id in merged_sats:
            if self._calculate_hop_distance(sat_id, potential_master, sat_edges) > max_jump:
                return 0
        
        # 3. 计算多维度评分
        scores = {}
        
        # 链路强度提升
        before_link = (self._calculate_intra_cluster_link_strength(cluster1['satellites'], sat_edges) +
                      self._calculate_intra_cluster_link_strength(cluster2['satellites'], sat_edges))
        after_link = self._calculate_intra_cluster_link_strength(merged_sats, sat_edges)
        scores['link_improvement'] = (after_link - before_link) / max(before_link, 1)
        
        # 观测效率提升
        merged_targets = cluster1['targets'] | cluster2['targets']
        observation_efficiency = self._calculate_observation_efficiency(merged_sats, merged_targets, target_edges)
        scores['observation_efficiency'] = observation_efficiency
        
        # 簇规模合理性
        target_count = len(merged_targets)
        sat_count = len(merged_sats)
        if self.strategy == ClusteringStrategy.BALANCED:
            size_ratio = sat_count / max(target_count, 1)
            scores['size_fitness'] = 1.0 - abs(1.0 - size_ratio)  # 越接近1:1越好
        else:
            size_ratio = sat_count / max(target_count * 2, 1)
            scores['size_fitness'] = 1.0 - abs(0.75 - size_ratio)  # 理想比例1.5:2
        
        # 拓扑结构改善
        scores['topology'] = self._evaluate_topology_quality(merged_sats, sat_edges)
        
        # 综合评分
        total_score = (
            scores['link_improvement'] * 0.3 +
            scores['observation_efficiency'] * 0.3 +
            scores['size_fitness'] * 0.2 +
            scores['topology'] * 0.2
        ) * inter_strength
        
        return total_score

    def _optimize_clusters_for_quality(
        self, clusters: List[Dict], satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> List[Dict]:
        """
        质量模式优化 - 增强版
        """
        max_jump = self.quality_mode_connection_jump
        
        # 获取所有已使用的卫星
        used_satellites = set()
        for cluster in clusters:
            used_satellites.update(cluster['satellites'])
        
        # 获取未分配的健康卫星
        available_satellites = []
        for sat_id, sat_info in satellites.items():
            if sat_id not in used_satellites:
                health = sat_info.get('health', 1.0)
                if health > 0.5:  # 只考虑健康度较高的卫星
                    available_satellites.append((sat_id, health))
        
        # 按健康度排序
        available_satellites.sort(key=lambda x: x[1], reverse=True)
        
        # 为每个簇进行优化扩展
        for cluster in clusters:
            target_count = len(cluster['targets'])
            max_satellites = int(2 * target_count)  # quality模式允许2倍
            current_size = len(cluster['satellites'])
            
            if current_size >= max_satellites:
                continue
            
            # 计算簇的扩展潜力
            expansion_candidates = []
            
            for sat_id, health in available_satellites:
                if sat_id in cluster['satellites']:
                    continue
                
                # 评估加入该卫星的收益
                benefit = self._evaluate_satellite_addition_benefit(
                    sat_id, cluster, satellites, sat_edges, target_edges, max_jump
                )
                
                if benefit > 0:
                    expansion_candidates.append((benefit, sat_id))
            
            # 按收益排序，逐个添加
            expansion_candidates.sort(reverse=True)
            
            for benefit, sat_id in expansion_candidates:
                if len(cluster['satellites']) >= max_satellites:
                    break
                    
                cluster['satellites'].add(sat_id)
                # 更新可用卫星列表
                available_satellites = [(s, h) for s, h in available_satellites if s != sat_id]
                
                # 可能需要更新主节点
                if len(cluster['satellites']) % 3 == 0:  # 每添加3个卫星重新评估主节点
                    cluster['master'] = self._select_cluster_master(
                        cluster['satellites'], sat_edges, satellites
                    )
        
        # 创建备份簇（提高系统冗余度）
        self._create_backup_clusters(clusters, available_satellites, satellites, sat_edges, target_edges)
        
        return clusters

    def _optimize_clusters_for_balanced(
        self, clusters: List[Dict], satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> List[Dict]:
        """
        平衡模式优化 - 精简卫星使用
        """
        optimized_clusters = []
        
        for cluster in clusters:
            # 尝试精简每个簇
            essential_satellites = self._find_essential_satellites(
                cluster, satellites, sat_edges, target_edges
            )
            
            if len(essential_satellites) < len(cluster['satellites']):
                cluster['satellites'] = essential_satellites
                cluster['master'] = self._select_cluster_master(
                    essential_satellites, sat_edges, satellites
                )
            
            optimized_clusters.append(cluster)
        
        return optimized_clusters

    def _evaluate_satellite_addition_benefit(
        self, sat_id: int, cluster: Dict, satellites: Dict, 
        sat_edges: Dict, target_edges: Dict, max_jump: int
    ) -> float:
        """
        评估添加卫星的收益
        """
        # 检查跳数约束
        hop_distance = self._calculate_hop_distance(sat_id, cluster['master'], sat_edges)
        if hop_distance > max_jump:
            return 0
        
        benefit = 0
        
        # 1. 链路强度改善
        link_improvement = 0
        for cluster_sat in cluster['satellites']:
            if (sat_id, cluster_sat) in sat_edges:
                link_improvement += sat_edges[(sat_id, cluster_sat)]['w']
        benefit += link_improvement * 0.4
        
        # 2. 新增观测能力
        new_observations = 0
        for (s_id, t_id), edge_info in target_edges.items():
            if s_id == sat_id and t_id not in cluster['targets']:
                new_observations += edge_info['q']
        benefit += new_observations * 0.3
        
        # 3. 拓扑结构改善（增加路径冗余）
        topology_improvement = self._calculate_topology_improvement(
            sat_id, cluster['satellites'], sat_edges
        )
        benefit += topology_improvement * 0.2
        
        # 4. 卫星健康度
        health = satellites.get(sat_id, {}).get('health', 1.0)
        benefit += health * 0.1
        
        return benefit

    def _validate_and_fix_clusters(
        self, clusters: List[Dict], satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> List[Dict]:
        """
        验证并修复簇 - 确保满足所有约束
        """
        valid_clusters = []
        
        # 检查目标覆盖
        all_targets = set(t_id for _, t_id in target_edges.keys())
        covered_targets = set()
        for cluster in clusters:
            covered_targets.update(cluster['targets'])
        
        missing_targets = all_targets - covered_targets
        if missing_targets:
            print(f"警告：发现未覆盖目标 {missing_targets}，正在修复...")
            # 为未覆盖目标创建补充簇
            for target_id in missing_targets:
                observers = [(s, q) for (s, t), edge in target_edges.items() 
                           if t == target_id for q in [edge['q']]]
                if observers:
                    observers.sort(key=lambda x: x[1], reverse=True)
                    sat_id = observers[0][0]
                    
                    # 尝试将目标添加到现有簇
                    added = False
                    for cluster in clusters:
                        if sat_id in cluster['satellites']:
                            cluster['targets'].add(target_id)
                            added = True
                            break
                    
                    if not added:
                        # 创建新簇
                        new_cluster = {
                            'id': len(clusters),
                            'master': sat_id,
                            'satellites': {sat_id},
                            'targets': {target_id}
                        }
                        clusters.append(new_cluster)
        
        # 验证每个簇
        for cluster in clusters:
            # 确保簇内实际能观测到声明的目标
            actual_targets = set()
            for sat_id in cluster['satellites']:
                for (s_id, t_id), _ in target_edges.items():
                    if s_id == sat_id:
                        actual_targets.add(t_id)
            
            cluster['targets'] = actual_targets  # 更新为实际能观测的目标
            
            if not cluster['targets']:  # 如果簇不能观测任何目标，跳过
                continue
            
            # 验证约束
            if self._validate_cluster_constraints(cluster, satellites, sat_edges, target_edges):
                valid_clusters.append(cluster)
            else:
                # 尝试修复
                fixed_cluster = self._fix_cluster_constraints(
                    cluster, satellites, sat_edges, target_edges
                )
                if fixed_cluster:
                    valid_clusters.append(fixed_cluster)
        
        return valid_clusters

    def _validate_cluster_constraints(
        self, cluster: Dict, satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> bool:
        """
        验证簇是否满足所有约束
        """
        # 检查卫星数量约束
        target_count = len(cluster['targets'])
        sat_count = len(cluster['satellites'])
        
        if self.strategy == ClusteringStrategy.BALANCED:
            if sat_count > target_count:
                return False
        else:
            if sat_count > 2 * target_count:
                return False
        
        # 检查跳数约束
        max_jump = (self.balanced_mode_connection_jump if self.strategy == ClusteringStrategy.BALANCED 
                   else self.quality_mode_connection_jump)
        
        for sat_id in cluster['satellites']:
            if sat_id != cluster['master']:
                hop_distance = self._calculate_hop_distance(sat_id, cluster['master'], sat_edges)
                if hop_distance > max_jump or hop_distance == float('inf'):
                    return False
        
        return True

    def _fix_cluster_constraints(
        self, cluster: Dict, satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> Optional[Dict]:
        """
        尝试修复不满足约束的簇
        """
        # 如果簇太大，移除边缘卫星
        target_count = len(cluster['targets'])
        max_sats = target_count if self.strategy == ClusteringStrategy.BALANCED else 2 * target_count
        
        while len(cluster['satellites']) > max_sats:
            # 找到最远的卫星
            farthest_sat = None
            max_distance = -1
            
            for sat_id in cluster['satellites']:
                if sat_id != cluster['master']:
                    distance = self._calculate_hop_distance(sat_id, cluster['master'], sat_edges)
                    if distance > max_distance:
                        max_distance = distance
                        farthest_sat = sat_id
            
            if farthest_sat:
                cluster['satellites'].remove(farthest_sat)
            else:
                break
        
        # 重新选择主节点
        cluster['master'] = self._select_cluster_master(
            cluster['satellites'], sat_edges, satellites
        )
        
        return cluster if self._validate_cluster_constraints(
            cluster, satellites, sat_edges, target_edges
        ) else None

    def _find_essential_satellites(
        self, cluster: Dict, satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> Set[int]:
        """
        找到簇中必要的卫星（去除冗余）
        """
        essential = set()
        
        # 每个目标至少需要一个观测者
        for target_id in cluster['targets']:
            observers = []
            for sat_id in cluster['satellites']:
                if (sat_id, target_id) in target_edges:
                    quality = target_edges[(sat_id, target_id)]['q']
                    observers.append((sat_id, quality))
            
            if observers:
                # 选择最佳观测者
                observers.sort(key=lambda x: x[1], reverse=True)
                essential.add(observers[0][0])
        
        # 确保连通性
        # 使用最小生成树算法找到连接所有必要卫星的最小集合
        if len(essential) > 1:
            connected_sats = self._find_minimum_connected_set(
                essential, cluster['satellites'], sat_edges
            )
            essential.update(connected_sats)
        
        return essential

    def _find_minimum_connected_set(
        self, required_sats: Set[int], available_sats: Set[int], sat_edges: Dict
    ) -> Set[int]:
        """
        找到连接所有必要卫星的最小卫星集合
        """
        # 使用Steiner树近似算法
        G = nx.Graph()
        
        # 构建图
        for sat1 in available_sats:
            for sat2 in available_sats:
                if sat1 != sat2 and (sat1, sat2) in sat_edges:
                    weight = 1.0 / sat_edges[(sat1, sat2)]['w']  # 权重取倒数
                    G.add_edge(sat1, sat2, weight=weight)
        
        if not G.has_node(list(required_sats)[0]):
            return set()
        
        # 找到连接所有必要节点的近似最小树
        steiner_nodes = set()
        
        # 简化实现：找到所有必要节点之间的最短路径
        required_list = list(required_sats)
        for i in range(len(required_list)):
            for j in range(i + 1, len(required_list)):
                try:
                    path = nx.shortest_path(G, required_list[i], required_list[j])
                    steiner_nodes.update(path)
                except nx.NetworkXNoPath:
                    pass
        
        return steiner_nodes - required_sats

    def _create_backup_clusters(
        self, clusters: List[Dict], available_satellites: List[Tuple[int, float]], 
        satellites: Dict, sat_edges: Dict, target_edges: Dict
    ):
        """
        创建备份簇以提高系统冗余度
        """
        if len(available_satellites) < 3:  # 至少需要3个卫星才考虑创建备份簇
            return
        
        # 识别关键目标（被多个簇观测的目标）
        target_coverage = defaultdict(int)
        for cluster in clusters:
            for target_id in cluster['targets']:
                target_coverage[target_id] += 1
        
        # 为覆盖不足的目标创建备份
        for target_id, coverage in target_coverage.items():
            if coverage < 2:  # 覆盖不足
                # 找到能观测该目标的可用卫星
                backup_candidates = []
                for sat_id, health in available_satellites:
                    if (sat_id, target_id) in target_edges:
                        quality = target_edges[(sat_id, target_id)]['q']
                        backup_candidates.append((sat_id, quality * health))
                
                if backup_candidates:
                    backup_candidates.sort(key=lambda x: x[1], reverse=True)
                    # 创建小型备份簇
                    backup_cluster = {
                        'id': len(clusters),
                        'master': backup_candidates[0][0],
                        'satellites': {backup_candidates[0][0]},
                        'targets': {target_id}
                    }
                    clusters.append(backup_cluster)

    def _calculate_observation_efficiency(
        self, satellites: Set[int], targets: Set[int], target_edges: Dict
    ) -> float:
        """
        计算观测效率
        """
        total_quality = 0
        observation_count = 0
        
        for sat_id in satellites:
            for target_id in targets:
                if (sat_id, target_id) in target_edges:
                    total_quality += target_edges[(sat_id, target_id)]['q']
                    observation_count += 1
        
        # 效率 = 平均观测质量 * 覆盖率
        avg_quality = total_quality / max(observation_count, 1)
        coverage_rate = observation_count / max(len(satellites) * len(targets), 1)
        
        return avg_quality * coverage_rate

    def _evaluate_topology_quality(self, satellites: Set[int], sat_edges: Dict) -> float:
        """
        评估拓扑结构质量
        """
        if len(satellites) <= 1:
            return 1.0
        
        # 构建子图
        G = nx.Graph()
        for sat1 in satellites:
            for sat2 in satellites:
                if sat1 != sat2 and (sat1, sat2) in sat_edges:
                    G.add_edge(sat1, sat2, weight=sat_edges[(sat1, sat2)]['w'])
        
        # 确保所有节点都在图中
        for sat in satellites:
            if sat not in G:
                G.add_node(sat)
        
        # 评估指标
        scores = {}
        
        # 连通性
        if nx.is_connected(G):
            scores['connectivity'] = 1.0
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            scores['connectivity'] = len(largest_cc) / len(satellites)
        
        # 平均度
        if G.number_of_edges() > 0:
            avg_degree = sum(dict(G.degree()).values()) / len(satellites)
            scores['avg_degree'] = min(avg_degree / 3.0, 1.0)  # 归一化
        else:
            scores['avg_degree'] = 0
        
        # 聚类系数
        scores['clustering'] = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
        
        return scores['connectivity'] * 0.5 + scores['avg_degree'] * 0.3 + scores['clustering'] * 0.2

    def _calculate_topology_improvement(
        self, new_sat: int, existing_sats: Set[int], sat_edges: Dict
    ) -> float:
        """
        计算添加新卫星对拓扑的改善
        """
        # 计算新增的连接数
        new_connections = 0
        for sat in existing_sats:
            if (new_sat, sat) in sat_edges:
                new_connections += 1
        
        # 评估是否增加了冗余路径
        redundancy_score = min(new_connections / 3.0, 1.0)
        
        return redundancy_score

    def _find_connected_satellites(
        self, start_sat: int, sat_edges: Dict, max_hops: int
    ) -> Set[int]:
        """
        找到指定跳数内的所有连接卫星
        """
        connected = set()
        visited = {start_sat}
        current_level = {start_sat}
        
        for hop in range(max_hops):
            next_level = set()
            for sat in current_level:
                for (s1, s2), _ in sat_edges.items():
                    if s1 == sat and s2 not in visited:
                        next_level.add(s2)
                        visited.add(s2)
                        connected.add(s2)
                    elif s2 == sat and s1 not in visited:
                        next_level.add(s1)
                        visited.add(s1)
                        connected.add(s1)
            
            current_level = next_level
            if not current_level:
                break
        
        return connected

    def _select_cluster_master(
        self, satellites: Set[int], sat_edges: Dict, sat_info: Dict
    ) -> int:
        """
        选择最佳主节点 - 改进版
        """
        if len(satellites) == 1:
            return list(satellites)[0]
        
        best_master = None
        best_score = -1
        
        for sat_id in satellites:
            score = 0
            
            # 1. 连接度评分
            connection_score = 0
            for other_sat in satellites:
                if other_sat != sat_id and (sat_id, other_sat) in sat_edges:
                    connection_score += sat_edges[(sat_id, other_sat)]['w']
            score += connection_score * 0.5
            
            # 2. 中心性评分（到其他节点的平均跳数）
            total_hops = 0
            reachable_count = 0
            for other_sat in satellites:
                if other_sat != sat_id:
                    hops = self._calculate_hop_distance(sat_id, other_sat, sat_edges)
                    if hops != float('inf'):
                        total_hops += hops
                        reachable_count += 1
            
            if reachable_count > 0:
                avg_hops = total_hops / reachable_count
                centrality_score = 1.0 / (1.0 + avg_hops)  # 平均跳数越小越好
                score += centrality_score * 0.3
            
            # 3. 健康度评分
            health = sat_info.get(sat_id, {}).get('health', 1.0)
            score += health * 0.2
            
            if score > best_score:
                best_score = score
                best_master = sat_id
        
        return best_master if best_master is not None else list(satellites)[0]

    def _perform_cluster_merge(
        self, cluster1: Dict, cluster2: Dict, satellites: Dict, sat_edges: Dict
    ) -> Dict:
        """
        执行簇合并
        """
        merged_satellites = cluster1['satellites'] | cluster2['satellites']
        merged_targets = cluster1['targets'] | cluster2['targets']
        
        # 选择新的主节点
        new_master = self._select_cluster_master(merged_satellites, sat_edges, satellites)
        
        return {
            'id': min(cluster1['id'], cluster2['id']),
            'master': new_master,
            'satellites': merged_satellites,
            'targets': merged_targets
        }

    def _print_clusters_summary(self, clusters: List[Dict]):
        """
        打印簇摘要信息
        """
        for cluster in clusters:
            print(f"  簇 {cluster['id']}: 卫星数={len(cluster['satellites'])}, "
                  f"目标数={len(cluster['targets'])}, 主节点={cluster['master']}")

    def _convert_to_cluster_info(
        self, clusters: List[Dict], satellites: Dict, sat_edges: Dict, target_edges: Dict
    ) -> List[ClusterInfo]:
        """
        转换为ClusterInfo格式并计算额外指标
        """
        cluster_infos = []
        
        for cluster in clusters:
            # 计算簇内平均链路强度
            link_count = 0
            total_strength = 0
            for sat1 in cluster['satellites']:
                for sat2 in cluster['satellites']:
                    if sat1 != sat2 and (sat1, sat2) in sat_edges:
                        total_strength += sat_edges[(sat1, sat2)]['w']
                        link_count += 1
            
            avg_link_strength = total_strength / link_count if link_count > 0 else 0
            
            # 计算总观测质量
            total_obs_quality = 0
            for sat_id in cluster['satellites']:
                for target_id in cluster['targets']:
                    if (sat_id, target_id) in target_edges:
                        total_obs_quality += target_edges[(sat_id, target_id)]['q']
            
            # 计算平均健康度
            total_health = sum(satellites.get(sat_id, {}).get('health', 1.0) 
                             for sat_id in cluster['satellites'])
            avg_health = total_health / len(cluster['satellites']) if cluster['satellites'] else 1.0
            
            cluster_info = ClusterInfo(
                cluster_id=cluster['id'],
                master=cluster['master'],
                sats=sorted(list(cluster['satellites'])),
                targets=sorted(list(cluster['targets'])),
                avg_intra_link_strength=avg_link_strength,
                total_observation_quality=total_obs_quality,
                avg_health=avg_health
            )
            
            cluster_infos.append(cluster_info)
        
        return sorted(cluster_infos, key=lambda x: x.cluster_id)

    # [保留原有的辅助方法，如 _parse_time_slice, _normalize_id, _calculate_hop_distance 等]
    
    def _parse_time_slice(self, time_slice: Dict) -> Tuple[Dict, Dict, Dict]:
        """解析时间切片数据"""
        satellites = {}
        sat_edges = {}
        target_edges = {}
        
        # 解析卫星信息
        for sat in time_slice.get("satellites", []):
            sat_id = self._normalize_id(sat["id"])
            satellites[sat_id] = {
                'health': sat.get('health', 1.0),
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

    def _calculate_hop_distance(self, sat1: int, sat2: int, sat_edges: Dict) -> int:
        """计算两个卫星之间的跳数距离"""
        if sat1 == sat2:
            return 0
        
        # 构建图
        graph = defaultdict(list)
        for (s1, s2), edge_info in sat_edges.items():
            if edge_info['w'] > 0:
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
        
        return float('inf')

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

    def _calculate_intra_cluster_link_strength(self, satellites: Set[int], sat_edges: Dict) -> float:
        """计算簇内链路强度"""
        total_strength = 0
        for sat1 in satellites:
            for sat2 in satellites:
                if sat1 != sat2 and (sat1, sat2) in sat_edges:
                    total_strength += sat_edges[(sat1, sat2)]['w']
        return total_strength

    def calculate_overall_score(
        self, cluster_infos: List[ClusterInfo], satellites: Dict, 
        sat_edges: Dict, target_edges: Dict
    ) -> Dict[str, float]:
        """
        计算整体优化分数 - 返回详细评分
        """
        scores = {}
        
        # 1. 簇内链路强度
        total_link_strength = sum(info.avg_intra_link_strength * len(info.sats) 
                                 for info in cluster_infos)
        scores['link_strength'] = total_link_strength / max(len(sat_edges), 1)
        
        # 2. 观测质量
        total_observation_quality = sum(info.total_observation_quality for info in cluster_infos)
        scores['observation_quality'] = total_observation_quality / max(len(target_edges), 1)
        
        # 3. 簇数量（归一化）
        scores['cluster_efficiency'] = 1.0 / max(len(cluster_infos), 1)
        
        # 4. 健康度
        total_health = sum(info.avg_health * len(info.sats) for info in cluster_infos)
        scores['health'] = total_health / max(len(satellites), 1)
        
        # 5. 目标覆盖率
        all_targets = set(t for _, t in target_edges.keys())
        covered_targets = set()
        for info in cluster_infos:
            covered_targets.update(info.targets)
        scores['target_coverage'] = len(covered_targets) / max(len(all_targets), 1)
        
        # 6. 负载均衡
        cluster_sizes = [len(info.sats) for info in cluster_infos]
        if cluster_sizes:
            avg_size = sum(cluster_sizes) / len(cluster_sizes)
            variance = sum((size - avg_size) ** 2 for size in cluster_sizes) / len(cluster_sizes)
            scores['load_balance'] = 1.0 / (1.0 + variance)
        else:
            scores['load_balance'] = 0
        
        # 综合分数
        scores['overall'] = (
            self.weight_link_strength * scores['link_strength'] +
            self.weight_observation_quality * scores['observation_quality'] +
            self.weight_cluster_count * scores['cluster_efficiency'] +
            self.weight_satellite_health * scores['health']
        )
        
        return scores

    def visualize_clusters_with_map(self, time_slice: Dict, cluster_infos: List[ClusterInfo]):
        """
        在地图上可视化分簇结果 - 集成地图背景
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            
            # 解析数据
            satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # 左图：3D可视化
            ax1.remove()
            ax1 = fig.add_subplot(121, projection='3d')
            self._plot_3d_clusters(ax1, time_slice, cluster_infos)
            
            # 右图：2D地图可视化（如果有地理坐标转换功能）
            try:
                self._plot_map_clusters(ax2, time_slice, cluster_infos)
            except Exception as e:
                print(f"地图可视化失败，使用拓扑图代替: {e}")
                self._plot_topology(ax2, time_slice, cluster_infos)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib不可用，跳过可视化")
        except Exception as e:
            print(f"可视化过程中出现错误: {e}")

    def _plot_map_clusters(self, ax, time_slice: Dict, cluster_infos: List[ClusterInfo]):
        """
        在地图上绘制分簇结果（需要地理坐标转换）
        """
        # 这里需要实现地理坐标转换
        # 如果没有转换功能，将使用简化的平面投影
        
        satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
        
        # 简化投影：直接使用XY坐标
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_infos)))
        
        for i, cluster_info in enumerate(cluster_infos):
            cluster_sats = []
            cluster_targets = []
            
            # 收集簇内卫星位置
            for sat_id in cluster_info.sats:
                if sat_id in satellites:
                    pos = satellites[sat_id]['position']
                    cluster_sats.append((pos[0], pos[1]))  # 使用X,Y坐标
            
            # 收集簇观测的目标位置
            for (sat_id, target_id), _ in target_edges.items():
                if sat_id in cluster_info.sats and target_id in cluster_info.targets:
                    # 从target_edges中获取目标位置（需要从time_slice中查找）
                    for obs in time_slice.get('target_visibility', []):
                        if (obs.get('from_satellite', {}).get('id') == sat_id and 
                            obs.get('to_target', {}).get('id') == target_id):
                            target_pos = obs.get('to_target', {}).get('position', [])
                            if len(target_pos) >= 2:
                                cluster_targets.append((target_pos[0], target_pos[1]))
                            break
            
            # 绘制簇内卫星
            if cluster_sats:
                sat_x, sat_y = zip(*cluster_sats)
                ax.scatter(sat_x, sat_y, c=[colors[i]], s=100, alpha=0.8, 
                          label=f'Cluster {cluster_info.cluster_id} Sats', marker='o')
                
                # 标记主节点
                master_pos = None
                if cluster_info.master in satellites:
                    master_pos = satellites[cluster_info.master]['position']
                    ax.scatter(master_pos[0], master_pos[1], c='red', s=200, 
                              marker='^', edgecolors='black', linewidth=2)
            
            # 绘制簇观测的目标
            if cluster_targets:
                target_x, target_y = zip(*cluster_targets)
                ax.scatter(target_x, target_y, c=[colors[i]], s=150, alpha=0.9,
                          marker='*', edgecolors='black', linewidth=1)
        
        # 绘制簇内连接
        for i, cluster_info in enumerate(cluster_infos):
            for sat1 in cluster_info.sats:
                for sat2 in cluster_info.sats:
                    if (sat1 != sat2 and (sat1, sat2) in sat_edges and 
                        sat1 in satellites and sat2 in satellites):
                        pos1 = satellites[sat1]['position']
                        pos2 = satellites[sat2]['position']
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                               color=colors[i], alpha=0.5, linewidth=1)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Satellite Clusters Map View ({self.strategy.value} mode)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def visualize_clusters(self, time_slice: Dict, cluster_infos: List[ClusterInfo]):
        """增强的可视化功能 - 兼容原有接口"""
        fig = plt.figure(figsize=(20, 15))
        
        # 3D视图
        ax1 = fig.add_subplot(221, projection='3d')
        self._plot_3d_clusters(ax1, time_slice, cluster_infos)
        
        # 拓扑视图
        ax2 = fig.add_subplot(222)
        self._plot_topology(ax2, time_slice, cluster_infos)
        
        # 性能指标
        ax3 = fig.add_subplot(223)
        self._plot_performance_metrics(ax3, time_slice, cluster_infos)
        
        # 簇详情
        ax4 = fig.add_subplot(224)
        self._plot_cluster_details(ax4, cluster_infos)
        
        plt.tight_layout()
        plt.show()

    def _plot_3d_clusters(self, ax, time_slice: Dict, cluster_infos: List[ClusterInfo]):
        """3D簇可视化"""
        satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
        
        # 颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_infos)))
        
        # 绘制卫星
        for i, cluster_info in enumerate(cluster_infos):
            cluster_positions = []
            
            for sat_id in cluster_info.sats:
                if sat_id in satellites:
                    pos = satellites[sat_id]['position']
                    cluster_positions.append(pos)
                    
                    # 标记主节点
                    if sat_id == cluster_info.master:
                        ax.scatter(pos[0], pos[1], pos[2], c='red', marker='^', 
                                 s=200, edgecolors='black', linewidth=2)
            
            if cluster_positions:
                cluster_positions = np.array(cluster_positions)
                ax.scatter(cluster_positions[:, 0], cluster_positions[:, 1], 
                         cluster_positions[:, 2], c=[colors[i]], s=100, alpha=0.8,
                         label=f'Cluster {cluster_info.cluster_id}')
        
        # 绘制连接
        for (sat1, sat2), edge_info in sat_edges.items():
            if sat1 in satellites and sat2 in satellites:
                # 只绘制簇内连接
                cluster1 = cluster2 = None
                for info in cluster_infos:
                    if sat1 in info.sats:
                        cluster1 = info.cluster_id
                    if sat2 in info.sats:
                        cluster2 = info.cluster_id
                
                if cluster1 == cluster2 and cluster1 is not None:
                    pos1 = satellites[sat1]['position']
                    pos2 = satellites[sat2]['position']
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                           'gray', alpha=0.5, linewidth=edge_info['w'] * 2)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")
        ax.set_title(f"3D Satellite Clustering ({self.strategy.value} mode)")
        ax.legend()

    def _plot_topology(self, ax, time_slice: Dict, cluster_infos: List[ClusterInfo]):
        """拓扑结构可视化"""
        satellites, sat_edges, _ = self._parse_time_slice(time_slice)
        
        G = nx.Graph()
        pos = {}
        node_colors = []
        node_sizes = []
        
        # 构建图
        for i, cluster_info in enumerate(cluster_infos):
            color = plt.cm.tab10(i / len(cluster_infos))
            
            for sat_id in cluster_info.sats:
                G.add_node(sat_id)
                node_colors.append(color)
                node_sizes.append(200 if sat_id == cluster_info.master else 100)
        
        # 添加边
        for (sat1, sat2), edge_info in sat_edges.items():
            if G.has_node(sat1) and G.has_node(sat2):
                G.add_edge(sat1, sat2, weight=edge_info['w'])
        
        # 布局
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 绘制
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=node_sizes, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title("Cluster Topology")
        ax.axis('off')

    def _plot_performance_metrics(self, ax, time_slice: Dict, cluster_infos: List[ClusterInfo]):
        """性能指标可视化"""
        satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
        scores = self.calculate_overall_score(cluster_infos, satellites, sat_edges, target_edges)
        
        metrics = list(scores.keys())
        values = list(scores.values())
        
        # 移除'overall'用于单独显示
        if 'overall' in metrics:
            overall_idx = metrics.index('overall')
            overall_score = values.pop(overall_idx)
            metrics.pop(overall_idx)
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        values = np.array(values)
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Score')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Metrics\nOverall Score: {overall_score:.3f}')
        ax.grid(True)

    def _plot_cluster_details(self, ax, cluster_infos: List[ClusterInfo]):
        """簇详情表格"""
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        headers = ['Cluster', 'Master', 'Sats', 'Targets', 'Link Str', 'Obs Qual']
        cell_data = []
        
        for info in cluster_infos:
            row = [
                info.cluster_id,
                info.master,
                len(info.sats),
                len(info.targets),
                f'{info.avg_intra_link_strength:.2f}',
                f'{info.total_observation_quality:.2f}'
            ]
            cell_data.append(row)
        
        table = ax.table(cellText=cell_data, colLabels=headers,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax.set_title('Cluster Details')

    def export_results(self, clustering_results: List[List[ClusterInfo]], 
                      time_series_data: List[Dict], filename: str = None):
        """导出详细结果"""
        if filename is None:
            filename = f"clustering_results_{self.strategy.value}_enhanced.json"
        
        export_data = []
        
        for i, (cluster_infos, time_slice) in enumerate(
            zip(clustering_results, time_series_data)
        ):
            timestamp = time_slice.get("timestamp", i)
            
            # 计算详细评分
            satellites, sat_edges, target_edges = self._parse_time_slice(time_slice)
            scores = self.calculate_overall_score(cluster_infos, satellites, sat_edges, target_edges)
            
            time_slice_data = {
                "timestamp": timestamp,
                "strategy": self.strategy.value,
                "scores": scores,
                "cluster_count": len(cluster_infos),
                "total_satellites": len(satellites),
                "total_targets": len(set(t for _, t in target_edges.keys())),
                "clusters": [cluster_info.dict() for cluster_info in cluster_infos],
                "statistics": {
                    "avg_cluster_size": sum(len(c.sats) for c in cluster_infos) / len(cluster_infos) if cluster_infos else 0,
                    "max_cluster_size": max(len(c.sats) for c in cluster_infos) if cluster_infos else 0,
                    "min_cluster_size": min(len(c.sats) for c in cluster_infos) if cluster_infos else 0,
                }
            }
            
            export_data.append(time_slice_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {filename}")
        
        # 生成摘要报告
        self._generate_summary_report(export_data, filename.replace('.json', '_summary.txt'))

    def _generate_summary_report(self, export_data: List[Dict], filename: str):
        """生成文本摘要报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"卫星动态分簇系统报告\n")
            f.write(f"策略: {self.strategy.value}\n")
            f.write(f"=" * 50 + "\n\n")
            
            for data in export_data:
                f.write(f"时间戳: {data['timestamp']}\n")
                f.write(f"簇数量: {data['cluster_count']}\n")
                f.write(f"卫星总数: {data['total_satellites']}\n")
                f.write(f"目标总数: {data['total_targets']}\n")
                f.write(f"综合评分: {data['scores']['overall']:.4f}\n")
                f.write(f"目标覆盖率: {data['scores']['target_coverage']:.2%}\n")
                f.write(f"平均簇大小: {data['statistics']['avg_cluster_size']:.1f}\n")
                f.write("-" * 30 + "\n")
            
            f.write(f"\n分析完成！\n")
        
        print(f"Summary report generated: {filename}")


def load_data(file_path: Path) -> List[dict]:
    """
    按时间切片加载数据，将同一时间戳下的所有卫星数据聚合在一起

    Args:
        file_path: 数据文件路径

    Returns:
        List[dict]: 每个元素是一个时间切片的字典，包含：
        - timestamp: 时间戳
        - time_offset_from_scenario_start: 相对场景开始时间的偏移
        - satellites: 该时间戳下所有卫星的信息列表
        - inter_satellite_connectivity: 该时间戳下所有卫星间连接关系
        - target_visibility: 该时间戳下所有卫星-目标观测关系
    """
    # 加载原始数据
    data = json.loads(file_path.read_text())

    # 按时间戳分组数据
    timestamp_groups = defaultdict(list)
    for record in data:
        timestamp = record['timestamp']
        timestamp_groups[timestamp].append(record)

    # 构建时间切片数据
    time_slices = []
    for timestamp, records in sorted(timestamp_groups.items()):
        # 聚合同一时间戳下的所有数据
        satellites = []
        all_inter_connectivity = []
        all_target_visibility = []
        time_offset = None

        for record in records:
            # 收集卫星信息
            satellite_info = record.get('satellite_info', {})
            if satellite_info:
                satellites.append(satellite_info)

            # 收集卫星间连接关系
            inter_connectivity = record.get('inter_satellite_connectivity', [])
            for conn in inter_connectivity:
                # 构建新的连接数据结构
                enhanced_conn = {
                    'from_satellite': {
                        'id': satellite_info.get('id'),
                        'position': satellite_info.get('position')
                    },
                    'to_satellite': {
                        'id': conn.get('to_satellite_id'),
                        'position': conn.get('position')
                    },
                    'connection_quality': conn.get('connection_quality'),
                    'visibility_time_window': conn.get('visibility_time_window')
                }
                all_inter_connectivity.append(enhanced_conn)

            # 收集目标观测关系
            target_visibility = record.get('target_visibility', [])
            for target in target_visibility:
                # 构建新的目标观测数据结构
                enhanced_target = {
                    'from_satellite': {
                        'id': satellite_info.get('id'),
                        'position': satellite_info.get('position')
                    },
                    'to_target': {
                        'id': target.get('target_id'),
                        'position': target.get('position')
                    },
                    'target_value': target.get('target_value'),
                    'observation_priority': target.get('observation_priority'),
                    'visibility_time_window': target.get('visibility_time_window')
                }
                all_target_visibility.append(enhanced_target)

            # 获取时间偏移（所有记录应该相同）
            if time_offset is None:
                time_offset = record.get('time_offset_from_scenario_start')

        # 构建时间切片数据
        slice_data = {
            'timestamp': timestamp,
            'time_offset_from_scenario_start': time_offset,
            'satellites': satellites,
            'inter_satellite_connectivity': all_inter_connectivity,
            'target_visibility': all_target_visibility
        }

        time_slices.append(slice_data)

    return time_slices


def create_global_target_colors(time_slices: List[dict]) -> dict:
    """
    为所有时间切片中出现的目标创建全局颜色映射，确保同一目标在不同时间切片中使用相同颜色

    Args:
        time_slices: 所有时间切片数据

    Returns:
        dict: 目标ID到颜色的映射字典
    """
    # 现代化配色方案 - 使用更柔和、更专业的颜色
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
              '#7209B7', '#F72585', '#4361EE', '#F77F00', '#FCBF49',
              '#06FFA5', '#FB8500', '#8ECAE6', '#219EBC', '#023047',
              '#FFB3C6', '#FB8500', '#8B5CF6', '#10B981', '#F59E0B']

    # 收集所有目标ID
    all_targets = set()
    for slice_data in time_slices:
        for target_obs in slice_data.get('target_visibility', []):
            to_target = target_obs.get('to_target', {})
            target_id = to_target.get('id')
            if target_id:
                all_targets.add(target_id)

    # 为每个目标分配固定颜色
    target_colors = {}
    for i, target_id in enumerate(sorted(all_targets)):  # 使用sorted确保顺序一致
        target_colors[target_id] = colors[i % len(colors)]

    return target_colors


def get_time_slice_summary(time_slices: List[dict]) -> dict:
    """
    获取时间切片数据的摘要信息

    Args:
        time_slices: 时间切片数据列表

    Returns:
        dict: 包含摘要信息的字典
    """
    if not time_slices:
        return {}

    summary = {
        'total_time_slices': len(time_slices),
        'time_range': [],
        'unique_satellites': set(),
        'unique_targets': set(),
        'connectivity_stats': {
            'total_connections': 0,
            'avg_connections_per_slice': 0
        },
        'target_stats': {
            'total_observations': 0,
            'avg_observations_per_slice': 0
        }
    }

    # 统计信息
    total_connections = 0
    total_observations = 0

    for slice_data in time_slices:
        # 时间范围
        time_offset = slice_data['time_offset_from_scenario_start']
        if not summary['time_range']:
            summary['time_range'] = [time_offset, time_offset]
        else:
            summary['time_range'][0] = min(summary['time_range'][0], time_offset)
            summary['time_range'][1] = max(summary['time_range'][1], time_offset)

        # 卫星信息
        satellites = slice_data['satellites']
        for satellite_info in satellites:
            if 'id' in satellite_info:
                summary['unique_satellites'].add(satellite_info['id'])

        # 连接统计
        connections = slice_data['inter_satellite_connectivity']
        total_connections += len(connections)
        for conn in connections:
            # 处理新的数据结构
            if 'from_satellite' in conn and conn['from_satellite'].get('id'):
                summary['unique_satellites'].add(conn['from_satellite']['id'])
            if 'to_satellite' in conn and conn['to_satellite'].get('id'):
                summary['unique_satellites'].add(conn['to_satellite']['id'])

        # 目标观测统计
        observations = slice_data['target_visibility']
        total_observations += len(observations)
        for obs in observations:
            # 处理新的数据结构
            if 'to_target' in obs and obs['to_target'].get('id'):
                summary['unique_targets'].add(obs['to_target']['id'])
            if 'from_satellite' in obs and obs['from_satellite'].get('id'):
                summary['unique_satellites'].add(obs['from_satellite']['id'])

            # 兼容旧的数据结构
            if 'target_id' in obs:
                summary['unique_targets'].add(obs['target_id'])
            if 'satellite_id' in obs:
                summary['unique_satellites'].add(obs['satellite_id'])

    # 计算平均值
    summary['connectivity_stats']['total_connections'] = total_connections
    summary['connectivity_stats']['avg_connections_per_slice'] = total_connections / len(time_slices)

    summary['target_stats']['total_observations'] = total_observations
    summary['target_stats']['avg_observations_per_slice'] = total_observations / len(time_slices)

    # 转换set为list以便JSON序列化
    summary['unique_satellites'] = list(summary['unique_satellites'])
    summary['unique_targets'] = list(summary['unique_targets'])

    return summary


# 模拟工具函数（如果实际环境中不存在）
def get_data_dir():
    """返回数据目录路径"""
    return Path("./data")  # 请根据实际路径修改

def get_documents_dir():
    """返回文档目录路径"""
    return Path("./documents")  # 请根据实际路径修改

def get_project_root():
    """返回项目根目录路径"""
    return Path(".")  # 请根据实际路径修改


# 示例使用
if __name__ == "__main__":
    # 加载数据
    data_file = get_data_dir() / "satellite_target_visibility_data.json"

    if not data_file.exists():
        print("数据文件不存在，请检查路径...")
        print(f"期望路径: {data_file.absolute()}")
        exit()

    print("正在加载数据...")
    time_slices = load_data(data_file)
    print(f"成功加载 {len(time_slices)} 个时间切片")

    # 获取数据摘要
    print("\n正在分析数据摘要...")
    summary = get_time_slice_summary(time_slices)
    print("\n=== 数据摘要 ===")
    print(f"- 总时间切片数: {summary['total_time_slices']}")
    print(f"- 时间范围: {summary['time_range'][0]:.1f}s - {summary['time_range'][1]:.1f}s")
    print(f"- 唯一卫星数: {len(summary['unique_satellites'])}")
    print(f"- 唯一目标数: {len(summary['unique_targets'])}")
    print(f"- 总连接数: {summary['connectivity_stats']['total_connections']}")
    print(f"- 平均每切片连接数: {summary['connectivity_stats']['avg_connections_per_slice']:.2f}")
    print(f"- 总观测数: {summary['target_stats']['total_observations']}")
    print(f"- 平均每切片观测数: {summary['target_stats']['avg_observations_per_slice']:.2f}")

    # 显示前几个时间切片的详细信息
    print("\n=== 前3个时间切片详情 ===")
    for i, slice_data in enumerate(time_slices[:3]):
        print(f"\n时间切片 {i+1}:")
        print(f"  时间戳: {slice_data['timestamp']}")
        print(f"  时间偏移: {slice_data['time_offset_from_scenario_start']}秒")
        
        # 显示卫星信息
        satellites = slice_data['satellites']
        satellite_ids = [sat.get('id', 'N/A') for sat in satellites]
        print(f"  卫星数量: {len(satellites)}")
        print(f"  卫星ID: {satellite_ids[:5]}{'...' if len(satellite_ids) > 5 else ''}")
        
        print(f"  卫星间连接数: {len(slice_data['inter_satellite_connectivity'])}")
        print(f"  目标观测数: {len(slice_data['target_visibility'])}")
        
        # 显示目标信息
        if slice_data['target_visibility']:
            targets = [obs.get('to_target', {}).get('id', obs.get('target_id', 'Unknown'))
                      for obs in slice_data['target_visibility']]
            unique_targets = list(set(targets))
            print(f"  观测目标: {unique_targets[:3]}{'...' if len(unique_targets) > 3 else ''}")

    # 测试两种策略
    print("\n=== 开始动态分簇测试 ===")
    
    # 选择用于测试的时间切片（可以选择不同的切片进行测试）
    test_slice_indices = [0, min(2, len(time_slices)-1), min(4, len(time_slices)-1)]
    test_slice_indices = [i for i in test_slice_indices if i < len(time_slices)]
    
    for strategy in [ClusteringStrategy.BALANCED, ClusteringStrategy.QUALITY]:
        print(f"\n{'='*50}")
        print(f"测试 {strategy.value.upper()} 策略")
        print(f"{'='*50}")
        
        # 创建聚类系统
        clustering_system = SatelliteClusteringSystem(strategy=strategy)
        
        strategy_results = []
        
        for slice_idx in test_slice_indices:
            print(f"\n--- 处理时间切片 {slice_idx + 1} ---")
            
            # 对时间切片进行聚类
            cluster_infos = clustering_system.cluster_time_slice(time_slices[slice_idx])
            
            if not cluster_infos:
                print("  警告：未能生成任何簇")
                continue
            
            # 打印结果摘要
            print(f"\n  生成 {len(cluster_infos)} 个簇：")
            total_sats = sum(len(info.sats) for info in cluster_infos)
            total_targets = sum(len(info.targets) for info in cluster_infos)
            
            for cluster_info in cluster_infos:
                print(f"    簇 {cluster_info.cluster_id}: "
                      f"主节点={cluster_info.master}, "
                      f"卫星数={len(cluster_info.sats)}, "
                      f"目标数={len(cluster_info.targets)}, "
                      f"链路强度={cluster_info.avg_intra_link_strength:.3f}")
            
            # 计算详细评分
            satellites, sat_edges, target_edges = clustering_system._parse_time_slice(time_slices[slice_idx])
            scores = clustering_system.calculate_overall_score(cluster_infos, satellites, sat_edges, target_edges)
            
            # 打印详细评分
            print(f"\n  性能评估:")
            print(f"    综合分数: {scores['overall']:.4f}")
            print(f"    链路强度: {scores['link_strength']:.4f}")
            print(f"    观测质量: {scores['observation_quality']:.4f}")
            print(f"    簇效率: {scores['cluster_efficiency']:.4f}")
            print(f"    健康度: {scores['health']:.4f}")
            print(f"    目标覆盖率: {scores['target_coverage']:.2%}")
            print(f"    负载均衡: {scores['load_balance']:.4f}")
            
            # 验证约束条件
            print(f"\n  约束验证:")
            all_cluster_sats = set()
            for info in cluster_infos:
                all_cluster_sats.update(info.sats)
            
            all_targets_in_data = set(t for _, t in target_edges.keys())
            all_targets_in_clusters = set()
            for info in cluster_infos:
                all_targets_in_clusters.update(info.targets)
            
            print(f"    卫星利用率: {len(all_cluster_sats)}/{len(satellites)} ({len(all_cluster_sats)/len(satellites):.1%})")
            print(f"    目标覆盖: {len(all_targets_in_clusters)}/{len(all_targets_in_data)} ({len(all_targets_in_clusters)/len(all_targets_in_data):.1%})")
            
            # 检查约束违反
            constraint_violations = []
            for info in cluster_infos:
                target_count = len(info.targets)
                sat_count = len(info.sats)
                
                if strategy == ClusteringStrategy.BALANCED:
                    if sat_count > target_count:
                        constraint_violations.append(f"簇{info.cluster_id}: 卫星数({sat_count}) > 目标数({target_count})")
                else:  # QUALITY
                    if sat_count > 2 * target_count:
                        constraint_violations.append(f"簇{info.cluster_id}: 卫星数({sat_count}) > 2*目标数({2*target_count})")
            
            if constraint_violations:
                print(f"    约束违反: {constraint_violations}")
            else:
                print(f"    约束检查: ✓ 通过")
            
            strategy_results.append((slice_idx, cluster_infos))
        
        # 可视化（仅对第一个时间切片）
        if strategy_results:
            print(f"\n--- 生成可视化图表 ---")
            try:
                slice_idx, cluster_infos = strategy_results[0]
                clustering_system.visualize_clusters(time_slices[slice_idx], cluster_infos)
                print("  ✓ 可视化生成成功")
            except Exception as e:
                print(f"  ✗ 可视化生成失败: {e}")
        
        # 导出结果
        if strategy_results:
            print(f"\n--- 导出结果 ---")
            try:
                results_for_export = [result[1] for result in strategy_results]
                slices_for_export = [time_slices[result[0]] for result in strategy_results]
                
                filename = f"clustering_results_{strategy.value}_enhanced.json"
                clustering_system.export_results(results_for_export, slices_for_export, filename)
                print(f"  ✓ 结果已导出到 {filename}")
            except Exception as e:
                print(f"  ✗ 结果导出失败: {e}")

    print(f"\n{'='*50}")
    print("动态分簇系统测试完成！")
    print(f"{'='*50}")
    
    # 生成对比报告
    print(f"\n=== 策略对比摘要 ===")
    print("建议：")
    print("- BALANCED策略：适合资源受限场景，每簇卫星数≤目标数")
    print("- QUALITY策略：适合高可靠性需求，每簇卫星数≤2×目标数，增加冗余")
    print("- 根据实际需求选择合适的策略和参数调整")