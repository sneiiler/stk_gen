import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt
import matplotlib
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import networkx as nx
from typing import List, Dict, Tuple
import pandas as pd

from matplotlib.font_manager import FontProperties
from tqdm import tqdm

from stk_server.Packages.Tools import ecef2lla

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import community as community_louvain  # python-louvain
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import List, Dict, Tuple, Optional
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

root_dir = Path(__file__).parent.parent
print(root_dir)
sys.path.append(str(root_dir))

from utils.misc_utils import get_data_dir, get_documents_dir, get_project_root


class GraphBasedSatelliteClusteringSystem:
    """
    基于图的卫星星座动态分簇系统
    
    核心思想：
    1. 将卫星网络建模为动态图，节点是卫星，边是通信链路
    2. 结合多种图聚类算法：谱聚类、Louvain社区检测、标签传播
    3. 构建多层图：卫星-卫星连接层 + 卫星-目标观测层
    4. 优化多个目标：网络连通性、观测覆盖、负载均衡
    """
    
    def __init__(self, method='spectral', n_clusters=5, 
                 alpha=0.6, beta=0.3, gamma=0.1, temporal_smoothing=0.8):
        """
        初始化图聚类系统
        
        Args:
            method: 聚类方法 {'spectral', 'louvain', 'leiden', 'label_propagation', 'hybrid'}
            n_clusters: 目标聚类数量（仅对spectral有效）
            alpha: 卫星连接权重
            beta: 观测任务权重  
            gamma: 位置距离权重
            temporal_smoothing: 时间平滑系数 [0,1]
        """
        self.method = method
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temporal_smoothing = temporal_smoothing
        
        # 历史记录
        self.graph_history = []
        self.cluster_history = []
        self.quality_history = []
        
    def build_satellite_graph(self, time_slice: Dict) -> nx.Graph:
        """
        构建卫星网络图
        
        图结构：
        - 节点：卫星（带属性：位置、观测能力等）
        - 边：卫星间连接（权重：连接质量、距离等）
        - 虚拟目标节点：用于建模观测约束
        """
        G = nx.Graph()
        
        # 1. 添加卫星节点
        satellites = self._extract_all_satellites(time_slice)
        for sat_id, sat_info in satellites.items():
            G.add_node(sat_id, 
                      node_type='satellite',
                      position=sat_info['position'],
                      observation_count=sat_info.get('observation_count', 0),
                      total_target_value=sat_info.get('total_target_value', 0))
        
        # 2. 添加卫星间连接边
        self._add_inter_satellite_edges(G, time_slice)
        
        # 3. 添加虚拟目标节点和观测边（用于约束）
        self._add_observation_constraints(G, time_slice)
        
        # 4. 添加基于距离的边（增强连通性）
        self._add_distance_based_edges(G, time_slice)
        
        return G
    
    def _extract_all_satellites(self, time_slice: Dict) -> Dict:
        """提取所有卫星信息"""
        satellites = {}
        
        # 从连接数据中提取
        for conn in time_slice.get('inter_satellite_connectivity', []):
            from_sat = conn['from_satellite']
            to_sat = conn['to_satellite']
            
            satellites[from_sat['id']] = {
                'position': from_sat['position'],
                'observation_count': 0,
                'total_target_value': 0
            }
            satellites[to_sat['id']] = {
                'position': to_sat['position'], 
                'observation_count': 0,
                'total_target_value': 0
            }
        
        # 从观测数据中补充信息
        for obs in time_slice.get('target_visibility', []):
            sat_id = obs['from_satellite']['id']
            if sat_id not in satellites:
                satellites[sat_id] = {
                    'position': obs['from_satellite']['position'],
                    'observation_count': 0,
                    'total_target_value': 0
                }
            
            satellites[sat_id]['observation_count'] += 1
            satellites[sat_id]['total_target_value'] += obs['target_value']
        
        # 从satellites列表中补充
        for sat in time_slice.get('satellites', []):
            if sat['id'] not in satellites:
                satellites[sat['id']] = {
                    'position': sat['position'],
                    'observation_count': 0,
                    'total_target_value': 0
                }
        
        return satellites
    
    def _add_inter_satellite_edges(self, G: nx.Graph, time_slice: Dict):
        """添加卫星间连接边"""
        for conn in time_slice.get('inter_satellite_connectivity', []):
            from_id = conn['from_satellite']['id']
            to_id = conn['to_satellite']['id']
            
            if from_id in G.nodes and to_id in G.nodes:
                # 计算边权重：连接质量 + 可见性时间窗口
                visibility_duration = (conn['visibility_time_window'][1] - 
                                     conn['visibility_time_window'][0])
                
                edge_weight = (self.alpha * conn['connection_quality'] + 
                             0.3 * min(visibility_duration / 3600, 1.0))  # 标准化到小时
                
                G.add_edge(from_id, to_id, 
                          weight=edge_weight,
                          edge_type='communication',
                          connection_quality=conn['connection_quality'],
                          visibility_duration=visibility_duration)
    
    def _add_observation_constraints(self, G: nx.Graph, time_slice: Dict):
        """添加观测约束（虚拟目标节点）"""
        # 为每个目标创建虚拟节点
        target_groups = defaultdict(list)
        
        for obs in time_slice.get('target_visibility', []):
            target_id = obs['to_target']['id']
            target_groups[target_id].append(obs)
        
        # 为高价值目标创建虚拟节点
        for target_id, observations in target_groups.items():
            total_value = sum(obs['target_value'] for obs in observations)
            avg_priority = np.mean([obs['observation_priority'] for obs in observations])
            
            # 只为高价值目标创建约束节点
            if total_value > 50 or avg_priority > 5:  # 阈值可调
                virtual_node_id = f"target_{target_id}"
                G.add_node(virtual_node_id,
                          node_type='virtual_target',
                          target_value=total_value,
                          avg_priority=avg_priority)
                
                # 连接观测该目标的卫星
                for obs in observations:
                    sat_id = obs['from_satellite']['id']
                    if sat_id in G.nodes:
                        constraint_weight = (self.beta * obs['target_value'] / 100 + 
                                           self.beta * obs['observation_priority'] / 10)
                        
                        G.add_edge(sat_id, virtual_node_id,
                                  weight=constraint_weight,
                                  edge_type='observation_constraint',
                                  target_value=obs['target_value'],
                                  priority=obs['observation_priority'])
    
    def _add_distance_based_edges(self, G: nx.Graph, time_slice: Dict):
        """基于距离添加补充边（增强连通性）"""
        satellite_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'satellite']
        
        # 计算距离矩阵
        positions = {}
        for sat_id in satellite_nodes:
            positions[sat_id] = np.array(G.nodes[sat_id]['position'])
        
        # 为距离较近的卫星添加弱连接
        for i, sat1 in enumerate(satellite_nodes):
            for sat2 in satellite_nodes[i+1:]:
                if not G.has_edge(sat1, sat2):  # 避免重复边
                    distance = np.linalg.norm(positions[sat1] - positions[sat2])
                    
                    # 距离阈值（可根据通信距离调整）
                    max_distance = 2000  # km
                    if distance < max_distance:
                        # 距离越近，权重越大
                        distance_weight = self.gamma * (1 - distance / max_distance)
                        
                        G.add_edge(sat1, sat2,
                                  weight=distance_weight,
                                  edge_type='distance_based',
                                  distance=distance)
    
    def cluster_satellites(self, time_slice: Dict) -> Dict[str, int]:
        """
        对卫星进行图聚类
        """
        # 构建图
        G = self.build_satellite_graph(time_slice)
        self.graph_history.append(G.copy())
        
        # 只对卫星节点进行聚类
        satellite_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'satellite']
        
        if len(satellite_nodes) < 2:
            return {sat: 0 for sat in satellite_nodes}
        
        # 创建卫星子图
        satellite_subgraph = G.subgraph(satellite_nodes).copy()
        
        # 根据方法选择聚类算法
        if self.method == 'spectral':
            clusters = self._spectral_clustering(satellite_subgraph)
        elif self.method == 'louvain':
            clusters = self._louvain_clustering(satellite_subgraph)
        elif self.method == 'label_propagation':
            clusters = self._label_propagation_clustering(satellite_subgraph)
        elif self.method == 'hybrid':
            clusters = self._hybrid_clustering(satellite_subgraph, G)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        return clusters
    
    def _spectral_clustering(self, G: nx.Graph) -> Dict[str, int]:
        """谱聚类实现"""
        if len(G.nodes) < self.n_clusters:
            return {node: i for i, node in enumerate(G.nodes)}
        
        # 构建邻接矩阵
        adjacency_matrix = nx.adjacency_matrix(G, weight='weight')
        
        # 应用谱聚类
        try:
            spectral = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='precomputed',
                random_state=42,
                n_init=10
            )
            labels = spectral.fit_predict(adjacency_matrix)
            
            return {node: labels[i] for i, node in enumerate(G.nodes)}
        except Exception as e:
            print(f"Spectral clustering failed: {e}")
            # 回退到简单分组
            return {node: i % self.n_clusters for i, node in enumerate(G.nodes)}
    
    def _louvain_clustering(self, G: nx.Graph) -> Dict[str, int]:
        """Louvain社区检测"""
        try:
            # 使用python-louvain库
            partition = community_louvain.best_partition(G, weight='weight', random_state=42)
            return partition
        except Exception as e:
            print(f"Louvain clustering failed: {e}")
            # 回退到连通分量
            components = list(nx.connected_components(G))
            clusters = {}
            for i, component in enumerate(components):
                for node in component:
                    clusters[node] = i
            return clusters
    
    def _label_propagation_clustering(self, G: nx.Graph) -> Dict[str, int]:
        """标签传播聚类"""
        try:
            communities = nx.algorithms.community.label_propagation_communities(G)
            clusters = {}
            for i, community in enumerate(communities):
                for node in community:
                    clusters[node] = i
            return clusters
        except Exception as e:
            print(f"Label propagation failed: {e}")
            return {node: 0 for node in G.nodes}
    
    def _hybrid_clustering(self, satellite_subgraph: nx.Graph, full_graph: nx.Graph) -> Dict[str, int]:
        """
        混合聚类方法：
        1. 首先用Louvain检测基础社区结构
        2. 然后用谱聚类优化边界
        3. 最后考虑观测约束进行调整
        """
        # 步骤1：Louvain社区检测
        base_clusters = self._louvain_clustering(satellite_subgraph)
        
        # 步骤2：如果社区数量过多，用谱聚类合并
        n_communities = len(set(base_clusters.values()))
        if n_communities > self.n_clusters:
            # 重新映射到目标聚类数
            spectral_clusters = self._spectral_clustering(satellite_subgraph)
            # 选择模块度更高的结果
            base_modularity = self._calculate_modularity(satellite_subgraph, base_clusters)
            spectral_modularity = self._calculate_modularity(satellite_subgraph, spectral_clusters)
            
            if spectral_modularity > base_modularity:
                base_clusters = spectral_clusters
        
        # 步骤3：基于观测约束调整
        adjusted_clusters = self._adjust_clusters_for_observations(full_graph, base_clusters)
        
        return adjusted_clusters
    
    def _calculate_modularity(self, G: nx.Graph, clusters: Dict[str, int]) -> float:
        """计算网络模块度"""
        try:
            # 创建社区列表
            communities = defaultdict(list)
            for node, cluster in clusters.items():
                communities[cluster].append(node)
            
            community_list = [set(nodes) for nodes in communities.values()]
            return nx.algorithms.community.modularity(G, community_list, weight='weight')
        except:
            return 0.0
    
    def _adjust_clusters_for_observations(self, full_graph: nx.Graph, 
                                        base_clusters: Dict[str, int]) -> Dict[str, int]:
        """基于观测约束调整聚类"""
        adjusted_clusters = base_clusters.copy()
        
        # 分析虚拟目标节点的连接
        target_nodes = [n for n in full_graph.nodes 
                       if full_graph.nodes[n]['node_type'] == 'virtual_target']
        
        for target_node in target_nodes:
            # 获取观测该目标的卫星
            connected_satellites = [n for n in full_graph.neighbors(target_node)
                                  if full_graph.nodes[n]['node_type'] == 'satellite']
            
            if len(connected_satellites) > 1:
                # 获取目标价值
                target_value = full_graph.nodes[target_node]['target_value']
                
                # 如果是高价值目标，倾向于将观测卫星分配到同一簇
                if target_value > 80:  # 高价值阈值
                    # 找到这些卫星当前所在的簇
                    current_clusters = [adjusted_clusters[sat] for sat in connected_satellites]
                    most_common_cluster = Counter(current_clusters).most_common(1)[0][0]
                    
                    # 将所有观测卫星移动到最常见的簇
                    for sat in connected_satellites:
                        adjusted_clusters[sat] = most_common_cluster
        
        return adjusted_clusters
    
    def dynamic_clustering(self, time_series_data: List[Dict]) -> List[Dict[str, int]]:
        """
        动态图聚类
        """
        clustering_results = []
        
        for i, time_slice in enumerate(time_series_data):
            print(f"Processing time slice {i+1}/{len(time_series_data)}")
            
            # 当前时间切片聚类
            current_clusters = self.cluster_satellites(time_slice)
            
            # 时间平滑处理
            if i > 0 and clustering_results:
                current_clusters = self._temporal_smoothing(
                    clustering_results[-1], current_clusters
                )
            
            clustering_results.append(current_clusters)
            self.cluster_history.append(current_clusters)
            
            # 计算质量指标
            quality = self.evaluate_clustering_quality(time_slice, current_clusters)
            self.quality_history.append(quality)
            
        return clustering_results
    
    def _temporal_smoothing(self, prev_clusters: Dict[str, int], 
                           curr_clusters: Dict[str, int]) -> Dict[str, int]:
        """
        时间平滑：基于卫星运动的连续性调整聚类
        """
        smoothed_clusters = curr_clusters.copy()
        
        # 计算聚类变化的卫星
        common_satellites = set(prev_clusters.keys()) & set(curr_clusters.keys())
        
        changed_satellites = []
        for sat in common_satellites:
            if prev_clusters[sat] != curr_clusters[sat]:
                changed_satellites.append(sat)
        
        # 如果变化过于剧烈，进行平滑
        change_ratio = len(changed_satellites) / len(common_satellites) if common_satellites else 0
        
        if change_ratio > (1 - self.temporal_smoothing):
            # 对部分卫星保持原聚类
            satellites_to_keep = np.random.choice(
                changed_satellites, 
                size=int(len(changed_satellites) * self.temporal_smoothing),
                replace=False
            )
            
            for sat in satellites_to_keep:
                smoothed_clusters[sat] = prev_clusters[sat]
        
        return smoothed_clusters
    
    def evaluate_clustering_quality(self, time_slice: Dict, clusters: Dict[str, int]) -> Dict:
        """
        评估图聚类质量
        """
        G = self.graph_history[-1] if self.graph_history else self.build_satellite_graph(time_slice)
        
        # 只考虑卫星节点
        satellite_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'satellite']
        satellite_subgraph = G.subgraph(satellite_nodes)
        
        # 1. 网络模块度
        modularity = self._calculate_modularity(satellite_subgraph, clusters)
        
        # 2. 簇内连通性
        intra_cluster_connectivity = self._calculate_intra_cluster_connectivity(
            satellite_subgraph, clusters
        )
        
        # 3. 观测覆盖质量
        observation_coverage = self._calculate_observation_coverage_quality(
            time_slice, clusters
        )
        
        # 4. 负载均衡
        load_balance = self._calculate_load_balance(clusters)
        
        # 5. 簇间分离度
        inter_cluster_separation = self._calculate_inter_cluster_separation(
            satellite_subgraph, clusters
        )
        
        return {
            'modularity': modularity,
            'intra_cluster_connectivity': intra_cluster_connectivity,
            'observation_coverage_quality': observation_coverage,
            'load_balance': load_balance,
            'inter_cluster_separation': inter_cluster_separation,
            'n_clusters': len(set(clusters.values())),
            'cluster_sizes': dict(Counter(clusters.values()))
        }
    
    def _calculate_intra_cluster_connectivity(self, G: nx.Graph, clusters: Dict[str, int]) -> float:
        """计算簇内连通性"""
        if not clusters:
            return 0.0
        
        total_weight = 0
        total_possible_weight = 0
        
        for cluster_id in set(clusters.values()):
            cluster_nodes = [n for n, c in clusters.items() if c == cluster_id]
            if len(cluster_nodes) < 2:
                continue
                
            # 簇内实际连接权重
            cluster_weight = 0
            for i, node1 in enumerate(cluster_nodes):
                for node2 in cluster_nodes[i+1:]:
                    if G.has_edge(node1, node2):
                        cluster_weight += G[node1][node2].get('weight', 1)
            
            total_weight += cluster_weight
            # 可能的最大权重（假设簇内完全连接，权重为1）
            total_possible_weight += len(cluster_nodes) * (len(cluster_nodes) - 1) / 2
        
        return total_weight / total_possible_weight if total_possible_weight > 0 else 0
    
    def _calculate_observation_coverage_quality(self, time_slice: Dict, clusters: Dict[str, int]) -> float:
        """计算观测覆盖质量"""
        target_coverage = defaultdict(set)
        target_values = {}
        
        for obs in time_slice.get('target_visibility', []):
            target_id = obs['to_target']['id']
            sat_id = obs['from_satellite']['id']
            
            target_values[target_id] = obs['target_value']
            
            if sat_id in clusters:
                target_coverage[target_id].add(clusters[sat_id])
        
        if not target_coverage:
            return 0.0
        
        # 计算加权覆盖质量
        total_weighted_coverage = 0
        total_weight = 0
        
        for target_id, covering_clusters in target_coverage.items():
            target_value = target_values.get(target_id, 1)
            coverage_score = min(len(covering_clusters) / self.n_clusters, 1.0)  # 标准化
            
            total_weighted_coverage += target_value * coverage_score
            total_weight += target_value
        
        return total_weighted_coverage / total_weight if total_weight > 0 else 0
    
    def _calculate_load_balance(self, clusters: Dict[str, int]) -> float:
        """计算负载均衡度"""
        if not clusters:
            return 0.0
        
        cluster_sizes = list(Counter(clusters.values()).values())
        if len(cluster_sizes) <= 1:
            return 1.0
        
        # 使用变异系数衡量负载均衡（越小越好）
        mean_size = np.mean(cluster_sizes)
        std_size = np.std(cluster_sizes)
        
        if mean_size == 0:
            return 0.0
        
        cv = std_size / mean_size
        # 转换为0-1分数（越高越好）
        return 1 / (1 + cv)
    
    def _calculate_inter_cluster_separation(self, G: nx.Graph, clusters: Dict[str, int]) -> float:
        """计算簇间分离度"""
        if len(set(clusters.values())) <= 1:
            return 1.0
        
        inter_cluster_edges = 0
        total_edges = 0
        
        for edge in G.edges():
            node1, node2 = edge
            if node1 in clusters and node2 in clusters:
                total_edges += 1
                if clusters[node1] != clusters[node2]:
                    inter_cluster_edges += 1
        
        if total_edges == 0:
            return 1.0
        
        # 簇间连接越少，分离度越高
        return 1 - (inter_cluster_edges / total_edges)
    
    def visualize_satellite_network(self, time_slice: Dict, clusters: Dict[str, int], 
                                   figsize=(15, 10)):
        """
        可视化卫星网络和聚类结果
        """
        G = self.build_satellite_graph(time_slice)
        
        # 分离卫星节点和虚拟目标节点
        satellite_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'satellite']
        target_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'virtual_target']
        
        # 创建布局
        pos = {}
        
        # 卫星节点使用3D位置投影到2D
        for sat in satellite_nodes:
            position = G.nodes[sat]['position']
            pos[sat] = (position[0], position[1])  # 使用x, y坐标
        
        # 虚拟目标节点放在边缘
        for i, target in enumerate(target_nodes):
            angle = 2 * np.pi * i / len(target_nodes)
            radius = 1.5 * max([np.linalg.norm(list(pos.values())[j]) 
                               for j in range(len(pos))]) if pos else 1000
            pos[target] = (radius * np.cos(angle), radius * np.sin(angle))
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 子图1：网络拓扑
        ax1.set_title("Satellite Network Topology", fontsize=14, fontweight='bold')
        
        # 绘制卫星节点
        satellite_colors = [clusters.get(sat, 0) for sat in satellite_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=satellite_nodes, 
                             node_color=satellite_colors, node_size=100,
                             cmap='tab10', ax=ax1, alpha=0.8)
        
        # 绘制虚拟目标节点
        if target_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=target_nodes,
                                 node_color='red', node_size=200, 
                                 node_shape='s', ax=ax1, alpha=0.6)
        
        # 绘制边
        communication_edges = [(u, v) for u, v, d in G.edges(data=True) 
                              if d.get('edge_type') == 'communication']
        observation_edges = [(u, v) for u, v, d in G.edges(data=True) 
                            if d.get('edge_type') == 'observation_constraint']
        distance_edges = [(u, v) for u, v, d in G.edges(data=True) 
                         if d.get('edge_type') == 'distance_based']
        
        if communication_edges:
            nx.draw_networkx_edges(G, pos, edgelist=communication_edges,
                                 edge_color='blue', width=2, alpha=0.6, ax=ax1)
        if observation_edges:
            nx.draw_networkx_edges(G, pos, edgelist=observation_edges,
                                 edge_color='red', width=1, alpha=0.4, 
                                 style='dashed', ax=ax1)
        if distance_edges:
            nx.draw_networkx_edges(G, pos, edgelist=distance_edges,
                                 edge_color='gray', width=0.5, alpha=0.3, ax=ax1)
        
        # 添加标签
        labels = {node: node for node in satellite_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax1)
        
        ax1.axis('off')
        
        # 子图2：聚类统计
        ax2.set_title("Clustering Statistics", fontsize=14, fontweight='bold')
        
        # 簇大小分布
        cluster_sizes = Counter(clusters.values())
        clusters_list = list(cluster_sizes.keys())
        sizes_list = list(cluster_sizes.values())
        
        bars = ax2.bar(clusters_list, sizes_list, color=plt.cm.tab10(np.arange(len(clusters_list))))
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Satellites')
        ax2.set_title('Cluster Size Distribution')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # 打印网络统计
        print(f"\n=== Network Statistics ===")
        print(f"Total satellites: {len(satellite_nodes)}")
        print(f"Virtual targets: {len(target_nodes)}")
        print(f"Communication links: {len(communication_edges)}")
        print(f"Observation constraints: {len(observation_edges)}")
        print(f"Distance-based links: {len(distance_edges)}")
        print(f"Number of clusters: {len(set(clusters.values()))}")
        
    def plot_quality_evolution(self):
        """绘制质量指标随时间的变化"""
        if not self.quality_history:
            print("No quality history available")
            return
        
        metrics = ['modularity', 'intra_cluster_connectivity', 
                  'observation_coverage_quality', 'load_balance', 
                  'inter_cluster_separation']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [q[metric] for q in self.quality_history]
            axes[i].plot(values, marker='o', linewidth=2)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Score')
            axes[i].grid(True, alpha=0.3)
        
        # 最后一个子图显示综合得分
        composite_scores = []
        for q in self.quality_history:
            score = (q['modularity'] + q['intra_cluster_connectivity'] + 
                    q['observation_coverage_quality'] + q['load_balance'] + 
                    q['inter_cluster_separation']) / 5
            composite_scores.append(score)
        
        axes[5].plot(composite_scores, marker='o', linewidth=2, color='red')
        axes[5].set_title('Composite Quality Score')
        axes[5].set_xlabel('Time Step')
        axes[5].set_ylabel('Score')
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, clustering_results: List[Dict[str, int]], 
                      time_series_data: List[Dict], filename: str = "graph_clustering_results.csv"):
        """导出结果"""
        export_data = []
        
        for i, (clusters, time_slice) in enumerate(zip(clustering_results, time_series_data)):
            timestamp = time_slice.get('timestamp', i)
            quality = self.quality_history[i] if i < len(self.quality_history) else {}
            
            for sat_id, cluster_id in clusters.items():
                satellites = self._extract_all_satellites(time_slice)
                pos = satellites.get(sat_id, {}).get('position', [0, 0, 0])
                
                export_data.append({
                    'timestamp': timestamp,
                    'satellite_id': sat_id,
                    'cluster_id': cluster_id,
                    'x_position': pos[0],
                    'y_position': pos[1],
                    'z_position': pos[2],
                    'modularity': quality.get('modularity', 0),
                    'connectivity': quality.get('intra_cluster_connectivity', 0),
                    'coverage_quality': quality.get('observation_coverage_quality', 0)
                })
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

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
        - inter_satellite_connectivity: 该时间戳下所有卫星间连接关系，格式为：
          [{
            'from_satellite': {'id': str, 'position': [x, y, z]},
            'to_satellite': {'id': str, 'position': [x, y, z]},
            'connection_quality': float,
            'visibility_time_window': [start, end]
          }]
        - target_visibility: 该时间戳下所有卫星-目标观测关系，格式为：
          [{
            'from_satellite': {'id': str, 'position': [x, y, z]},
            'to_target': {'id': str, 'position': [x, y, z]},
            'target_value': int,
            'observation_priority': int,
            'visibility_time_window': [start, end]
          }]
    """
    # 加载原始数据
    data = json.loads(file_path.read_text())

    # 按时间戳分组数据
    timestamp_groups = defaultdict(list)
    for record in data:
        timestamp = record["timestamp"]
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
            satellite_info = record.get("satellite_info", {})
            if satellite_info:
                satellites.append(satellite_info)

            # 收集卫星间连接关系
            inter_connectivity = record.get("inter_satellite_connectivity", [])
            for conn in inter_connectivity:
                # 构建新的连接数据结构
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

            # 收集目标观测关系
            target_visibility = record.get("target_visibility", [])
            for target in target_visibility:
                # 构建新的目标观测数据结构
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

            # 获取时间偏移（所有记录应该相同）
            if time_offset is None:
                time_offset = record.get("time_offset_from_scenario_start")

        # 构建时间切片数据
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
    # 加载真实数据
    data_file = get_data_dir() / "satellite_target_visibility_data_sc1.json"

    if not data_file.exists():
        exit("数据文件不存在，使用模拟数据...")

    time_slices = load_data(data_file)

    print(f"成功加载 {len(time_slices)} 个时间切片")

    print("=== 基于图的卫星星座动态分簇系统 ===\n")
    
    # 创建不同方法的聚类系统
    methods = ['spectral', 'louvain', 'hybrid']
    
    for method in methods:
        print(f"\n--- 测试 {method.upper()} 方法 ---")
        
        clustering_system = GraphBasedSatelliteClusteringSystem(
            method=method,
            n_clusters=3,
            alpha=0.6,  # 连接权重
            beta=0.3,   # 观测权重
            gamma=0.1   # 距离权重
        )
        
        # 单时间切片聚类
        clusters = clustering_system.cluster_satellites(time_slices[0])
        print(f"聚类结果: {clusters}")
        
        # 评估质量
        quality = clustering_system.evaluate_clustering_quality(time_slices[0], clusters)
        print(f"聚类质量: {quality}")
        
        # 可视化（可选）
        # clustering_system.visualize_satellite_network(test_data, clusters)

