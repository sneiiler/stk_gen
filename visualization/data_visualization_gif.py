import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

from data_classes.sft_data_models import ClusterInfo, SatelliteClusterClearOutput, SatelliteClusterOutput

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from misc_tools.sharegpt_utils import load_sharegpt_data
from stk_server.Packages.Tools import ecef2lla

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from utils.misc_utils import get_data_dir, get_documents_dir, get_project_root


def load_cluster_data(file_path:Path)-> List[SatelliteClusterOutput]:
    data = json.loads(file_path.read_text())
    return data


def load_raw_data(file_path: Path) -> List[dict]:
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


def create_global_cluster_colors(clustor_data) -> dict:
    """
    为所有时间切片中出现的分簇创建全局颜色映射，确保同一分簇在不同时间切片中使用相同颜色
    
    Args:
        clustor_data: 分簇数据对象
        
    Returns:
        dict: 分簇ID到颜色的映射字典
    """
    # 使用新的分簇连续性分析
    continuity_analysis = analyze_cluster_continuity(clustor_data)
    return continuity_analysis['stable_cluster_colors']


def analyze_cluster_continuity(clustor_data) -> dict:
    """
    分析分簇的连续性，基于目标重叠度进行分簇跟踪和重新编号
    
    Args:
        clustor_data: 分簇数据对象
        
    Returns:
        dict: 包含稳定分簇映射的字典，结构为：
        {
            'cluster_mapping': {timestamp: {original_id: stable_id}},
            'stable_cluster_colors': {stable_id: color},
            'stable_cluster_shapes': {stable_id: shape}
        }
    """
    # 提取所有时间切片的分簇数据
    time_cluster_data = []
    for clustor in clustor_data.output_result_data:
        timestamp_clusters = {}
        for cluster_item in clustor:
            timestamp = cluster_item.get('timestamp')
            if timestamp not in timestamp_clusters:
                timestamp_clusters[timestamp] = []
            
            if 'clusters' in cluster_item:
                for cluster_dict in cluster_item['clusters']:
                    cluster_info = {
                        'cluster_id': cluster_dict.get('cluster_id'),
                        'targets': set(cluster_dict.get('targets', [])),
                        'sats': set(cluster_dict.get('sats', []))
                    }
                    timestamp_clusters[timestamp].append(cluster_info)
            elif isinstance(cluster_item, dict) and 'cluster_id' in cluster_item:
                cluster_info = {
                    'cluster_id': cluster_item.get('cluster_id'),
                    'targets': set(cluster_item.get('targets', [])),
                    'sats': set(cluster_item.get('sats', []))
                }
                timestamp_clusters[timestamp].append(cluster_info)
        
        time_cluster_data.append(timestamp_clusters)
    
    # 按时间戳排序，获取所有时间切片
    all_timestamps = set()
    for timestamp_clusters in time_cluster_data:
        all_timestamps.update(timestamp_clusters.keys())
    sorted_timestamps = sorted(all_timestamps)
    
    # 初始化稳定分簇跟踪
    stable_cluster_counter = 0
    cluster_mapping = {}  # {timestamp: {original_id: stable_id}}
    stable_clusters_history = {}  # {stable_id: [targets_set_per_timestamp]}
    
    # 颜色和形状列表
    base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                   '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
                   '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D5A6BD']
    shapes = ['rectangle', 'circle', 'diamond', 'hexagon', 'rounded_rect', 'triangle', 'octagon']
    
    def calculate_overlap_ratio(targets1, targets2):
        """计算两个目标集合的重叠率"""
        if not targets1 or not targets2:
            return 0.0
        intersection = len(targets1.intersection(targets2))
        union = len(targets1.union(targets2))
        return intersection / union if union > 0 else 0.0
    
    def find_best_match(current_cluster, previous_stable_clusters, threshold=0.3):
        """为当前分簇找到最佳的稳定分簇匹配"""
        best_match = None
        best_ratio = threshold
        
        for stable_id, history in previous_stable_clusters.items():
            if history:  # 确保历史记录存在
                last_targets = history[-1]
                overlap_ratio = calculate_overlap_ratio(current_cluster['targets'], last_targets)
                if overlap_ratio > best_ratio:
                    best_ratio = overlap_ratio
                    best_match = stable_id
        
        return best_match, best_ratio
    
    # 逐时间切片处理
    for timestamp in sorted_timestamps:
        cluster_mapping[timestamp] = {}
        current_clusters = []
        
        # 收集当前时间戳的所有分簇
        for timestamp_clusters in time_cluster_data:
            if timestamp in timestamp_clusters:
                current_clusters.extend(timestamp_clusters[timestamp])
        
        if not current_clusters:
            continue
        
        # 第一个时间切片，直接分配稳定ID
        if not stable_clusters_history:
            for cluster in current_clusters:
                stable_id = stable_cluster_counter
                cluster_mapping[timestamp][cluster['cluster_id']] = stable_id
                stable_clusters_history[stable_id] = [cluster['targets']]
                stable_cluster_counter += 1
        else:
            # 为当前分簇寻找最佳匹配
            used_stable_ids = set()
            unmatched_clusters = []
            
            # 按目标数量从大到小排序，优先处理大分簇
            current_clusters.sort(key=lambda x: len(x['targets']), reverse=True)
            
            for cluster in current_clusters:
                best_match, overlap_ratio = find_best_match(cluster, stable_clusters_history)
                
                if best_match is not None and best_match not in used_stable_ids:
                    # 找到匹配，使用已有的稳定ID
                    cluster_mapping[timestamp][cluster['cluster_id']] = best_match
                    stable_clusters_history[best_match].append(cluster['targets'])
                    used_stable_ids.add(best_match)
                    print(f"时间戳 {timestamp}: 分簇 {cluster['cluster_id']} -> 稳定分簇 {best_match} (重叠率: {overlap_ratio:.2f})")
                else:
                    # 没有找到匹配，标记为未匹配
                    unmatched_clusters.append(cluster)
            
            # 为未匹配的分簇分配新的稳定ID
            for cluster in unmatched_clusters:
                stable_id = stable_cluster_counter
                cluster_mapping[timestamp][cluster['cluster_id']] = stable_id
                stable_clusters_history[stable_id] = [cluster['targets']]
                stable_cluster_counter += 1
                print(f"时间戳 {timestamp}: 分簇 {cluster['cluster_id']} -> 新稳定分簇 {stable_id}")
    
    # 生成稳定分簇的颜色和形状映射
    all_stable_ids = list(stable_clusters_history.keys())
    stable_cluster_colors = {}
    stable_cluster_shapes = {}
    
    for i, stable_id in enumerate(sorted(all_stable_ids)):
        stable_cluster_colors[stable_id] = base_colors[i % len(base_colors)]
        stable_cluster_shapes[stable_id] = shapes[i % len(shapes)]
    
    print(f"\n分簇连续性分析完成:")
    print(f"- 总共识别出 {len(all_stable_ids)} 个稳定分簇")
    print(f"- 分簇映射关系: {len(cluster_mapping)} 个时间戳")
    
    return {
        'cluster_mapping': cluster_mapping,
        'stable_cluster_colors': stable_cluster_colors,
        'stable_cluster_shapes': stable_cluster_shapes
    }


def create_global_cluster_shapes(clustor_data) -> dict:
    """
    为所有分簇创建全局形状映射，确保同一分簇在不同时间切片中使用相同形状
    
    Args:
        clustor_data: 分簇数据对象
        
    Returns:
        dict: 分簇ID到形状的映射字典
    """
    # 使用新的分簇连续性分析
    continuity_analysis = analyze_cluster_continuity(clustor_data)
    return continuity_analysis['stable_cluster_shapes']


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


def visualize_satellites_and_targets_on_map(all_data_frame: List[dict], 
                                            clustor_data,  # 类型为ValidationInput
                                            time_slice_index: int = 0,
                                            save_path: str|Path|None = None, 
                                            global_target_colors: dict|None = None,
                                            global_cluster_colors: dict|None = None,
                                            global_cluster_shapes: dict|None = None,
                                            cluster_mapping: dict|None = None):
    """
    在平面地图上可视化卫星和目标的位置，并显示卫星-目标可见性连接线

    Args:
        all_data_frame: 时间切片数据列表
        clustor_data: 分簇数据
        time_slice_index: 要可视化的时间切片索引，默认为0（第一个时间切片）
        save_path: 保存图片的路径，如果为None则显示图片
        global_target_colors: 全局目标颜色映射字典，确保不同时间切片中同一目标使用相同颜色
        global_cluster_colors: 全局分簇颜色映射字典，确保不同时间切片中同一分簇使用相同颜色
        global_cluster_shapes: 全局分簇形状映射字典，确保不同时间切片中同一分簇使用相同形状
        cluster_mapping: 分簇连续性映射字典，{timestamp: {original_id: stable_id}}
    """
    if not all_data_frame or time_slice_index >= len(all_data_frame):
        print("无效的时间切片索引或空数据")
        return

    data_frame = all_data_frame[time_slice_index]

    current_clustor = None
    for clustor in clustor_data.output_result_data:
        if clustor[0]['timestamp'] == data_frame['timestamp']:
            current_clustor = clustor
            break
    
    # 解析分簇数据
    current_clusters = []
    if current_clustor:
        for cluster_item in current_clustor:
            if 'clusters' in cluster_item:
                # 将字典转换为ClusterInfo对象
                for cluster_dict in cluster_item['clusters']:
                    cluster_info = ClusterInfo(**cluster_dict)
                    current_clusters.append(cluster_info)
            # 处理直接的分簇列表（从ShareGPT数据中解析出来的）
            elif isinstance(cluster_item, dict) and 'cluster_id' in cluster_item:
                cluster_info = ClusterInfo(**cluster_item)
                current_clusters.append(cluster_info)

    # 提取卫星位置并转换为经纬度
    satellite_lats = []
    satellite_lons = []
    satellite_ids = []
    satellite_positions = {}  # 用于存储卫星ID到位置的映射

    # 处理主要卫星数据
    for satellite in data_frame['satellites']:
        position = satellite.get('position', [])
        if len(position) >= 3:
            # ECEF坐标转换为经纬度（position单位为km，需要转换为m）
            x, y, z = position[0] * 1000, position[1] * 1000, position[2] * 1000
            lat, lon, _ = ecef2lla(x, y, z)
            satellite_lats.append(lat)
            satellite_lons.append(lon)
            sat_id = satellite.get('id', 'Unknown')
            satellite_ids.append(sat_id)
            satellite_positions[sat_id] = (lon, lat)

    # 处理卫星间连接中的额外卫星
    for satellite in data_frame['inter_satellite_connectivity']:
        to_satellite = satellite.get('to_satellite', {})
        sat_id = to_satellite.get('id', 'Unknown')

        # 避免重复添加已存在的卫星
        if sat_id not in satellite_positions:
            position = to_satellite.get('position', [])
            if len(position) >= 3:
                x, y, z = position[0] * 1000, position[1] * 1000, position[2] * 1000
                lat, lon, _ = ecef2lla(x, y, z)
                satellite_lats.append(lat)
                satellite_lons.append(lon)
                satellite_ids.append(sat_id)
                satellite_positions[sat_id] = (lon, lat)

    # 提取目标位置并转换为经纬度
    target_lats = []
    target_lons = []
    target_ids = []
    target_positions = {}  # 用于存储目标ID到位置的映射

    for target_obs in data_frame['target_visibility']:
        to_target = target_obs.get('to_target', {})
        position = to_target.get('position', [])
        if len(position) >= 3:
            # ECEF坐标转换为经纬度（position单位为km，需要转换为m）
            x, y, z = position[0] * 1000, position[1] * 1000, position[2] * 1000
            lat, lon, alt = ecef2lla(x, y, z)
            target_lats.append(lat)
            target_lons.append(lon)
            target_id = to_target.get('id', 'Unknown')
            target_ids.append(target_id)
            target_positions[target_id] = (lon, lat)

    # 尝试使用系统中已安装的中文字体
    try:
        font_path = get_project_root() / "utils/simhei.ttf"
        chinese_font = FontProperties(fname=str(font_path))
    except Exception:
        print("警告：无法加载中文字体，将使用系统默认字体")
        chinese_font = FontProperties()

    # 加载背景图片
    try:
        bg_image_path = get_project_root() / "documents/Specular.png"
        bg_image = Image.open(bg_image_path)

        # 获取图片尺寸
        img_width, img_height = bg_image.size
        aspect_ratio = img_width / img_height

        # 根据图片比例调整figure大小
        if aspect_ratio > 1:
            # 宽图
            fig_width = 15
            fig_height = 15 / aspect_ratio
        else:
            # 高图
            fig_height = 10
            fig_width = 10 * aspect_ratio

        # 创建图形
        plt.figure(figsize=(fig_width, fig_height))

        # 显示背景图片 - 调整透明度使其更柔和
        plt.imshow(bg_image, extent=(-180, 180, -90, 90), aspect='auto', alpha=0.8)

    except Exception as e:
        print(f"警告：无法加载背景图片 {bg_image_path}: {e}")
        # 如果无法加载背景图片，使用默认设置
        plt.figure(figsize=(15, 10))

    # 绘制世界地图边界并固定坐标轴范围
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)

    # 固定坐标轴范围，防止标签影响绘图区域
    ax = plt.gca()
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal', adjustable='box')  # 保持纵横比

    # 手动设置绘图区域的固定位置，确保不受后续元素影响
    # 这样可以为标签留出足够的空间，同时保持绘图区域位置固定
    fixed_position = (0.1, 0.15, 0.75, 0.7)  # (left, bottom, width, height)
    ax.set_position(fixed_position)

    # 添加网格 - 使用更现代的样式
    plt.grid(True, alpha=0.4, color='#E9ECEF', linewidth=0.8, linestyle='-')

    # 检查分簇数据
    if current_clusters:
        print(f"找到 {len(current_clusters)} 个分簇，正在绘制分簇框...")
        draw_clusters_after_positions = True
    else:
        draw_clusters_after_positions = False

    # 绘制卫星 - 使用原来的统一样式
    if satellite_lats:
        # 使用统一的卫星样式绘制所有卫星，调整边框颜色和粗细
        plt.scatter(satellite_lons, satellite_lats, 
                   c='#4361EE', s=80, alpha=0.9, marker='o', 
                   edgecolors='#2C3E50', linewidth=0.8, 
                   label=f'卫星 ({len(satellite_lats)})', zorder=5)

        # 添加卫星ID标签
        for lon, lat, sat_id in zip(satellite_lons, satellite_lats, satellite_ids):
            plt.annotate(sat_id, (lon, lat), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.9, color='#023047', fontproperties=chinese_font,
                         clip_on=False, weight='bold')

    # 绘制目标 - 使用原来的统一样式
    if target_lats:
        # 使用统一的目标样式绘制所有目标，调整边框颜色和粗细
        plt.scatter(target_lons, target_lats, 
                   c='#F72585', s=120, alpha=0.9, marker='*', 
                   edgecolors='#8B2635', linewidth=0.8, 
                   label=f'目标 ({len(target_lats)})', zorder=5)

        # 添加目标ID标签
        for lon, lat, target_id in zip(target_lons, target_lats, target_ids):
            plt.annotate(target_id, (lon, lat), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.9, color='#C73E1D', fontproperties=chinese_font,
                         clip_on=False, weight='bold')

    # 绘制分簇形状框（在收集完所有位置信息后）
    cluster_legend_info = []
    if draw_clusters_after_positions and current_clusters:
        try:
            # 获取当前时间戳的分簇映射
            current_timestamp = data_frame['timestamp']
            current_cluster_mapping = cluster_mapping.get(current_timestamp, {}) if cluster_mapping else {}
            
            cluster_legend_info = create_cluster_colors_and_draw_shapes(
                current_clusters, satellite_positions, target_positions, 
                global_cluster_colors, global_cluster_shapes, current_cluster_mapping)
            print(f"成功绘制了 {len(current_clusters)} 个分簇的形状框")
        except Exception as e:
            print(f"绘制分簇形状框时出错: {e}")

    # 绘制卫星-目标可见性连接线（不同目标使用不同颜色）
    visibility_count = 0

    # 现代化配色方案 - 使用更柔和、更专业的颜色
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
              '#7209B7', '#F72585', '#4361EE', '#F77F00', '#FCBF49',
              '#06FFA5', '#FB8500', '#8ECAE6', '#219EBC', '#023047',
              '#FFB3C6', '#FB8500', '#8B5CF6', '#10B981', '#F59E0B']

    # 使用全局颜色映射或创建新的映射
    if global_target_colors is None:
        target_colors = {}
        unique_targets = list(set(target_ids))
        for i, target_id in enumerate(unique_targets):
            target_colors[target_id] = colors[i % len(colors)]
    else:
        target_colors = global_target_colors

    # 按目标分组绘制连接线
    target_visibility_count = {}  # 记录每个目标的连接数

    for target_obs in data_frame['target_visibility']:
        from_satellite = target_obs.get('from_satellite', {})
        to_target = target_obs.get('to_target', {})

        sat_id = from_satellite.get('id')
        target_id = to_target.get('id')

        # 获取卫星和目标的经纬度坐标
        if sat_id in satellite_positions and target_id in target_positions:
            sat_lon, sat_lat = satellite_positions[sat_id]
            target_lon, target_lat = target_positions[target_id]

            # 获取该目标的颜色
            color = target_colors.get(target_id, 'green')

            # 绘制连接线
            plt.plot([sat_lon, target_lon], [sat_lat, target_lat],
                     color=color, linewidth=1.5, alpha=0.7, linestyle='-')

            # 统计连接数
            if target_id not in target_visibility_count:
                target_visibility_count[target_id] = 0
            target_visibility_count[target_id] += 1
            visibility_count += 1

    # 添加可见性连接线的图例项（总统计）
    if visibility_count > 0:
        plt.plot([], [], color='#6C757D', linewidth=2, alpha=0.8, linestyle='-',
                 label=f'可见性连接 ({visibility_count}条)')

    # 添加分簇图例项
    if cluster_legend_info:
        for cluster_info in cluster_legend_info:
            # 为每个分簇创建一个图例项，显示形状、颜色和统计信息
            # 不需要实际绘制patch，只使用颜色信息
            plt.plot([], [], color=cluster_info['color'], linewidth=3, alpha=0.9,
                    label=f'分簇{cluster_info["cluster_id"]} ({cluster_info["shape_name"]}) - '
                          f'{cluster_info["satellite_count"]}卫星, {cluster_info["target_count"]}目标')

    # 设置标题和标签
    plt.title(
        f'卫星和目标位置分布图（含可见性连接）\n时间切片: {time_slice_index + 1}, 时间戳: {data_frame["timestamp"]}',
        fontsize=14, fontproperties=chinese_font, pad=20)
    plt.xlabel('经度 (度)', fontproperties=chinese_font, fontsize=12)
    plt.ylabel('纬度 (度)', fontproperties=chinese_font, fontsize=12)

    # 添加图例到绘图区域上方 - 使用现代样式
    plt.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=3, fontsize=10,
               frameon=True, fancybox=True, shadow=True, framealpha=0.9,
               edgecolor='#E9ECEF', facecolor='white')

    # 设置坐标轴刻度 - 使用现代样式
    plt.xticks(range(-180, 181, 30), fontsize=10, color='#495057')
    plt.yticks(range(-90, 91, 30), fontsize=10, color='#495057')

    # 设置坐标轴样式
    ax.spines['top'].set_color('#DEE2E6')
    ax.spines['right'].set_color('#DEE2E6')
    ax.spines['bottom'].set_color('#DEE2E6')
    ax.spines['left'].set_color('#DEE2E6')
    ax.tick_params(colors='#495057')

    # # 添加统计信息到图例旁边（绘图区域外）- 使用现代样式
    # info_text = f"卫星: {len(satellite_lats)} | 目标: {len(set(target_ids))} | 卫星间连接: {len(data_frame['inter_satellite_connectivity'])} | 可见性连接: {visibility_count}"
    # plt.figtext(0.5, 0.07, info_text, ha='center', fontproperties=chinese_font, fontsize=10,
    #             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', alpha=0.95,
    #                      edgecolor='#DEE2E6', linewidth=1))

    # 在调整布局前，再次确保绘图区域位置固定
    ax = plt.gca()
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    # 重新设置固定位置，确保绘图区域不受标签影响
    ax.set_position(fixed_position)

    # 不使用自动布局调整，保持手动设置的固定位置

    # 保存或显示图片
    if save_path:
        # 使用固定的边界框，确保图片大小一致
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1,
                    facecolor='white', edgecolor='none')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def create_cluster_colors_and_draw_shapes(clusters: List[ClusterInfo], 
                                         satellite_positions: dict, 
                                         target_positions: dict,
                                         global_cluster_colors: dict|None = None,
                                         global_cluster_shapes: dict|None = None,
                                         cluster_mapping: dict|None = None) -> list:
    """
    为分簇创建同色系颜色映射并绘制形状框
    
    Args:
        clusters: 分簇信息列表
        satellite_positions: 卫星位置映射
        target_positions: 目标位置映射
        global_cluster_colors: 全局分簇颜色映射字典，确保不同时间切片中同一分簇使用相同颜色
        global_cluster_shapes: 全局分簇形状映射字典，确保不同时间切片中同一分簇使用相同形状
        cluster_mapping: 当前时间戳的分簇映射，{original_id: stable_id}
        
    Returns:
        list: 图例信息列表
    """
    from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
    import numpy as np
    
    # 使用全局颜色映射或创建新的映射
    if global_cluster_colors is None:
        # 基础颜色（浅色用于填充/内框）
        base_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                       '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        cluster_color_map = {}
        for i, cluster in enumerate(clusters):
            cluster_color_map[cluster.cluster_id] = base_colors[i % len(base_colors)]
    else:
        cluster_color_map = global_cluster_colors
    
    # 使用全局形状映射或创建新的映射
    if global_cluster_shapes is None:
        shapes = ['rectangle', 'circle', 'diamond', 'hexagon', 'rounded_rect', 'triangle', 'octagon']
        cluster_shape_map = {}
        for i, cluster in enumerate(clusters):
            cluster_shape_map[cluster.cluster_id] = shapes[i % len(shapes)]
    else:
        cluster_shape_map = global_cluster_shapes
    
    # 对应的深色（用于边框）
    def darken_color(color_hex):
        """将颜色变深85%，使边框更深"""
        color_hex = color_hex.lstrip('#')
        r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
        r, g, b = int(r * 0.15), int(g * 0.15), int(b * 0.15)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    # 形状名称映射
    shape_names = {'rectangle': '矩形', 'circle': '圆形', 'diamond': '菱形', 
                   'hexagon': '六边形', 'rounded_rect': '圆角矩形', 'triangle': '三角形', 'octagon': '八边形'}
    
    legend_info = []
    
    for cluster in clusters:
        # 使用稳定的分簇ID获取颜色和形状
        original_id = cluster.cluster_id
        stable_id = cluster_mapping.get(original_id, original_id) if cluster_mapping else original_id
        
        base_color = cluster_color_map.get(stable_id, '#FF6B6B')
        dark_color = darken_color(base_color)
        shape_style = cluster_shape_map.get(stable_id, 'rectangle')
        
        # 记录图例信息 - 显示稳定ID
        legend_info.append({
            'cluster_id': stable_id,  # 使用稳定ID显示
            'original_id': original_id,  # 保留原始ID用于调试
            'color': base_color,
            'shape_style': shape_style,
            'shape_name': shape_names[shape_style],
            'satellite_count': len(cluster.sats),
            'target_count': len(cluster.targets)
        })
        
        # 绘制分簇中的卫星和目标
        all_positions = []
        
        # 收集卫星位置
        for sat_id in cluster.sats:
            sat_id_str = f"Satellite{sat_id}"
            if sat_id_str in satellite_positions:
                all_positions.append(satellite_positions[sat_id_str])
        
        # 收集目标位置  
        for target_id in cluster.targets:
            target_id_str = f"m{target_id}"
            if target_id_str in target_positions:
                all_positions.append(target_positions[target_id_str])
        
        # 为每个位置绘制形状
        for lon, lat in all_positions:
            create_and_add_shape(lon, lat, shape_style, base_color, dark_color, 4.5)
    
    return legend_info


def create_and_add_shape(lon, lat, shape_style, fill_color, border_color, size):
    """创建并添加单个形状到图中"""
    from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
    import numpy as np
    
    if shape_style == 'rectangle':
        shape = Rectangle((lon-size, lat-size), size*2, size*2, 
                         linewidth=2, edgecolor=border_color, facecolor=fill_color,
                         alpha=0.3, zorder=3)
    
    elif shape_style == 'circle':
        shape = Circle((lon, lat), size, 
                      linewidth=2, edgecolor=border_color, facecolor=fill_color,
                      alpha=0.3, zorder=3)
    
    elif shape_style == 'diamond':
        points = np.array([[lon, lat + size], [lon + size, lat], 
                          [lon, lat - size], [lon - size, lat]])
        shape = Polygon(points, closed=True, linewidth=2, 
                       edgecolor=border_color, facecolor=fill_color,
                       alpha=0.3, zorder=3)
    
    elif shape_style == 'hexagon':
        angles = np.linspace(0, 2*np.pi, 7)
        points = np.array([[lon + size*np.cos(a), lat + size*np.sin(a)] for a in angles])
        shape = Polygon(points, closed=True, linewidth=2,
                       edgecolor=border_color, facecolor=fill_color,
                       alpha=0.3, zorder=3)
    
    elif shape_style == 'rounded_rect':
        shape = FancyBboxPatch((lon-size, lat-size), size*2, size*2,
                              boxstyle="round,pad=0.1", linewidth=2,
                              edgecolor=border_color, facecolor=fill_color,
                              alpha=0.3, zorder=3)
    
    elif shape_style == 'triangle':
        h = size * np.sqrt(3) / 2
        points = np.array([[lon, lat + h], [lon - size, lat - h/2], [lon + size, lat - h/2]])
        shape = Polygon(points, closed=True, linewidth=2,
                       edgecolor=border_color, facecolor=fill_color,
                       alpha=0.3, zorder=3)
    
    elif shape_style == 'octagon':
        angles = np.linspace(0, 2*np.pi, 9)
        points = np.array([[lon + size*np.cos(a), lat + size*np.sin(a)] for a in angles])
        shape = Polygon(points, closed=True, linewidth=2,
                       edgecolor=border_color, facecolor=fill_color,
                       alpha=0.3, zorder=3)
    
    else:
        return
        
    plt.gca().add_patch(shape)


# 示例使用
if __name__ == "__main__":
    # 加载真实数据
    data_file = get_data_dir() / "satellite_target_visibility_data_sc1.json"

    if not data_file.exists():
        exit("数据文件不存在，使用模拟数据...")

    time_slices = load_raw_data(data_file)
    clustor_data = load_sharegpt_data(get_data_dir() / "clustering_results_cmax_20001.jsonl")
    # time_slices = time_slices[1:19]

    print(f"成功加载 {len(time_slices)} 个时间切片")

    # 获取数据摘要
    summary = get_time_slice_summary(time_slices)
    print("\n数据摘要:")
    print(f"- 总时间切片数: {summary['total_time_slices']}")
    print(f"- 时间范围: {summary['time_range']}")
    print(f"- 唯一卫星数: {len(summary['unique_satellites'])}")
    print(f"- 唯一目标数: {len(summary['unique_targets'])}")
    print(f"- 总连接数: {summary['connectivity_stats']['total_connections']}")
    print(f"- 平均每切片连接数: {summary['connectivity_stats']['avg_connections_per_slice']:.2f}")
    print(f"- 总观测数: {summary['target_stats']['total_observations']}")
    print(f"- 平均每切片观测数: {summary['target_stats']['avg_observations_per_slice']:.2f}")

    # # 显示前几个时间切片的详细信息
    # print("\n前3个时间切片详情:")
    # for i, slice_data in enumerate(time_slices[:3]):
    #     print(f"\n时间切片 {i + 1}:")
    #     print(f"  时间戳: {slice_data['timestamp']}")
    #     print(f"  时间偏移: {slice_data['time_offset_from_scenario_start']}秒")

    #     # 显示卫星信息
    #     satellites = slice_data['satellites']
    #     satellite_ids = [sat.get('id', 'N/A') for sat in satellites]
    #     print(f"  卫星数量: {len(satellites)}")
    #     print(f"  卫星ID: {satellite_ids}")

    #     print(f"  卫星间连接数: {len(slice_data['inter_satellite_connectivity'])}")
    #     print(f"  目标观测数: {len(slice_data['target_visibility'])}")

    #     # 显示目标信息
    #     if slice_data['target_visibility']:
    #         targets = [obs.get('to_target', {}).get('id', obs.get('target_id', 'Unknown'))
    #                    for obs in slice_data['target_visibility']]
    #         print(f"  观测目标: {list(set(targets))}")

    #     # 显示连接信息
    #     if slice_data['inter_satellite_connectivity']:
    #         connections = slice_data['inter_satellite_connectivity']
    #         print(f"  连接关系示例: {connections[0] if connections else 'None'}")

    # 可视化第一个时间切片的卫星和目标分布
    print("\n生成卫星和目标位置分布图...")
    try:
        # 创建可视化图片保存目录
        visualize_dir = get_project_root() / "visualization/visualize_figs_scenario_1_test"
        visualize_dir.mkdir(exist_ok=True)

        # 进行全局分簇连续性分析
        print("\n进行分簇连续性分析...")
        continuity_analysis = analyze_cluster_continuity(clustor_data)
        
        # 创建全局目标颜色映射，确保所有时间切片中同一目标使用相同颜色
        global_target_colors = create_global_target_colors(time_slices)
        print(f"为 {len(global_target_colors)} 个目标分配了固定颜色: {list(global_target_colors.keys())}")

        # 使用连续性分析结果创建稳定的分簇颜色和形状映射
        global_cluster_colors = continuity_analysis['stable_cluster_colors']
        global_cluster_shapes = continuity_analysis['stable_cluster_shapes']
        cluster_mapping = continuity_analysis['cluster_mapping']
        
        print(f"为 {len(global_cluster_colors)} 个稳定分簇分配了固定颜色")
        print(f"为 {len(global_cluster_shapes)} 个稳定分簇分配了固定形状")

        for index in tqdm(range(len(time_slices)), desc="生成可视化图片"):
            visualize_fig_save_path = visualize_dir / f"satellite_target_map_with_clusters_{index:03d}.png"
            # visualize_fig_save_path = None  # 取消注释这行来直接显示而不保存
            visualize_satellites_and_targets_on_map(time_slices,
                                                    clustor_data,  # 直接传递ValidationInput对象
                                                    time_slice_index=index,
                                                    save_path=visualize_fig_save_path,
                                                    global_target_colors=global_target_colors,
                                                    global_cluster_colors=global_cluster_colors,
                                                    global_cluster_shapes=global_cluster_shapes,
                                                    cluster_mapping=cluster_mapping)
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print("请检查数据格式和依赖库是否正确安装")
