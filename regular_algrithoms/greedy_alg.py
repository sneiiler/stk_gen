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

from matplotlib.font_manager import FontProperties
from tqdm import tqdm

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from stk_server.Packages.Tools import ecef2lla

root_dir = Path(__file__).parent.parent
print(root_dir)
sys.path.append(str(root_dir))

from utils.misc_utils import get_data_dir, get_documents_dir, get_project_root


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

def compute_overlap(A, i, j):
    """
    计算两个卫星 i 和 j 的 overlap。
    A: 观测质量矩阵 (N_sat x N_target)
    """
    min_sum = np.sum(np.minimum(A[i], A[j]))
    max_sum = np.sum(np.maximum(A[i], A[j]))
    if max_sum == 0:
        return 0.0
    return min_sum / max_sum

def compute_insight(cluster, overlap_matrix, lambda_param=0.5):
    """
    计算簇的 insight 分数。
    cluster: 簇内卫星 ID 列表
    overlap_matrix: 预计算的 overlap 矩阵
    """
    score = 0.0
    for p in range(len(cluster)):
        for q in range(p + 1, len(cluster)):
            score += overlap_matrix[cluster[p], cluster[q]]
    score += lambda_param * len(cluster)  # 簇大小奖励/惩罚
    return score

def satellite_clustering(A, sat_ids, prev_clusters=None, lambda_param=0.5, threshold=0.3, max_iterations=100):
    """
    卫星分簇算法主函数。
    A: 观测质量矩阵 (N_sat x N_target)
    sat_ids: 卫星 ID 列表 (e.g., [0,1,2,...])
    prev_clusters: 前次簇结果 (dict {sat_id: cluster_id}) for stability (optional)
    lambda_param: insight 中的簇大小系数
    threshold: overlap 阈值，用于种子选择
    max_iterations: 最大迭代次数

    返回: dict {sat_id: cluster_id}, dict {cluster_id: insight_score}
    """
    N = len(sat_ids)
    # 预计算 overlap 矩阵
    overlap_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ov = compute_overlap(A, i, j)
            overlap_matrix[i, j] = ov
            overlap_matrix[j, i] = ov
    
    # 初始化簇：每个卫星单独成簇
    clusters = [{i} for i in range(N)]
    cluster_ids = {i: i for i in range(N)}  # 临时簇 ID
    
    # 如果有前次簇，注入稳定性：优先合并与前簇一致的
    if prev_clusters:
        # 示例：计算与前簇的重叠奖励（可自定义）
        stability_bonus = defaultdict(float)
        for sat, prev_cid in prev_clusters.items():
            stability_bonus[sat] = 1.0  # 简化；实际可基于重叠率
    
    iteration = 0
    while iteration < max_iterations:
        merged = False
        best_insight_gain = 0.0
        best_pair = None
        
        # 寻找能最大化 insight 增益的簇对
        for p in range(len(clusters)):
            for q in range(p + 1, len(clusters)):
                if overlap_matrix[clusters[p].pop(), clusters[q].pop()] < threshold:  # 检查代表性 overlap
                    continue
                # 模拟合并
                temp_cluster = clusters[p] | clusters[q]
                gain = compute_insight(temp_cluster, overlap_matrix, lambda_param) - \
                       compute_insight(clusters[p], overlap_matrix, lambda_param) - \
                       compute_insight(clusters[q], overlap_matrix, lambda_param)
                
                # 注入稳定性奖金（如果适用）
                if prev_clusters:
                    gain += sum(stability_bonus[s] for s in temp_cluster) * 0.1  # 权重可调
                
                if gain > best_insight_gain:
                    best_insight_gain = gain
                    best_pair = (p, q)
        
        if best_pair and best_insight_gain > 0:
            # 合并
            p, q = best_pair
            new_cluster = clusters[p] | clusters[q]
            clusters[p] = new_cluster
            del clusters[q]
            # 更新簇 ID（简化：使用最小 ID）
            for s in new_cluster:
                cluster_ids[s] = min(cluster_ids[s] for s in new_cluster)
            merged = True
        
        if not merged:
            break  # 无进一步改进
        iteration += 1
    
    # 最终输出：标准化簇 ID 从 0 开始
    cluster_map = {}
    insight_scores = defaultdict(float)
    unique_cids = sorted(set(cluster_ids.values()))
    cid_remap = {old: new for new, old in enumerate(unique_cids)}
    for s in range(N):
        old_cid = cluster_ids[s]
        new_cid = cid_remap[old_cid]
        cluster_map[sat_ids[s]] = new_cid
        # 计算最终 insight
        if new_cid not in insight_scores:
            cluster_sats = [sat_ids[k] for k in range(N) if cluster_map[sat_ids[k]] == new_cid]
            insight_scores[new_cid] = compute_insight(cluster_sats, overlap_matrix, lambda_param)
    
    # 解释：打印简单总结（可扩展为详细输出）
    print("分簇结果:", cluster_map)
    print("每个簇的 insight 分数:", insight_scores)
    
    return cluster_map, insight_scores


def visualize_satellites_and_targets_on_map(all_data_frame: List[dict], time_slice_index: int = 0, save_path: str = None, global_target_colors: dict = None):
    """
    在平面地图上可视化卫星和目标的位置，并显示卫星-目标可见性连接线

    Args:
        all_data_frame: 时间切片数据列表
        time_slice_index: 要可视化的时间切片索引，默认为0（第一个时间切片）
        save_path: 保存图片的路径，如果为None则显示图片
        global_target_colors: 全局目标颜色映射字典，确保不同时间切片中同一目标使用相同颜色
    """
    if not all_data_frame or time_slice_index >= len(all_data_frame):
        print("无效的时间切片索引或空数据")
        return

    data_frame = all_data_frame[time_slice_index]

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

    # 绘制卫星（小圆点）- 使用现代蓝色
    if satellite_lats:
        plt.scatter(satellite_lons, satellite_lats, c='#4361EE', s=60, marker='o',
                    label=f'卫星 ({len(satellite_lats)}个)', alpha=0.8, edgecolors='#023047', linewidth=1.5)

        # 添加卫星ID标签（保持标签完整性，不影响绘图区域布局）
        for lon, lat, sat_id in zip(satellite_lons, satellite_lats, satellite_ids):
            plt.annotate(sat_id, (lon, lat), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.9, color='#023047', fontproperties=chinese_font,
                         clip_on=False, weight='bold')  # 允许标签超出绘图区域，保持完整性

    # 绘制目标（五角星）- 使用现代红色
    if target_lats:
        unique_targets = len(set(zip(target_lons, target_lats)))
        plt.scatter(target_lons, target_lats, c='#F72585', s=120, marker='*',
                    label=f'目标 ({unique_targets}个)', alpha=0.9, edgecolors='#C73E1D', linewidth=1.5)

        # 添加目标ID标签（保持标签完整性，不影响绘图区域布局）
        for lon, lat, target_id in zip(target_lons, target_lats, target_ids):
            plt.annotate(target_id, (lon, lat), xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.9, color='#C73E1D', fontproperties=chinese_font,
                         clip_on=False, weight='bold')  # 允许标签超出绘图区域，保持完整性

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

    # 设置标题和标签
    plt.title(f'卫星和目标位置分布图（含可见性连接）\n时间切片: {time_slice_index + 1}, 时间戳: {data_frame["timestamp"]}',
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
        # plt.show()
    else:
        plt.show()

    plt.close()

# 示例使用
if __name__ == "__main__":
    # 加载真实数据
    data_file = get_data_dir() / "satellite_target_visibility_data.json"

    if not data_file.exists():
        exit("数据文件不存在，使用模拟数据...")

    time_slices = load_data(data_file)

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

    # 显示前几个时间切片的详细信息
    print("\n前3个时间切片详情:")
    for i, slice_data in enumerate(time_slices[:3]):
        print(f"\n时间切片 {i+1}:")
        print(f"  时间戳: {slice_data['timestamp']}")
        print(f"  时间偏移: {slice_data['time_offset_from_scenario_start']}秒")

        # 显示卫星信息
        satellites = slice_data['satellites']
        satellite_ids = [sat.get('id', 'N/A') for sat in satellites]
        print(f"  卫星数量: {len(satellites)}")
        print(f"  卫星ID: {satellite_ids}")

        print(f"  卫星间连接数: {len(slice_data['inter_satellite_connectivity'])}")
        print(f"  目标观测数: {len(slice_data['target_visibility'])}")

        # 显示目标信息
        if slice_data['target_visibility']:
            targets = [obs.get('to_target', {}).get('id', obs.get('target_id', 'Unknown'))
                      for obs in slice_data['target_visibility']]
            print(f"  观测目标: {list(set(targets))}")

        # 显示连接信息
        if slice_data['inter_satellite_connectivity']:
            connections = slice_data['inter_satellite_connectivity']
            print(f"  连接关系示例: {connections[0] if connections else 'None'}")
    exit()
    # 可视化第一个时间切片的卫星和目标分布
    print("\n生成卫星和目标位置分布图...")
    try:
        # 创建可视化图片保存目录
        visualize_dir = get_documents_dir() / "visualize_figs_scenario_1"
        visualize_dir.mkdir(exist_ok=True)

        # 创建全局目标颜色映射，确保所有时间切片中同一目标使用相同颜色
        global_target_colors = create_global_target_colors(time_slices)
        print(f"为 {len(global_target_colors)} 个目标分配了固定颜色: {list(global_target_colors.keys())}")

        for index in tqdm(range(len(time_slices)),desc="生成可视化图片"):
            visualize_fig_save_path = visualize_dir / f"satellite_target_map_{index:03d}.png"
            visualize_satellites_and_targets_on_map(time_slices, time_slice_index=index,
                                                   save_path=str(visualize_fig_save_path),
                                                   global_target_colors=global_target_colors)
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        print("请检查数据格式和依赖库是否正确安装")
