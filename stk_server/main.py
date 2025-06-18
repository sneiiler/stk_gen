import math
from typing import List, Dict, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from stk_server.Packages import STKConnector, Tools
from icecream import ic, install
from data_models.observation_target_models import ObservationTargetInfo, RevisitAnalysisInfo, MissileInfo
from data_models.payload_models import PayloadInfo

from data_models.satellite_models import (
    SatelliteInfo,
    LifetimeEstimationInfo,
    LightingTimeData,
    OrbitData,
    OrbitalElementsInfo,
    RegressionAnalysisInfo,
    SunBetaAngleInfo,
)
from utils.misc_utils import get_documents_dir

install()
stk_conn = STKConnector.STKConnector()

def visualize_satellites_mutual_access(access_data, output_file: str | None = None):
    """
    将卫星之间的可见性时长数据可视化为热力图。

    Args:
        access_data: 卫星可见性数据
        output_file (str, optional): 输出图片的文件路径。如果为None，则显示图片而不保存。
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 获取所有卫星名称
    satellites = set()
    for key in access_data.keys():
        sat1, sat2 = key.split('-')
        satellites.add(sat1)
        satellites.add(sat2)
    satellites = sorted(list(satellites))

    # 创建空的数据矩阵
    n = len(satellites)
    data = np.zeros((n, n))

    # 填充数据矩阵
    for key, intervals in access_data.items():
        sat1, sat2 = key.split('-')
        i = satellites.index(sat1)
        j = satellites.index(sat2)
        # 计算总可见时长（秒）
        total_duration = sum(duration for _, _, duration in intervals)
        # 转换为分钟
        total_duration_minutes = total_duration / 60
        data[i, j] = total_duration_minutes
        data[j, i] = total_duration_minutes  # 对称矩阵

    # 创建DataFrame
    df = pd.DataFrame(data, index=satellites, columns=satellites)

    # 设置图形大小
    plt.figure(figsize=(15, 12))

    # 创建热力图
    sns.heatmap(df, 
               annot=True,  # 显示数值
               fmt='.0f',   # 数值格式：不保留小数
               cmap='YlOrRd',  # 颜色映射
               cbar_kws={'label': 'Duration (min)'},  # 颜色条标签
               square=True,  # 保持正方形
               annot_kws={'size': 8},  # 设置数值标签的字体大小
               xticklabels=True,  # 显示x轴标签
               yticklabels=True)  # 显示y轴标签

    # 设置标题和标签
    plt.title('Satellite Visibility Duration Heatmap', pad=20, fontsize=12)
    plt.xlabel('Satellite', fontsize=10)
    plt.ylabel('Satellite', fontsize=10)

    # 倒置y轴
    plt.gca().invert_yaxis()

    # 调整布局
    plt.tight_layout()

    # 保存或显示图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_satellites_to_targets_access(access_data: Dict[str, List[Tuple[str, str, float]]], output_file: str | None = None):
    """
    可视化卫星对目标的可见性持续时间数据
    """
    # 检查数据是否为空
    if not access_data:
        print("警告：没有卫星对目标的可见性数据")
        return
        
    # 提取所有卫星和目标名称
    satellites = set()
    targets = set()
    for key in access_data.keys():
        sat, tgt = key.split('-')
        satellites.add(sat)
        targets.add(tgt)
    
    # 创建数据矩阵
    data = []
    for sat in sorted(satellites):
        row = []
        for tgt in sorted(targets):
            key = f"{sat}-{tgt}"
            if key in access_data:
                # 计算总可见时间（分钟）    
                total_duration = sum(duration for _, _, duration in access_data[key]) / 60
                row.append(total_duration)
            else:
                row.append(0)
        data.append(row)
    
    # 检查数据矩阵是否为空
    if not data or not data[0]:
        print("警告：无法生成可见性数据矩阵")
        return
        
    # 创建DataFrame
    df = pd.DataFrame(data, 
                     index=sorted(satellites),
                     columns=sorted(targets))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图形
    plt.figure(figsize=(15, 12))
    
    # 创建热力图
    sns.heatmap(df,
                annot=True,  # 显示数值
                fmt='.0f',  # 数值格式为整数
                cmap='YlOrRd',  # 颜色映射
                cbar_kws={'label': '可见时间(分钟)'},  # 设置颜色条标签
                annot_kws={'size': 8},  # 设置数值字体大小
                mask=df == 0,  # 将0值标记为白色
                vmin=0.1)  # 设置最小值，使0值显示为白色
    
    # 设置标题和标签
    plt.title('卫星对目标的可见时间', pad=20, fontsize=16)
    plt.xlabel('目标', fontsize=14)
    plt.ylabel('卫星', fontsize=14)
    
    # 设置颜色条标签大小
    plt.gcf().axes[-1].tick_params(labelsize=12)
    plt.gcf().axes[-1].set_ylabel('可见时间(分钟)', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
   
    # # 读取并转为 MissileInfo 对象
    # with open(get_documents_dir() / "missile_info2.json", "r", encoding="utf-8") as f:
    #     missile_data_loaded = json.load(f)
    # missile_list = [MissileInfo(**item) for item in missile_data_loaded]
    # # ic(missile_list)
    # # exit()

    # # 添加导弹到STK场景
    # stk_conn.add_missile(missile_list)
    # satellites_mutual_access = stk_conn.get_satellites_mutual_access()
    # satellites_to_missiles_access = stk_conn.get_satellites_to_missiles_access()
    import csv
    from datetime import datetime

    # # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # csv_filename = f"satellites_mutual_access_{timestamp}.csv"
    # png_filename = f"satellites_mutual_access_{timestamp}.png"
    # targets_csv_filename = f"satellites_to_targets_access_{timestamp}.csv"
    # targets_png_filename = f"satellites_to_targets_access_{timestamp}.png"

    # # 写入卫星互可见性CSV文件
    # with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     # 写入表头
    #     writer.writerow(['卫星对', '开始时间', '结束时间', '持续时长(秒)'])
        
    #     # 写入数据
    #     for sat_pair, access_times in satellites_mutual_access.items():
    #         for start_time, end_time, duration in access_times:
    #             writer.writerow([sat_pair, start_time, end_time, duration])
    
    # print(f"卫星互可见性数据已保存到CSV文件: {csv_filename}")
    
    # # 生成并保存卫星互可见性热力图
    # visualize_satellites_mutual_access(satellites_mutual_access, output_file=png_filename)
    # print(f"卫星互可见性热力图已保存到文件: {png_filename}")

    # # 写入卫星对目标可见性CSV文件
    # with open(targets_csv_filename, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     # 写入表头
    #     writer.writerow(['卫星-目标对', '开始时间', '结束时间', '持续时长(秒)'])
        
    #     # 写入数据
    #     for sat_target_pair, access_times in satellites_to_missiles_access.items():
    #         for start_time, end_time, duration in access_times:
    #             writer.writerow([sat_target_pair, start_time, end_time, duration])
    
    # print(f"卫星对目标可见性数据已保存到CSV文件: {targets_csv_filename}")
    # # 生成并保存卫星对目标可见性热力图
    # visualize_satellites_to_targets_access(satellites_to_missiles_access, output_file=targets_png_filename)
    # print(f"卫星对目标可见性热力图已保存到文件: {targets_png_filename}")

    # === 新增：生成结构化可见性数据并实时保存到json ===
    from stk_server.data_processing import generate_satellite_target_visibility_data
    json_output_filename = f"satellite_target_visibility_data_{timestamp}.json"
    
    structured_data = generate_satellite_target_visibility_data(
        "satellites_to_targets_access_20250609_153841.csv",
        stk_conn.scenario_begin_time,
        step=10,
        output_file=json_output_filename  # 实时保存到文件
    )
    print(f"结构化卫星-目标可见性数据已保存到: {json_output_filename}")