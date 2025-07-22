import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from icecream import install
from tqdm import tqdm

from data_classes.observation_target_models import (
    MissileInfo,
)
from data_classes.visibility_data_models import (
    SatelliteInfo,
    TargetVisibility,
    InterSatelliteConnectivity,
    SatelliteVisibilityData
)
from stk_server.Packages import STKConnector, Tools
from utils.misc_utils import get_current_timestamp, get_data_dir

install()
stk_conn = STKConnector.STKConnector()


def visualize_satellites_mutual_access(access_data: Dict[str, Dict[str, List[Tuple[str, str, float]]]], output_file: str | Path):
    """
    将卫星之间的可见性时长数据可视化为热力图。

    Args:
        access_data: 卫星可见性数据
        output_file (str, optional): 输出图片的文件路径。如果为None，则显示图片而不保存。
    """
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

    # 获取所有卫星名称
    satellites = set(access_data.keys())
    for sat_data in access_data.values():
        satellites.update(sat_data.keys())
    satellites = sorted(list(satellites))

    # 创建空的数据矩阵
    n = len(satellites)
    data = np.zeros((n, n))

    # 填充数据矩阵
    for sat1, sat_targets in access_data.items():
        for sat2, intervals in sat_targets.items():
            i = satellites.index(sat1)
            j = satellites.index(sat2)
            # 计算总可见时长（秒）
            total_duration = sum(duration for _, _, duration in intervals)
            # 转换为分钟
            total_duration_minutes = total_duration / 60
            data[i, j] = total_duration_minutes

    # 创建DataFrame
    df = pd.DataFrame(data, index=satellites, columns=satellites)

    # 设置图形大小
    plt.figure(figsize=(15, 12))

    # 创建热力图
    sns.heatmap(
        df,
        annot=True,  # 显示数值
        fmt=".0f",  # 数值格式：不保留小数
        cmap="YlOrRd",  # 颜色映射
        cbar_kws={"label": "Duration (min)"},  # 颜色条标签
        square=True,  # 保持正方形
        annot_kws={"size": 8},  # 设置数值标签的字体大小
        xticklabels=True,  # 显示x轴标签
        yticklabels=True,
    )  # 显示y轴标签

    # 设置标题和标签
    plt.title("Satellite Visibility Duration Heatmap", pad=20, fontsize=12)
    plt.xlabel("Satellite", fontsize=10)
    plt.ylabel("Satellite", fontsize=10)

    # 倒置y轴
    plt.gca().invert_yaxis()

    # 调整布局
    plt.tight_layout()

    # 保存或显示图片
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()
    else:
        plt.show()


def visualize_satellites_to_targets_access(
        access_data: Dict[str, Dict[str, List[Tuple[str, str, float]]]], output_file: str | Path
):
    """
    可视化卫星对目标的可见性持续时间数据
    """
    # 检查数据是否为空
    if not access_data:
        print("警告：没有卫星对目标的可见性数据")
        return

    # 提取所有卫星和目标名称
    satellites = set(access_data.keys())
    targets = set()
    for sat_data in access_data.values():
        targets.update(sat_data.keys())

    # 创建数据矩阵
    data = []
    for sat in sorted(satellites):
        row = []
        for tgt in sorted(targets):
            if sat in access_data and tgt in access_data[sat]:
                # 计算总可见时间（分钟）
                total_duration = (
                        sum(duration for _, _, duration in access_data[sat][tgt]) / 60
                )
                row.append(total_duration)
            else:
                row.append(0)
        data.append(row)

    # 检查数据矩阵是否为空
    if not data or not data[0]:
        print("警告：无法生成可见性数据矩阵")
        return

    # 创建DataFrame
    df = pd.DataFrame(data, index=sorted(satellites), columns=sorted(targets))

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

    # 创建图形
    plt.figure(figsize=(15, 12))

    # 创建热力图
    sns.heatmap(
        df,
        annot=True,  # 显示数值
        fmt=".0f",  # 数值格式为整数
        cmap="YlOrRd",  # 颜色映射
        cbar_kws={"label": "可见时间(分钟)"},  # 设置颜色条标签
        annot_kws={"size": 8},  # 设置数值字体大小
        # mask=df == 0,  # 将0值标记为白色
        vmin=0.1,
    )  # 设置最小值，使0值显示为白色

    # 设置标题和标签
    plt.title("卫星对目标的可见时间", pad=20, fontsize=16)
    plt.xlabel("目标", fontsize=14)
    plt.ylabel("卫星", fontsize=14)

    # 设置颜色条标签大小
    plt.gcf().axes[-1].tick_params(labelsize=12)
    plt.gcf().axes[-1].set_ylabel("可见时间(分钟)", fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存或显示图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()
    else:
        plt.show()

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

def generate_satellite_target_visibility_data(
    satellites_to_targets_access: Dict[str, Dict[str, List[Tuple[str, str, float]]]],
    satellites_mutual_access: Dict[str, Dict[str, List[Tuple[str, str, float]]]],
    scenario_begin_time: str,
    step: int = 10,
    output_file: str | Path | None = None,
    default_target_value: int = 3,
    default_observation_priority: int = 8,
) -> List[SatelliteVisibilityData]:
    """
    基于卫星可见性数据，按可见时段每隔step秒采样卫星和目标的ECEF坐标，生成结构化数据。
    每个采样时刻，target_visibility为该卫星此刻能看到的所有目标。
    所有采样时间点均以scenario_begin_time为基准。

    Args:
        satellites_to_targets_access (Dict): 卫星对目标的可见性数据，格式为 {卫星ID: {目标ID: [(开始时间, 结束时间, 持续时长)]}}
        satellites_mutual_access (Dict): 卫星间的可见性数据，格式为 {卫星A: {卫星B: [(开始时间, 结束时间, 持续时长)]}}
        scenario_begin_time (str): 场景开始时间（如 '6 Jun 2025 04:00:00.000'）
        step (int): 采样步长（秒）
        output_file (str | Path): 输出JSON文件路径，如果指定则实时保存结果
        default_target_value (int): 默认目标价值
        default_observation_priority (int): 默认观测优先级

    Returns:
        List[SatelliteVisibilityData]: 结构化数据列表
    """
    # 使用全局STK连接器，避免重复创建连接
    global stk_conn
    scenario_begin_ts = Tools.get_ms_timestamp_by_date_string(scenario_begin_time)

    # 1. 预处理：构建可见性字典 {sat_id: {tgt_id: [(start, end, duration), ...]}}
    visibility_dict = {}
    all_sats = set()
    all_targets = set()
    min_time = float("inf")
    max_time = float("-inf")

    # 处理卫星-目标可见性数据
    for sat_id, targets_dict in satellites_to_targets_access.items():
        all_sats.add(sat_id)

        # 构建可见性字典
        if sat_id not in visibility_dict:
            visibility_dict[sat_id] = {}

        for tgt_id, intervals in targets_dict.items():
            all_targets.add(tgt_id)

            if tgt_id not in visibility_dict[sat_id]:
                visibility_dict[sat_id][tgt_id] = []

            for start_time, end_time, duration in intervals:
                # 将时间转换为相对于场景开始时间的偏移量（秒）
                t0 = Tools.get_ms_timestamp_by_date_string(start_time) - scenario_begin_ts
                t1 = Tools.get_ms_timestamp_by_date_string(end_time) - scenario_begin_ts
                if t0 < 0 or t1 < 0:
                    continue

                # 更新时间范围
                min_time = min(min_time, t0)
                max_time = max(max_time, t1)

                visibility_dict[sat_id][tgt_id].append((t0, t1, duration))

    # 2. 处理卫星间可见性数据
    sat_sat_visibility_dict = {}
    for sat1, satellites_dict in satellites_mutual_access.items():
        if sat1 not in sat_sat_visibility_dict:
            sat_sat_visibility_dict[sat1] = {}

        for sat2, intervals in satellites_dict.items():
            if sat2 not in sat_sat_visibility_dict[sat1]:
                sat_sat_visibility_dict[sat1][sat2] = []

            for start_time, end_time, duration in intervals:
                # 将时间转换为相对于场景开始时间的偏移量（秒）
                t0 = Tools.get_ms_timestamp_by_date_string(start_time) - scenario_begin_ts
                t1 = Tools.get_ms_timestamp_by_date_string(end_time) - scenario_begin_ts
                if t0 < 0 or t1 < 0:
                    continue

                # 更新时间范围
                min_time = min(min_time, t0)
                max_time = max(max_time, t1)

                sat_sat_visibility_dict[sat1][sat2].append((t0, t1, duration))

    # 3. 生成全局采样时间点（等间隔）
    if min_time == float("inf"):
        return []

    # 从最小时间开始，按step间隔生成时间点
    start_sample_time = int(min_time // step) * step  # 向下取整到step的倍数
    end_sample_time = int(max_time // step + 1) * step  # 向上取整到step的倍数
    sample_time_points = list(range(start_sample_time, end_sample_time + 1, step))

    print(f"采样时间范围: {start_sample_time} 到 {end_sample_time}, 步长: {step}")
    print(f"采样时间点数量: {len(sample_time_points)}")

    # 4. 对每个卫星在每个采样时间点，统计可见目标和可见卫星
    result = []

    # 如果指定了输出文件，初始化文件
    if output_file:
        # 如果文件已存在，先读取已有数据
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                print(f"从现有文件加载了 {len(result)} 条记录")
            except:
                result = []

        # 创建备份文件名
        backup_file = str(output_file).replace(".json", "_backup.json")

    processed_count = 0
    for sat_id in tqdm(sorted(all_sats),desc="Processing satellites"):
        if sat_id not in visibility_dict:
            continue

        for t_offset in tqdm(sample_time_points,desc="Processing time offsets"):
            # 统计此刻可见的所有目标
            target_visibility = []
            for tgt_id, intervals in visibility_dict[sat_id].items():
                for t0, t1, duration in intervals:
                    if t0 <= t_offset <= t1:
                        # 获取目标坐标 - 指定特定的导弹实例
                        missile_ecef_data = stk_conn.get_missile_ecef_by_time_shift(
                            start_time_shift=t_offset,
                            period=10,
                            step=1,
                            ret_single_point=True,
                            instance_names=[tgt_id],  # 只获取当前目标的坐标
                        )
                        tgt_ecef = missile_ecef_data.get(
                            tgt_id, [[None, None, None, None]]
                        )[0][1:]

                        # 确保位置数据是有效的浮点数列表
                        if tgt_ecef and all(x is not None for x in tgt_ecef):
                            position = [float(x) for x in tgt_ecef if x is not None]  # 确保类型为float
                        else:
                            position = [0.0, 0.0, 0.0]  # 默认位置

                        target_visibility.append(
                            TargetVisibility(
                                target_id=tgt_id,
                                target_value=default_target_value,
                                observation_priority=default_observation_priority,
                                position=position,
                                visibility_time_window=[
                                    t_offset,
                                    min(t1, t_offset + step),  # 修正时间窗口计算
                                ],
                            )
                        )
                        break  # 一个目标只加一次

            # 如果该卫星在此时刻没有可见目标和可见卫星，跳过
            if not target_visibility:
                continue

            # 获取卫星坐标 - 指定特定的卫星实例
            satellite_ecef_data = stk_conn.get_satellite_ecef_by_time_shift(
                start_time_shift=t_offset,
                period=10,
                step=1,
                ret_single_point=True,
                instance_names=[sat_id],  # 只获取当前卫星的坐标
            )
            sat_ecef = satellite_ecef_data.get(sat_id, [[None, None, None, None]])[0][
                1:
            ]

            # 统计此刻可见的所有卫星
            inter_satellite_connectivity = []
            if sat_id in sat_sat_visibility_dict:
                for other_sat_id, intervals in sat_sat_visibility_dict[sat_id].items():
                    for t0, t1, duration in intervals:
                        if t0 <= t_offset <= t1:
                            # 获取其他卫星的坐标
                            other_sat_ecef_data = (
                                stk_conn.get_satellite_ecef_by_time_shift(
                                    start_time_shift=t_offset,
                                    period=10,
                                    step=1,
                                    ret_single_point=True,
                                    instance_names=[other_sat_id],
                                )
                            )
                            other_sat_ecef = other_sat_ecef_data.get(
                                other_sat_id, [[None, None, None, None]]
                            )[0][1:]

                            # 计算连接质量（基于距离的归一化值）
                            # 这里使用一个简单的示例：假设最大距离为10000km，最小距离为100km
                            # 实际应用中应该根据具体场景调整这些参数
                            distance = Tools.ecef_distance(
                                {"x": sat_ecef[0], "y": sat_ecef[1], "z": sat_ecef[2]},
                                {
                                    "x": other_sat_ecef[0],
                                    "y": other_sat_ecef[1],
                                    "z": other_sat_ecef[2],
                                },
                            )
                            connection_quality = max(
                                0,
                                min(
                                    100,
                                    int(
                                        100
                                        * (
                                            1
                                            - (distance - 100000) / (10000000 - 100000)
                                        )
                                    ),
                                ),
                            )

                            # 确保卫星位置数据是有效的浮点数列表
                            if other_sat_ecef and all(x is not None for x in other_sat_ecef):
                                sat_position = [float(x) for x in other_sat_ecef if x is not None]
                            else:
                                sat_position = [0.0, 0.0, 0.0]  # 默认位置

                            inter_satellite_connectivity.append(
                                InterSatelliteConnectivity(
                                    to_satellite_id=other_sat_id,
                                    position=sat_position,
                                    connection_quality=connection_quality,
                                    visibility_time_window=[
                                        t_offset,
                                        min(t1, t_offset + step),  # 修正时间窗口计算
                                    ],
                                )
                            )
                            break  # 一个卫星只加一次

            # 确保卫星位置数据是有效的浮点数列表
            if sat_ecef and all(x is not None for x in sat_ecef):
                satellite_position = [float(x) for x in sat_ecef if x is not None]
            else:
                satellite_position = [0.0, 0.0, 0.0]  # 默认位置

            data = SatelliteVisibilityData(
                satellite_info=SatelliteInfo(
                    id=sat_id,
                    position=satellite_position,
                    health_status="good",  # 可根据实际情况调整
                    full_visibility_time_window_length=step,
                ),
                inter_satellite_connectivity=inter_satellite_connectivity,
                target_visibility=target_visibility,
                timestamp=Tools.get_date_string_by_timestamp(
                    t_offset + scenario_begin_ts
                ),
                time_offset_from_scenario_start=t_offset,
            )
            result.append(data)
            processed_count += 1

            # 实时保存到文件
            if output_file:
                try:
                    # 先保存到备份文件
                    with open(backup_file, "w", encoding="utf-8") as f:
                        # 将BaseModel对象转换为字典再序列化
                        result_dicts = [item.model_dump() for item in result]
                        json.dump(result_dicts, f, ensure_ascii=False, indent=2)

                    # 备份成功后，重命名为正式文件
                    if os.path.exists(backup_file):
                        if os.path.exists(output_file):
                            os.remove(output_file)
                        os.rename(backup_file, output_file)

                except Exception as e:
                    print(f"保存文件时出错: {e}")

    print(f"生成的数据点数量: {len(result)}")
    if output_file:
        print(f"最终结果已保存到: {output_file}")
    return result

if __name__ == "__main__":

    # 读取并转为 MissileInfo 对象
    with open(
            get_data_dir() / "missile_route_info_scenario_3.json",
            "r",
            encoding="utf-8",
    ) as f:
        missile_data_loaded = json.load(f)
    missile_list = [MissileInfo(**item) for item in missile_data_loaded]

    # missile_list=missile_list[:2]

    # 添加导弹到STK场景
    stk_conn.add_missile(missile_list)
    satellites_mutual_access = stk_conn.get_satellites_mutual_access()
    satellites_to_missiles_access = stk_conn.get_satellites_to_missiles_access()

    # 生成带时间戳的文件名
    timestamp = get_current_timestamp()
    png_filename = get_data_dir() / f"stk_satellites_mutual_access_scenario_3_{timestamp}.png"
    targets_png_filename = get_data_dir() / f"satellites_to_targets_access_scenario_3_{timestamp}.png"
    json_output_filename = get_data_dir() / f"satellite_target_visibility_data_scenario_3_{timestamp}.json"

    # 生成并保存卫星互可见性热力图
    visualize_satellites_mutual_access(
        satellites_mutual_access, output_file=png_filename
    )
    print(f"卫星互可见性热力图已保存到文件: {png_filename}")

    # 生成并保存卫星对目标可见性热力图
    visualize_satellites_to_targets_access(
        satellites_to_missiles_access, output_file=targets_png_filename
    )
    print(f"卫星对目标可见性热力图已保存到文件: {targets_png_filename}")

    # === 生成结构化可见性数据并实时保存到json ===

    structured_data = generate_satellite_target_visibility_data(
        satellites_to_targets_access=satellites_to_missiles_access,
        satellites_mutual_access=satellites_mutual_access,
        scenario_begin_time=stk_conn.scenario_begin_time,
        step=10,
        output_file=json_output_filename,  # 实时保存到文件
    )
    print(f"结构化卫星-目标可见性数据已保存到: {json_output_filename}")
