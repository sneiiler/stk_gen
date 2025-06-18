import csv
import json
import os
from typing import List, Dict, Any
from stk_server.Packages import STKConnector, Tools


def generate_satellite_target_visibility_data(
    csv_path: str, scenario_begin_time: str, step: int = 10, output_file: str | None = None
) -> List[Dict[str, Any]]:
    """
    解析卫星-目标可见性CSV，按可见时段每隔step秒采样卫星和目标的ECEF坐标，生成结构化数据。
    每个采样时刻，target_visibility为该卫星此刻能看到的所有目标。
    所有采样时间点均以scenario_begin_time为基准。

    Args:
        csv_path (str): CSV文件路径
        scenario_begin_time (str): 场景开始时间（如 '6 Jun 2025 04:00:00.000'）
        step (int): 采样步长（秒）
        output_file (str | None): 输出JSON文件路径，如果指定则实时保存结果

    Returns:
        List[dict]: 结构化数据列表
    """
    # 初始化STK连接器
    stk_conn = STKConnector.STKConnector()
    scenario_begin_ts = Tools.get_ms_timestamp_by_date_string(scenario_begin_time)

    # 1. 预处理：构建可见性字典 {sat_id: {tgt_id: [(start, end, duration), ...]}}
    visibility_dict = {}
    all_sats = set()
    all_targets = set()
    min_time = float("inf")
    max_time = float("-inf")

    # 读取卫星-目标可见性数据
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sat_target = row.get("卫星-目标对")
            if not sat_target:
                continue
            sat_id, tgt_id = sat_target.split("-")
            all_sats.add(sat_id)
            all_targets.add(tgt_id)
            start_time = row["开始时间"]
            end_time = row["结束时间"]
            duration = float(row["持续时长(秒)"])
            t0 = Tools.get_ms_timestamp_by_date_string(start_time) - scenario_begin_ts
            t1 = Tools.get_ms_timestamp_by_date_string(end_time) - scenario_begin_ts
            if t0 < 0 or t1 < 0:
                continue

            # 更新时间范围
            min_time = min(min_time, t0)
            max_time = max(max_time, t1)

            if sat_id not in visibility_dict:
                visibility_dict[sat_id] = {}
            if tgt_id not in visibility_dict[sat_id]:
                visibility_dict[sat_id][tgt_id] = []
            visibility_dict[sat_id][tgt_id].append((t0, t1, duration))

    # 2. 读取卫星间可见性数据
    sat_sat_visibility_dict = {}
    sat_sat_csv_path = csv_path.replace("satellites_to_targets_access", "satellites_mutual_access")
    if os.path.exists(sat_sat_csv_path):
        with open(sat_sat_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sat_pair = row.get("卫星对")
                if not sat_pair:
                    continue
                sat1, sat2 = sat_pair.split("-")
                start_time = row["开始时间"]
                end_time = row["结束时间"]
                duration = float(row["持续时长(秒)"])
                t0 = Tools.get_ms_timestamp_by_date_string(start_time) - scenario_begin_ts
                t1 = Tools.get_ms_timestamp_by_date_string(end_time) - scenario_begin_ts
                if t0 < 0 or t1 < 0:
                    continue

                # 更新时间范围
                min_time = min(min_time, t0)
                max_time = max(max_time, t1)

                if sat1 not in sat_sat_visibility_dict:
                    sat_sat_visibility_dict[sat1] = {}
                if sat2 not in sat_sat_visibility_dict[sat1]:
                    sat_sat_visibility_dict[sat1][sat2] = []
                sat_sat_visibility_dict[sat1][sat2].append((t0, t1, duration))

                # 对称性：如果卫星1能看到卫星2，那么卫星2也能看到卫星1
                if sat2 not in sat_sat_visibility_dict:
                    sat_sat_visibility_dict[sat2] = {}
                if sat1 not in sat_sat_visibility_dict[sat2]:
                    sat_sat_visibility_dict[sat2][sat1] = []
                sat_sat_visibility_dict[sat2][sat1].append((t0, t1, duration))

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
        backup_file = output_file.replace('.json', '_backup.json')
        
    total_combinations = len(all_sats) * len(sample_time_points)
    processed_count = 0
    for sat_id in sorted(all_sats):
        if sat_id not in visibility_dict:
            continue

        for t_offset in sample_time_points:
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
                        tgt_ecef = missile_ecef_data.get(tgt_id, [[None, None, None, None]])[0][1:]

                        # 计算剩余可见时间
                        remaining_visibility = t1 - t_offset

                        target_visibility.append(
                            {
                                "target_id": tgt_id,
                                "target_value": 3,
                                "observation_priority": 8,
                                "position": tgt_ecef,
                                "visibility_time_window": [t_offset, t_offset + min(t1, step)],
                            }
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
            sat_ecef = satellite_ecef_data.get(sat_id, [[None, None, None, None]])[0][1:]

            # 统计此刻可见的所有卫星
            inter_satellite_connectivity = []
            if sat_id in sat_sat_visibility_dict:
                for other_sat_id, intervals in sat_sat_visibility_dict[sat_id].items():
                    for t0, t1, duration in intervals:
                        if t0 <= t_offset <= t1:
                            # 获取其他卫星的坐标
                            other_sat_ecef_data = stk_conn.get_satellite_ecef_by_time_shift(
                                start_time_shift=t_offset,
                                period=10,
                                step=1,
                                ret_single_point=True,
                                instance_names=[other_sat_id],
                            )
                            other_sat_ecef = other_sat_ecef_data.get(other_sat_id, [[None, None, None, None]])[0][1:]

                            # 计算连接质量（基于距离的归一化值）
                            # 这里使用一个简单的示例：假设最大距离为10000km，最小距离为100km
                            # 实际应用中应该根据具体场景调整这些参数
                            distance = Tools.ecef_distance(
                                {'x': sat_ecef[0], 'y': sat_ecef[1], 'z': sat_ecef[2]},
                                {'x': other_sat_ecef[0], 'y': other_sat_ecef[1], 'z': other_sat_ecef[2]}
                            )
                            connection_quality = max(0, min(100, int(100 * (1 - (distance - 100000) / (10000000 - 100000)))))

                            inter_satellite_connectivity.append({
                                "to_satellite_id": other_sat_id,
                                "position": other_sat_ecef,
                                "connection_quality": connection_quality,
                                "visibility_time_window": [t_offset, t_offset + min(t1, step)]
                            })
                            break  # 一个卫星只加一次

            data = {
                "satellite_info": {
                    "id": sat_id,
                    "position": sat_ecef,
                    "health_status": "good",  # 可根据实际情况调整
                    "full_visibility_time_window_length": step,
                },
                "inter_satellite_connectivity": inter_satellite_connectivity,
                "target_visibility": target_visibility,
                "timestamp":Tools.get_date_string_by_timestamp(t_offset + scenario_begin_ts),
                "time_offset_from_scenario_start": t_offset,
            }
            result.append(data)
            processed_count += 1
            
            # 实时保存到文件
            if output_file:
                try:
                    # 先保存到备份文件
                    with open(backup_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    # 备份成功后，重命名为正式文件
                    if os.path.exists(backup_file):
                        if os.path.exists(output_file):
                            os.remove(output_file)
                        os.rename(backup_file, output_file)
                    
                    # 每10条记录打印一次进度
                    if processed_count % 10 == 0:
                        progress = (processed_count / total_combinations) * 100
                        print(f"进度: {processed_count}/{total_combinations} ({progress:.1f}%) - 已保存到 {output_file}")
                        
                except Exception as e:
                    print(f"保存文件时出错: {e}")

    print(f"生成的数据点数量: {len(result)}")
    if output_file:
        print(f"最终结果已保存到: {output_file}")
    return result
