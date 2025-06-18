"""
STKConnector - 通过COM连接STK 11

用于提供Python与STK的连接支持能力，并提供部分函数调用支持。

Author: Yin Kaifeng <yinkaifeng@cast>
Created: 2024-06-16
"""

import datetime
import math
import time
import numpy as np
import os
import winreg  # 添加winreg模块导入
from pathlib import Path

# This file requires Python 3.12 or later
import win32com.client
from typing import List, Dict, Tuple
from icecream import ic

from data_models.observation_target_models import (
    ObservationTargetInfo,
    RevisitAnalysisConstraints,
    RevisitAnalysisInfo,
    STKAccessEvent,
    MissileInfo,
)
from data_models.payload_models import PayloadInfo

from data_models.satellite_models import (
    SatelliteInfo,
    LifetimeEstimationInfo,
    LightingDuration,
    LightingTimeData,
    RegressionAnalysisInfo,
    SunBetaAngleInfo,
)

from stk_server.Packages.stkutil import (
    AgEOrbitStateType,
    AgECoordinateSystem,
)
from stk_server.Packages.stkobjects import (
    AgESTKObjectType,
    AgELeadTrailData,
    AgEVePropagatorType,
    AgEClassicalSizeShape,
    AgEClassicalLocation,
    AgEOrientationAscNode,
    AgECnstrLighting,
    AgESnPattern,
    AgEAccessConstraints,
)

from stk_server.Packages import Tools
from utils.misc_utils import get_documents_dir, measure_time
from tqdm import tqdm


from tqdm import tqdm

class STKConnector:
    def __init__(self):
        """
        初始化STKConnector类，创建STK应用实例并获取IAgStkObjectRoot接口

        Args:
            无参数

        Returns:
            无返回值

        """
        self.scenario_begin_time = ""
        self.scenario_end_time = ""
        self.earth_radius = 6371  # km

        # 获取当前的场景
        try:
            uiApp = win32com.client.GetActiveObject("STK12.Application")
        except Exception as e:
            print(f"Get STK11.Application object error {e}")

        uiApp.Visible = True
        uiApp.UserControl = True  # 可以用鼠标和STK GUI交互

        self.root = uiApp.Personality2
        # 获取STK安装路径
        stk_versions = ["12.0", "11.0"]  # 尝试多个STK版本
        self.stk_install_path = None

        for version in stk_versions:
            try:
                # 尝试打开不同版本的STK注册表键
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"SOFTWARE\\AGI\\STK\\{version}")
                # 读取InstallPath值
                self.stk_install_path = winreg.QueryValueEx(key, "InstallHome")[0]
                winreg.CloseKey(key)
                break
            except Exception:
                continue

        # 如果所有版本都失败，使用默认路径
        if self.stk_install_path is None:
            print("从注册表获取STK安装路径失败，使用默认路径")
            self.stk_install_path = r"C:\Program Files\AGI\STK 12"

        # Set date format 设置日期格式，UTCG为'24 Sep 2020 16:00:00.00'格式的时间
        self.root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")

        # Create new scenario
        try:
            print("Creating scenario...")
            self.scenario = self.root.CloseScenario
            self.root.NewScenario("MCP_Created_Scenario")
            # 获取当前场景的时间
        except Exception as e:
            print(f"     ↳ Info: {e.excepinfo[2]}")
            print(f"Scenario existed, using current scenario...")
            self.scenario = self.root.CurrentScenario
            self.scenario_begin_time = self.scenario.StartTime
            self.scenario_end_time = self.scenario.StopTime

    def set_scenario_time(self, scenario_begin_time, scenario_end_time):
        """
        设置场景的时间范围，并重置动画至开始时间

        此方法会将场景的开始时间、结束时间以及时间基准（Epoch）设置为提供的参数。
        同时，执行命令重置动画，从新设置的开始时间开始播放。

        Args:
            scenario_begin_time (str): 场景的开始时间，格式为 "YYYY/MM/DD HH:MM:SS"
            scenario_end_time (str): 场景的结束时间，格式为 "YYYY/MM/DD HH:MM:SS"

        Returns:
            None: 此方法不返回任何值

        Example:
            set_scenario_time("2024/01/01 00:00:00", "2024/01/02 00:00:00")
        """

        self.scenario.StartTime = scenario_begin_time  # 修改场景的起止时间
        self.scenario.StopTime = scenario_end_time
        self.scenario.Epoch = scenario_begin_time

        self.root.ExecuteCommand("Animate * Reset")

        self.scenario_begin_time = scenario_begin_time
        self.scenario_end_time = scenario_end_time

    def get_satellite_ecef_by_time_shift(
        self, start_time_shift: float, period: float, step=0.125, ret_single_point=True, instance_names=None
    ) -> Dict[str, List[Tuple[str, float, float, float]]]:
        """根据时间偏移量获取指定卫星的ECEF坐标。

        该方法计算场景中指定卫星在指定时间段内的地心地固坐标系(ECEF)位置。
        时间基于场景开始时间加上指定的偏移量。

        Args:
            start_time_shift (float): 相对于场景开始时间的偏移量，单位为秒
            period (float): 计算的时间段长度，单位为秒
            step (float, optional): 时间采样步长，单位为秒，默认为0.125秒
            ret_single_point (bool, optional): 是否只返回第一个时间点的坐标。
                默认为True，仅返回第一个点
            instance_names (List[str], optional): 指定要获取坐标的卫星实例名称列表。
                如果为None，则获取所有卫星的坐标

        Returns:
            dict: 以卫星名称为键的字典，值为卫星位置数据列表。
                每个列表包含元组 (时间, x坐标, y坐标, z坐标)，其中：
                - t: STK时间格式字符串，如 '22 Mar 2025 06:15:58.220'
                - x, y, z: ECEF坐标系中的位置，单位为千米

                若ret_single_point为True，则每个卫星的列表仅包含一个元组
                若ret_single_point为False，则包含整个时间段内按step采样的所有点

        Note:
            ECEF (Earth-Centered, Earth-Fixed) 是一个以地球质心为原点，
            随地球自转的坐标系统。x轴指向赤道与本初子午线交点，
            z轴指向北极，y轴与x、z轴构成右手系。
        """
        # 获取场景中的所有卫星对象
        paths = self.get_objects("Satellite")
        resp = {}
        for path in tqdm(paths, desc="Calculating satellite ECEF"):
            satellite = self.root.GetObjectFromPath(path)
            satellite_name = satellite.InstanceName
            
            # 如果指定了instance_names，只处理指定的卫星
            if instance_names is not None and satellite_name not in instance_names:
                continue
                
            print(f"处理satellite: {satellite_name}")
            orbit_begin_time = Tools.get_date_string_by_timestamp(
                Tools.get_ms_timestamp_by_date_string(self.scenario_begin_time) + start_time_shift
            )
            orbit_end_time = Tools.get_date_string_by_timestamp(
                Tools.get_ms_timestamp_by_date_string(self.scenario_begin_time) + start_time_shift + period
            )

            satelliteDP = satellite.DataProviders.Item("Cartesian Position").Group.Item("Fixed")
            result = satelliteDP.Exec(orbit_begin_time, orbit_end_time, step)
            times = result.DataSets.GetDataSetByName("Time").GetValues()
            x_pos = result.DataSets.GetDataSetByName("x").GetValues()
            y_pos = result.DataSets.GetDataSetByName("y").GetValues()
            z_pos = result.DataSets.GetDataSetByName("z").GetValues()
            _temp = []
            for t, x, y, z in zip(times, x_pos, y_pos, z_pos):
                _temp.append(
                    (
                        Tools.get_date_string_by_timestamp(Tools.get_ms_timestamp_by_date_string(t)),
                        round(x, 3),
                        round(y, 3),
                        round(z, 3),
                    )
                )
                if ret_single_point:
                    break
            resp.update({satellite_name: _temp})

        return resp

    def get_satellite_lla_by_time_shift(
        self, start_time_shift: float, period: float, step=0.125, ret_single_point=True
    ) -> Dict[str, List[Tuple[str, float, float, float]]]:
        """获取卫星在给定时间偏移下的经纬度和高度（LLA）。

        该方法通过在场景的开始时间基础上添加时间偏移，获取指定时间段内所有卫星的经纬度和高度
        （LLA: Latitude, Longitude, Altitude）。

        Args:
            start_time_shift (float): 相对于场景开始时间的偏移量，单位为秒
            period (float): 计算的时间段长度，单位为秒
            step (float, optional): 时间采样步长，单位为秒，默认为0.125秒
            ret_single_point (bool, optional): 是否只返回第一个时间点的坐标。
                默认为True，仅返回第一个点

        Returns:
            dict: 以卫星名称为键的字典，值为卫星位置数据列表。
                每个列表元素是包含以下键的字典:
                - 'time': STK时间格式字符串，如 '22 Mar 2025 06:15:58.220'
                - 'latitude': 纬度，单位为度，范围 -90° 到 90°
                - 'longitude': 经度，单位为度，范围 -180° 到 180°
                - 'altitude': 高度，单位为米，表示海拔高度

                若ret_single_point为True，则每个卫星只返回第一个时间点的数据
                若ret_single_point为False，则返回整个时间段内按step采样的所有点数据

        Example:
            get_satellite_lla_by_time_shift(3600, 1800)  # 获取场景开始1小时后，持续30分钟内的所有卫星位置
        """
        # 获取场景中的所有卫星对象
        paths = self.get_objects("Satellite")
        resp = {}

        for path in paths:
            satellite = self.root.GetObjectFromPath(path)

            # 计算开始和结束时间
            orbit_begin_time = Tools.get_date_string_by_timestamp(
                Tools.get_ms_timestamp_by_date_string(self.scenario_begin_time) + start_time_shift
            )
            orbit_end_time = Tools.get_date_string_by_timestamp(
                Tools.get_ms_timestamp_by_date_string(self.scenario_begin_time) + start_time_shift + period
            )

            # 获取LLA数据提供器并执行查询
            satelliteDP = satellite.DataProviders.Item("LLA State").Group.Item("Fixed")
            result = satelliteDP.Exec(orbit_begin_time, orbit_end_time, step)

            # 获取时间、经纬度和高度数据
            times = result.DataSets.GetDataSetByName("Time").GetValues()
            lat_pos = result.DataSets.GetDataSetByName("Lat").GetValues()
            lon_pos = result.DataSets.GetDataSetByName("Lon").GetValues()
            alt_pos = result.DataSets.GetDataSetByName("Alt").GetValues()

            # 构建该卫星的位置数据列表
            satellite_data = []
            for t, lat, lon, alt in zip(times, lat_pos, lon_pos, alt_pos):
                satellite_data.append(
                    (
                        Tools.get_date_string_by_timestamp(Tools.get_ms_timestamp_by_date_string(t)),
                        round(lat, 3),
                        round(lon, 3),
                        round(alt, 3),
                    )
                )
                if ret_single_point:
                    break

            # 将该卫星的数据添加到结果字典中
            resp[satellite.InstanceName] = satellite_data

        return resp

    def calculate_sensor_fov_imaging_swath(
        self, sensor_info: PayloadInfo, satellite_ecef_data: Dict[str, List[Tuple[str, float, float, float]]]
    ) -> Dict[str, float]:
        """
        计算载荷视场角对应的地面覆盖幅宽。

        Args:
            sensor_info (PayloadInfo): 载荷信息，包含视场角等参数
            satellite_ecef_data (Dict[str, List[Tuple[str, float, float, float]]]): 卫星ECEF坐标数据
                {
                    "satellite_name1": [(time, x, y, z), ...],
                    "satellite_name2": [(time, x, y, z), ...],
                }

        Returns:
            Dict[str, float]: 每个卫星载荷的地面覆盖幅宽(km)
                {
                    "satellite_name1": imaging_swath,
                    "satellite_name2": imaging_swath,
                }

        Note:
            - 地面覆盖幅宽基于卫星高度和载荷视场角计算
            - 使用地球平均半径6371km作为参考
        """
        resp = {}
        # 提取卫星坐标
        for satellite_name, ecef_data in satellite_ecef_data.items():
            x_s, y_s, z_s = ecef_data[0][1], ecef_data[0][2], ecef_data[0][3]

            # 计算卫星到地球中心的距离（模长）
            distance_to_center = math.sqrt(x_s**2 + y_s**2 + z_s**2)

            # 计算卫星的高度（轨道高度）
            orbit_height = distance_to_center - self.earth_radius

            # 将视场半角转换为弧度
            fov_half_angle_rad = math.radians(sensor_info.fov_angle)

            # 计算地面覆盖半径
            imaging_swath = round(orbit_height * math.tan(fov_half_angle_rad), 2)
            resp.update({satellite_name: imaging_swath})

        return resp

    def get_satellite_pass_data(self, epoch: int = 30) -> Dict:
        """获取卫星过境数据，包括轨道特性和降交点信息。

        该方法计算卫星在指定时间范围内的轨道特性，包括最大/最小纬度、
        降交点时间、降交点经度以及对应的地方时。计算周期默认为30天。

        Args:
            epoch (int, optional): 计算周期，单位为天，默认为30天

        Returns:
            dict: 包含以下键值对的字典:
                - min_lat (float): 卫星轨道最小纬度，单位为度
                - max_lat (float): 卫星轨道最大纬度，单位为度
                - time_of_descen_node (str): 降交点时间，UTC格式
                - descending_node (list(Tuple[str, float])): 降交点经度，单位为度
                - descending_node_local_time (str): 降交点对应的地方时，格式为"HH:MM"

        Note:
            地方时的计算基于经度和UTC时间，每15度经度对应1小时时差。
        """
        # 设置场景时间，默认30天为一个计算周期
        resp = {}

        # 获取场景中的所有卫星对象
        paths = self.get_objects("Satellite")
        for path in paths:
            satellite = self.root.GetObjectFromPath(path)

            # 获取卫星数据提供器，用于计算过境数据
            satelliteDP = satellite.DataProviders.Item("Passes")

            # 计算场景结束时间（当前时间+epoch天）
            scenario_end_time = Tools.get_date_string_by_timestamp(
                Tools.get_timestamp_by_date_string(self.scenario_begin_time) + epoch * 86400  # epoch天的秒数
            )

            # 设置当前场景的时间范围
            self.set_scenario_time(self.scenario_begin_time, scenario_end_time)

            # 执行数据查询，获取卫星过境数据
            result = satelliteDP.Exec(self.scenario_begin_time, scenario_end_time)

            # 提取并四舍五入卫星轨道的最小纬度
            min_lat = round(result.DataSets.GetDataSetByName("Min Lat").GetValues()[1], 2)

            # 提取并四舍五入卫星轨道的最大纬度
            max_lat = round(result.DataSets.GetDataSetByName("Max Lat").GetValues()[1], 2)

            # 如果场景的开始时间不是卫星的降交点时间，则需要判断后面的数据是不是在pass里面
            raw_lon_descen_node = result.DataSets.GetDataSetByName("Lon Descen Node").GetValues()
            raw_time_of_descen_node = result.DataSets.GetDataSetByName("Time of Descen Node").GetValues()
            valid_pass_index = 0
            for x in raw_lon_descen_node:
                try:
                    float(x)
                    break
                except:
                    valid_pass_index += 1

            # 获取降交点时间（卫星穿过赤道从北向南的时刻）
            time_of_descen_node = raw_time_of_descen_node[valid_pass_index:-1]

            # 获取降交点经度
            lon_descen_node = raw_lon_descen_node[valid_pass_index:-1]

            # 格式化时间字符串，保留3位小数
            formatted_times = []
            for t in time_of_descen_node:
                timestamp = Tools.get_ms_timestamp_by_date_string(t)
                formatted_time = Tools.get_date_string_by_timestamp(timestamp)
                formatted_times.append(formatted_time)

            # 解析UTC时间字符串为datetime对象
            utc_time = datetime.datetime.strptime(
                Tools.get_date_string_by_timestamp(
                    Tools.get_ms_timestamp_by_date_string(time_of_descen_node[0])
                ),
                "%d %b %Y %H:%M:%S.%f",
            )
            # 计算地方时偏移（经度/15小时） 360度/24小时=15度/小时
            offset = datetime.timedelta(hours=lon_descen_node[0] / 15)
            # 加上偏移
            local_time = utc_time + offset
            # 地方时在24小时制内循环
            lon_descen_node_local_time = (
                local_time.strftime("%H:%M")
                if local_time.hour >= 0
                else local_time + datetime.timedelta(days=1)
            )
            lon_descen_node_local_time = (
                local_time.strftime("%H:%M")
                if local_time.hour < 24
                else local_time - datetime.timedelta(days=1)
            )

            # 构造并返回结果字典
            resp.update(
                {
                    satellite.InstanceName: {
                        "min_lat": min_lat,  # 轨道最小纬度
                        "max_lat": max_lat,  # 轨道最大纬度
                        "descending_node": list(zip(formatted_times, lon_descen_node)),  # 降交点经度
                        "descending_node_local_time": lon_descen_node_local_time,  # 降交点地方时
                    }
                }
            )
        return resp

    def calculate_satellite_pass(self):
        """
        计算卫星的过境时间窗口

        该方法通过获取卫星的降交点（Descend Node）过境时间，并计算每次过境的起始和结束时间。
        每次过境的时间窗口由当前降交点的时间与相邻降交点的时间平均值计算得到。

        返回的数据包含每次过境的起始和结束时间，格式如下：
        {
            索引: [过境开始时间, 过境结束时间]
        }

        Args:


        Returns:
            list: 每次过境的时间窗口列表，每个元素是一个字典，格式为：
                [
                    {0: [开始时间, 结束时间]},
                    {1: [开始时间, 结束时间]},
                    ...
                ]

        Example:
            pass_data = calculate_satellite_pass()
        """
        path = self.get_objects("Satellite")
        satellite = self.root.GetObjectFromPath(path[0])
        # satelliteDP = satellite.DataProviders.Item('LLR State').Group.Item('Fixed')
        # result = satelliteDP.Exec(self.scenario_begin_time, self.scenario_end_time, 60)
        # times = result.DataSets.GetDataSetByName('Time').GetValues()
        # lat_pos = result.DataSets.GetDataSetByName('Lat').GetValues()
        # lon_pos = result.DataSets.GetDataSetByName('Lon').GetValues()
        #
        # resp = list(zip(times, lat_pos, lon_pos))

        satelliteDP = satellite.DataProviders.Item("Pass Event Times")
        result = satelliteDP.Exec(self.scenario_begin_time, self.scenario_end_time)
        descen_times = result.DataSets.GetDataSetByName("Time of Descen Node").GetValues()

        pass_data = []
        for idx, t in enumerate(descen_times):
            current_pass_descending_t = Tools.get_timestamp_by_date_string(t)
            if idx < len(descen_times) - 1:
                delta_t = (
                    Tools.get_timestamp_by_date_string(descen_times[idx + 1]) - current_pass_descending_t
                ) / 2
            else:
                delta_t = (
                    current_pass_descending_t - Tools.get_timestamp_by_date_string(descen_times[idx - 1])
                ) / 2
            current_pass = [
                Tools.get_date_string_by_timestamp(int(current_pass_descending_t - delta_t)),
                Tools.get_date_string_by_timestamp(int(current_pass_descending_t + delta_t)),
            ]
            pass_data.append({idx: current_pass})

        return pass_data

    def set_scenario_time_period(self, begin_time, end_time):
        """
        重置仿真场景开始时间
        Args:
            begin_time:
            end_time:

        Returns:

        """
        self.scenario.SetTimePeriod(begin_time, end_time)
        self.root.ExecuteCommand("Animate * Reset")
        self.scenario.Epoch = begin_time

    def add_annotations(self, message):
        """
        增加注释
        Args:
            message:

        Returns:

        """
        # 在地图上找个地方显示现在正在做什么
        annotation_target = [
            {
                "name": message,
                "longitude": 80,
                "latitude": 70,
            }
        ]
        # 先删除原有的标签
        self.delete_objects(obj_type="Place", delete_all=True)

        # 再添加新的标签
        self.add_objects(
            "Place",
            annotation_target,
            label_visible=True,
            stk_sync_show=True,
            echo=False,
            delete_origin_targets=False,
        )

    def add_objects(
        self,
        obj_type: str,
        obj_list: List,
        label_visible=True,
        stk_sync_show=True,
        echo=False,
        delete_origin_targets=True,
    ):
        """
        增加目标点到STK
        Args:
            stk_sync_show: 在STK中同步显示，会降低性能
            obj_type: 对象类型，如'Satellite'、'Sensor'、'Target' 等。
            obj_list: 同 `target_list` 结构

            后续所有的`targets_list_x`都具有相同的数据结构，可能长度不同：
                target_list:[{
                                "target_name": "Beijing",
                                "target_coords":[39.9042,116.4074,1] # lat, lon, alt
                            },...]
            target_status:可选值--0:可选 --1：已观测
            echo: 是否打开显示当前添加点的进度
            delete_origin_targets: 是否删除原来的点
        Returns:

        """
        start_time = time.time()
        # 关闭stk中的界面显示，提高添加点的性能
        if not stk_sync_show:
            self.root.BeginUpdate()
        # delete all exists targets before insert new target
        if delete_origin_targets:
            self.delete_objects(obj_type=obj_type, delete_all=True)

        default_altitude = 0

        len_obj_list = len(obj_list)
        for idx, obj in tqdm(enumerate(obj_list), desc="Adding objects"):
            try:
                if type(obj.target_name) is List:
                    obj_target_name = obj.target_name[0]
                else:
                    obj_target_name = obj.target_name
            except Exception as e:
                raise BaseException(f"解析target_name失败：{e}")

            target = self.scenario.Children.New(AgESTKObjectType[obj_type], str(obj_target_name))
            target.Graphics.LabelVisible = False
            target.Position.AssignGeodetic(obj.target_coords[0], obj.target_coords[1], obj.target_coords[2])

            if echo:
                print(
                    f"\r Process: {idx / len_obj_list * 100:.2f}%, add {obj_type}-{obj_target_name},"
                    f" Longitude={obj.target_coords[1]}, Latitude={obj.target_coords[0]}",
                    end="",
                )

            # 设置目标约束
            target_constraints = target.AccessConstraints
            light = target_constraints.AddConstraint(AgEAccessConstraints["eCstrLighting"])
            light.Condition = AgECnstrLighting["eUmbraOrDirectSun"]
            target.UseTerrain = False

            if obj_type == "Place":
                target.Graphics.LabelColor = 16777215  # 白色
            target.Graphics.LabelVisible = label_visible
            target.HeightAboveGround = 0

        # 结束更新，打开界面显示
        if not stk_sync_show:
            self.root.EndUpdate()

        end_time = time.time()
        execution_time = end_time - start_time
        ic(execution_time)

    def add_missile(self, missile_info: List[MissileInfo]):
        """
        增加导弹到STK
        Args:
            missile_info: List[MissileInfo]类型，包含导弹的轨道信息等数据
        """
        for missile_info_ in tqdm(missile_info, desc="添加导弹"):
            try:
                missile = self.scenario.Children.New(AgESTKObjectType["eMissile"], missile_info_.name)
            except Exception as e:
                self.delete_objects("eMissile", obj_list=[missile_info_.model_dump()], delete_all=False)
                missile = self.scenario.Children.New(AgESTKObjectType["eMissile"], missile_info_.name)
            missile.SetTrajectoryType(AgEVePropagatorType.ePropagatorBallistic)  # ePropagatorBallistic
            trajectory = missile.Trajectory
            trajectory.EphemerisInterval.SetExplicitInterval(
                Tools.get_date_string_by_timestamp(
                    Tools.get_timestamp_by_date_string(self.scenario_begin_time)
                    + missile_info_.trajectory_epoch_second
                ),
                Tools.get_date_string_by_timestamp(
                    Tools.get_timestamp_by_date_string(self.scenario_begin_time)
                    + missile_info_.trajectory_epoch_second
                ),
            )  # stop time later computed based on propagation
            trajectory.Launch.Lat = missile_info_.latitude  # deg
            trajectory.Launch.Lon = missile_info_.longitude  # deg
            trajectory.ImpactLocation.Impact.Lat = missile_info_.impact_latitude  # deg
            trajectory.ImpactLocation.Impact.Lon = missile_info_.impact_longitude  # deg
            trajectory.ImpactLocation.SetLaunchControlType(0)  # eLaunchControlFixedApogeeAlt
            trajectory.ImpactLocation.LaunchControl.ApogeeAlt = missile_info_.altitude  # km
            trajectory.Propagate()
            # 设置3D视图的地面轨迹不显示
            missile.VO.Trajectory.TrackData.PassData.GroundTrack.SetLeadDataType(0)
            missile.VO.Trajectory.TrackData.PassData.GroundTrack.SetTrailDataType(0)
            

    def add_satellite(
        self,
        satellite_info: List[SatelliteInfo],
        hpop=False,
        stk_sync_show=True,
        delete_duplicate_targets=True,
    ):
        """
        增加卫星到STK
        Args:
            satellite_info: List[SatelliteInfo]类型，包含卫星的轨道信息等数据
                [{
                 "name":"satellite1",
                 "orbit_epoch_time":"22 Mar 2025 04:00:00.000",
                 "position":[234,1234,234],
                 "velocity":[234,1234,234],
                 "orbit_elements":{
                        "semi_axis":, #km
                        "eccentricity": ,
                        "inclination":, #degrees
                        "arg_of_perigee":, #degrees
                        "raan":, #degrees
                        "mean_anomaly":, #degrees
                    }
                },...]
            stk_sync_show: 在STK中同步显示，会降低性能
            label_visible: 显示标签。

            echo: 是否打开显示当前添加点的进度
            delete_duplicate_targets: 如果场景里面有这个卫星，是否删除
        Returns:


        其他的命令参考：

        --To set animation times so that animation starts at 12 a.m. on November 2, 2011, the step time is every 10 seconds, the refresh delta is 0.5 seconds and refresh mode is set to use the refresh delta:
        ret=stkExec(cons,'SetAnimation * StartTimeOnly "1 Nov 2000 01:02:00.00" TimeStep 10 RefreshDelta 0.5 RefreshMode RefreshDelta')

        ret=stkExec(cons,'New / */Satellite sat1')
        ret=stkExec(cons,'SetState */Satellite/sat1 Classical J2Perturbation "1 Nov 2000 00:00:00.00" "1 Nov 2000 04:00:00.00" 60 J2000 "1 Nov 2000 00:00:00.00" 7163000.137079 0.0 98.5 0.0 139.7299 360.0')

        ret=stkExec(cons,'New / */Satellite sat2')
        ret=stkExec(cons,'SetState */Satellite/sat2 TLE "1 10637U 78012A 04113.40484266 .00000000 00000-0 10000-3 0 9453" "2 10637 40.1939 40.6913 1575431 119.1582 257.4005 1.00053877 48625" TimePeriod "1 Jun 2004 12:00:00.00" "1 Jun 2004 18:00:00.00"')

        """
        # 关闭stk中的界面显示，提高添加点的性能
        if not stk_sync_show:
            self.root.BeginUpdate()

        # 获取场景中现有的卫星
        obj_type = "Satellite"
        existing_satellites = {}
        self.delete_objects("Satellite", delete_all=True)
        # for path in self.get_objects(obj_type):
        #     if path != "None":
        #         stk_satellite = self.root.GetObjectFromPath(path)
        #         existing_satellites[stk_satellite.InstanceName] = path

        # # 处理卫星删除和过滤
        # satellites_to_create = []
        # for sate in satellite_info:
        #     if sate.name in existing_satellites:
        #         if delete_duplicate_targets:
        #             # 如果需要删除重复的卫星，则删除场景中的卫星
        #             target = self.root.GetObjectFromPath(existing_satellites[sate.name])
        #             target.Unload()
        #             satellites_to_create.append(sate)
        #     else:
        #         # 如果场景中没有这个卫星，则添加到待创建列表
        #         satellites_to_create.append(sate)

        # 创建新的卫星
        for sate in tqdm(satellite_info, desc="Adding satellites"):
            satellite = self.scenario.Children.New(AgESTKObjectType["eSatellite"], str(sate.name))
            # satellite.Graphics.LabelVisible = False

            if hpop:
                satellite.SetPropagatorType(AgEVePropagatorType.ePropagatorHPOP)
                satellite.Propagator.Step = 60
                keplerian = satellite.Propagator.InitialState.Representation.ConvertTo(
                    AgEOrbitStateType["eOrbitStateClassical"]
                )
                keplerian.SizeShapeType = AgEClassicalSizeShape["eSizeShapeSemimajorAxis"]
                keplerian.LocationType = AgEClassicalLocation["eLocationMeanAnomaly"]
                keplerian.Orientation.AscNodeType = AgEOrientationAscNode["eAscNodeRAAN"]

                # Assign the perigee and apogee altitude values:
                keplerian.SizeShape.SemiMajorAxis = sate.orbit_elements.semi_axis  # km
                keplerian.SizeShape.Eccentricity = sate.orbit_elements.eccentricity  # km

                # Assign the other desired orbital parameters:
                keplerian.Orientation.Inclination = sate.orbit_elements.inclination  # deg
                keplerian.Orientation.ArgOfPerigee = sate.orbit_elements.arg_of_perigee  # deg
                keplerian.Orientation.AscNode.Value = sate.orbit_elements.raan  # deg
                keplerian.Location.Value = sate.orbit_elements.mean_anomaly  # deg

                # Apply the changes made to the satellite's state and propagate:
                satellite.Propagator.InitialState.Representation.Assign(keplerian)

                forceModel = satellite.Propagator.ForceModel
                # 使用动态获取的STK安装路径
                gravity_file = os.path.join(
                    str(self.stk_install_path), "STKData", "CentralBodies", "Earth", "WGS84_EGM96.grv"
                )
                # hpop预设参数
                forceModel.CentralBodyGravity.File = gravity_file
                forceModel.CentralBodyGravity.MaxDegree = 21
                forceModel.CentralBodyGravity.MaxOrder = 21
                forceModel.Drag.Use = 1
                forceModel.Drag.DragModel.Cd = 0.01
                forceModel.Drag.DragModel.AreaMassRatio = 0.01
                forceModel.SolarRadiationPressure.Use = 0

                integrator = satellite.Propagator.Integrator
                integrator.DoNotPropagateBelowAlt = -1e6
                integrator.IntegrationModel = 3
                integrator.StepSizeControl.Method = 1
                integrator.StepSizeControl.ErrorTolerance = 1e-13
                integrator.StepSizeControl.MinStepSize = 0.1
                integrator.StepSizeControl.MaxStepSize = 30
                integrator.Interpolation.Method = 1
                integrator.Interpolation.Order = 7

                satellite.Propagator.Propagate()
            else:
                # propagator = satellite.Propagator
                # #  XPosition:float, YPosition:float, ZPosition:float, XVelocity:float, YVelocity:float, ZVelocity:float
                # # 这里暂时不用，只使用六根数
                # propagator.InitialState.Representation.AssignCartesian(
                #     AgECoordinateSystem["eCoordinateSystemTrueOfDate"],
                #     sate.position[0],
                #     sate.position[1],
                #     sate.position[2],
                #     sate.velocity[0],
                #     sate.velocity[1],
                #     sate.velocity[2],
                # )
                # propagator.Propagate()
                satellite.SetPropagatorType(AgEVePropagatorType.ePropagatorJ2Perturbation)
                keplerian = satellite.Propagator.InitialState.Representation.ConvertTo(
                    AgEOrbitStateType["eOrbitStateClassical"]
                )
                keplerian.SizeShapeType = AgEClassicalSizeShape["eSizeShapeSemimajorAxis"]
                keplerian.LocationType = AgEClassicalLocation["eLocationMeanAnomaly"]
                keplerian.Orientation.AscNodeType = AgEOrientationAscNode["eAscNodeRAAN"]

                # Assign the perigee and apogee altitude values:
                keplerian.SizeShape.SemiMajorAxis = sate.orbit_elements.semi_axis  # km
                keplerian.SizeShape.Eccentricity = sate.orbit_elements.eccentricity  # km

                # Assign the other desired orbital parameters:
                keplerian.Orientation.Inclination = sate.orbit_elements.inclination  # deg
                keplerian.Orientation.ArgOfPerigee = sate.orbit_elements.arg_of_perigee  # deg
                keplerian.Orientation.AscNode.Value = sate.orbit_elements.raan  # deg
                keplerian.Location.Value = sate.orbit_elements.mean_anomaly  # deg

                # Apply the changes made to the satellite's state and propagate:
                satellite.Propagator.InitialState.Representation.Assign(keplerian)
                satellite.Propagator.Propagate()

            # show pass
            passdata = satellite.Graphics.PassData
            groundTrack = passdata.GroundTrack
            groundTrack.SetLeadDataType(AgELeadTrailData["eDataAll"])
            groundTrack.SetTrailSameAsLead()

        # 结束更新，打开界面显示
        if not stk_sync_show:
            self.root.EndUpdate()

    def add_sensor_attach_to_all_satellite(self, sensor_params, label_visible=True, stk_sync_show=True):
        """为所有卫星对象添加传感器。

        该函数遍历场景中的所有卫星对象，并为每个卫星添加指定参数的传感器。
        如果卫星已有载荷，会先删除现有载荷再添加新的载荷。

        Args:
            sensor_params (PayloadInfo): 传感器的参数配置，包含名称、锥角、约束等信息。
            label_visible (bool, optional): 是否在STK中显示标签。Defaults to True.
            stk_sync_show (bool, optional): 是否在STK中同步显示操作。Defaults to True.

        Returns:
            None
        """
        obj_type = "Satellite"
        obj_paths = self.get_objects(obj_type)
        for path in tqdm(obj_paths, desc="Adding sensors"):
            if path != "None":
                _satellite = self.root.GetObjectFromPath(path)
                # 检查卫星是否已有载荷
                existing_sensors = self.get_objects("Sensor")
                for sensor_path in existing_sensors:
                    if sensor_path != "None" and path in sensor_path:
                        # 如果找到该卫星的载荷，先删除
                        sensor = self.root.GetObjectFromPath(sensor_path)
                        sensor.Unload()

                # 添加新的载荷
                sensor_name = "MCP_Created_Sensor"
                sensor = _satellite.Children.New(AgESTKObjectType["eSensor"], sensor_name)
                # def SetPatternSimpleConic(self, ConeAngle:typing.Any, AngularResolution:typing.Any) -> "IAgSnSimpleConicPattern":
                # 半张角30°，角分辨率0.5°。
                sensor.SetPatternType(AgESnPattern["eSnSimpleConic"])
                # todo 角分辨率默认0.5°
                angular_resolution = 0.5
                sensor.CommonTasks.SetPatternSimpleConic(
                    sensor_params.sensor_cone_half_angle, angular_resolution
                )
                # 可见性约束与之前介绍的卫星对象、地面站对象使用类似，这里只给个视线的例子
                senConstraints = sensor.AccessConstraints
                LOS = senConstraints.AddConstraint(AgEAccessConstraints["eCstrLOSSunExclusion"])
                LOS.Angle = sensor_params.los_angle

                # 添加光照条件约束
                light = senConstraints.AddConstraint(AgEAccessConstraints["eCstrLighting"])
                light.Condition = AgECnstrLighting[sensor_params.light_condition]

    def add_line_target(
        self,
        points: List,
        stk_sync_show=True,
        delete_origin_targets=False,
        echo=False,
        line_name="Line",
    ):
        """
        增加Line目标到STK
        Args:
            stk_sync_show: 在STK中同步显示，会降低性能
            points: 点结构
                [{
                  'latitude':21,
                  'longitude':123
                },{},{}]
            echo: 是否打开显示当前添加点的进度
        Returns:

        """
        start_time = time.time()
        # 关闭stk中的界面显示，提高添加点的性能
        if not stk_sync_show:
            self.root.BeginUpdate()
        # delete all exists targets before insert new target
        if delete_origin_targets:
            self.delete_objects(obj_type="LineTarget", delete_all=True)

        line_target = self.scenario.Children.New(AgESTKObjectType["eLine"], line_name)  # eLineTarget.
        line_target.Graphics.Color = 16777215  # 白色
        line_target.Graphics.LineWidth = 1.5  # 白色
        for idx, point in enumerate(points):
            line_target.Points.Add(point["latitude"], point["longitude"])

            if echo:
                print(
                    f"\r Process: {idx / len(points) * 100:.2f}%, add Line Target,"
                    f" Longitude={point['longitude']}, Latitude={point['latitude']}",
                    end="",
                )

        end_time = time.time()
        execution_time = end_time - start_time

        # 结束更新，打开界面显示
        if not stk_sync_show:
            self.root.EndUpdate()

    def get_objects(self, obj_type: str) -> List:
        """
        获取STK中指定类型的所有对象名称列表。

        Args:
            obj_type (str): 对象类型，如'Satellite'、'Target' 等。

        Returns:
            List[str]: 返回一个包含所有指定类型对象名称的字符串列表。

        """
        obj_items = self.root.ExecuteCommand(f"ShowNames * Class {obj_type}")
        return obj_items[0].strip().split(" ")

    def get_satellite_sensor(self) -> List:
        """
        获取STK中所有传感器的列表，包含卫星名称、传感器名称和完整路径。

        Args:
            无

        Returns:
            List[Dict[str, str]]: 包含传感器信息的列表，每个元素是一个字典，包含以下键：
                - "sensor": 传感器的名称
                - "satellite": 所属卫星的名称
                - "full_path": 传感器的完整路径

        """
        sensor_paths = self.get_objects("Sensor")
        sensors_list = []
        for path in sensor_paths:
            # 分割路径以获取卫星和传感器的部分
            parts = path.split("/")

            # 假设卫星名称在'Satellite/'之后，紧接着的下一个部分
            satellite_index = parts.index("Satellite") + 1
            # 假设传感器名称在'Sensor/'之后，紧接着的下一个部分
            sensor_index = parts.index("Sensor") + 1
            sensors_list.append(
                {
                    "sensor": parts[sensor_index],
                    "satellite": parts[satellite_index],
                    "full_path": path,
                }
            )
        return sensors_list

    def delete_objects(self, obj_type: str, delete_all=False, obj_list=None, delay=0):
        """
        删除指定类型的目标
        Args:
            obj_type: "Target"/"Sensor"/"Satellite" 等，但是也可能输入eTarget等
            obj_list:  数据结构与`target_list`一致
            delete_all: Boolean
            delay: 延迟删除目标

        Returns:

        """
        if obj_type.startswith("e"):
            obj_type = obj_type[1:]

        if obj_list is None:
            obj_list = []

        obj_paths = self.get_objects(obj_type)
        for path in obj_paths:
            if path != "None":
                if delete_all:
                    target = self.root.GetObjectFromPath(path)
                    target.Unload()
                else:
                    # 分割路径以获取卫星和传感器的部分
                    parts = path.split("/")

                    # 假设传感器名称在'obj_type/'之后，紧接着的下一个部分
                    obj_name = parts[parts.index(obj_type) + 1]
                    matches = [obj["name"] for obj in obj_list if obj["name"] == obj_name]
                    if matches:
                        target = self.root.GetObjectFromPath(path)
                        target.Unload()
                        time.sleep(delay)

    def get_targets_to_satellites_access(
        self,
        scenario_begin_time: str,
        scenario_end_time: str,
        sensor_info: PayloadInfo,
        target_list: List[ObservationTargetInfo],
        satellite_info: List[SatelliteInfo],
        revisit_epoch: int,
        stk_sync_show: bool = False,
    ) -> List[RevisitAnalysisInfo]:
        """
        计算所有目标对所有传感器的可见性。

        Args:
            scenario_begin_time (str): 场景开始时间。
            scenario_end_time (str): 场景结束时间。
            target_list (List[ObservationTargetInfo]): 目标列表，每个目标包含名称、纬度、经度和高度。
            sensor_info (PayloadInfo): 传感器信息
            satellite_info (List[SatelliteInfo]): 卫星信息
            revisit_epoch (int): 重访计算周期（天）。
            stk_sync_show (bool, optional): 是否同步显示STK中的界面更新。
        Returns:
            List[RevisitAnalysisInfo]: 可见性计算结果的列表，每个 `RevisitAnalysisInfo` 对象包含以下信息：
                - location_name (str): 观测点名称。
                - satellite_name (str): 卫星名称。
                - latitude (float): 纬度（°）。
                - longitude (float): 经度（°）。
                - access_events (Optional[List[STKAccessEvent]]): 详细访问数据。
                - revisit_epoch (float): 重访计算周期（天）。
                - avg_revisit (float): 平均重访时间（小时）。
                - max_revisit (float): 最大重访间隔（小时）。
                - min_revisit (float): 最小重访间隔（小时）。
                - revisit_time_unit (str): 重访时间单位。
                - revisit_constraints (STKRevisitConstraints): 优化约束条件。
        """
        # 关闭stk中的界面显示，提高添加点的性能
        if not stk_sync_show:
            self.root.BeginUpdate()

        # 根据卫星信息先计算这个卫星的最优gsd
        gsd_info = {}
        for sate in satellite_info:
            min_range = sate.orbit_elements.semi_axis - self.earth_radius
            max_elevation = 90 * math.pi / 180
            gsd = round(sensor_info.gsd_factor * min_range / math.sqrt(math.sin(max_elevation)), 3)
            gsd_info.update({sate.name: gsd})

        # 计算卫星视场角对应的地面覆盖幅宽
        propagation_data_ecef = self.get_satellite_ecef_by_time_shift(
            start_time_shift=0,  # start_time_shift
            period=12,
            step=1,
            ret_single_point=True,
        )
        imaging_swath = self.calculate_sensor_fov_imaging_swath(
            sensor_info=sensor_info, satellite_ecef_data=propagation_data_ecef
        )

        # 计算对目标的可见性
        # ic(target_list)
        revisit_analysis_list: List[RevisitAnalysisInfo] = []

        # 重置场景时间，计算可见性
        sensors_list = self.get_satellite_sensor()
        self.set_scenario_time_period(scenario_begin_time, scenario_end_time)

        targets_path = {}
        _targets_path = self.get_objects("Target")
        # 找sensor对应的path
        for path in _targets_path:
            # 分割路径以获取卫星和传感器的部分
            parts = path.split("/")
            # 假设传感器名称在'obj_type/'之后，紧接着的下一个部分
            obj_name = parts[parts.index("Target") + 1]
            targets_path[obj_name] = path

        # 计算所有sensor对所有target的可见性
        # 目前仅支持一个卫星+一个传感器的组合情况
        for i in range(len(target_list)):
            for sensor in tqdm(sensors_list, desc="Processing revisit_analysis"):
                revisit_analysis_list.append(
                    RevisitAnalysisInfo(
                        target=target_list[i],
                        satellite_name=sensor["satellite"],
                        imaging_swath=imaging_swath[sensor["satellite"]],
                        access_events=self.compute_access(
                            sensor, targets_path[target_list[i].target_name], sensor_info
                        ),
                        revisit_epoch=revisit_epoch,
                        gsd=gsd_info[sensor["satellite"]],
                        avg_revisit=0.0,
                        max_revisit=0.0,
                        min_revisit=0.0,
                        revisit_time_unit="hour",
                        revisit_constraints=RevisitAnalysisConstraints(),
                    )
                )

        for idx, revisit_analysis in enumerate(revisit_analysis_list):
            # 按 'start_time_dt' 排序
            access_data = []
            if revisit_analysis.access_events:
                for access_event in revisit_analysis.access_events:
                    access_data.append(access_event.__dict__)
                access_data.sort(key=lambda _x: _x["start_timestamp"])

            # 计算重访时间间隔
            time_diffs = []

            # 计算两次访问的时间间隔：本次的 start_time 减去上次的 end_time
            for i in range(1, len(access_data)):
                end_time_prev = access_data[i - 1]["end_timestamp"]
                start_time_curr = access_data[i]["start_timestamp"]
                # 计算时间间隔
                time_diff = start_time_curr - end_time_prev  # 返回间隔的秒数
                time_diffs.append(time_diff)

            # 计算最大、最小和平均重访时间，单位s
            try:
                revisit_analysis_list[idx].avg_revisit = round(sum(time_diffs) / len(time_diffs) / 3600, 2)
                revisit_analysis_list[idx].max_revisit = round(max(time_diffs) / 3600, 2)
                revisit_analysis_list[idx].min_revisit = round(min(time_diffs) / 3600, 2)
            except Exception as e:
                print(f"计算重访出错，错误信息：{e}")
                revisit_analysis_list[idx].avg_revisit = 99999
                revisit_analysis_list[idx].max_revisit = 99999
                revisit_analysis_list[idx].min_revisit = 99999
            # 因为后面access_events就不用到了，为了便于查看结果，把access_events的冗余信息去掉
            # revisit_analysis.access_events = None

        # 结束更新，打开界面显示
        if not stk_sync_show:
            self.root.EndUpdate()
        return revisit_analysis_list

    def compute_access(
        self, sensor_path: dict, target_path: str, sensor_info: PayloadInfo
    ) -> List[STKAccessEvent]:
        """
        计算sensor_path对target_path的可见性，并返回结果列表
        Args:
            sensor_path (dict): 星上载荷的path
                "sensor": parts[sensor_index],
                "satellite": parts[satellite_index],
                "full_path": path
            target_path (str): target 的 path
            sensor_info (PayloadInfo): 传感器信息

        Returns:
            List[STKAccessEvent]: 包含结果的类
                end_time: str
                end_timestamp: int
                start_time: str
                start_timestamp: int
                satellite: str
                sensor: str
                gsd: float
                max_elevation: float
                min_range: float
        """
        access_list = []
        from_target = self.root.GetObjectFromPath(target_path)
        to_sensor = self.root.GetObjectFromPath(sensor_path["full_path"])

        access = from_target.GetAccessToObject(to_sensor)

        # Compute access
        access.ComputeAccess()  # 时间花费较多
        access_intervals = access.ComputedAccessIntervalTimes

        # 如果有可见弧段，则计算可见的elevation以及range，并计算gsd
        if access_intervals.Count:
            access_intervals_time_cell = access_intervals.ToArray(0, -1)

            for time_cell in access_intervals_time_cell:
                access_aer = (
                    access.DataProviders.Item("AER Data")
                    .Group.Item("VVLH CBF")
                    .Exec(time_cell[0], time_cell[1], 5)
                )
                max_elevation = max(access_aer.DataSets.GetDataSetByName("Elevation").GetValues())
                min_range = min(access_aer.DataSets.GetDataSetByName("Range").GetValues())

                # compute the GSD   *地面采样距离，较小的 GSD 表示更高的分辨率
                gsd = round(
                    sensor_info.gsd_factor * min_range / math.sqrt(math.sin(Tools.deg2rad(max_elevation))),
                    3,
                )

                access_list.append(
                    STKAccessEvent(
                        start_time=time_cell[0],
                        end_time=time_cell[1],
                        max_elevation=round(max_elevation, 3),
                        min_range=round(min_range, 3),
                        gsd=gsd,
                        sensor=sensor_path["sensor"],
                        satellite_name=sensor_path["satellite"],
                        start_timestamp=Tools.get_timestamp_by_date_string(time_cell[0]),
                        end_timestamp=Tools.get_timestamp_by_date_string(time_cell[1]),
                    )
                )

        return access_list


    def get_satellites_mutual_access(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        计算场景中所有卫星两两之间的可见性。

        Returns:
            Dict[str, List[Tuple[str, str, float]]]:
                键为"卫星A-卫星B"，值为可见弧段的(开始时间, 结束时间, 持续时长(秒))元组列表。
        """
        paths = self.get_objects("Satellite")
        result = {}
        total = (len(paths) * (len(paths) - 1)) // 2
        with tqdm(total=total, desc="卫星两两可见性计算") as pbar:
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    sat_a = self.root.GetObjectFromPath(paths[i])
                    sat_b = self.root.GetObjectFromPath(paths[j])
                    access = sat_a.GetAccessToObject(sat_b)
                    access.ComputeAccess()
                    intervals = access.ComputedAccessIntervalTimes
                    key = f"{sat_a.InstanceName}-{sat_b.InstanceName}"
                    result[key] = []
                    if intervals.Count:
                        for t0, t1 in intervals.ToArray(0, -1):
                            # 将时间字符串转换为datetime对象
                            t0_dt = datetime.datetime.strptime(t0, "%d %b %Y %H:%M:%S.%f")
                            t1_dt = datetime.datetime.strptime(t1, "%d %b %Y %H:%M:%S.%f")
                            # 计算持续时长（秒）
                            duration = (t1_dt - t0_dt).total_seconds()
                            result[key].append((t0, t1, duration))
                    pbar.update(1)
        return result

    def get_satellites_to_missiles_access(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        计算所有卫星传感器对目标的可见性。

        Returns:
            Dict[str, List[Tuple[str, str, float]]]:
                键为"卫星-目标"，值为可见弧段的(开始时间, 结束时间, 持续时长(秒))元组列表。
        """
        # 获取所有卫星的传感器
        sensors_list = self.get_satellite_sensor()
        tgt_paths = self.get_objects("Missile")
        result = {}
        total = len(sensors_list) * len(tgt_paths)
        with tqdm(total=total, desc="卫星传感器对目标可见性计算") as pbar:
            for sensor_info in sensors_list:
                if sensor_info["full_path"] == "None":
                    continue
                sensor = self.root.GetObjectFromPath(sensor_info["full_path"])
                for tgt_path in tgt_paths:
                    if tgt_path == "None":
                        continue
                    try:
                        tgt = self.root.GetObjectFromPath(tgt_path)
                        access = tgt.GetAccessToObject(sensor)
                        access.ComputeAccess()
                        intervals = access.ComputedAccessIntervalTimes
                        key = f"{sensor_info['satellite']}-{tgt.InstanceName}"
                        result[key] = []
                        if intervals.Count:
                            for t0, t1 in intervals.ToArray(0, -1):
                                # 将时间字符串转换为datetime对象
                                t0_dt = datetime.datetime.strptime(t0, "%d %b %Y %H:%M:%S.%f")
                                t1_dt = datetime.datetime.strptime(t1, "%d %b %Y %H:%M:%S.%f")
                                # 计算持续时长（秒）
                                duration = (t1_dt - t0_dt).total_seconds()
                                result[key].append((t0, t1, duration))
                    except Exception as e:
                        print(f"处理目标 {tgt_path} 时出错: {str(e)}")
                        continue
                    pbar.update(1)
        return result

    def get_missile_ecef_by_time_shift(
        self, start_time_shift: float, period: float, step=0.125, ret_single_point=True, instance_names=None
    ) -> Dict[str, List[Tuple[str, float, float, float]]]:
        """
        获取指定导弹在指定时间段内的ECEF坐标。

        Args:
            start_time_shift (float): 相对于场景开始时间的偏移量，单位为秒
            period (float): 计算的时间段长度，单位为秒
            step (float, optional): 时间采样步长，单位为秒，默认为0.125秒
            ret_single_point (bool, optional): 是否只返回第一个时间点的坐标。
                默认为True，仅返回第一个点
            instance_names (List[str], optional): 指定要获取坐标的导弹实例名称列表。
                如果为None，则获取所有导弹的坐标

        Returns:
            dict: 以导弹名称为键的字典，值为导弹位置数据列表。
                每个列表包含元组 (时间, x坐标, y坐标, z坐标)，其中：
                - t: STK时间格式字符串，如 '22 Mar 2025 06:15:58.220'
                - x, y, z: ECEF坐标系中的位置，单位为千米

                若ret_single_point为True，则每个导弹的列表仅包含一个元组
                若ret_single_point为False，则包含整个时间段内按step采样的所有点
        """
        paths = self.get_objects("Missile")
        resp = {}
        for path in paths:
            if path == "None":
                continue
            missile = self.root.GetObjectFromPath(path)
            missile_name = missile.InstanceName
            
            # 如果指定了instance_names，只处理指定的导弹
            if instance_names is not None and missile_name not in instance_names:
                # print(f"跳过missile: {missile_name}")
                continue
                
            print(f"处理missile: {missile_name}")
            orbit_begin_time = Tools.get_date_string_by_timestamp(
                Tools.get_ms_timestamp_by_date_string(self.scenario_begin_time) + start_time_shift
            )
            orbit_end_time = Tools.get_date_string_by_timestamp(
                Tools.get_ms_timestamp_by_date_string(self.scenario_begin_time) + start_time_shift + period
            )
            ic(orbit_begin_time, orbit_end_time)
            missileDP = missile.DataProviders.Item("Cartesian Position").Group.Item("Fixed")
            result = missileDP.Exec(orbit_begin_time, orbit_end_time, step)
            times = result.DataSets.GetDataSetByName("Time").GetValues()
            x_pos = result.DataSets.GetDataSetByName("x").GetValues()
            y_pos = result.DataSets.GetDataSetByName("y").GetValues()
            z_pos = result.DataSets.GetDataSetByName("z").GetValues()
            _temp = []
            ic(_temp)
            for t, x, y, z in zip(times, x_pos, y_pos, z_pos):
                _temp.append(
                    (
                        Tools.get_date_string_by_timestamp(Tools.get_ms_timestamp_by_date_string(t)),
                        round(x, 3),
                        round(y, 3),
                        round(z, 3),
                    )
                )
                if ret_single_point:
                    break
            resp.update({missile_name: _temp})
            ic(resp)
        return resp
