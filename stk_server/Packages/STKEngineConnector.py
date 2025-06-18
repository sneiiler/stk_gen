import os
import platform
import time
from typing import List

import numpy as np
from agi.stk12.stkengine import STKEngine
# from agi.stk12.stkobjects import (
#     AgEClassicalLocation,
#     AgEClassicalSizeShape,
#     AgECvBounds,
#     AgECvResolution,
#     AgEFmCompute,
#     AgEFmDefinitionType,
#     AgEOrientationAscNode,
#     AgESTKObjectType,
#     AgEVePropagatorType,
#     AgECoordinateSystem,
#     AgELeadTrailData,
# )
from Packages.stkobjects import (IAgScenario,
                                 AgESTKObjectType,
                                 AgEOrbitStateType,
                                 AgELeadTrailData,
                                 AgECoordinateSystem,
                                 AgEClassicalSizeShape,
                                 AgEVePropagatorType,
                                 AgEClassicalLocation,
                                 AgEOrientationAscNode,
                                 AgECnstrLighting,
                                 AgESnPattern,
                                 AgEAccessConstraints)
from agi.stk12.stkutil import AgEOrbitStateType
from icecream import ic

from Packages import Tools


class STKEngineConnector:
    def __init__(self):
        """
        初始化STKConnector类，创建STK应用实例并获取IAgStkObjectRoot接口

        Args:
            无参数

        Returns:
            无返回值

        """

        startTime = time.time()

        """
        SET TO TRUE TO USE ENGINE, FALSE TO USE GUI
        """
        if platform.system() == "Linux":
            # Only STK Engine is available on Linux
            useStkEngine = True
        else:
            # Change to true to run engine on Windows
            useStkEngine = False
        ############################################################################
        # Scenario Setup
        ############################################################################

        if useStkEngine:
            from agi.stk12.stkengine import STKEngine

            # Launch STK Engine with NoGraphics mode
            print("Launching STK Engine...")
            stk = STKEngine.StartApplication(noGraphics=True)

            # Create root object
            self.stk_root = stk.NewObjectRoot()

        else:
            from agi.stk12.stkdesktop import STKDesktop

            # Launch GUI
            print("Launching STK...")
            stk = STKDesktop.StartApplication(visible=True, userControl=True)

            # Get root object
            self.stk_root = stk.Root

        # Set date format
        self.stk_root.UnitPreferences.SetCurrentUnit("DateFormat", "UTCG")

        # Create new scenario
        print("Creating scenario...")
        self.stk_root.NewScenario("MCP_Created_Scenario")
        self.scenario = self.stk_root.CurrentScenario

        if not useStkEngine:
            # Graphics calls are not available when running STK Engine in NoGraphics mode
            self.stk_root.Rewind()

    def set_scenario_time(self, scenario_begin_time, scenario_end_time):
        # 设置场景开始时间，并将场景重置为开始

        self.scenario.SetTimePeriod(scenario_begin_time, scenario_end_time)

        self.stk_root.ExecuteCommand('Animate * Reset')

        self.scenario_begin_time = scenario_begin_time
        self.scenario_end_time = scenario_end_time

    def add_satellite(self, satellite_info, classsic_elements=True, label_visible=True, stk_sync_show=True,
                      delete_origin_targets=True):
        """
        增加卫星到STK
        Args:
            satellite_info:卫星列表
                {
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
                }
            stk_sync_show: 在STK中同步显示，会降低性能
            label_visible: 显示标签。

            echo: 是否打开显示当前添加点的进度
            delete_origin_targets: 是否删除原来的点
        Returns:


        其他的命令参考：

        --To set animation times so that animation starts at 12 a.m. on November 2, 2011, the step time is every 10 seconds, the refresh delta is 0.5 seconds and refresh mode is set to use the refresh delta:
        ret=stkExec(cons,'SetAnimation * StartTimeOnly "1 Nov 2000 01:02:00.00" TimeStep 10 RefreshDelta 0.5 RefreshMode RefreshDelta')

        ret=stkExec(cons,'New / */Satellite sat1')
        ret=stkExec(cons,'SetState */Satellite/sat1 Classical J2Perturbation "1 Nov 2000 00:00:00.00" "1 Nov 2000 04:00:00.00" 60 J2000 "1 Nov 2000 00:00:00.00" 7163000.137079 0.0 98.5 0.0 139.7299 360.0')

        ret=stkExec(cons,'New / */Satellite sat2')
        ret=stkExec(cons,'SetState */Satellite/sat2 TLE "1 10637U 78012A 04113.40484266 .00000000 00000-0 10000-3 0 9453" "2 10637 40.1939 40.6913 1575431 119.1582 257.4005 1.00053877 48625" TimePeriod "1 Jun 2004 12:00:00.00" "1 Jun 2004 18:00:00.00"')

        """
        start_time = time.time()
        # 关闭stk中的界面显示，提高添加点的性能
        if not stk_sync_show:
            self.stk_root.BeginUpdate()
        # delete all exists targets before insert new satellite
        if delete_origin_targets:
            self.delete_objects(obj_type="Satellite", delete_all=True)

        satellite = self.scenario.Children.New(AgESTKObjectType.eSatellite, str(satellite_info['name']))
        satellite.SetPropagatorType(AgEVePropagatorType.ePropagatorJ2Perturbation)
        # satellite.Graphics.LabelVisible = False

        if classsic_elements:
            # @todo 现在设置轨道参数有点问题，应该是使用TrueOfDate,这个设置出来是ICRF
            # Get orbit state
            keplerian = satellite.Propagator.InitialState.Representation.ConvertTo(
                AgEOrbitStateType['eOrbitStateClassical'])
            keplerian.SizeShapeType = AgEClassicalSizeShape['eSizeShapeSemimajorAxis']
            keplerian.LocationType = AgEClassicalLocation['eLocationMeanAnomaly']
            keplerian.Orientation.AscNodeType = AgEOrientationAscNode['eAscNodeRAAN']

            # Assign the perigee and apogee altitude values:
            keplerian.SizeShape.SemiMajorAxis = satellite_info['orbit_elements']['semi_axis']  # km
            keplerian.SizeShape.Eccentricity = satellite_info['orbit_elements']['eccentricity']  # km

            # Assign the other desired orbital parameters:
            keplerian.Orientation.Inclination = satellite_info['orbit_elements']['inclination']  # deg
            keplerian.Orientation.ArgOfPerigee = satellite_info['orbit_elements']['arg_of_perigee']  # deg
            keplerian.Orientation.AscNode.Value = satellite_info['orbit_elements']['raan']  # deg
            keplerian.Location.Value = satellite_info['orbit_elements']['mean_anomaly']  # deg

            # Apply the changes made to the satellite's state and propagate:
            satellite.Propagator.InitialState.Representation.Assign(keplerian)
            satellite.Propagator.Propagate()
        else:
            propagator = satellite.Propagator
            #  XPosition:float, YPosition:float, ZPosition:float, XVelocity:float, YVelocity:float, ZVelocity:float
            propagator.InitialState.Representation.AssignCartesian(
                AgECoordinateSystem['eCoordinateSystemTrueOfDate'],
                satellite_info['position'][0], satellite_info['position'][1],
                satellite_info['position'][2], satellite_info['velocity'][0],
                satellite_info['velocity'][1], satellite_info['velocity'][2])
            propagator.Propagate()

        # show pass
        passdata = satellite.Graphics.PassData
        groundTrack = passdata.GroundTrack
        groundTrack.SetLeadDataType(AgELeadTrailData['eDataAll'])
        groundTrack.SetTrailSameAsLead()
        if not stk_sync_show:
            self.stk_root.EndUpdate()

        end_time = time.time()
        execution_time = end_time - start_time

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
        if obj_type.startswith('e'):
            obj_type = obj_type[1:]

        if obj_list is None:
            obj_list = []

        obj_paths = self.get_objects(obj_type)
        for path in obj_paths:
            if path != 'None':
                if delete_all:
                    target = self.stk_root.GetObjectFromPath(path)
                    target.Unload()
                else:
                    # 分割路径以获取卫星和传感器的部分
                    parts = path.split('/')

                    # 假设传感器名称在'obj_type/'之后，紧接着的下一个部分
                    obj_name = parts[parts.index(obj_type) + 1]
                    matches = [obj["name"] for obj in obj_list if obj["name"] == obj_name]
                    if matches:
                        target = self.stk_root.GetObjectFromPath(path)
                        target.Unload()
                        time.sleep(delay)

    def get_objects(self, obj_type: str) -> List:
        """
        获取STK中指定类型的所有对象名称列表。

        Args:
            obj_type (str): 对象类型，如'Satellite'、'Target' 等。

        Returns:
            List[str]: 返回一个包含所有指定类型对象名称的字符串列表。

        """
        obj_items = self.stk_root.ExecuteCommand(f'ShowNames * Class {obj_type}')
        return obj_items[0].strip().split(' ')

    def get_sun_beta(self, scenario_begin_time, scenario_end_time):
        """计算卫星的太阳beta角。

        太阳beta角是太阳矢量与卫星轨道平面之间的角度，该角度对卫星的热控制和
        太阳能电池板效率有重要影响。

        Args:
            scenario_begin_time (str): 场景开始时间。
            scenario_end_time (str): 场景结束时间。

        Returns:
            dict: 包含beta角时间序列数据及统计信息的字典，数据结构:
                  {
                      "time": [...],       # 时间点列表
                      "beta_angle": [...], # 对应的beta角值列表(度)
                      "min": float,         # 最小beta角值(度)
                      "max": float,         # 最大beta角值(度)
                      "avg": float          # 平均beta角值(度)
                  }
        """

        path = self.get_objects("Satellite")
        satellite = self.root.GetObjectFromPath(path[0])

        # 使用STK的数据提供者获取太阳beta角数据
        # beta角通常在'Vectors'组的'Sun'类别下，提供太阳与轨道平面的角度
        data_provider = satellite.DataProviders.Item('Beta Angle').Exec(
            scenario_begin_time,
            scenario_end_time,
            600  # 采样间隔，单位为秒，可根据需要调整
        )

        beta_angle_time = data_provider.DataSets.GetDataSetByName('Time').GetValues()
        beta_angle = np.round(data_provider.DataSets.GetDataSetByName('Beta Angle').GetValues(), 2)

        result = {
            "time": list(beta_angle_time),
            "beta_angle": beta_angle.tolist(),
            "min": beta_angle.min().item(),
            "max": beta_angle.max().item(),
            "avg": np.round(beta_angle.mean(), 2).item(),
        }

        return result
        ############################################################################
        # Simple Access
        ############################################################################
        #
        # # Create faciliy
        # facility = scenario.Children.New(AgESTKObjectType.eFacility, "MyFacility")
        #
        # # Set position
        # facility.Position.AssignGeodetic(28.62, -80.62, 0.03)
        #
        # # Compute access between satellite and facility
        # print("\nComputing access...")
        # access = satellite.GetAccessToObject(facility)
        # access.ComputeAccess()
        #
        # # Get access interval data
        # self.stk_root.UnitPreferences.SetCurrentUnit("Time", "Min")
        # accessDataProvider = access.DataProviders.GetDataPrvIntervalFromPath("Access Data")
        # elements = ["Start Time", "Stop Time", "Duration"]
        # accessResults = accessDataProvider.ExecElements(
        #     scenario.StartTime, scenario.StopTime, elements
        # )
        #
        # startTimes = accessResults.DataSets.GetDataSetByName("Start Time").GetValues()
        # stopTimes = accessResults.DataSets.GetDataSetByName("Stop Time").GetValues()
        # durations = accessResults.DataSets.GetDataSetByName("Duration").GetValues()
        #
        # # Print data to console
        # print("\nAccess Intervals")
        # print(
        #     "{a:<29s}  {b:<29s}  {c:<14s}".format(
        #         a="Start Time", b="Stop Time", c="Duration (min)"
        #     )
        # )
        # for i in range(len(startTimes)):
        #     print(
        #         "{a:<29s}  {b:<29s}  {c:<4.2f}".format(
        #             a=startTimes[i], b=stopTimes[i], c=durations[i]
        #         )
        #     )
        #
        # print("\nThe maximum access duration is {a:4.2f} minutes.".format(a=max(durations)))
        #
        # # Print computation time
        # totalTime = time.time() - startTime
        # sectionTime = time.time() - splitTime
        # splitTime = time.time()
        # print(
        #     "--- Access computation: {a:4.3f} sec\t\tTotal time: {b:4.3f} sec ---".format(
        #         a=sectionTime, b=totalTime
        #     )
        # )
        #
        # ############################################################################
        # # Constellations and Chains
        # ############################################################################
        #
        # # Remove initial satellite
        # satellite.Unload()
        #
        # # Create constellation object
        # constellation = scenario.Children.New(
        #     AgESTKObjectType.eConstellation, "SatConstellation"
        # )
        #
        # # Insert the constellation of Satellites
        # numOrbitPlanes = 4
        # numSatsPerPlane = 8
        #
        # self.stk_root.BeginUpdate()
        # for orbitPlaneNum, RAAN in enumerate(
        #         range(0, 180, 180 // numOrbitPlanes), 1
        # ):  # RAAN in degrees
        #
        #     for satNum, trueAnomaly in enumerate(
        #             range(0, 360, 360 // numSatsPerPlane), 1
        #     ):  # trueAnomaly in degrees
        #
        #         # Insert satellite
        #         satellite = scenario.Children.New(
        #             AgESTKObjectType.eSatellite, f"Sat{orbitPlaneNum}{satNum}"
        #         )
        #
        #         # Select Propagator
        #         satellite.SetPropagatorType(AgEVePropagatorType.ePropagatorTwoBody)
        #
        #         # Set initial state
        #         twoBodyPropagator = satellite.Propagator
        #         keplarian = twoBodyPropagator.InitialState.Representation.ConvertTo(
        #             AgEOrbitStateType.eOrbitStateClassical.eOrbitStateClassical
        #         )
        #
        #         keplarian.SizeShapeType = AgEClassicalSizeShape.eSizeShapeSemimajorAxis
        #         keplarian.SizeShape.SemiMajorAxis = 8200  # km
        #         keplarian.SizeShape.Eccentricity = 0
        #
        #         keplarian.Orientation.Inclination = 60  # degrees
        #         keplarian.Orientation.ArgOfPerigee = 0  # degrees
        #         keplarian.Orientation.AscNodeType = AgEOrientationAscNode.eAscNodeRAAN
        #         keplarian.Orientation.AscNode.Value = RAAN  # degrees
        #
        #         keplarian.LocationType = AgEClassicalLocation.eLocationTrueAnomaly
        #         keplarian.Location.Value = trueAnomaly + (360 // numSatsPerPlane / 2) * (
        #                 orbitPlaneNum % 2
        #         )  # Stagger true anomalies (degrees) for every other orbital plane
        #
        #         # Propagate
        #         satellite.Propagator.InitialState.Representation.Assign(keplarian)
        #         satellite.Propagator.Propagate()
        #
        #         # Add to constellation object
        #         constellation.Objects.AddObject(satellite)
        #
        # self.stk_root.EndUpdate()
        # # Create chain
        # chain = scenario.Children.New(AgESTKObjectType.eChain, "Chain")
        #
        # # Add satellite constellation and facility
        # chain.Objects.AddObject(constellation)
        # chain.Objects.AddObject(facility)
        #
        # # Compute chain
        # chain.ComputeAccess()
        #
        # # Find satellite with most access time
        # chainDataProvider = chain.DataProviders.GetDataPrvIntervalFromPath("Object Access")
        # chainResults = chainDataProvider.Exec(scenario.StartTime, scenario.StopTime)
        #
        # objectList = []
        # durationList = []
        #
        # # Loop through all satellite access intervals
        # for intervalNum in range(chainResults.Intervals.Count - 1):
        #     # Get interval
        #     interval = chainResults.Intervals[intervalNum]
        #
        #     # Get data for interval
        #     objectName = interval.DataSets.GetDataSetByName("Strand Name").GetValues()[0]
        #     durations = interval.DataSets.GetDataSetByName("Duration").GetValues()
        #
        #     # Add data to list
        #     objectList.append(objectName)
        #     durationList.append(sum(durations))
        #
        # # Find object with longest total duration
        # index = durationList.index(max(durationList))
        # print(
        #     "\n{a:s} has the longest total duration: {b:4.2f} minutes.".format(
        #         a=objectList[index], b=durationList[index]
        #     )
        # )
        #
        # # Print computation time
        # totalTime = time.time() - startTime
        # sectionTime = time.time() - splitTime
        # splitTime = time.time()
        # print(
        #     "--- Chain computation: {a:4.2f} sec\t\tTotal time: {b:4.2f} sec ---".format(
        #         a=sectionTime, b=totalTime
        #     )
        # )
        #
        # ############################################################################
        # # Coverage
        # ############################################################################
        #
        # # Create coverage definition
        # coverageDefinition = scenario.Children.New(
        #     AgESTKObjectType.eCoverageDefinition, "CoverageDefinition"
        # )
        #
        # # Set grid bounds type
        # grid = coverageDefinition.Grid
        # grid.BoundsType = AgECvBounds.eBoundsCustomRegions
        #
        # # Add US shapefile to bounds
        # bounds = coverageDefinition.Grid.Bounds
        #
        # if platform.system() == "Linux":
        #     install_path = os.getenv("STK_INSTALL_DIR")
        # else:
        #     import winreg
        #
        #     registry = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        #     key = winreg.OpenKey(registry, r"Software\AGI\STK\12.0")
        #     install_path = winreg.QueryValueEx(key, "InstallHome")
        #
        # bounds.RegionFiles.Add(
        #     os.path.join(
        #         install_path[0],
        #         r"Data/Shapefiles/Countries/United_States_of_America\United_States_of_America.shp",
        #     )
        # )
        #
        # # Set resolution
        # grid.ResolutionType = AgECvResolution.eResolutionDistance
        # resolution = grid.Resolution
        # resolution.Distance = 75
        #
        # # Add constellation as asset
        # coverageDefinition.AssetList.Add("Constellation/SatConstellation")
        # coverageDefinition.ComputeAccesses()
        #
        # # Create figure of merit
        # figureOfMerit = coverageDefinition.Children.New(
        #     AgESTKObjectType.eFigureOfMerit, "FigureOfMerit"
        # )
        #
        # # Set the definition and compute type
        # figureOfMerit.SetDefinitionType(AgEFmDefinitionType.eFmAccessDuration)
        # definition = figureOfMerit.Definition
        # definition.SetComputeType(AgEFmCompute.eAverage)
        #
        # fomDataProvider = figureOfMerit.DataProviders.GetDataPrvFixedFromPath("Overall Value")
        # fomResults = fomDataProvider.Exec()
        #
        # minAccessDuration = fomResults.DataSets.GetDataSetByName("Minimum").GetValues()[0]
        # maxAccessDuration = fomResults.DataSets.GetDataSetByName("Maximum").GetValues()[0]
        # avgAccessDuration = fomResults.DataSets.GetDataSetByName("Average").GetValues()[0]
        #
        # # Computation time
        # totalTime = time.time() - startTime
        # sectionTime = time.time() - splitTime
        #
        # # Print data to console
        # print("\nThe minimum coverage duration is {a:4.2f} min.".format(a=minAccessDuration))
        # print("The maximum coverage duration is {a:4.2f} min.".format(a=maxAccessDuration))
        # print("The average coverage duration is {a:4.2f} min.".format(a=avgAccessDuration))
        # print(
        #     "--- Coverage computation: {a:0.3f} sec\t\tTotal time: {b:0.3f} sec ---".format(
        #         a=sectionTime, b=totalTime
        #     )
        # )
        #
        # # stkRoot.CloseScenario()
        # # stk.ShutDown()
        #
        # print("\nClosed STK successfully.")


if __name__ == '__main__':
    # 运行之前需要启动STK
    _satellite_info = {
        "name": "satellite1",
        "orbit_epoch_time": "22 Mar 2025 04:00:00.000",
        "orbit_elements": {
            "semi_axis": 6916.05,
            "eccentricity": 0,
            "inclination": 100,
            "arg_of_perigee": 0,
            "raan": 315,
            "mean_anomaly": 0
        }
    }
    _sensor_info = {
        "sensor_type": "Optic",
        "constraints":
            {
                "light.Condition": "eDirectSun",
                "LOS.Angle": 30
            },
        "sensor_cone_half_angle": 30.0
    }

    _target_list = [{"name": "Beijing", "lat": 39.9042, "lon": 116.4074}]

    epoch = 30  # days

    # resp = orbit_info_analysis(satellite_info)
    # resp = target_revisit(_satellite_info,
    #                       _sensor_info,
    #                       _target_list,
    #                       _epoch)
    # resp = lighting_times(_satellite_info,
    #                       _epoch)
    # 创建STK连接器
    stk_conn = STKEngineConnector()

    # 设置时间范围
    scenario_begin_time = _satellite_info['orbit_epoch_time']
    scenario_begin_time_timestamp = Tools.get_timestamp_by_date_string(scenario_begin_time)
    # 默认计算30天
    scenario_end_time = Tools.get_date_string_by_timestamp(
        scenario_begin_time_timestamp + epoch * 24 * 3600
    )

    # 配置STK场景
    stk_conn.set_scenario_time(scenario_begin_time, scenario_end_time)
    stk_conn.add_satellite(_satellite_info, delete_origin_targets=True)

    # 执行访问计算
    resp = stk_conn.get_sun_beta(scenario_begin_time, scenario_end_time)

    ic(resp)
