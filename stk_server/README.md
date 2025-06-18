# STK 卫星计算工具

本文档提供了使用 STK 进行卫星相关计算的工具函数说明。

## 轨道计算

### orbit_info_analysis

进行卫星轨道传播，计算并返回指定时间段内的轨道数据。

```python
orbit_info_analysis(
    satellite_info: SatelliteInfo,
    step_size: int = 10,
    propagate_duration: int = 30
) -> OrbitData
```

-   `satellite_info`: `SatelliteInfo` 对象，包含卫星的名称、历元时间、轨道六根数等信息。
    例如:
    ```json
    {
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
    ```
-   `step_size` (可选): 轨道递推步长 (秒)，默认为 10 秒。
-   `propagate_duration` (可选): 轨道预报时长 (秒)，默认为 30 秒。
-   **返回**: `OrbitData` 对象，包含时间、纬度、经度和高度列表。

### target_revisit

计算卫星在指定周期内对一组目标的重访情况，包括访问时间窗口和重访统计信息。

```python
target_revisit(
    satellite_info: SatelliteInfo,
    sensor_info: PayloadInfo,
    target_list: List[ObservationTargetInfo],
    revisit_epoch: int = 30
) -> List[RevisitAnalysisInfo]
```

-   `satellite_info`: `SatelliteInfo` 对象，包含卫星信息。 (结构同 `orbit_info_analysis`)
-   `sensor_info`: `PayloadInfo` 对象，包含传感器类型、约束条件（如光照、视场角）等信息。
    例如:
    ```json
    {
        "sensor_type": "Optic",
        "gsd_factor": 2.0,
        "fov_angle": 1e-5,
        "light_condition": "eDirectSun",
        "los_angle": 30,
        "sensor_cone_half_angle": 30.0
    }
    ```
-   `target_list`: `ObservationTargetInfo` 对象列表，每个对象包含目标名称和坐标（纬度、经度、高度）。
    例如:
    ```json
    [{
        "target_name": "Beijing",
        "target_coords":[39.9042, 116.4074, 1]
    }, ...]
    ```
-   `revisit_epoch` (可选): 重访计算周期 (天)，默认为 30 天。
-   **返回**: `RevisitAnalysisInfo` 对象列表，每个对象包含对应目标的访问事件列表 (`access_events`) 和重访时间统计信息 (`avg_revisit`, `max_revisit`, `min_revisit`)。

### lighting_times

计算卫星在指定周期内的光照时间段（进入/离开阴影区的时间）。

```python
lighting_times(
    satellite_info: SatelliteInfo,
    epoch: int = 30
) # -> 返回类型未在代码中明确指定，根据函数名推测可能返回光照事件列表
```

-   `satellite_info`: `SatelliteInfo` 对象，包含卫星信息。 (结构同 `orbit_info_analysis`)
-   `epoch` (可选): 光照计算周期 (天)，默认为 30 天。
-   **返回**: 光照计算结果（具体结构需参考 `_lighting_times` 函数的实际返回值）。

### sun_beta_angle

计算卫星在指定周期内的太阳 Beta 角变化情况。

```python
sun_beta_angle(
    satellite_info: SatelliteInfo,
    epoch: int = 30
) -> SunBetaAngleInfo
```

-   `satellite_info`: `SatelliteInfo` 对象，包含卫星信息。 (结构同 `orbit_info_analysis`)
-   `epoch` (可选): 太阳 Beta 角计算周期 (天)，默认为 30 天。
-   **返回**: `SunBetaAngleInfo` 对象，包含时间列表、对应的 Beta 角值列表以及最小、最大、平均 Beta 角值。

## 数据结构说明

工具函数中使用了以下主要的数据结构（定义于 `data_class/state_class.py`）：

-   **`SatelliteInfo`**: 包含卫星名称 (`name`)、轨道历元时间 (`orbit_epoch_time`) 和轨道根数 (`orbit_elements: OrbitalElementsInfo`)。
-   **`OrbitalElementsInfo`**: 包含轨道六根数（半长轴 `semi_axis`、偏心率 `eccentricity`、倾角 `inclination`、升交点赤经 `raan`、近地点幅角 `arg_of_perigee`、平近点角 `mean_anomaly`）。
-   **`PayloadInfo`**: 包含传感器类型 (`sensor_type`)、焦距 (`focal_length`)、像元尺寸 (`pixel_size`)、成像幅宽 (`imaging_swath`)、光照条件 (`light_condition`)、视线角 (`los_angle`) 和视场半角 (`sensor_cone_half_angle`)。
-   **`ObservationTargetInfo`**: 包含目标名称 (`target_name`) 和目标坐标 (`target_coords`: 纬度、经度、高度元组）。
-   **`OrbitData`**: 包含轨道数据列表：时间 (`time`)、纬度 (`latitude`)、经度 (`longitude`)、高度 (`altitude`)。
-   **`RevisitAnalysisInfo`**: 包含单个目标的重访分析结果，如目标位置信息、访问事件列表 (`access_events: List[STKAccessEvent]`) 和重访统计（`avg_revisit`, `max_revisit`, `min_revisit`）。
-   **`STKAccessEvent`**: 包含单次访问事件的详细信息，如起止时间、卫星/传感器名称、地面分辨率 (`gsd`) 等。
-   **`SunBetaAngleInfo`**: 包含太阳 Beta 角随时间变化的数据列表 (`time`, `beta_angle`) 以及统计值（`min`, `max`, `avg`）。
