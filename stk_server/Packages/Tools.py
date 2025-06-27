# -*- coding: utf-8 -*-
# ***************************************
# * Author      : yinkaifeng
# * Email       : yinkaifeng@cast.casc
# * Create Time : 2024-6-12 09:27:27
# * Description ：工具类
# ***************************************
import math
import time
from datetime import datetime

# import pyproj
import icecream
import matplotlib.pyplot as plt

icecream.install()

# 地球长半轴半径
earth_radius = 6378137.0
# 地球扁率
f = 1 / 298.257223563


def deg2rad(deg):
    return deg * math.pi / 180


def current_timestamp():
    return int(str(datetime.now().timestamp()).split(".")[0])


def haversine_distance(p1, p2):
    """
    计算两点之间的Haversine距离 in Km
    :param p1: {"latitude":45.21, "longitude": 125.1}
    :param p2: {"latitude":45.21, "longitude": 125.1}
    :return:
    """
    # 地球半径
    r = 6371

    lat1, lon1 = deg2rad(p1["latitude"]), deg2rad(p1["longitude"])
    lat2, lon2 = deg2rad(p2["latitude"]), deg2rad(p2["longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = r * c

    return distance


def ecef_distance(p1, p2):
    """
    计算ecef坐标系两个点的距离， in Km
    Args:
        p1: {'x':-3032.841340, 'y':4887.240465, 'z':2747.094073 }  in Km
        p2: {'x':-2972.295151, 'y':5087.882663, 'z':2433.032611 } in Km

    Returns:
        distance:  in Km
    """
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    dz = p1["z"] - p2["z"]

    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance


# def ecef2lla(x, y, z):
#     """
#     根据地心地固系xyz坐标，转为经纬高
#     Args:
#         x: 心地固系xyz坐标
#         y: 心地固系xyz坐标
#         z: 心地固系xyz坐标

#     Returns:
#         经纬高
#     """
#     transformer = pyproj.Transformer.from_crs(
#         {"proj": "geocent", "ellps": "WGS84", "dataum": "WGS84"},
#         {"proj": "latlon", "ellps": "WGS84", "dataum": "WGS84"},
#     )
#     lon, lat, alt = transformer.transform(x, y, z, radians=False)
#     return lat, lon, alt


def lla2ecef(lat, lon, h):
    """
    经纬度坐标lla转为地心地固系ecef
    Args:
        lat: 纬度，deg
        lon: 经度，deg
        h: 海拔高度

    Returns:
        xyz: 单位 km
    """
    # 地球短半轴长度
    earth_radius_b = earth_radius * (1 - f)
    # 计算地球偏心率
    e = math.sqrt(1 - (earth_radius_b / earth_radius) ** 2)
    # 曲率半径
    N = earth_radius / math.sqrt(1 - (e * math.sin(deg2rad(lat))) ** 2)

    x = (N + h) * math.cos(deg2rad(lat)) * math.cos(deg2rad(lon))
    x = round(x, 5)
    y = (N + h) * math.cos(deg2rad(lat)) * math.sin(deg2rad(lon))
    y = round(y, 5)
    z = (N * (1 - e**2) + h) * math.sin(deg2rad(lat))
    z = round(z, 5)

    return x, y, z


def data_plot(x_data, y_data, title=""):
    """
    绘图曲线
    Args:
        x_data:
        y_data:
        title:

    Returns:

    """
    plt.plot(x_data[1:], y_data[1:])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


def get_timestamp_by_date_string(date_string: str):
    """
    将STK的场景时间转为时间戳
    Args:
        date_string: STK的场景时间，如：'31 May 2024 04:00:00.000'

    Returns:

    """
    dt_part = date_string.split(".")
    dt_obj = datetime.strptime(dt_part[0], "%d %b %Y %H:%M:%S")
    return int(time.mktime(dt_obj.timetuple()))


def get_ms_timestamp_by_date_string(date_string: str):
    """
    将STK的场景时间转为时间戳
    Args:
        date_string: STK的场景时间，如：'31 May 2024 04:00:00.000'

    Returns:
        1717099200.000
    """
    dt_part = date_string.split(".")
    dt_obj = datetime.strptime(dt_part[0], "%d %b %Y %H:%M:%S")
    
    # 将小数部分作为浮点数处理，确保正确转换各种长度的毫秒值
    ms_part = 0
    if len(dt_part) > 1 and dt_part[1]:
        ms_part = float("0." + dt_part[1])
        ms_part = int(ms_part * 1000)  # 转换为毫秒
        
    return int(time.mktime(dt_obj.timetuple())) + ms_part / 1000


def get_date_string_by_timestamp(timestamp: float):
    """
    将时间戳转为STK的场景时间
    Args:
        timestamp: 时间戳

    Returns:

    """
    dt_obj = datetime.fromtimestamp(timestamp)
    formatted_time = dt_obj.strftime("%d %b %Y %H:%M:%S.%f")[:-3]
    return formatted_time
