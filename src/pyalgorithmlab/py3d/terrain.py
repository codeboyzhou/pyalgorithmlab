import math

import numpy as np
from scipy.ndimage import gaussian_filter

from pyalgorithmlab.py3d.types import Peak, Point


def init_mountain_terrain(xx: np.ndarray, yy: np.ndarray, peaks: list[Peak]) -> np.ndarray:
    """
    初始化山脉地形

    Args:
        xx: X网格坐标
        yy: Y网格坐标
        peaks: 山峰参数列表

    Returns:
        生成的山脉地形对应的Z网格坐标
    """
    # 初始化Z网格
    zz = np.zeros_like(xx)

    # 生成每个山峰的山脉地形
    for peak in peaks:
        zz += peak.amplitude * np.exp(-((xx - peak.center_x) ** 2 + (yy - peak.center_y) ** 2) / (2 * peak.width**2))

    # 添加一些随机噪声和基础波动，增强山峰的真实性
    zz += 0.2 * np.sin(0.5 * np.sqrt(xx**2 + yy**2)) + 0.1 * np.random.normal(size=xx.shape)

    # 使用高斯滤波，保持山峰独立性的同时也保证平滑性
    zz = gaussian_filter(zz, sigma=3)

    # 确保Z网格的最小值为0，避免负高度
    zz = np.clip(zz, 0, None)

    return zz


def get_mountain_terrain_height(x: float, y: float, peak: Peak) -> float:
    """
    获取指定坐标点的山脉地形高度

    Args:
        x: X坐标
        y: Y坐标
        peak: 山峰参数

    Returns:
        该坐标点的山脉地形高度
    """
    z = 0
    z += peak.amplitude * np.exp(-((x - peak.center_x) ** 2 + (y - peak.center_y) ** 2) / (2 * peak.width**2))
    z += 0.2 * np.sin(0.5 * np.sqrt(x**2 + y**2)) + 0.1 * np.random.normal()
    z = gaussian_filter(z, sigma=3)
    return z.item() if z > 0 else 0


def check_point_collision(point: Point, peaks: list[Peak]) -> bool:
    """
    检查指定坐标点是否与山峰碰撞

    Args:
        point: 待检查的坐标点
        peaks: 山峰参数列表

    Returns:
        True: 碰撞
        False: 不碰撞
    """
    safe_distance = 0.5
    for peak in peaks:
        terrain_height = get_mountain_terrain_height(point.x, point.y, peak)
        if point.z <= terrain_height + safe_distance:
            return True
    return False


def check_line_segment_collision(start: Point, end: Point, peaks: list[Peak]) -> list[Point]:
    """
    使用平面投影检查指定线段是否与山峰碰撞

    将山峰剖面简化为半径为peak.width的圆
    通过二次方程求解线段与圆的交点得到结果

    线段方程：
    x = start.x + t * (end.x - start.x)
    y = start.y + t * (end.y - start.y)

    剖面圆方程：
    (x - peak.center_x)**2 + (y - peak.center_y)**2 = peak.width**2

    联立两个方程：
    A * t**2 + B * t + C = 0

    其中：
    A = (end.x - start.x)**2 + (end.y - start.y)**2
    B = 2 * ((end.x - start.x) * (start.x - peak.center_x) + (end.y - start.y) * (start.y - peak.center_y))
    C = (start.x - peak.center_x)**2 + (start.y - peak.center_y)**2 - peak.width**2

    求解这个一元二次方程即可得到参数t的值

    参考链接：https://thejinchao.github.io/blog/2025/02/SegmentCircle.html

    Args:
        start: 线段起点
        end: 线段终点
        peaks: 山峰参数列表

    Returns:
        线段与山峰的交点列表，为空表示二者没有相交，即没有碰撞
    """
    # 计算射线与每个山峰的交点
    intersection_points: list[Point] = []
    for peak in peaks:
        # 计算一元二次方程的系数
        a = (end.x - start.x) ** 2 + (end.y - start.y) ** 2
        b = 2 * ((end.x - start.x) * (start.x - peak.center_x) + (end.y - start.y) * (start.y - peak.center_y))
        c = (start.x - peak.center_x) ** 2 + (start.y - peak.center_y) ** 2 - peak.width**2

        # 计算判别式
        discriminant = b**2 - 4 * a * c

        # 如果判别式小于0，说明线段与山峰无交点
        if discriminant < 0:
            return intersection_points

        # 计算交点参数t1和t2
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)

        # 计算交点坐标
        if 0 <= t1 <= 1:
            intersection = Point(x=start.x + t1 * (end.x - start.x), y=start.y + t1 * (end.y - start.y), z=start.z)
            intersection_points.append(intersection)
        if 0 <= t2 <= 1:
            intersection = Point(x=start.x + t2 * (end.x - start.x), y=start.y + t2 * (end.y - start.y), z=start.z)
            intersection_points.append(intersection)

    return intersection_points
