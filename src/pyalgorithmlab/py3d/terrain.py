import math

import numpy as np
from loguru import logger
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from pyalgorithmlab.py3d.types import Peak, Point


class Terrain:
    """地形类"""

    def __init__(self, xx: np.ndarray, yy: np.ndarray, peaks: list[Peak]) -> None:
        """
        初始化地形类

        Args:
            xx: X网格坐标
            yy: Y网格坐标
            peaks: 山峰列表
        """
        self.xx = xx
        self.yy = yy
        self.peaks = peaks
        self.height_interpolator = None

    def init_mountain_terrain(self) -> np.ndarray:
        """
        初始化山脉地形

        Returns:
            生成的山脉地形对应的Z网格坐标
        """
        # 初始化Z网格
        zz = np.zeros_like(self.xx)

        # 生成每个山峰的山脉地形
        for peak in self.peaks:
            zz += peak.amplitude * np.exp(
                -((self.xx - peak.center_x) ** 2 + (self.yy - peak.center_y) ** 2) / (2 * peak.width**2)
            )

        # 添加一些随机噪声和基础波动，增强山峰的真实性
        zz += 0.2 * np.sin(0.5 * np.sqrt(self.xx**2 + self.yy**2)) + 0.1 * np.random.normal(size=self.xx.shape)

        # 使用高斯滤波，保持山峰独立性的同时也保证平滑性
        zz = gaussian_filter(zz, sigma=3)

        # 确保Z网格的最小值为0，避免负高度
        zz = np.clip(zz, 0, None)

        # 构建高度插值函数，用于查询任意坐标点的地形高度
        self.height_interpolator = RegularGridInterpolator(
            points=(self.yy[:, 0], self.xx[0, :]),  # 注意：网格方向是 (y, x)
            values=zz,
            bounds_error=False,
            fill_value=zz.max() + 100,  # 填充为超高值，避免越界时漏判
        )

        return zz

    def get_mountain_terrain_height(self, x: float, y: float) -> float:
        """
        获取指定坐标点的山脉地形高度

        Args:
            x: X坐标
            y: Y坐标

        Returns:
            该坐标点的山脉地形高度
        """
        if self.height_interpolator is None:
            raise ValueError("self.height_interpolator is None. Please call init_mountain_terrain() first.")
        return self.height_interpolator([[y, x]])[0]  # 注意输入是 [[y, x]]

    def min_horizontal_distance_to_peak_centers(self, point: Point) -> float:
        """
        计算指定坐标点到所有山峰中心的最小水平距离

        Args:
            point: 待计算的坐标点

        Returns:
            该坐标点到所有山峰中心的最小水平距离
        """
        return min(math.sqrt((point.x - peak.center_x) ** 2 + (point.y - peak.center_y) ** 2) for peak in self.peaks)

    def check_point_collision(self, point: Point, safe_distance: float = 0.5) -> bool:
        """
        检查指定坐标点是否与地形碰撞

        Args:
            point: 待检查的坐标点
            safe_distance: 安全距离

        Returns:
            True: 碰撞
            False: 不碰撞
        """
        return point.z <= self.get_mountain_terrain_height(point.x, point.y) + safe_distance

    def check_line_segment_collision_points(
        self, start_point: Point, end_point: Point, safe_distance: float = 0.5
    ) -> list[Point]:
        """
        通过对线段进行中间点采样，检查两点之间的线段是否会与地形发生碰撞

        Args:
            start_point: 线段起点
            end_point: 线段终点
            safe_distance: 安全距离

        Returns:
            碰撞点列表，为空时表示线段与地形无碰撞
        """
        collision_points: list[Point] = []

        start = start_point.to_ndarray()
        end = end_point.to_ndarray()

        num_sample_points = 20
        t_values = np.linspace(0, 1, num_sample_points)

        for t in t_values:
            sample_point = start + t * (end - start)
            point = Point.from_ndarray(sample_point)
            if self.check_point_collision(point, safe_distance):
                collision_points.append(point)

        return collision_points

    def try_correct_collision_point(self, point: Point) -> Point:
        """
        尝试对碰撞点的坐标进行修正

        Args:
            point: 碰撞点坐标

        Returns:
            修正后的碰撞点坐标
        """
        # 没有碰撞不需要修正
        if not self.check_point_collision(point):
            return point

        # 计算地形高度范围
        x_min, x_max = self.xx.min(), self.xx.max()
        y_min, y_max = self.yy.min(), self.yy.max()
        z_max = self.get_mountain_terrain_height(x_max, y_max)

        # 定义坐标修正方向（只考虑水平方向）
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # 初始化搜索半径
        search_radius = 1

        # 初始化搜索次数
        attempt = 0
        max_attempts = 100

        while attempt < max_attempts:
            attempt += 1
            best_safe_point = None
            min_distance = float("inf")

            for dx, dy in directions:
                # 动态步长（随尝试次数增加搜索范围）
                step = search_radius if attempt < 20 else search_radius * (attempt / 20)
                # 保证每个方向搜索结果互不干扰
                new_point = point.clone()
                # 移动位置
                new_point.x += dx * step
                new_point.y += dy * step
                # 边界约束
                new_point.x = x_min if new_point.x < x_min else x_max if new_point.x > x_max else new_point.x
                new_point.y = y_min if new_point.y < y_min else y_max if new_point.y > y_max else new_point.y
                # 高度约束（保持在地形高度以上）
                new_point.z = max(new_point.z, self.get_mountain_terrain_height(new_point.x, new_point.y))
                # 每移动一次位置都检查是否已经安全
                if not self.check_point_collision(new_point):
                    # 计算新点与原始点的距离，选择最近安全点
                    distance = np.linalg.norm(new_point.to_ndarray() - point.to_ndarray())
                    if distance < min_distance:
                        min_distance = distance
                        best_safe_point = new_point

            # 找到最近安全点
            if best_safe_point is not None:
                logger.success(f"碰撞点{point}已被成功修正为安全点{best_safe_point}")
                return best_safe_point

            # 扩大搜索范围
            search_radius += 1

        # 没有搜索到任何安全点，返回到原始点，并抬升高度到最大海拔
        new_point = Point(x=point.x, y=point.y, z=z_max)
        logger.warning(f"未找到安全点，返回原始点，并抬升高度到最大海拔{new_point}")

        return new_point
