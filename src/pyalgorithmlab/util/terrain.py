import numpy as np
from scipy.ndimage import gaussian_filter

from pyalgorithmlab.util.model.peak import Peak


def generate_simulated_mountain_peaks(x_grid: np.ndarray, y_grid: np.ndarray, peaks: list[Peak]) -> np.ndarray:
    """
    生成模拟山峰的地形图在Z网格的坐标

    Args:
        x_grid: X网格坐标
        y_grid: Y网格坐标
        peaks: 山峰参数列表

    Returns:
        生成的地形图在Z网格的坐标
    """
    # 初始化Z网格
    z_grid = np.zeros_like(x_grid)

    # 生成每个山峰的山脉地形
    for peak in peaks:
        z_grid += peak.amplitude * np.exp(
            -((x_grid - peak.center_x) ** 2 + (y_grid - peak.center_y) ** 2) / (2 * peak.width**2)
        )

    # 添加一些随机噪声和基础波动，增强山峰的真实性
    z_grid += 0.2 * np.sin(0.5 * np.sqrt(x_grid**2 + y_grid**2)) + 0.1 * np.random.normal(size=x_grid.shape)

    # 使用高斯滤波，保持山峰独立性的同时也保证平滑性
    z_grid = gaussian_filter(z_grid, sigma=3)

    return z_grid


def is_point_collision_detected(point: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> bool:
    """
    检测空间中的某个点是否与地形碰撞

    Args:
        point: 待检测的点坐标
        x_grid: X网格坐标
        y_grid: Y网格坐标
        z_grid: Z网格坐标

    Returns:
        bool: 如果发生碰撞则返回True，否则返回False
    """
    # 获取网格的实际坐标值范围
    x_vals = np.unique(x_grid)
    y_vals = np.unique(y_grid)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    x, y, z = point[0], point[1], point[2]

    # 超出地形范围视为碰撞
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return True

    # 在边界点上视为不碰撞
    if x == x_min or x == x_max or y == y_min or y == y_max or z == 0:
        return False

    # 计算实际坐标步长并获取网格索引
    x_step = x_vals[1] - x_vals[0]
    y_step = y_vals[1] - y_vals[0]
    x_index = np.clip(int((x - x_min) / x_step), 0, len(x_vals) - 1)
    y_index = np.clip(int((y - y_min) / y_step), 0, len(y_vals) - 1)

    # 获取地形高度并增加安全余量（10%地形高度）
    terrain_z = z_grid[y_index, x_index]  # 注意meshgrid的索引顺序是(y, x)
    safety_margin = 0.1 * terrain_z  # 增加10%的安全高度避免贴地穿模

    # 点高度低于地形高度+安全余量则判定为碰撞
    result = z < (terrain_z + safety_margin)

    return result.item()


def is_line_collision_detected(
    line_start_point: np.ndarray,
    line_end_point: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    num_sample_points: int = 20,
) -> list[tuple[float, float, float]]:
    """
    检测空间中的一条线段是否与地形碰撞

    Args:
        line_start_point: 线段起点坐标
        line_end_point: 线段终点坐标
        x_grid: X网格坐标
        y_grid: Y网格坐标
        z_grid: Z网格坐标
        num_sample_points: 采样点数量

    Returns:
        list[tuple[float, float, float]]: 如果发生碰撞则返回碰撞点列表，否则返回空列表
    """
    # 计算线的长度
    line_length = np.linalg.norm(line_end_point - line_start_point)

    # 计算线的方向向量
    line_direction = (line_end_point - line_start_point) / line_length

    # 对线段进行点采样
    step = line_length / num_sample_points

    # 采样点碰撞检测
    collision_points = []
    # 排除起点本身
    for i in range(1, num_sample_points + 1):
        # 计算当前点的坐标
        current_point = line_start_point + i * step * line_direction
        # 检查当前点是否与地形碰撞
        if is_point_collision_detected(current_point, x_grid, y_grid, z_grid):
            x, y, z = current_point[0], current_point[1], current_point[2]
            collision_points.append((x.item(), y.item(), z.item()))

    return collision_points
