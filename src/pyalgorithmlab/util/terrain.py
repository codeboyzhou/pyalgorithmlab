import numpy as np

from pyalgorithmlab.util.model.peak import Peak


def generate_simulated_mountain_peaks(gridx: np.ndarray, gridy: np.ndarray, peaks: list[Peak]) -> np.ndarray:
    """
    生成模拟的山脉地形

    Args:
        gridx: 网格x坐标
        gridy: 网格y坐标
        peaks: 山峰列表

    Returns:
        模拟的山脉地形
    """
    # 初始化山脉地形
    mountain_terrain = np.zeros_like(gridx)
    # 生成每个山峰的山脉地形
    for peak in peaks:
        # 计算当前山峰对网格的影响
        mountain_terrain += peak.amplitude * np.exp(
            -((gridx - peak.center_x) ** 2 + (gridy - peak.center_y) ** 2) / (2 * peak.width**2)
        )
    return mountain_terrain


def is_collision_detected(point: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> bool:
    """
    检测空间中的某个点 (x, y, z) 是否与高度场发生了碰撞

    Args:
        point (np.ndarray): 待检测的点的坐标 (x, y, z)
        x_grid (np.ndarray): X平面网格
        y_grid (np.ndarray): Y平面网格
        z_grid (np.ndarray): Z平面网格

    Returns:
        bool: 如果发生碰撞则返回True，否则返回False
    """
    x, y, z = point[0], point[1], point[2]

    # 边界值处理，高度为0，无需考虑碰撞
    if z == 0:
        return False

    # 获取网格的实际坐标值范围
    x_vals = np.unique(x_grid)
    y_vals = np.unique(y_grid)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # 超出地形范围视为碰撞
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return True

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
