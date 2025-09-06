import math

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs import Scatter3d
from scipy.interpolate import make_interp_spline

from pyalgorithmlab.py3d.types import Point


def compute_plotly_camera_eye(elev: int = 0, azim: int = 0, camera_distance: float = 2.5) -> dict:
    """
    计算Plotly库的相机视角坐标

    Args:
        elev: 相机视角的仰角（单位：度）
        azim: 相机视角的方位角（单位：度）
        camera_distance: 相机距离

    Returns:
        相机视角坐标字典
    """
    azim_rad = math.radians(azim)
    elev_rad = math.radians(elev)
    x = math.cos(azim_rad) * math.cos(elev_rad) * camera_distance
    y = math.sin(azim_rad) * math.cos(elev_rad) * camera_distance
    z = math.sin(elev_rad) * camera_distance
    return {"x": x, "y": y, "z": z}


def circle_scatter3d(point: Point, name: str, color: str, size: int = 5) -> Scatter3d:
    """
    生成一个圆形的3D散点，用于表示一个点

    Args:
        point: 点的坐标
        name: 点的名称
        color: 点的颜色
        size: 点的大小

    Returns:
        3D散点图对象
    """
    return go.Scatter3d(
        x=[point.x],
        y=[point.y],
        z=[point.z],
        name=name,
        mode="markers",
        marker={"color": color, "size": size, "symbol": "circle"},
    )


def smooth_path(
    x: list[float], y: list[float], z: list[float], degree: int = 3
) -> tuple[list[float], list[float], list[float]]:
    """
    使用B样条曲线平滑路径，默认阶数为3

    Args:
        x: 路径的x坐标列表
        y: 路径的y坐标列表
        z: 路径的z坐标列表
        degree: B样条曲线的阶数

    Returns:
        平滑后的路径坐标列表
    """
    # 计算B样条曲线
    t = np.linspace(start=0, stop=1, num=len(x))
    x_spline = make_interp_spline(t, np.array(x), k=degree)
    y_spline = make_interp_spline(t, np.array(y), k=degree)
    z_spline = make_interp_spline(t, np.array(z), k=degree)

    # 生成平滑后的路径坐标
    t_smooth = np.linspace(t.min(), t.max(), num=100)
    x_smooth = x_spline(t_smooth)
    y_smooth = y_spline(t_smooth)
    z_smooth = z_spline(t_smooth)

    # 返回平滑后的路径坐标
    return x_smooth.tolist(), y_smooth.tolist(), z_smooth.tolist()
