import math

import plotly.graph_objects as go
from plotly.graph_objs import Scatter3d

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
