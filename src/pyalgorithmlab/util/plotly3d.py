import math


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
