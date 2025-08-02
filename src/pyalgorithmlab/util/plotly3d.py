import math


def compute_plotly_camera_eye(elev: int = 0, azim: int = 0, camera_distance: float = 2.5) -> dict:
    azim_rad = math.radians(azim)
    elev_rad = math.radians(elev)
    x = math.cos(azim_rad) * math.cos(elev_rad) * camera_distance
    y = math.sin(azim_rad) * math.cos(elev_rad) * camera_distance
    z = math.sin(elev_rad) * camera_distance
    return {"x": x, "y": y, "z": z}
