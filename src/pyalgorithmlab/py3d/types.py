import numpy as np
from pydantic import BaseModel, ConfigDict


class Peak(BaseModel):
    """山峰模型"""

    center_x: float
    """山峰中心点的x坐标"""

    center_y: float
    """山峰中心点的y坐标"""

    amplitude: float
    """山峰的振幅"""

    width: float
    """山峰的宽度"""


class Point(BaseModel):
    """三维点模型"""

    x: float
    """点的x坐标"""

    y: float
    """点的y坐标"""

    z: float
    """点的z坐标"""

    @classmethod
    def from_array(cls, point: list[float]) -> "Point":
        return cls(x=point[0], y=point[1], z=point[2])

    def to_array(self):
        return [self.x, self.y, self.z]

    @classmethod
    def from_ndarray(cls, point: np.ndarray) -> "Point":
        return cls(x=point[0].item(), y=point[1].item(), z=point[2].item())

    def to_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class Grid(BaseModel):
    """三维网格模型"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """模型配置"""

    x: np.ndarray
    """X网格"""

    y: np.ndarray
    """Y网格"""

    z: np.ndarray
    """Z网格"""
