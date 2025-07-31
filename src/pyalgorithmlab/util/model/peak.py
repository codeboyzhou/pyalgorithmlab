from pydantic import BaseModel


class Peak(BaseModel):
    """三维路径规划中的模拟山峰参数定义"""

    center_x: float
    """山峰中心的x坐标"""

    center_y: float
    """山峰中心的y坐标"""

    amplitude: float
    """山峰的振幅"""

    width: float
    """山峰的宽度"""
