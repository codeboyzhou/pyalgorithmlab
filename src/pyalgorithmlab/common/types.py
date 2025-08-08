from enum import Enum, unique


@unique
class ProblemType(Enum):
    """定义问题类型枚举"""

    MIN = 1
    """最小化问题"""

    MAX = 2
    """最大化问题"""
