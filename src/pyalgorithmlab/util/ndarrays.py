import numpy as np

from pyalgorithmlab.common.consts import Consts


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    对输入的向量进行归一化处理，使每行的值在0到1之间

    Args:
        vectors: 输入的向量，形状为 (n, m)

    Returns:
        np.ndarray: 归一化后的向量，形状与输入相同
    """
    # 使用5%到95%分位数剪裁，避免极值影响
    clipped = np.clip(vectors, np.percentile(vectors, q=5), np.percentile(vectors, q=95))
    min_val = clipped.min()
    max_val = clipped.max()
    return (clipped - min_val) / (max_val - min_val + Consts.EPSILON)


def angle_degrees(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> np.ndarray:
    """
    计算向量 point1 -> point2 与 向量 point2 -> point3 的夹角角度值

    Args:
        point1: 输入点1
        point2: 输入点2
        point3: 输入点3

    Returns:
        np.ndarray: 向量 point1 -> point2 与 向量 point2 -> point3 的夹角角度值，形状为 (n,)
    """
    vector12 = point2 - point1
    vector23 = point3 - point2
    vector12_norm = np.linalg.norm(vector12)
    vector23_norm = np.linalg.norm(vector23)
    cos_angles = np.dot(vector12, vector23) / (vector12_norm * vector23_norm.astype(np.int64))
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angle_rad = np.arccos(cos_angles)
    return np.degrees(angle_rad)


def point_to_line_distance(points: np.ndarray, line_endpoint_a: np.ndarray, line_endpoint_b: np.ndarray) -> np.ndarray:
    """
    计算空间中多个点到直线ab的垂直距离

    Args:
        points: 形状为 (n, 3) 的点集
        line_endpoint_a: 直线起点，形状为 (3,)
        line_endpoint_b: 直线终点，形状为 (3,)

    Returns:
        np.ndarray: 每个点到直线的距离，形状为 (n,)
    """
    if np.array_equal(line_endpoint_a, line_endpoint_b):
        raise ZeroDivisionError("line_endpoint_a should be different from line_endpoint_b")

    ab = line_endpoint_b - line_endpoint_a
    ap = points - line_endpoint_a
    cross = np.cross(ap, ab)
    distance = np.linalg.norm(cross, axis=1) / np.linalg.norm(ab).astype(np.int64)
    return distance
