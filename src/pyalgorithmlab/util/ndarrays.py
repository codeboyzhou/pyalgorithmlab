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


def cos_angles(vectors: np.ndarray, direction_start: np.ndarray, direction_destination: np.ndarray) -> np.ndarray:
    """
    计算多个向量与某个方向向量的余弦夹角

    Args:
        vectors: 输入的向量，形状为 (n, m)
        direction_start: 方向向量起点，形状为 (m,)
        direction_destination: 方向向量终点，形状为 (m,)

    Returns:
        np.ndarray: 每个向量与方向向量的余弦夹角，形状为 (n,)
    """
    direction = direction_destination - direction_start
    unit_direction = direction / np.linalg.norm(direction)
    vectors_to_direction = vectors - direction_start
    vectors_to_direction_norms = np.linalg.norm(vectors_to_direction, axis=1) + Consts.EPSILON
    vectors_cos_angles = np.dot(vectors_to_direction, unit_direction) / vectors_to_direction_norms
    angles = np.arccos(np.clip(vectors_cos_angles, -1.0, 1.0))
    return angles


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
    ab = line_endpoint_b - line_endpoint_a
    ap = points - line_endpoint_a
    cross = np.cross(ap, ab)
    distance = np.linalg.norm(cross, axis=1) / np.linalg.norm(ab).astype(np.int64)
    return distance
