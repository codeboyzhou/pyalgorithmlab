import numpy as np


def min_max_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    对输入的向量进行最小-最大归一化处理，使每行的值在0到1之间

    Args:
        vectors: 输入的向量，形状为 (n, m)

    Returns:
        np.ndarray: 归一化后的向量，形状与输入相同
    """
    min_values = np.min(vectors)
    max_values = np.max(vectors)
    normalized_vectors = (vectors - min_values) / (max_values - min_values + 1e-6)  # 防止除0
    return normalized_vectors


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
