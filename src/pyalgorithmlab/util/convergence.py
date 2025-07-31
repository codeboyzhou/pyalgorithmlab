def is_converged(values: list[float], consecutive: int = 10, threshold: float = 1e-6) -> bool:
    """
    判断values包含的值是否已经收敛

    Args:
        values: 包含值的列表
        consecutive: 连续n次元素值变化都很小
        threshold: 可以接受的元素值变化阈值

    Returns:
        bool: 是否已经收敛
    """
    length = len(values)
    if length < consecutive:
        return False
    for i in range(length - consecutive + 1, length):
        diff = abs(values[i] - values[i - 1])
        if diff > threshold:
            return False
    return True
