def is_converged(
    values: list[float], consecutive: int = 10, threshold: float = 1e-6
) -> bool:
    length = len(values)
    if length < consecutive:
        return False
    for i in range(length - consecutive + 1, length):
        diff = abs(values[i] - values[i - 1])
        if diff > threshold:
            return False
    return True
