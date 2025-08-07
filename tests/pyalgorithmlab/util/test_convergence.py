from pyalgorithmlab.util import convergence


def test_is_converged__should_be_converged():
    assert convergence.is_converged(values=[1, 1, 1], consecutive=3)


def test_is_converged__consecutive_greater_than_values_length():
    assert not convergence.is_converged(values=[1, 1], consecutive=3)


def test_is_converged__increasing_values():
    assert not convergence.is_converged(values=[1, 1, 2], consecutive=3)


def test_is_converged__decreasing_values():
    assert not convergence.is_converged(values=[1, 1, 0], consecutive=3)


def test_is_converged__increasing_then_decreasing_values():
    assert not convergence.is_converged(values=[1, 2, 1], consecutive=3)


def test_is_converged__continuous_increasing_values():
    assert not convergence.is_converged(values=[1, 2, 3], consecutive=3)
