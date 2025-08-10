import numpy as np
import pytest

from pyalgorithmlab.util import ndarrays

############################# test_normalize #############################


def test_normalize_basic():
    vectors = np.array([[1, 2], [3, 4]])
    result = ndarrays.normalize(vectors)
    assert result.shape == vectors.shape
    assert np.all(result >= 0) and np.all(result <= 1)


def test_normalize_with_outliers():
    vectors = np.array([[1, 2, 3], [100, 200, 300]])
    result = ndarrays.normalize(vectors)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_normalize_all_same_values():
    vectors = np.full((3, 3), 5)
    result = ndarrays.normalize(vectors)
    assert np.all(result == 0)


def test_normalize_shape_preserved():
    vectors = np.random.rand(10, 5)
    result = ndarrays.normalize(vectors)
    assert result.shape == (10, 5)


def test_normalize_float_and_int():
    vectors_float = np.array([[0.5, 1.5], [2.5, 3.5]])
    vectors_int = np.array([[1, 2], [3, 4]])
    result_float = ndarrays.normalize(vectors_float)
    result_int = ndarrays.normalize(vectors_int)
    assert result_float.shape == vectors_float.shape
    assert result_int.shape == vectors_int.shape
    assert np.all(result_float >= 0) and np.all(result_float <= 1)
    assert np.all(result_int >= 0) and np.all(result_int <= 1)


########################### test_angle_degrees ###########################


def test_angle_degrees_0():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([2, 0])
    angle = ndarrays.angle_degrees(p1, p2, p3)
    assert np.round(angle) == 0


def test_angle_degrees_45():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([1 + np.sqrt(2) / 2, np.sqrt(2) / 2])
    angle = ndarrays.angle_degrees(p1, p2, p3)
    assert np.round(angle) == 45


def test_angle_degrees_90():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([1, 1])
    angle = ndarrays.angle_degrees(p1, p2, p3)
    assert np.round(angle) == 90


def test_angle_degrees_180():
    p1 = np.array([0, 0])
    p2 = np.array([1, 0])
    p3 = np.array([-1, 0])
    angle = ndarrays.angle_degrees(p1, p2, p3)
    assert np.round(angle) == 180


####################### test_point_to_line_distance ######################


def test_point_to_line_distance__points_on_line():
    a = np.array([0, 0, 0])
    b = np.array([1, 1, 1])
    points = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
    distances = ndarrays.point_to_line_distance(points, a, b)
    assert np.all(distances == 0)


def test_point_to_line_distance__points_off_line():
    a = np.array([0, 0, 0])
    b = np.array([1, 0, 0])
    points = np.array([[0, 1, 0], [0, 0, 2], [3, 4, 0]])
    distances = ndarrays.point_to_line_distance(points, a, b)
    expected = np.array([1, 2, 4])
    assert np.all(distances == expected)


def test_point_to_line_distance__multiple_points():
    a = np.array([0, 0, 0])
    b = np.array([0, 0, 1])
    points = np.array([[1, 0, 0], [0, 2, 0], [-1, -1, 0], [0, 0, 5]])
    distances = ndarrays.point_to_line_distance(points, a, b)
    expected = np.array([1, 2, np.sqrt(2), 0])
    assert np.all(distances == expected)


def test_point_to_line_distance__large_coordinates():
    a = np.array([1e9, 0, 0])
    b = np.array([1e9, 1, 0])
    points = np.array(
        [
            [1e9 + 1, 0, 0],
            [1e9 - 1, 0, 0],
        ]
    )
    distances = ndarrays.point_to_line_distance(points, a, b)
    expected = np.array([1.0, 1.0])
    assert np.all(distances == expected)


def test_point_to_line_distance__same_endpoints_should_raise():
    a = np.array([1, 1, 1])
    b = np.array([1, 1, 1])
    points = np.array([[2, 2, 2]])
    with pytest.raises(ZeroDivisionError):
        ndarrays.point_to_line_distance(points, a, b)
