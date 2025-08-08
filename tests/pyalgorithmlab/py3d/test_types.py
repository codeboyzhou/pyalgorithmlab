import numpy as np

from pyalgorithmlab.py3d.types import Point


def test_point_from_array():
    p = Point.from_array([1, 2, 3])
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3


def test_point_to_array():
    p = Point(x=1, y=2, z=3)
    assert p.to_array() == [1, 2, 3]


def test_point_from_ndarray():
    p = Point.from_ndarray(np.array([1, 2, 3]))
    assert p.x == 1
    assert p.y == 2
    assert p.z == 3


def test_point_to_ndarray():
    p = Point(x=1, y=2, z=3)
    assert np.array_equal(p.to_ndarray(), np.array([1, 2, 3]))


def test_point_clone():
    p = Point(x=1, y=2, z=3)
    p2 = p.clone()
    assert p2.x == 1
    assert p2.y == 2
    assert p2.z == 3
    assert p2 is not p
