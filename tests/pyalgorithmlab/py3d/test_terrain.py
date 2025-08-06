import numpy as np
import pytest
from loguru import logger

from pyalgorithmlab.py3d.terrain import Terrain
from pyalgorithmlab.py3d.types import Peak, Point

peak = Peak(center_x=20, center_y=20, amplitude=6, width=6)


@pytest.fixture
def terrain():
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    xx, yy = np.meshgrid(x, y)
    terrain = Terrain(xx=xx, yy=yy, peaks=[peak])
    terrain.init_mountain_terrain()
    return terrain


def test_check_point_collision__point_above_terrain(terrain):
    point = Point(x=20, y=20, z=7)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y)
    logger.debug(f"point_above_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert terrain.check_point_collision(point) is False


def test_check_point_collision__point_below_terrain(terrain):
    point = Point(x=20, y=20, z=5)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y)
    logger.debug(f"point_below_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert terrain.check_point_collision(point) is True


def test_check_point_collision__point_on_terrain(terrain):
    point = Point(x=20, y=20, z=6)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y)
    point.z = terrain_height
    logger.debug(f"point_on_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert terrain.check_point_collision(point) is True


def test_check_point_collision__point_far_away(terrain):
    point = Point(x=100, y=100, z=6)
    assert terrain.check_point_collision(point) is False
