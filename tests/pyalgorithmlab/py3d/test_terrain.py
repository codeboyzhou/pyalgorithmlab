import numpy as np
from loguru import logger

import pytest
from pyalgorithmlab.py3d.terrain import Terrain
from pyalgorithmlab.py3d.types import Peak, Point


@pytest.fixture
def terrain():
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    xx, yy = np.meshgrid(x, y)
    terrain = Terrain(xx=xx, yy=yy, peaks=[Peak(center_x=20, center_y=20, amplitude=6, width=6)])
    terrain.init_mountain_terrain()
    return terrain


def test_min_horizontal_distance_to_peak_centers(terrain):
    point = Point(x=20, y=20, z=6)
    distance = terrain.min_horizontal_distance_to_peak_centers(point)
    assert distance == 0


def test_check_point_collision__point_above_terrain(terrain):
    point = Point(x=20, y=20, z=7)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y)
    logger.debug(f"point_above_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert not terrain.check_point_collision(point)


def test_check_point_collision__point_below_terrain(terrain):
    point = Point(x=20, y=20, z=5)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y)
    logger.debug(f"point_below_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert terrain.check_point_collision(point)


def test_check_point_collision__point_on_terrain(terrain):
    point = Point(x=20, y=20, z=6)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y)
    point.z = terrain_height
    logger.debug(f"point_on_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert terrain.check_point_collision(point)


def test_check_point_collision__point_far_away(terrain):
    point = Point(x=100, y=100, z=6)
    assert not terrain.check_point_collision(point)


def test_check_line_segment_collision_points__line_above_terrain(terrain):
    start_point = Point(x=20, y=20, z=7)
    end_point = Point(x=20, y=20, z=8)
    collision_points = terrain.check_line_segment_collision_points(start_point, end_point)
    assert len(collision_points) == 0


def test_check_line_segment_collision_points__line_below_terrain(terrain):
    start_point = Point(x=20, y=20, z=5)
    end_point = Point(x=20, y=20, z=6)
    collision_points = terrain.check_line_segment_collision_points(start_point, end_point)
    assert len(collision_points) > 0


def test_check_line_segment_collision_points__line_intersect_terrain(terrain):
    start_point = Point(x=20, y=20, z=5)
    end_point = Point(x=20, y=20, z=7)
    collision_points = terrain.check_line_segment_collision_points(start_point, end_point)
    assert len(collision_points) > 0


def test_try_correct_collision_point__point_above_terrain(terrain):
    point = Point(x=20, y=20, z=7)
    corrected_point = terrain.try_correct_collision_point(point)
    assert corrected_point.x == corrected_point.y == 20 and corrected_point.z == 7


def test_try_correct_collision_point__point_below_terrain(terrain):
    point = Point(x=20, y=20, z=1)
    corrected_point = terrain.try_correct_collision_point(point)
    assert corrected_point.z == 1
