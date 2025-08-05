from loguru import logger

from pyalgorithmlab.py3d import terrain
from pyalgorithmlab.py3d.types import Peak, Point

peak = Peak(center_x=20, center_y=20, amplitude=6, width=6)


def test_check_point_collision__point_above_terrain():
    point = Point(x=20, y=20, z=7)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y, peak)
    logger.debug(f"point_above_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert terrain.check_point_collision(point, [peak]) is False


def test_check_point_collision__point_below_terrain():
    point = Point(x=20, y=20, z=5)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y, peak)
    logger.debug(f"point_below_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert terrain.check_point_collision(point, [peak]) is True


def test_check_point_collision__point_on_terrain():
    point = Point(x=20, y=20, z=6)
    terrain_height = terrain.get_mountain_terrain_height(point.x, point.y, peak)
    point.z = terrain_height
    logger.debug(f"point_on_terrain -> point.z={point.z}, terrain_height={terrain_height}")
    assert terrain.check_point_collision(point, [peak]) is True


def test_check_point_collision__point_far_away():
    point = Point(x=100, y=100, z=6)
    assert terrain.check_point_collision(point, [peak]) is False
