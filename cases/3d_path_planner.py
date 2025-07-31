from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import make_interp_spline

from pyalgorithmlab.pso.core import ParticleSwarmOptimizer
from pyalgorithmlab.pso.types import AlgorithmArguments, ProblemType
from pyalgorithmlab.util import terrain
from pyalgorithmlab.util.model.peak import Peak


class PathPlanner3D:
    """这是一个借助PSO算法实现3D路径规划的案例"""

    def __init__(
        self,
        algorithm_args: AlgorithmArguments,
        start_point: tuple[float, float, float],
        peaks: list[Peak],
    ) -> None:
        # 初始化坐标网格
        x = np.linspace(algorithm_args.position_bounds_min[0], algorithm_args.position_bounds_max[0], 100)
        y = np.linspace(algorithm_args.position_bounds_min[1], algorithm_args.position_bounds_max[1], 100)
        self.x_grid, self.y_grid = np.meshgrid(x, y)

        # 生成模拟的山脉地形
        self.z_grid = terrain.generate_simulated_mountain_peaks(self.x_grid, self.y_grid, peaks)

        # 初始化最优路径点
        self.best_path_points: list[tuple[float, float, float]] = [start_point]

    def plot_terrain_and_best_path(self, mark_waypoints: bool = True) -> None:
        """绘制地形和最优路径"""
        # 绘制地形
        fig = plt.figure(figsize=(10, 8))
        axes3d = cast(Axes3D, fig.add_subplot(111, projection="3d"))
        surface = axes3d.plot_surface(self.x_grid, self.y_grid, self.z_grid, cmap="viridis")
        fig.colorbar(surface, shrink=0.5, aspect=5)

        # 标记路径点
        for i, point in enumerate(self.best_path_points):
            px, py, pz = point[0], point[1], point[2]
            # 起点
            if i == 0:
                axes3d.scatter(px, py, int(pz), c="green", s=100, marker="o", label="Starting Point")
            # 终点
            elif i == len(self.best_path_points) - 1:
                axes3d.scatter(px, py, int(pz), c="red", s=100, marker="*", label="Destination")
            # 途经点
            elif mark_waypoints:
                axes3d.scatter(px, py, int(pz), c="orange", s=100, marker="^", label="Waypoint" if i == 1 else None)
                axes3d.text(x=px + 0.2, y=py + 0.2, z=pz + 0.2, color="red", s=f"P{i}", fontsize=10)

        # 使用三阶B样条曲线绘制平滑路径
        np_best_path_points = np.array(self.best_path_points)
        px, py, pz = np_best_path_points[:, 0], np_best_path_points[:, 1], np_best_path_points[:, 2]
        # 计算路径点的参数化变量
        t_for_spline = np.arange(len(px))
        # 创建三阶B样条曲线（k=3表示三阶）
        spline = make_interp_spline(t_for_spline, np.column_stack((px, py, pz)), k=3)
        # 生成平滑路径点，使用参数化变量进行插值，可以根据需要调整点的数量
        smooth_path = spline(np.linspace(t_for_spline.min(), t_for_spline.max(), 100))
        px, py, pz = smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2]
        axes3d.plot(px, py, pz, "b-", linewidth=5, label="Best Path")

        axes3d.view_init(elev=30, azim=240)
        axes3d.set_xlabel("X")
        axes3d.set_ylabel("Y")
        axes3d.set_zlabel("Z")
        plt.show()

    def plan_best_path(self, positions: np.ndarray) -> np.ndarray:
        """规划最优路径"""
        # 初始化粒子路径成本
        costs = np.zeros(positions.shape[0])

        # 选择成本最小的点
        best_point_index = np.argmin(costs)
        best_point = positions[best_point_index]

        # 碰撞点纠偏
        while terrain.is_collision_detected(np.array(best_point), self.x_grid, self.y_grid, self.z_grid):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 -x 方向纠偏")
            best_point[0] -= 1
            if best_point[0] < 0:
                best_point[0] = 0
                break
        while terrain.is_collision_detected(np.array(best_point), self.x_grid, self.y_grid, self.z_grid):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 +x 方向纠偏")
            best_point[0] += 1
            if best_point[0] > 100:
                best_point[0] = 100
                break
        while terrain.is_collision_detected(np.array(best_point), self.x_grid, self.y_grid, self.z_grid):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 -y 方向纠偏")
            best_point[1] -= 1
            if best_point[1] < 0:
                best_point[1] = 0
                break
        while terrain.is_collision_detected(np.array(best_point), self.x_grid, self.y_grid, self.z_grid):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 +y 方向纠偏")
            best_point[1] += 1
            if best_point[1] > 100:
                best_point[1] = 100
                break

        best_x, best_y, best_z = best_point[0], best_point[1], best_point[2]
        self.best_path_points.append((best_x.item(), best_y.item(), best_z.item()))

        return costs


if __name__ == "__main__":
    pso_args = AlgorithmArguments(
        num_particles=100,
        num_dimensions=3,
        max_iterations=100,
        position_bounds_min=(0, 0, 0),
        position_bounds_max=(100, 100, 1),
        velocity_bound_max=1,
        inertia_weight_max=2,
        inertia_weight_min=0.5,
        cognitive_coefficient=1.6,
        social_coefficient=1.8,
    )
    path_planner = PathPlanner3D(
        algorithm_args=pso_args,
        start_point=(0, 0, 0),
        peaks=[
            Peak(center_x=20, center_y=20, amplitude=6, width=6),
            Peak(center_x=20, center_y=60, amplitude=7, width=7),
            Peak(center_x=60, center_y=20, amplitude=5, width=8),
            Peak(center_x=80, center_y=60, amplitude=5, width=8),
        ],
    )
    pso_optimizer = ParticleSwarmOptimizer(
        args=pso_args,
        problem_type=ProblemType.MIN,
        objective_function=path_planner.plan_best_path,
    )
    pso_optimizer.start_iterating()
    path_planner.plot_terrain_and_best_path()
