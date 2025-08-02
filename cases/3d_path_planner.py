from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray
from scipy.interpolate import make_interp_spline

from pyalgorithmlab.pso.core import ParticleSwarmOptimizer
from pyalgorithmlab.pso.types import AlgorithmArguments, ProblemType
from pyalgorithmlab.pso.util import plot
from pyalgorithmlab.util import ndarrays, terrain
from pyalgorithmlab.util.model.peak import Peak


class PathPlanner3D:
    """这是一个借助PSO算法实现3D路径规划的案例"""

    def __init__(
        self,
        algorithm_args: AlgorithmArguments,
        start_point: tuple[float, float, float],
        destination: tuple[float, float, float],
        peaks: list[Peak],
    ) -> None:
        """
        初始化3D路径规划器

        Args:
            algorithm_args: PSO算法参数
            start_point: 起点坐标
            destination: 终点坐标
            peaks: 山脉峰值列表
        """
        self.algorithm_args = algorithm_args

        # 初始化坐标网格
        x = np.linspace(algorithm_args.position_bounds_min[0], algorithm_args.position_bounds_max[0], 100)
        y = np.linspace(algorithm_args.position_bounds_min[1], algorithm_args.position_bounds_max[1], 100)
        self.x_grid, self.y_grid = np.meshgrid(x, y)

        # 生成模拟的山脉地形
        self.z_grid = terrain.generate_simulated_mountain_peaks(self.x_grid, self.y_grid, peaks)

        # 初始化起点和终点
        self.start_point = start_point
        self.destination = destination

        # 初始化最优路径点
        self.best_path_points: list[tuple[float, float, float]] = [start_point]

    def plot_terrain_and_best_path(self, mark_waypoints: bool = True) -> None:
        """
        绘制地形和最优路径

        Args:
            mark_waypoints: 是否标记途经点. Defaults to True.
        """
        # 点坐标排序
        self.best_path_points.sort(key=lambda p: p[0] + p[1] + p[2])
        # 追加终点
        self.best_path_points.append(self.destination)
        logger.success(f"最优路径点为{self.best_path_points}")

        # 绘制地形
        fig = plt.figure(figsize=(10, 8))
        axes3d = cast(Axes3D, fig.add_subplot(111, projection="3d"))
        surface = axes3d.plot_surface(self.x_grid, self.y_grid, self.z_grid, cmap="viridis", alpha=0.6)
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
        axes3d.legend()
        plt.show()

    def correct_collision_point(self, point: np.ndarray, max_attempts: int = 100) -> ndarray:
        """
        纠偏碰撞点

        Args:
            point: 碰撞点坐标
            max_attempts: 最大尝试次数

        Returns:
            纠偏后的碰撞点坐标
        """
        x_min, x_max = np.min(self.x_grid), np.max(self.x_grid)
        y_min, y_max = np.min(self.y_grid), np.max(self.y_grid)
        z_min, z_max = 0, np.max(self.z_grid)

        # 8方向搜索（含对角线）+ Z轴调整
        directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        # 初始步长
        step_size = 3

        for i in range(max_attempts):
            # 动态步长衰减（指数级减小）
            current_step = step_size * (0.8**i)
            current_step = min(current_step, 0.5)

            # 尝试搜索所有方向
            for dx, dy, dz in directions:
                new_point = point.copy()
                new_point[0] += dx * current_step
                new_point[1] += dy * current_step
                new_point[2] += dz * current_step

                # 边界检查
                new_point[0] = np.clip(new_point[0], x_min, x_max)
                new_point[1] = np.clip(new_point[1], y_min, y_max)
                new_point[2] = np.clip(new_point[2], z_min, z_max)

                # 检查是否找到无碰撞点
                if not terrain.is_point_collision_detected(new_point, self.x_grid, self.y_grid, self.z_grid):
                    logger.warning(f"原碰撞点{point}已经纠偏为无碰撞点{new_point}")
                    return new_point

        # 若所有方向均失败，强制抬升Z轴
        point[2] = z_max
        logger.warning(f"无法完全规避碰撞，已强制抬升高度至{z_max}")

        return point

    def calculate_best_path_costs(self, positions: np.ndarray) -> np.ndarray:
        """
        PSO算法目标函数：计算粒子路径成本，规划最优路径点

        Args:
            positions: 粒子位置数组，形状为 (num_particles, num_dimensions)

        Returns:
            粒子路径成本数组，形状为 (num_particles,)
        """
        # 初始化粒子路径成本
        costs = np.zeros(positions.shape[0])

        # 定义惩罚权重
        iteration = len(self.best_path_points) - 1
        iteration_progress = iteration / self.algorithm_args.max_iterations
        distance_to_destination_weight = 0.1 + 0.3 * iteration_progress
        point_deviation_weight = 0.4 - 0.1 * (iteration_progress**3)
        point_gathering_weight = 0.1 + 0.4 * iteration_progress
        terrain_collision_weight = 0.7 + 0.3 * iteration_progress

        # 终点距离成本
        distance_to_destination_costs = np.linalg.norm(positions - self.destination, axis=1)
        costs += ndarrays.min_max_normalize(np.array(distance_to_destination_costs)) * distance_to_destination_weight

        # 粒子偏离成本
        point_deviation_costs = ndarrays.point_to_line_distance(
            positions, np.array(self.start_point), np.array(self.destination)
        )
        costs += ndarrays.min_max_normalize(point_deviation_costs) * point_deviation_weight

        # 粒子聚集成本
        previous_point = np.array(self.best_path_points[-1])
        point_gathering_costs = np.linalg.norm(positions - previous_point, axis=1)
        costs += ndarrays.min_max_normalize(point_gathering_costs) * point_gathering_weight

        # 地形碰撞成本
        terrain_collision_costs = np.array(
            [
                1 if terrain.is_point_collision_detected(point, self.x_grid, self.y_grid, self.z_grid) else 0
                for point in positions
            ]
        )
        costs += terrain_collision_costs * terrain_collision_weight

        # 选择成本最小的点
        best_point = positions[np.argmin(costs)]

        # 路径点碰撞纠偏
        if terrain.is_point_collision_detected(best_point, self.x_grid, self.y_grid, self.z_grid):
            best_point = self.correct_collision_point(best_point)

        # 路径点连接后的线段碰撞纠偏
        collision_points = terrain.is_line_collision_detected(
            previous_point, best_point, self.x_grid, self.y_grid, self.z_grid
        )
        # 如果没有碰撞点，直接添加当前最佳点
        if len(collision_points) == 0:
            logger.success(f"路径点{previous_point} -> {best_point}之间没有发生连接后穿透地形碰撞风险")
            bx, by, bz = best_point[0], best_point[1], best_point[2]
            self.best_path_points.append((bx.item(), by.item(), bz.item()))
        else:
            # 否则就对采样检测到的碰撞点进行纠偏
            logger.warning(f"检测到路径点{previous_point} -> {best_point}之间连接后发生穿透地形碰撞风险")
            logger.warning(f"已经对连接线段进行采样，发现{len(collision_points)}个采样碰撞点，正在对碰撞点进行纠偏")
            for point in collision_points:
                corrected_collision_point = self.correct_collision_point(np.array(point))
                cx, cy, cz = corrected_collision_point[0], corrected_collision_point[1], corrected_collision_point[2]
                self.best_path_points.append((cx.item(), cy.item(), cz.item()))

        return costs


if __name__ == "__main__":
    pso_args = AlgorithmArguments(
        num_particles=100,
        num_dimensions=3,
        max_iterations=100,
        position_bounds_min=(0, 0, 0),
        position_bounds_max=(100, 100, 1),
        velocity_bound_max=1,
        inertia_weight_max=1.8,
        inertia_weight_min=0.5,
        cognitive_coefficient=1.6,
        social_coefficient=1.2,
    )

    path_planner = PathPlanner3D(
        algorithm_args=pso_args,
        start_point=(0, 0, 0),
        destination=(100, 100, 1),
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
        objective_function=path_planner.calculate_best_path_costs,
    )

    best_fitness_values = pso_optimizer.start_iterating()
    plot.plot_fitness_change_curve(best_fitness_values)
    path_planner.plot_terrain_and_best_path()
