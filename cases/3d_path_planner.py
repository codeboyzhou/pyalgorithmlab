import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.interpolate import make_interp_spline

from pyalgorithmlab.pso.core import ParticleSwarmOptimizer
from pyalgorithmlab.pso.types import AlgorithmArguments, ProblemType
from pyalgorithmlab.util import terrain
from pyalgorithmlab.util.model.peak import Peak


class PathPlanner3D:

    def __init__(
        self,
        algorithm_args: AlgorithmArguments,
        start_point: tuple[float, float, float],
        peaks: list[Peak],
    ) -> None:
        # 初始化坐标网格
        x = np.linspace(
            algorithm_args.position_bounds_min, algorithm_args.position_bounds_max, 100
        )
        y = np.linspace(
            algorithm_args.position_bounds_min, algorithm_args.position_bounds_max, 100
        )
        self.gridx, self.gridy = np.meshgrid(x, y)
        # 生成模拟的山脉地形
        self.gridz = terrain.generate_simulated_mountain_peaks(
            self.gridx, self.gridy, peaks
        )
        # 初始化最优路径点
        self.best_path_points: list[tuple[float, float, float]] = [start_point]

    def plot_terrain(self) -> plt.Axes:
        """绘制地形"""
        fig = plt.figure(figsize=(10, 8))
        axes = fig.add_subplot(111, projection="3d")
        surface = axes.plot_surface(self.gridx, self.gridy, self.gridz, cmap="viridis")
        fig.colorbar(surface, shrink=0.5, aspect=5)
        axes.view_init(elev=30, azim=240)
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")
        plt.show()
        return axes

    def plot_path(self, ax: plt.Axes, mark_waypoints: bool = True) -> None:
        """绘制路径"""
        # 标记路径点
        for i, point in enumerate(self.best_path_points):
            px, py, pz = point[0], point[1], point[2]
            # 起点
            if i == 0:
                ax.scatter(
                    px, py, pz, c="green", s=100, marker="o", label="Starting Point"
                )
            # 终点
            elif i == len(self.best_path_points) - 1:
                ax.scatter(px, py, pz, c="red", s=100, marker="*", label="Destination")
            # 途经点
            elif mark_waypoints:
                ax.scatter(
                    px,
                    py,
                    pz,
                    c="orange",
                    s=100,
                    marker="^",
                    label="Waypoint" if i == 1 else None,
                )
                ax.text(
                    x=px + 0.2,
                    y=py + 0.2,
                    z=pz + 0.2,
                    color="red",
                    s=f"P{i}",
                    fontsize=10,
                )

        # 使用三阶B样条曲线绘制平滑路径
        np_best_path_points = np.array(self.best_path_points)
        bx, by, bz = (
            np_best_path_points[:, 0],
            np_best_path_points[:, 1],
            np_best_path_points[:, 2],
        )
        # 计算路径点的参数化变量
        t_for_spline = np.arange(len(bx))
        # 创建三阶B样条曲线（k=3表示三阶）
        spline = make_interp_spline(t_for_spline, np.column_stack((bx, by, bz)), k=3)
        # 生成平滑路径点，使用参数化变量进行插值，可以根据需要调整点的数量
        smooth_path = spline(np.linspace(t_for_spline.min(), t_for_spline.max(), 100))
        bx, by, bz = smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2]
        ax.plot(bx, by, bz, "b-", linewidth=5, label="Best Path")

    def plan_best_path(self, positions: np.ndarray) -> np.ndarray:
        """规划最优路径"""
        # 初始化粒子路径成本
        costs = np.zeros(positions.shape[0])

        # 选择成本最小的点
        best_point_index = np.argmin(costs)
        best_point = positions[best_point_index]

        # 碰撞点纠偏
        while terrain.is_collision_detected(
            np.array(best_point), self.gridx, self.gridy, self.gridz
        ):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 -x 方向纠偏")
            best_point[0] -= 1
            if best_point[0] < 0:
                best_point[0] = 0
                break
        while terrain.is_collision_detected(
            np.array(best_point), self.gridx, self.gridy, self.gridz
        ):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 +x 方向纠偏")
            best_point[0] += 1
            if best_point[0] > 100:
                best_point[0] = 100
                break
        while terrain.is_collision_detected(
            np.array(best_point), self.gridx, self.gridy, self.gridz
        ):
            logger.warning(f"当前点 {best_point} 会与地形发生碰撞，尝试向 -y 方向纠偏")
            best_point[1] -= 1
            if best_point[1] < 0:
                best_point[1] = 0
                break
        while terrain.is_collision_detected(
            np.array(best_point), self.gridx, self.gridy, self.gridz
        ):
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
        algorithm_args=pso_args, start_point=(0, 0, 0), peaks=[]
    )
    pso_optimizer = ParticleSwarmOptimizer(
        args=pso_args,
        problem_type=ProblemType.MIN,
        objective_function=path_planner.plan_best_path,
    )
    pso_optimizer.start_iterating()
    ax = path_planner.plot_terrain()
    path_planner.plot_path(ax)
