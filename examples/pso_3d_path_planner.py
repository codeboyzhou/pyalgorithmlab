import numpy as np
import plotly.graph_objects as go
from loguru import logger
from plotly import subplots
from scipy.interpolate import make_interp_spline

from pyalgorithmlab.algorithm.pso import AlgorithmArguments, ParticleSwarmOptimizer
from pyalgorithmlab.common.consts import Consts
from pyalgorithmlab.common.types import ProblemType
from pyalgorithmlab.py3d.terrain import Terrain
from pyalgorithmlab.py3d.types import Grid, Peak, Point
from pyalgorithmlab.util import ndarrays, plotly3d


class PathPlanner3D:
    """这是一个借助PSO算法实现3D路径规划的案例"""

    def __init__(
        self,
        algorithm_args: AlgorithmArguments,
        start_point: Point,
        destination: Point,
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

        # 初始化XY平面坐标
        x = np.linspace(algorithm_args.position_boundaries_min[0], algorithm_args.position_boundaries_max[0], 100)
        y = np.linspace(algorithm_args.position_boundaries_min[1], algorithm_args.position_boundaries_max[1], 100)
        xx, yy = np.meshgrid(x, y)

        # 初始化地形和山脉坐标
        self.terrain = Terrain(xx, yy, peaks)
        zz = self.terrain.init_mountain_terrain()

        # 初始化网格
        self.grid = Grid(x=xx, y=yy, z=zz)
        self.peaks = peaks

        # 初始化起点和终点
        self.start_point: Point = start_point
        self.destination: Point = destination

        # 初始化最优路径点
        self.best_path_points: list[Point] = [start_point]

    def plot_result(self, fitness_values: list[float], mark_waypoint: bool = True) -> None:
        """
        绘制实验结果

        Args:
            fitness_values: 适应度值列表
            mark_waypoint: 是否标记航路点

        Returns:
            None
        """
        # 坐标排序
        self.best_path_points.sort(key=lambda pt: pt.x + pt.y + pt.z)
        # 追加终点
        self.best_path_points.append(self.destination)

        # 尝试进行碰撞点坐标修正
        for i in range(1, len(self.best_path_points)):
            # 路径点碰撞
            current_point = self.best_path_points[i]
            self.best_path_points[i] = self.terrain.try_correct_collision_point(current_point)
            # 路径线段碰撞
            previous_point = self.best_path_points[i - 1]
            collision_points = self.terrain.check_line_segment_collision_points(previous_point, current_point)
            if len(collision_points) > 0:
                self.best_path_points[i:i] = [self.terrain.try_correct_collision_point(p) for p in collision_points]

        logger.success(f"最优路径点结果{self.best_path_points}")

        # 绘制算法迭代结果
        fig = subplots.make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.4, 0.6],
            subplot_titles=["适应度变化曲线", "路径规划效果图"],
            specs=[[{"type": "xy"}, {"type": "surface"}]],
        )

        # 绘制适应度变化曲线
        fitness_trace = go.Scatter(
            mode="lines",
            x=list(range(0, len(fitness_values))),
            y=fitness_values,
            line={"width": 3},
            name="适应度值",
        )
        fig.add_trace(fitness_trace, row=1, col=1)

        # 绘制地形图
        terrain_trace = go.Surface(x=self.grid.x, y=self.grid.y, z=self.grid.z, showscale=False)
        fig.add_trace(terrain_trace, row=1, col=2)

        # 使用三阶B样条曲线绘制平滑路径
        x = np.array([p.x for p in self.best_path_points])
        y = np.array([p.y for p in self.best_path_points])
        z = np.array([p.z for p in self.best_path_points])
        # 计算路径点的参数化变量
        t_for_spline = np.arange(len(x))
        # 创建三阶B样条曲线（k=3表示三阶）
        spline = make_interp_spline(t_for_spline, np.column_stack((x, y, z)), k=3)
        # 生成平滑路径点，使用参数化变量进行插值，可以根据需要调整点的数量
        smooth_path = spline(np.linspace(t_for_spline.min(), t_for_spline.max(), 100))
        x, y, z = smooth_path[:, 0], smooth_path[:, 1], smooth_path[:, 2]
        path_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            line={"width": 10, "color": "green"},
            marker={"size": 3, "color": "yellow", "symbol": "diamond"},
            mode="lines+markers" if mark_waypoint else "lines",
            name="目标路径",
        )
        fig.add_trace(path_trace, row=1, col=2)

        # 标记起点
        fig.add_trace(plotly3d.circle_scatter3d(point=self.start_point, name="起点", color="green"))
        # 标记终点
        fig.add_trace(plotly3d.circle_scatter3d(point=self.destination, name="终点", color="red"))

        # 配置全局绘图参数
        fig.update_layout(
            title={"text": "基于PSO算法的3D路径规划实验结果"},
            # 适应度变化曲线
            xaxis_title="X-迭代次数",
            yaxis_title="Y-适应度值",
            # 路径规划效果图
            scene={
                "xaxis_title": "X方向",
                "yaxis_title": "Y方向",
                "zaxis_title": "高度",
                "camera_eye": plotly3d.compute_plotly_camera_eye(elev=30, azim=240),
            },
        )

        # 显示图表
        fig.show(renderer="browser")

    def calculate_best_path_costs(self, positions: np.ndarray) -> np.ndarray:
        """
        PSO算法目标函数：计算粒子路径成本，规划最优路径点

        Args:
            positions: 粒子位置数组，形状为 (num_particles, num_dimensions)

        Returns:
            粒子路径成本数组，形状为 (num_particles,)
        """
        # 初始化粒子路径成本
        num_particles = positions.shape[0]
        costs = np.zeros(num_particles)

        # 当前迭代进度
        iteration = len(self.best_path_points) - 1
        iteration_progress = iteration / self.algorithm_args.max_iterations

        # 定义各项惩罚权重
        distance_to_destination_penalty_weight = 0.1 + 0.3 * iteration_progress
        distance_to_shortest_line_penalty_weight = 0.4 - 0.1 * (iteration_progress**3)
        path_direction_angle_penalty_weight = 0.5 + 0.5 * iteration_progress
        point_gathering_penalty_weight = (0.3 + 0.4 * iteration_progress) * (0.5 if iteration_progress < 0.3 else 1.0)
        terrain_collision_penalty_weight = 0.7 + 0.3 * iteration_progress

        ### 到终点的距离惩罚，鼓励路径点向终点靠拢
        destination = self.destination.to_ndarray()
        distance_to_destination = np.linalg.norm(positions - destination, axis=1)
        costs += ndarrays.normalize(distance_to_destination) * distance_to_destination_penalty_weight

        ### 偏离起点到终点的直线惩罚，鼓励路径点向最短路径靠拢
        start_point = self.start_point.to_ndarray()
        distance_to_line = ndarrays.point_to_line_distance(positions, start_point, destination)
        costs += ndarrays.normalize(distance_to_line) * distance_to_shortest_line_penalty_weight

        ### 路径方向约束惩罚，基于方向夹角，鼓励相邻路径线段不要有太大转折
        if len(self.best_path_points) >= 3:
            point1 = self.best_path_points[-2].to_ndarray()
            point2 = self.best_path_points[-1].to_ndarray()
            angles = np.array([ndarrays.angle_degrees(point1, point2, point3) for point3 in positions])
            path_direction_penalty = np.sin(angles)  # 越偏离方向惩罚越大
            costs += path_direction_penalty * path_direction_angle_penalty_weight

        ### 路径点聚集惩罚，鼓励路径点之间有更合理的间距
        previous_point = self.best_path_points[-1]
        point_gathering_costs = np.linalg.norm(positions - previous_point.to_ndarray(), axis=1)
        point_gathering_costs = np.clip(point_gathering_costs, 10, 40)
        costs += ndarrays.normalize(point_gathering_costs) * point_gathering_penalty_weight

        ### 路径点靠近山体惩罚，鼓励路径点远离山体
        terrain_proximity_costs = np.array(
            [self.terrain.min_horizontal_distance_to_peak_centers(Point.from_ndarray(p)) for p in positions]
        )
        terrain_proximity_costs = np.exp(-terrain_proximity_costs)  # 越近惩罚越大
        costs += terrain_proximity_costs * terrain_collision_penalty_weight

        ### 路径点碰撞山体惩罚，鼓励路径点不要与山体碰撞
        for p in positions:
            point = Point.from_ndarray(p)
            # 检查点是否碰撞
            if self.terrain.check_point_collision(point):
                costs += 10 * terrain_collision_penalty_weight
            # 检查线段是否碰撞
            collision_points = self.terrain.check_line_segment_collision_points(previous_point, point)
            if len(collision_points) > 0:
                costs += len(collision_points) * terrain_collision_penalty_weight

        # 选择成本最优且满足约束条件的点
        candidate_point = self.choose_candidate_point(costs, positions, previous_point)
        if candidate_point is not None:
            self.best_path_points.append(candidate_point)

        return costs

    def choose_candidate_point(self, costs: np.ndarray, positions: np.ndarray, previous_point: Point) -> Point | None:
        """
        选择成本最优且满足约束条件的路径点

        Args:
            costs: 当前路径总成本数组，形状为 (num_particles,)
            positions: 粒子位置数组，形状为 (num_particles, num_dimensions)
            previous_point: 前一个路径点

        Returns:
            成本最优且满足约束条件的路径点
        """
        sorted_indices = np.argsort(costs)
        for index in sorted_indices:
            candidate_point = Point.from_ndarray(positions[index])

            # 丢弃会发生地形碰撞的点
            if self.terrain.check_point_collision(candidate_point):
                continue

            # 丢弃会发生地形穿透的点
            if self.terrain.check_line_segment_collision_points(previous_point, candidate_point):
                continue

            # 丢弃偏离终点高度大于0.1的点
            height_diff_to_destination = abs(candidate_point.z - self.destination.z)
            if height_diff_to_destination - 0.1 > Consts.EPSILON:
                continue

            # 丢弃相邻路径段水平转角大于15度的点
            if len(self.best_path_points) >= 3:
                point1 = self.best_path_points[-2].to_ndarray()
                point2 = self.best_path_points[-1].to_ndarray()
                angle = ndarrays.angle_degrees(point1, point2, candidate_point.to_ndarray())
                if angle - 15 > Consts.EPSILON:
                    continue

            return candidate_point

        return None


if __name__ == "__main__":
    pso_args = AlgorithmArguments(
        num_particles=100,
        num_dimensions=3,
        max_iterations=100,
        position_boundaries_min=(0, 0, 1),
        position_boundaries_max=(100, 100, 5),
        velocity_bound_max=1,
        inertia_weight_max=1.8,
        inertia_weight_min=0.5,
        cognitive_coefficient=1.6,
        social_coefficient=1.2,
    )

    path_planner = PathPlanner3D(
        algorithm_args=pso_args,
        start_point=Point(x=0, y=0, z=1),
        destination=Point(x=80, y=80, z=2),
        peaks=[
            Peak(center_x=20, center_y=20, amplitude=5, width=8),
            Peak(center_x=20, center_y=70, amplitude=5, width=8),
            Peak(center_x=60, center_y=20, amplitude=5, width=8),
            Peak(center_x=60, center_y=70, amplitude=5, width=8),
        ],
    )

    pso_optimizer = ParticleSwarmOptimizer(
        args=pso_args,
        problem_type=ProblemType.MIN,
        objective_function=path_planner.calculate_best_path_costs,
    )

    best_fitness_values = pso_optimizer.start_iterating()
    path_planner.plot_result(best_fitness_values, mark_waypoint=False)
