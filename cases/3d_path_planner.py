import numpy as np
import plotly.graph_objects as go
from loguru import logger
from plotly import subplots
from scipy.interpolate import make_interp_spline

from pyalgorithmlab.pso.core import ParticleSwarmOptimizer
from pyalgorithmlab.pso.types import AlgorithmArguments, ProblemType
from pyalgorithmlab.py3d import terrain
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

        # 初始化坐标网格和地形
        x = np.linspace(algorithm_args.position_bounds_min[0], algorithm_args.position_bounds_max[0], 100)
        y = np.linspace(algorithm_args.position_bounds_min[1], algorithm_args.position_bounds_max[1], 100)
        xx, yy = np.meshgrid(x, y)
        zz = terrain.init_mountain_terrain(xx, yy, peaks)
        self.grid = Grid(x=xx, y=yy, z=zz)
        self.peaks = peaks

        # 初始化起点和终点
        self.start_point: Point = start_point
        self.destination: Point = destination

        # 初始化最优路径点
        self.best_path_points: list[Point] = [start_point]

    def plot_result(self, fitness_values: list[float]) -> None:
        """
        绘制实验结果

        Args:
            fitness_values: 适应度值列表

        Returns:
            None
        """
        # 追加终点
        self.best_path_points.append(self.destination)
        # 坐标排序
        self.best_path_points.sort(key=lambda pt: pt.x + pt.y + pt.z)

        # 碰撞纠偏
        for i in range(len(self.best_path_points)):
            # 单个点碰撞场景
            current_point = self.best_path_points[i]
            if terrain.check_point_collision(current_point, self.peaks):
                logger.warning(f"检测到点{current_point}存在地形碰撞风险，正在对碰撞点进行纠偏")
                self.best_path_points[i] = self.correct_collision_point(current_point)

            # 相邻点连接后穿透地形碰撞场景
            if i > 0:
                previous_point = self.best_path_points[i - 1]
                collision_points = terrain.check_line_segment_collision(previous_point, current_point, self.peaks)
                if len(collision_points) > 0:
                    logger.warning(f"检测到线段{previous_point} -> {current_point}在点{collision_points}处穿透地形")
                    self.best_path_points[i:i] = [self.correct_collision_point(p) for p in collision_points]

        self.best_path_points.sort(key=lambda pt: pt.x + pt.y + pt.z)
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

        # 标记起点
        fig.add_trace(
            go.Scatter3d(
                x=[self.start_point.x],
                y=[self.start_point.y],
                z=[self.start_point.z],
                marker={"size": 5, "color": "green", "symbol": "circle"},
                mode="markers",
                name="起点",
            )
        )

        # 标记终点
        fig.add_trace(
            go.Scatter3d(
                x=[self.destination.x],
                y=[self.destination.y],
                z=[self.destination.z],
                marker={"size": 5, "color": "red", "symbol": "x"},
                mode="markers",
                name="终点",
            )
        )

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
        path_trace = go.Scatter3d(x=x, y=y, z=z, line={"width": 10, "color": "green"}, mode="lines", name="目标路径")
        fig.add_trace(path_trace, row=1, col=2)

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

    def correct_collision_point(self, point: Point) -> Point:
        """
        纠偏碰撞点

        Args:
            point: 碰撞点坐标

        Returns:
            纠偏后的碰撞点坐标
        """
        x_min, x_max = self.algorithm_args.position_bounds_min[0], self.algorithm_args.position_bounds_max[0]
        y_min, y_max = self.algorithm_args.position_bounds_min[1], self.algorithm_args.position_bounds_max[1]
        z_min, z_max = self.algorithm_args.position_bounds_min[2], self.algorithm_args.position_bounds_max[2]

        # 定义搜索方向
        directions = [
            # X方向
            (1, 0, 0),
            (-1, 0, 0),
            # Y方向
            (0, 1, 0),
            (0, -1, 0),
            # 对角线方向
            (1, 1, 0),
            (1, -1, 0),
            (-1, 1, 0),
            (-1, -1, 0),
            # Z方向
            (0, 0, 1),
            (0, 0, -1),
        ]

        # 初始化搜索半径
        search_radius = 1

        # 初始化搜索次数
        attempt = 0
        max_attempts = 100

        while attempt < max_attempts:
            attempt += 1
            best_safe_point = None
            min_distance = float("inf")

            for dx, dy, dz in directions:
                # 动态步长：随尝试次数增加搜索范围
                step = search_radius if attempt < 20 else search_radius * (attempt / 20)
                # 保证每个方向搜索结果互不干扰
                new_point = Point(x=point.x, y=point.y, z=point.z)
                # 移动位置
                new_point.x += dx * step
                new_point.y += dy * step
                new_point.z += dz * step
                # 边界约束
                new_point.x = x_min if new_point.x < x_min else x_max if new_point.x > x_max else new_point.x
                new_point.y = y_min if new_point.y < y_min else y_max if new_point.y > y_max else new_point.y
                new_point.z = z_min if new_point.z < z_min else z_max if new_point.z > z_max else new_point.z
                # 每移动一次位置都检查是否已经安全
                if not terrain.check_point_collision(new_point, self.peaks):
                    # 计算距离，选择最近安全点
                    distance = np.linalg.norm(new_point.to_ndarray() - point.to_ndarray())
                    if distance < min_distance:
                        min_distance = distance
                        best_safe_point = new_point

            # 选择最近安全点
            if best_safe_point is not None:
                logger.success(f"碰撞点{point}已被成功纠偏为安全点{best_safe_point}")
                return best_safe_point

            # 扩大搜索范围
            search_radius += 1

        # 最终纠偏没有搜索到安全点，返回到原始点，并抬升高度到最大海拔
        new_point = Point(x=point.x, y=point.y, z=z_max)
        logger.warning(f"未找到安全点，返回原始点，并抬升高度到最大海拔{new_point}")
        return new_point

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
        distance_to_destination_costs = np.linalg.norm(positions - self.destination.to_ndarray(), axis=1)
        costs += ndarrays.min_max_normalize(np.array(distance_to_destination_costs)) * distance_to_destination_weight

        # 粒子偏离成本
        point_deviation_costs = ndarrays.point_to_line_distance(
            positions, self.start_point.to_ndarray(), self.destination.to_ndarray()
        )
        costs += ndarrays.min_max_normalize(point_deviation_costs) * point_deviation_weight

        # 粒子聚集成本
        previous_point = self.best_path_points[-1].to_ndarray()
        point_gathering_costs = np.linalg.norm(positions - previous_point, axis=1)
        costs += ndarrays.min_max_normalize(point_gathering_costs) * point_gathering_weight

        # 地形碰撞成本
        terrain_collision_costs = np.array(
            [1 if terrain.check_point_collision(Point.from_ndarray(p), self.peaks) else 0 for p in positions]
        )
        costs += terrain_collision_costs * terrain_collision_weight

        # 选择成本最小的点作为最优路径点
        best_point = positions[np.argmin(costs)]
        self.best_path_points.append(Point.from_ndarray(best_point))

        return costs


if __name__ == "__main__":
    pso_args = AlgorithmArguments(
        num_particles=100,
        num_dimensions=3,
        max_iterations=100,
        position_bounds_min=(0, 0, 1),
        position_bounds_max=(100, 100, 1),
        velocity_bound_max=1,
        inertia_weight_max=1.8,
        inertia_weight_min=0.5,
        cognitive_coefficient=1.6,
        social_coefficient=1.2,
    )

    path_planner = PathPlanner3D(
        algorithm_args=pso_args,
        start_point=Point(x=0, y=0, z=1),
        destination=Point(x=100, y=100, z=1),
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
    path_planner.plot_result(best_fitness_values)
