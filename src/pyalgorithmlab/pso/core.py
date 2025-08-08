from collections.abc import Callable

import numpy as np
from loguru import logger

from pyalgorithmlab.common.types import ProblemType
from pyalgorithmlab.pso.types import AlgorithmArguments
from pyalgorithmlab.util import convergence


class ParticleSwarmOptimizer:
    """粒子群优化算法（Particle Swarm Optimization）核心实现"""

    def __init__(
        self,
        args: AlgorithmArguments,
        problem_type: ProblemType,
        objective_function: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        初始化粒子群优化算法

        Args:
            args: 算法参数
            problem_type: 问题类型
            objective_function: 待优化问题的目标函数
                接受一个形状为 (num_particles, num_dimensions) 的 numpy 数组，表示每个粒子的位置
                返回一个形状为 (num_particles,) 的 numpy 数组，表示每个粒子的适应度值
        """
        logger.success(f"初始化PSO算法，使用以下参数：{args.model_dump_json(indent=4)}")

        shape = (args.num_particles, args.num_dimensions)
        self.positions = np.random.uniform(args.position_boundaries_min, args.position_boundaries_max, shape)
        self.velocities = np.random.uniform(-args.velocity_bound_max, args.velocity_bound_max, shape)

        self.individual_best_positions = self.positions.copy()
        initial_best_fitness = np.inf if problem_type == ProblemType.MIN else -np.inf
        self.individual_best_fitness = np.full(args.num_particles, initial_best_fitness)

        self.global_best_positions = self.individual_best_positions[0]
        self.global_best_fitness = initial_best_fitness

        self.args = args
        self.problem_type = problem_type
        self.objective_function = objective_function

        logger.success("PSO算法初始化成功")

    def _update_velocities(self, current_iteration: int) -> None:
        """
        更新粒子速度

        Args:
            current_iteration: 当前迭代次数
        """
        # 线性递减惯性权重
        inertia_weight = self.args.inertia_weight_max - (
            self.args.inertia_weight_max - self.args.inertia_weight_min
        ) * (current_iteration / self.args.max_iterations)
        # 计算速度更新
        r1 = np.random.rand(self.args.num_particles, self.args.num_dimensions)
        r2 = np.random.rand(self.args.num_particles, self.args.num_dimensions)
        self.velocities = (
            inertia_weight * self.velocities
            + self.args.cognitive_coefficient * r1 * (self.individual_best_positions - self.positions)
            + self.args.social_coefficient * r2 * (self.global_best_positions - self.positions)
        )
        # 限制速度边界
        self.velocities = np.clip(
            self.velocities,
            -self.args.velocity_bound_max,
            self.args.velocity_bound_max,
        )

    def _update_positions(self) -> None:
        """更新粒子位置"""
        self.positions += self.velocities
        # 限制位置边界
        self.positions = np.clip(self.positions, self.args.position_boundaries_min, self.args.position_boundaries_max)

    def _update_individual_best(self) -> None:
        """更新个体最优解"""
        fitness = self.objective_function(self.positions)
        better_indices = (
            fitness < self.individual_best_fitness
            if self.problem_type == ProblemType.MIN
            else fitness > self.individual_best_fitness
        )
        self.individual_best_fitness[better_indices] = fitness[better_indices]
        self.individual_best_positions[better_indices] = self.positions[better_indices]

    def _update_global_best(self) -> None:
        """更新全局最优解"""
        best_individual_index = (
            np.argmin(self.individual_best_fitness)
            if self.problem_type == ProblemType.MIN
            else np.argmax(self.individual_best_fitness)
        )
        individual_best_fitness = self.individual_best_fitness[best_individual_index]

        if self._compare_best_fitness(individual_best_fitness, self.global_best_fitness):
            self.global_best_fitness = individual_best_fitness
            self.global_best_positions = self.individual_best_positions[best_individual_index]

    def _compare_best_fitness(self, x, y):
        """
        根据问题类型比较两个适应度值

        Args:
            x: 第一个适应度值
            y: 第二个适应度值

        Returns:
            如果问题是最小化问题，返回 x < y
            如果问题是最大化问题，返回 x > y
        """
        return x < y if self.problem_type == ProblemType.MIN else x > y

    def start_iterating(self) -> list[float]:
        """
        开始执行算法迭代

        Returns:
            每次迭代后的最优适应度，全部记录下来用于绘制迭代曲线
        """
        best_fitness_values = []  # 每次迭代后的最优适应度，全部记录下来用于绘制迭代曲线
        for iteration in range(self.args.max_iterations):
            if convergence.is_converged(best_fitness_values):
                logger.success(f"PSO算法在第{iteration}次迭代后已经收敛")
                break
            self._update_velocities(iteration)
            self._update_positions()
            self._update_individual_best()
            self._update_global_best()
            best_fitness_values.append(self.global_best_fitness)
        logger.success(f"PSO算法迭代结束，当前最优适应度为{self.global_best_fitness:.6f}")
        return best_fitness_values
