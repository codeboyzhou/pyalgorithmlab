from collections.abc import Callable

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pyalgorithmlab.common.types import ProblemType
from pyalgorithmlab.util import convergence


class PSOAlgorithmArguments(BaseModel):
    """定义粒子群优化算法（Particle Swarm Optimization）改进版核心参数"""

    num_particles: int
    """粒子群规模，即粒子个数"""

    num_dimensions: int
    """待优化问题的维度，即问题涉及到的自变量个数"""

    max_iterations: int
    """最大迭代次数"""

    inertia_weight_max: float
    """
    最大惯性权重
    用于实现惯性权重的线性递减
    初始值较大可以增加算法的探索能力
    """

    inertia_weight_min: float
    """
    最小惯性权重
    用于实现惯性权重的线性递减
    最终值较小可以增加算法的开发能力
    """

    cognitive_coefficient_min: float
    """认知系数C1最小值"""

    cognitive_coefficient_max: float
    """认知系数C1最大值"""

    social_coefficient_min: float
    """社会系数C2最小值"""

    social_coefficient_max: float
    """社会系数C2最大值"""

    position_boundaries_min: tuple[float, ...]
    """
    粒子位置下界，即允许自变量可取的最小值
    tuple类型，有几个自变量，就有几个元素
    对于无约束的问题，可以设置一个很小的值
    """

    position_boundaries_max: tuple[float, ...]
    """
    粒子位置上界，即允许自变量可取的最大值
    tuple类型，有几个自变量，就有几个元素
    对于无约束的问题，可以设置一个很大的值
    """

    velocity_bound_max: float
    """
    速度上界，因为速度是矢量
    所以上界取反方向就可以得到速度下界
    目的是为了平衡算法的探索能力和开发能力
    """


class GeneticAlgorithmArguments(BaseModel):
    """定义遗传算法（Genetic Algorithm）改进版核心参数"""

    population_size: int
    """种群规模，即个体个数"""

    num_dimensions: int
    """待优化问题的维度，即问题涉及到的自变量个数"""

    tournament_size: int
    """锦标赛小组规模，即每次锦标赛中参与比较的个体个数"""

    max_generations: int
    """最大进化代数"""

    individual_available_boundaries_min: tuple[float, ...]
    """
    个体活动下界，即允许自变量可取的最小值
    tuple类型，有几个自变量，就有几个元素
    对于无约束的问题，可以设置一个很小的值
    """

    individual_available_boundaries_max: tuple[float, ...]
    """
    个体活动上界，即允许自变量可取的最大值
    tuple类型，有几个自变量，就有几个元素
    对于无约束的问题，可以设置一个很大的值
    """

    crossover_rate: float
    """交叉概率"""

    mutation_rate: float
    """变异概率"""


class EnhancedGeneticAlgorithm:
    """遗传算法（Genetic Algorithm）改进版核心实现，会将这个算法的实现作为粒子群算法的优化算子"""

    def __init__(self, args: GeneticAlgorithmArguments, problem_type: ProblemType) -> None:
        """
        初始化遗传算法

        Args:
            args: 遗传算法参数
            problem_type: 问题类型
        """
        logger.success(f"初始化遗传算法，使用以下参数：{args.model_dump_json(indent=4)}")

        self.args = args
        self.problem_type = problem_type

        initial_fitness = np.inf if problem_type == ProblemType.MIN else -np.inf
        self.fitness = np.full(self.args.population_size, initial_fitness)

        logger.success("遗传算法初始化成功")

    def select(self, advantage_individuals: np.ndarray) -> np.ndarray:
        """
        选择：从PSO算法产生的优势个体中选择优良个体作为父代

        策略：锦标赛选择

        Args:
            advantage_individuals: PSO算法产生的优势个体

        Returns:
            父代选择结果
        """
        individual_indices = np.arange(self.args.population_size)
        tournament_size = self.args.tournament_size
        selected_as_parents = []
        for i in range(self.args.population_size):
            random_indices = np.random.choice(individual_indices, size=tournament_size, replace=False)
            individual_fitness_values = self.fitness[random_indices]
            best_fitness_index = (
                np.argmin(individual_fitness_values)
                if self.problem_type == ProblemType.MIN
                else np.argmax(individual_fitness_values)
            )
            winner_index = random_indices[best_fitness_index]
            selected_as_parents.append(advantage_individuals[winner_index])
        return np.array(selected_as_parents)

    def crossover(
        self, selected_advantage_individual: np.ndarray, disadvantage_individuals: np.ndarray, current_iteration: int
    ) -> np.ndarray:
        """
        交叉：select()结果作为父本，disadvantage_individuals作为母本，进行交叉

        策略：采用基于交叉概率的算数交叉和模拟二进制交叉结合的交叉策略

        Args:
            selected_advantage_individual: 选择结果中的优势个体
            disadvantage_individuals: PSO算法产生的劣势个体
            current_iteration: 当前迭代次数

        Returns:
            子代选择结果，形状为 (population_size, num_dimensions)
        """
        # 动态计算交叉率
        sin_coefficient = 0.5
        golden_ratio = (1 + np.sqrt(5)) / 2
        crossover_rate_decay = 0.01
        computed_crossover_rate = (
            sin_coefficient
            * (1 + np.sin(2 * np.pi * self.args.crossover_rate / (1 + golden_ratio)))
            * np.exp(-crossover_rate_decay * current_iteration)
        )

        # 动态计算交叉参数beta
        particle_distribution_coefficient_eta = 2 * (1 + 0.2 * np.cos(2 * np.pi * current_iteration / 15))
        power = 1 / (particle_distribution_coefficient_eta + 1)
        random_num = np.random.random()
        if random_num <= 0.5:
            crossover_beta = np.power(2 * random_num, power)
        else:
            crossover_beta = np.power(1 / (2 - 2 * random_num), power)

        # 模拟二进制交叉（simulated binary crossover）
        random_mate_index = np.random.randint(self.args.population_size)
        mate = disadvantage_individuals[random_mate_index]
        child1 = computed_crossover_rate * selected_advantage_individual + (1 - computed_crossover_rate) * mate
        child2 = computed_crossover_rate * mate + (1 - computed_crossover_rate) * selected_advantage_individual

        # 子代增加t分布扰动，增加种群搜索多样性，防止算法陷入局部最优
        standard_t_value = np.random.standard_t(df=3, size=selected_advantage_individual.shape)
        child1 = 0.5 * ((1 + crossover_beta) * child1 + (1 - crossover_beta) * child2) + 2 * standard_t_value
        child2 = 0.5 * ((1 + crossover_beta) * child2 + (1 - crossover_beta) * child1) + 2 * standard_t_value

        # 限制子代基因值在有效范围内
        child1 = np.clip(
            child1, self.args.individual_available_boundaries_min, self.args.individual_available_boundaries_max
        )
        child2 = np.clip(
            child2, self.args.individual_available_boundaries_min, self.args.individual_available_boundaries_max
        )

        # 随机选择一个子代
        random_index = np.random.randint(2)
        return child1 if random_index == 0 else child2

    def mutate(self, individual: np.ndarray, current_iteration: int) -> np.ndarray:
        """
        变异：随机改变个体的基因，增加种群多样性，防止算法陷入局部最优

        策略：基于莱维飞行变异策略

        Args:
            individual: 个体
            current_iteration: 当前迭代次数

        Returns:
            变异后的个体
        """
        # 动态计算变异率
        sin_coefficient = 0.5
        golden_ratio = (1 + np.sqrt(5)) / 2
        mutation_rate_decay = 0.01
        computed_mutation_rate = (
            sin_coefficient
            * (1 + np.sin(2 * np.pi * self.args.mutation_rate / (1 + golden_ratio)))
            * np.exp(-mutation_rate_decay * current_iteration)
        )

        # 计算莱维飞行步长
        gaussian_distribution_random_num_u = np.random.normal(loc=0, scale=1, size=individual.shape)
        gaussian_distribution_random_num_v = np.random.normal(loc=0, scale=1, size=individual.shape)
        random_num1 = np.random.random()
        random_num2 = np.random.random()
        levy_flight_alpha = max(0.6 - 0.1 * current_iteration / self.args.max_generations, 0.5)
        levy_flight_beta = max(1.5 - 0.1 * current_iteration / self.args.max_generations, 1)
        levy_flight_step_size = (
            gaussian_distribution_random_num_u
            / np.power(np.abs(gaussian_distribution_random_num_v), 1 / levy_flight_beta)
            * levy_flight_alpha
            * random_num1
            * np.power(np.abs(random_num2), 1 / levy_flight_beta)
        )

        # 子代变异
        mutated_individual = individual + computed_mutation_rate * levy_flight_step_size
        # 限制子代基因值在有效范围内
        mutated_individual = np.clip(
            mutated_individual,
            self.args.individual_available_boundaries_min,
            self.args.individual_available_boundaries_max,
        )
        return mutated_individual

    def start_iterating(
        self,
        objective_function: Callable[[np.ndarray], np.ndarray],
        advantage_individuals: np.ndarray,
        disadvantage_individuals: np.ndarray,
    ) -> list[float]:
        """
        开始迭代

        Args:
            objective_function: 待优化问题的目标函数
            advantage_individuals: PSO算法产生的优势个体
            disadvantage_individuals: PSO算法产生的劣势个体

        Returns:
            每次迭代后的最优适应度，全部记录下来用于绘制迭代曲线
        """
        best_fitness_values = []  # 每次迭代后的最优适应度，全部记录下来用于绘制迭代曲线
        for generation in range(self.args.max_generations):
            # 提前收敛检查
            if convergence.is_converged(best_fitness_values):
                logger.success(f"遗传算法在第{generation}次迭代后已经收敛")
                break

            # 评估当前种群中所有个体的适应度
            self.fitness = objective_function(disadvantage_individuals)

            # 记录当前代的最优适应度
            best_fitness = np.min(self.fitness) if self.problem_type == ProblemType.MIN else np.max(self.fitness)
            best_fitness_values.append(best_fitness)

            # 选择父代
            selected_advantage_individuals = self.select(advantage_individuals)

            # 生成子代
            offspring = []
            for father in selected_advantage_individuals:
                child = self.crossover(father, disadvantage_individuals, generation)
                child = self.mutate(child, generation)
                offspring.append(child)

            # 个体进化
            disadvantage_individuals = np.array(offspring)

        logger.success(f"遗传算法迭代结束，当前最优适应度为{best_fitness_values[-1]:.6f}")
        return best_fitness_values


class EnhancedParticleSwarmOptimizer:
    """粒子群优化算法（Particle Swarm Optimization）改进版核心实现"""

    def __init__(
        self,
        args: PSOAlgorithmArguments,
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
        # 高斯扰动惯性权重
        inertia_weight = self.args.inertia_weight_min + (
            self.args.inertia_weight_max - self.args.inertia_weight_min
        ) * np.exp(-np.sin(2 * current_iteration**2 / self.args.max_iterations))
        # 高斯扰动认知系数
        cognitive_coefficient = self.args.cognitive_coefficient_min + (
            self.args.cognitive_coefficient_max - self.args.cognitive_coefficient_min
        ) * np.exp(-np.sin(2 * current_iteration**2 / self.args.max_iterations))
        # 高斯扰动社会系数
        social_coefficient = self.args.social_coefficient_min + (
            self.args.social_coefficient_max - self.args.social_coefficient_min
        ) * np.exp(-np.sin(2 * current_iteration**2 / self.args.max_iterations))
        # 计算速度更新
        r1 = np.random.rand(self.args.num_particles, self.args.num_dimensions)
        r2 = np.random.rand(self.args.num_particles, self.args.num_dimensions)
        self.velocities = (
            inertia_weight * self.velocities
            + cognitive_coefficient * r1 * (self.individual_best_positions - self.positions)
            + social_coefficient * r2 * (self.global_best_positions - self.positions)
            + 0.5 * np.random.randn(self.args.num_particles, self.args.num_dimensions)
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
        self.fitness = self.objective_function(self.positions)
        better_indices = (
            self.fitness < self.individual_best_fitness
            if self.problem_type == ProblemType.MIN
            else self.fitness > self.individual_best_fitness
        )
        self.individual_best_fitness[better_indices] = self.fitness[better_indices]
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

    def start_iterating(self, genetic_algorithm_operator: EnhancedGeneticAlgorithm) -> list[float]:
        """
        开始执行算法迭代

        Args:
            genetic_algorithm_operator: 遗传算法算子

        Returns:
            每次迭代后的最优适应度，全部记录下来用于绘制迭代曲线
        """
        best_fitness_values = []  # 每次迭代后的最优适应度，全部记录下来用于绘制迭代曲线

        for iteration in range(self.args.max_iterations):
            # PSO提前收敛检查
            if convergence.is_converged(best_fitness_values):
                logger.success(f"PSO算法在第{iteration}次迭代后已经收敛")
                break

            # PSO迭代核心流程
            self._update_velocities(iteration)
            self._update_positions()
            self._update_individual_best()
            self._update_global_best()
            best_fitness_values.append(self.global_best_fitness)

            # 按适应度值排序
            sorted_indices = np.argsort(self.fitness)
            sorted_individuals = self.positions[sorted_indices]

            # 前一半作为优势个体，后一半作为劣势个体
            advantage_individuals = sorted_individuals[: self.args.num_particles // 2]
            disadvantage_individuals = sorted_individuals[self.args.num_particles // 2 :]

            # PSO继续优化优势个体
            self.positions[: self.args.num_particles // 2] = advantage_individuals

            # 遗传算法优化劣势个体
            genetic_algorithm_operator.start_iterating(
                self.objective_function, advantage_individuals, disadvantage_individuals
            )

        logger.success(f"PSO算法迭代结束，当前最优适应度为{self.global_best_fitness:.6f}")
        return best_fitness_values
