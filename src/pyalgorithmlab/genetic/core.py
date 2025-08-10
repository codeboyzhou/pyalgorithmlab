from collections.abc import Callable

import numpy as np
from loguru import logger

from pyalgorithmlab.common.types import ProblemType
from pyalgorithmlab.genetic.types import AlgorithmArguments
from pyalgorithmlab.util import convergence


class GeneticAlgorithm:
    """遗传算法（Genetic Algorithm）核心实现"""

    def __init__(self, args: AlgorithmArguments, problem_type: ProblemType) -> None:
        """
        初始化遗传算法

        Args:
            args: 遗传算法参数
            problem_type: 问题类型
        """
        logger.success(f"初始化遗传算法，使用以下参数：{args.model_dump_json(indent=4)}")

        self.args = args
        self.problem_type = problem_type

        shape = (args.population_size, args.num_dimensions)
        self.individuals = np.random.uniform(
            args.individual_available_boundaries_min, args.individual_available_boundaries_max, shape
        )

        initial_fitness = np.inf if problem_type == ProblemType.MIN else -np.inf
        self.fitness = np.full(self.args.population_size, initial_fitness)

        logger.success("遗传算法初始化成功")

    def select(self) -> np.ndarray:
        """
        选择：从种群中选择优良个体作为父代

        策略：锦标赛选择

        Returns:
            父代选择结果，形状为 (population_size, num_dimensions)
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
            selected_as_parents.append(self.individuals[winner_index])
        return np.array(selected_as_parents)

    def crossover(self, parent: np.ndarray) -> np.ndarray:
        """
        交叉：由父代产生新的子代个体

        策略：中间交叉（子代是父代之间的线性组合）
        """
        if np.random.rand() < self.args.crossover_rate:
            random_mate_index = np.random.randint(self.args.population_size)
            mate = self.individuals[random_mate_index]
            alpha = np.random.rand(*parent.shape)
            child = alpha * parent + (1 - alpha) * mate
            return np.clip(
                child, self.args.individual_available_boundaries_min, self.args.individual_available_boundaries_max
            )
        return parent

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        变异：随机改变个体的基因，增加种群多样性

        策略：高斯变异
        """
        if np.random.rand() < self.args.mutation_rate:
            noise = np.random.normal(loc=0, scale=0.5, size=individual.shape)
            individual += noise
            individual = np.clip(
                individual, self.args.individual_available_boundaries_min, self.args.individual_available_boundaries_max
            )
        return individual

    def start_iterating(self, objective_function: Callable[[np.ndarray], np.ndarray]) -> list[float]:
        """
        开始迭代
        """
        best_fitness_values = []  # 每次迭代后的最优适应度，全部记录下来用于绘制迭代曲线
        for generation in range(self.args.max_generations):
            # 提前收敛检查
            if convergence.is_converged(best_fitness_values):
                logger.success(f"遗传算法在第{generation}次迭代后已经收敛")
                break

            # 评估当前种群中所有个体的适应度
            self.fitness = objective_function(self.individuals)

            # 记录当前代的最优适应度
            best_fitness = np.min(self.fitness) if self.problem_type == ProblemType.MIN else np.max(self.fitness)
            best_fitness_values.append(best_fitness)

            # 选择父代
            parents = self.select()

            # 生成子代
            offspring = []
            for p in parents:
                child = self.crossover(p)
                child = self.mutate(child)
                offspring.append(child)

            # 个体进化
            self.individuals = np.array(offspring)

        logger.success(f"遗传算法迭代结束，当前最优适应度为{best_fitness_values[-1]:.6f}")
        return best_fitness_values
