import numpy as np

from pyalgorithmlab.common.types import ProblemType
from pyalgorithmlab.genetic.core import GeneticAlgorithm
from pyalgorithmlab.genetic.types import AlgorithmArguments
from pyalgorithmlab.util import convergence


def xyz_square_sum_max_problem(individuals: np.ndarray) -> np.ndarray:
    """
    这个函数定义了一个简单的优化问题：最大化 f(x, y, z) = x^2 + y^2 + z^2

    Args:
        individuals: 遗传算法个体数组，形状为 (population_size, num_dimensions)

    Returns:
        np.ndarray: 每个个体的适应度值，形状为 (population_size,)
    """
    return np.sum(individuals**2, axis=1)


def test_xyz_square_sum_max_problem():
    """
    测试 xyz_square_sum_max_problem 函数
    该函数用于测试遗传算法在三维最大化问题上的表现
    """
    # 初始化遗传算法参数
    genetic_arguments = AlgorithmArguments(
        population_size=100,
        num_dimensions=3,
        tournament_size=3,
        max_generations=300,
        individual_available_boundaries_min=(-10, -10, -10),
        individual_available_boundaries_max=(10, 10, 10),
        crossover_rate=0.8,
        mutation_rate=0.2,
    )

    # 初始化遗传算法
    genetic_algorithm = GeneticAlgorithm(args=genetic_arguments, problem_type=ProblemType.MAX)

    # 执行算法迭代
    best_fitness_values = genetic_algorithm.start_iterating(objective_function=xyz_square_sum_max_problem)

    assert convergence.is_converged(best_fitness_values)
