import numpy as np

from pyalgorithmlab.common.types import ProblemType
from pyalgorithmlab.pso.core import ParticleSwarmOptimizer
from pyalgorithmlab.pso.types import AlgorithmArguments
from pyalgorithmlab.util import convergence


def xyz_square_sum_minimization_problem(positions: np.ndarray) -> np.ndarray:
    """
    这个函数定义了一个简单的优化问题：最小化 f(x, y, z) = x^2 + y^2 + z^2

    Args:
        positions: PSO算法粒子的位置数组，形状为 (num_particles, num_dimensions)

    Returns:
        np.ndarray: 每个粒子的适应度值，形状为 (num_particles,)
    """
    return np.sum(positions**2, axis=1)


def test_xyz_square_sum_minimization_problem():
    """
    测试 xyz_square_sum_minimization_problem 函数
    该函数用于测试 PSO 算法在三维最小化问题上的表现
    """
    # 初始化PSO算法参数
    pso_arguments = AlgorithmArguments(
        num_particles=100,
        num_dimensions=3,
        max_iterations=100,
        position_boundaries_min=(-10,),
        position_boundaries_max=(10,),
        velocity_bound_max=1,
        inertia_weight_max=2,
        inertia_weight_min=0.5,
        cognitive_coefficient=0.5,
        social_coefficient=0.5,
    )

    # 初始化PSO优化器
    pso_optimizer = ParticleSwarmOptimizer(
        args=pso_arguments,
        problem_type=ProblemType.MIN,
        objective_function=xyz_square_sum_minimization_problem,
    )

    # 执行算法迭代
    best_fitness_values = pso_optimizer.start_iterating()

    assert convergence.is_converged(best_fitness_values)
