from pydantic import BaseModel


class AlgorithmArguments(BaseModel):
    """定义遗传算法（Genetic Algorithm）核心参数"""

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
