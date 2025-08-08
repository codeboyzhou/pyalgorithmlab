from pydantic import BaseModel


class AlgorithmArguments(BaseModel):
    """定义粒子群优化算法（Particle Swarm Optimization）核心参数"""

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

    cognitive_coefficient: float
    """认知系数C1"""

    social_coefficient: float
    """社会系数C2"""

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
