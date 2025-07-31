from matplotlib import pyplot as plt


def plot_fitness_change_curve(fitness_history: list[float]) -> None:
    """
    绘制适应度值变化曲线

    Args:
        fitness_history: 适应度值历史记录
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fitness_history, "b-", linewidth=2)
    plt.title("Fitness Change Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.draw()
    plt.pause(1)
    plt.close()
