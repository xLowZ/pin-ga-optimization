import numpy as np
import matplotlib.pyplot as plt

from algorithm import Solution
from utils import RASTRIGIN, ACKLEY, SPHERE, EASOM, MCCORMICK

def plot_fitness_classification(classification: list[int]) -> None:
    """
        Plot a pie chart of the fitness classification.

    Args:
        classification (list[int]): List with the count of solutions in each classification.
    """

    LABELS = ['Excellent', 'Good', 'Average', 'Poor']
    COLORS = ['#ff6666', '#66ff66', '#6666ff', '#ff9966']
    
    filtered_labels = [label for label, value in zip(LABELS, classification) if value > 0]
    filtered_classification = [value for value in classification if value > 0]
    filtered_colors = [color for color, value in zip(COLORS, classification) if value > 0]

    plt.figure(figsize=(6, 6))
    plt.pie(filtered_classification, labels=filtered_labels, colors=filtered_colors, autopct='%1.1f%%', startangle=140)
    plt.title('Fitness Classification')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def classify_fitness(solutions: list[Solution], target_fnc: int) -> list[int]:
    """
        Classify the fitness value of the solutions.

        Args:
            solutions (list[Solution]): List of solutions.

        Returns:
            list[int]: List with the count of solutions in each classification.
    """

    classification = [0, 0, 0, 0]

    if target_fnc in (RASTRIGIN, ACKLEY, SPHERE):

        EXCELLENT_THRESHOLD = 1e-9

        GOOD_THRESHOLD = 1e-7

        AVERAGE_THRESHOLD = 1e-5

        for solution in solutions:

            fitness = solution.get_fitness()
            if fitness < EXCELLENT_THRESHOLD:
                classification[0] += 1  # Excellent
            elif EXCELLENT_THRESHOLD <= fitness < GOOD_THRESHOLD:
                classification[1] += 1  # Good
            elif GOOD_THRESHOLD <= fitness < AVERAGE_THRESHOLD:
                classification[2] += 1  # Average
            else:
                classification[3] += 1  # Poor

    elif target_fnc == EASOM:

        EXCELLENT_THRESHOLD = 1e-12

        GOOD_THRESHOLD = 1e-9

        AVERAGE_THRESHOLD = 1e-7

        for solution in solutions:
            fitness = solution.get_fitness()
            fitness_str = f"{fitness:.15f}"  # Convert fitness to string with high precision
            num_nines = fitness_str.count('9')
            abs_fitness = abs(fitness)
            distance_to_optimal = abs(abs_fitness - 1)  # Distance to the optimal value -1

            if distance_to_optimal < EXCELLENT_THRESHOLD:
                classification[0] += 1  # Excellent
            elif distance_to_optimal < GOOD_THRESHOLD:
                classification[1] += 1  # Good
            elif distance_to_optimal < AVERAGE_THRESHOLD:
                classification[2] += 1  # Average
            else:
                classification[3] += 1  # Poor

    elif target_fnc == MCCORMICK:

        EXCELLENT_THRESHOLD = -1.913322294

        GOOD_THRESHOLD = -1.913322292

        AVERAGE_THRESHOLD = -1.91332229

        for solution in solutions:
            fitness = solution.get_fitness()

            if fitness < EXCELLENT_THRESHOLD:
                classification[0] += 1  # Excellent
            elif EXCELLENT_THRESHOLD <= fitness < GOOD_THRESHOLD:
                classification[1] += 1  # Good
            elif GOOD_THRESHOLD <= fitness < AVERAGE_THRESHOLD:
                classification[2] += 1  # Average
            else:
                classification[3] += 1  # Poor

    else:
        raise ValueError("Unknown target function")

    return classification    


def eval_GA(results: list[Solution], target_fnc: int):
    
    solution_values = [sol.get_fitness() for sol in results]

    results = np.sort(results)

    # Basic Statistics
    median_value = np.median(solution_values)
    std_value = np.std(solution_values)
    best_value = np.min(solution_values)
    worst_value = np.max(solution_values)

    print(f"Median of solutions: {median_value}")
    print(f"Standard deviation of solutions: {std_value}")
    print(f"Best Genes found: {results[0].get_genes()}")
    print(f"Best Fitness found: {best_value}")
    print(f"Worst solution found: {worst_value}")
    

    # plot_fitness_classification(classify_fitness(results, target_fnc))

    # # Visualization
    # plt.hist(solution_values, bins=30, alpha=0.75, color='blue')
    # plt.title('Distribution of solution values')
    # plt.xlabel('Objective function value')
    # plt.ylabel('Frequency')
    # plt.show()


def conversion_visualization(trajectories: list[list[float]]) -> None:
    # Visualize the convergence
    for trajectory in trajectories:
        plt.plot(trajectory, alpha=0.5, color='blue')

    plt.title('Genetic Algorithm Convergence')
    plt.xlabel('Generation')
    plt.ylabel('Best solution value')
    plt.show()

if __name__ == "__main__":

    print('')
