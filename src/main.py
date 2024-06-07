import sys
import os
import json
import subprocess

sys.path.insert(0, os.path.abspath(os.curdir))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
import time
#import matplotlib.pyplot as plt

# from benchmark import *
from utils import RASTRIGIN, ACKLEY, SPHERE, EASOM, MCCORMICK
from algorithm import *
from UI.interface import get_user_inputs


def timer(elapsed_time: float) -> None:
    """
        Shows the elapsed time of the algorithm.
    """

    if elapsed_time < 60:
        print(f"Time: {elapsed_time:.4f} seconds")
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Time: {minutes} minutes and {seconds:.1f} seconds")
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        print(f"Time: {hours} hours, {minutes} minutes and {seconds:.1f} seconds")


# def plot_fitness_classification(classification: list[int]) -> None:
#     """
#         Plot a pie chart of the fitness classification.

#     Args:
#         classification (list[int]): List with the count of solutions in each classification.
#     """
#     labels = ['Excellent', 'Good', 'Average', 'Poor']
#     colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
#     plt.figure(figsize=(8, 8))
#     plt.pie(classification, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
#     plt.title('Fitness Classification')
#     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     plt.show()


def classify_fitness(solutions: list[Solution], target_fnc: int) -> list[int]:
    """
        Classify the fitness value of the solutions.

        Args:
            solutions (list[Solution]): List of solutions.

        Returns:
            list[int]: List with the count of solutions in each classification.
    """

    classification = [0, 0, 0, 0]

    if target_fnc == RASTRIGIN or ACKLEY or SPHERE:

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
        raise NotImplementedError

    else:
        raise NotImplementedError


    return classification    


def eval_GA(results: list[Solution], target_fnc):
    
    solution_values = [sol.get_fitness() for sol in results]

    # Basic Statistics
    mean_value = np.mean(solution_values)
    std_value = np.std(solution_values)
    best_value = np.min(solution_values)
    worst_value = np.max(solution_values)

    print(f"Mean of solutions: {mean_value}")
    print(f"Standard deviation of solutions: {std_value}")
    print(f"Best solution found: {best_value}")
    print(f"Worst solution found: {worst_value}")

    # plot_fitness_classification(classify_fitness(results, target_fnc))

    # # Visualization
    # plt.hist(solution_values, bins=30, alpha=0.75, color='blue')
    # plt.title('Distribution of solution values')
    # plt.xlabel('Objective function value')
    # plt.ylabel('Frequency')
    # plt.show()


# def conversion_visualization(trajectories: list[list[float]]) -> None:
#     # Visualize the convergence
#     for trajectory in trajectories:
#         plt.plot(trajectory, alpha=0.5, color='blue')

#     plt.title('Genetic Algorithm Convergence')
#     plt.xlabel('Generation')
#     plt.ylabel('Best solution value')
#     plt.show()


def genetic_algorithm(*, nIterations: int, pop_size: int, mutation_rate: float, target_function: int, dimensions: int) -> tuple[Solution, float]:

    best_solutions_per_generation = []

    population: list[Solution] = initialization(pop_size, nDimensions=dimensions, target_fnc=target_function)

    for generation in range(nIterations):

        population = create_new_gen(population, mutation_rate, pop_size, target_function)

        print(f"{generation+1}. Generation -->> Fitness --> {population[0].get_fitness()}")

        best_solutions_per_generation.append(population[0].get_fitness())

    return population[0], best_solutions_per_generation


def VARgenetic_algorithm(*, nIterations: int, pop_size: int, mutation_rate: float, final_mutation_rate, target_function: int, dimensions: int) -> Solution:

    best_solutions_per_generation = []

    population: list[Solution] = initialization(pop_size, nDimensions=dimensions, target_fnc=target_function)

    for generation in range(nIterations):

        current_mutation_rate = mutation_rate - (generation / nIterations) * (mutation_rate - final_mutation_rate)

        population = VAR_create_new_gen(population, current_mutation_rate, pop_size, target_function)

        print(f"{generation+1}. Generation -->> Fitness --> {population[0].get_fitness()}")

        best_solutions_per_generation.append(population[0].get_fitness())

    return population[0], best_solutions_per_generation


# def main() -> None:

#     np.random.seed(10)

#     solutions = []
#     trajectories = []

#     target_function = MCCORMICK
#     population_size: int = 250
#     number_of_generations: int = 500
#     dimensions: int = 2
#     mutation_rate: float = 0.1
#     final_mutation_rate = 0.01
#     nTests = 5

#     start = time.time()
#     for iteration in range(nTests):

#         best_solution, best_solutions_per_generation = genetic_algorithm(
#             nIterations=number_of_generations,
#             pop_size=population_size,
#             mutation_rate=mutation_rate,
#             target_function=target_function,
#             dimensions=dimensions,
#         )

#         # best_solution, best_solutions_per_generation = VARgenetic_algorithm(
#         #     nIterations=number_of_generations,
#         #     pop_size=population_size,
#         #     mutation_rate=mutation_rate,
#         #     final_mutation_rate=final_mutation_rate,
#         #     target_function=target_function,
#         #     dimensions=dimensions,
#         # )

#         print(f"{iteration+1}. Best solution: {best_solution}")

#         solutions.append(best_solution)
#         trajectories.append(best_solutions_per_generation)
#     end = time.time()

#     eval_GA(solutions, target_function)

#     conversion_visualization(trajectories)    

#     timer(end-start)

def run_ga(benchmark, population_size, number_of_generations, dimensions, mutation_rate, nTests):
    solutions = []
    trajectories = []
    for iteration in range(nTests):
        best_solution, best_solutions_per_generation = genetic_algorithm(
            nIterations=number_of_generations,
            pop_size=population_size,
            mutation_rate=mutation_rate,
            target_function=benchmark,
            dimensions=dimensions,
        )
        solutions.append(best_solution)
        trajectories.append(best_solutions_per_generation)
    return solutions, trajectories

def main() -> None:
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'user_inputs.json')
    subprocess.run(["python", os.path.join(os.path.dirname(__file__), 'UI', 'interface.py')])

    with open(config_path, "r") as f:
        user_inputs = json.load(f)

    benchmarks = user_inputs["benchmarks"]
    population_size = int(user_inputs["population_size"])
    number_of_generations = int(user_inputs["number_of_generations"])
    dimensions = int(user_inputs["dimensions"])
    mutation_rate = float(user_inputs["mutation_rate"])  # Garantir que seja float
    nTests = int(user_inputs["nTests"])

    for benchmark in benchmarks:
        solutions, trajectories = run_ga(eval(benchmark), population_size, number_of_generations, dimensions, mutation_rate, nTests)
        eval_GA(solutions, eval(benchmark))
        #conversion_visualization(trajectories)

if __name__ == "__main__":
    main()



# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Test Functions ---

def teste():
    np.random.seed(10)

    population_size: int = 100
    dimensions: int = 2
    mutation: float = 0.1

    new_gen = initialization(
        population_size, nDimensions=dimensions, target_fnc=RASTRIGIN
    )

    for individual in new_gen:

        individual.calc_fitness(RASTRIGIN)
        print(individual.get_fitness())

    new_gen.sort()

    print([ind.get_genes() for ind in new_gen])


def teste0():

    np.random.seed(10)

    population_size: int = 20_000
    dimensions: int = 2
    mutation: float = 0.1

    start = time.time()
    new_gen = initialization(
        population_size, nDimensions=dimensions, target_fnc=RASTRIGIN
    )
    new_gen2 = initialization(
        population_size, nDimensions=dimensions, target_fnc=ACKLEY
    )
    new_gen3 = initialization(
        population_size, nDimensions=dimensions, target_fnc=SPHERE
    )

    # for individual in new_gen:

    #     individual.calc_fitness(RASTRIGIN)

    # for individual in new_gen2:

    #     individual.calc_fitness(ACKLEY)

    # for individual in new_gen3:

    #     individual.calc_fitness(SPHERE)

    # Classe Population?
    ras_values = np.sort([individual.get_fitness() for individual in new_gen])
    ack_values = np.sort([individual.get_fitness() for individual in new_gen2])
    sph_values = np.sort([individual.get_fitness() for individual in new_gen3])

    # best_solution = np.min(fitness_values)
    end = time.time()

    print("Rastrigin best solutions:")
    for i in range(3):
        print(ras_values[i])

    print("Ackley best solutions:")
    for i in range(3):
        print(ack_values[i])

    print("Sphere best solutions:")
    for i in range(3):
        print(sph_values[i])

    print(Solution.info())
    print(f"Time: {end - start:.4f} seconds")


def variacao_mut():
    
    np.random.seed(10)

    solutions = []
    trajectories = []

    population_size: int = 300
    number_of_generations: int = 700
    dimensions: int = 2
    mutation_rate = 0.1
    final_mutation_rate = 0.02
    nTests = 1


    for iteration in range(nTests):

        # best_solution, best_solutions_per_generation = VARgenetic_algorithm(
        #     nIterations=number_of_generations,
        #     pop_size=population_size,
        #     mutation_rate=mutation_rate,
        #     final_mutation_rate=final_mutation_rate,
        #     target_function=RASTRIGIN,
        #     dimensions=dimensions,
        # )

        # print(f"{iteration+1}. Best solution VAR: {best_solution}")

        best_solution, best_solutions_per_generation = genetic_algorithm(
            nIterations=number_of_generations,
            pop_size=population_size,
            mutation_rate=mutation_rate,
            target_function=RASTRIGIN,
            dimensions=dimensions,
        )

        print(f"{iteration+1}. Best solution:     {best_solution}\n")

        solutions.append(best_solution)
        trajectories.append(best_solutions_per_generation)

    eval_GA(solutions)

    # conversion_visualization(trajectories)    


# if __name__ == "__main__":
#     main()
#     # variacao_mut()
