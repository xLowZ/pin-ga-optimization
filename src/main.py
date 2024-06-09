import sys
import os

sys.path.insert(0, os.path.abspath(os.curdir))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import time
import matplotlib.pyplot as plt

# from benchmark import *
from utils import RASTRIGIN, ACKLEY, SPHERE, EASOM, MCCORMICK, eval_GA, conversion_visualization
from algorithm import *


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


def genetic_algorithm(*, nIterations: int, pop_size: int, mutation_rate: float, target_function: int, dimensions: int) -> tuple[Solution, float]:

    best_solutions_per_generation = []

    population: list[Solution] = initialization(pop_size, nDimensions=dimensions, target_fnc=target_function)

    for generation in range(nIterations):

        # population = VAR_create_new_gen(population, mutation_rate, pop_size, target_function)
        population = create_new_gen(population, mutation_rate, pop_size, target_function)

        # print(f"{generation+1}. Generation -->> Fitness --> {population[0].get_fitness()}")

        best_solutions_per_generation.append(population[0].get_fitness())

    return population[0], best_solutions_per_generation



def msGA(*, nIterations: int, pop_size: int, mutation_rate: float, final_mutation_rate: float=0.01, ms_rate: float=0.2, final_ms_rate: float=0.0001, target_function: int, dimensions: int) -> Solution:

    best_solutions_per_generation = []

    population: list[Solution] = initialization(pop_size, nDimensions=dimensions, target_fnc=target_function)

    for generation in range(nIterations):

        def linear_decay(initial_rate, final_rate):
            return initial_rate - (generation / nIterations) * (initial_rate - final_rate)

        current_mutation_rate = linear_decay(mutation_rate, final_mutation_rate)
        mutation_strength = linear_decay(ms_rate, final_ms_rate)

        # population = VAR_create_new_gen(population, current_mutation_rate, pop_size, target_function, mutation_strength)
        population = create_new_gen(population, current_mutation_rate, pop_size, target_function, mutation_strength)

        # print(f"{generation+1}. Generation -->> Fitness --> {population[0].get_fitness()}")

        best_solutions_per_generation.append(population[0].get_fitness())

    return population[0], best_solutions_per_generation


def main() -> None:

    np.random.seed(10)

    solutions = []
    trajectories = []

    target_function = MCCORMICK
    population_size: int = 250
    number_of_generations: int = 2000
    dimensions: int = 2
    mutation_rate: float = 0.1
    final_mutation_rate: float = 0.1
    mutation_strength: float = 0.2
    final_mutation_strength: float = 0.001
    nTests = 4

    start = time.time()
    for iteration in range(nTests):

        # best_solution, best_solutions_per_generation = genetic_algorithm(
        #     nIterations=number_of_generations,
        #     pop_size=population_size,
        #     mutation_rate=mutation_rate,
        #     target_function=target_function,
        #     dimensions=dimensions,
        # )

        best_solution, best_solutions_per_generation = msGA(
            nIterations=number_of_generations,
            pop_size=population_size,
            mutation_rate=mutation_rate,
            final_mutation_rate=final_mutation_rate,
            ms_rate=mutation_strength,
            final_ms_rate=final_mutation_strength,
            target_function=target_function,
            dimensions=dimensions,
        )

        print(f"{iteration+1}. Best solution: {best_solution}")

        solutions.append(best_solution)
        trajectories.append(best_solutions_per_generation)
    end = time.time()

    eval_GA(solutions, target_function)

    # if nTests < 4:
    #     conversion_visualization(trajectories)    

    timer(end-start)


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


if __name__ == "__main__":
    main()
    # variacao_mut()
