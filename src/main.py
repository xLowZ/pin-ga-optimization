import sys
import os
import json
import subprocess
import multiprocessing

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

        print(f"{generation+1}. Generation -->> Fitness --> {population[0].get_fitness()}")

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

        print(f"{generation+1}. Generation -->> Fitness --> {population[0].get_fitness()}")

        best_solutions_per_generation.append(population[0].get_fitness())

    return population[0], best_solutions_per_generation



def run_ga(params):
    benchmark, population_size, number_of_generations, dimensions, mutation_rate, final_mutation_rate, mutation_strength, final_mutation_strength, nTests = params
    solutions = []
    trajectories = []
    for iteration in range(nTests):
        best_solution, best_solutions_per_generation = msGA(
            nIterations=number_of_generations,
            pop_size=population_size,
            mutation_rate=mutation_rate,
            final_mutation_rate=final_mutation_rate,
            ms_rate=mutation_strength,
            final_ms_rate=final_mutation_strength,
            target_function=benchmark,
            dimensions=dimensions,
        )
        solutions.append(best_solution)
        trajectories.append(best_solutions_per_generation)
    return benchmark, solutions, trajectories

def run_operations():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'user_inputs.json')
    results_path = os.path.join(os.path.dirname(__file__), 'config', 'results.json')

    with open(config_path, "r") as f:
        user_inputs = json.load(f)

    benchmarks = user_inputs["benchmarks"]
    population_size = user_inputs["population_size"]
    number_of_generations = user_inputs["number_of_generations"]
    dimensions = user_inputs["dimensions"]
    mutation_rate = user_inputs["mutation_rate"]
    final_mutation_rate = user_inputs["final_mutation_rate"]
    mutation_strength = user_inputs["mutation_strength"]
    final_mutation_strength = user_inputs["final_mutation_strength"]
    nTests = user_inputs["nTests"]

    benchmark_map = {
        "RASTRIGIN": RASTRIGIN,
        "ACKLEY": ACKLEY,
        "SPHERE": SPHERE,
        "EASOM": EASOM,
        "MCCORMICK": MCCORMICK
    }

    all_results = []

    # Preparar parâmetros para multiprocessing
    params_list = [
        (benchmark_map[benchmark], population_size, number_of_generations, dimensions, mutation_rate, final_mutation_rate, mutation_strength, final_mutation_strength, nTests)
        for benchmark in benchmarks
    ]

    with multiprocessing.Pool() as pool:
        results = pool.map(run_ga, params_list)

    for benchmark, solutions, trajectories in results:
        benchmark_name = [k for k, v in benchmark_map.items() if v == benchmark][0]
        benchmark_results = []
        for i, solution in enumerate(solutions):
            result = {
                "test_number": i + 1,
                "mean_value": float(np.mean([sol.get_fitness() for sol in solutions])),
                "std_value": float(np.std([sol.get_fitness() for sol in solutions])),
                "best_genes": solution.get_genes().tolist(),
                "best_value": float(solution.get_fitness()),
                "worst_value": float(max([sol.get_fitness() for sol in solutions]))
            }
            benchmark_results.append(result)
        
        all_results.append({
            "benchmark_name": benchmark_name,
            "results": benchmark_results
        })

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

def clear_user_inputs():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'user_inputs.json')
    with open(config_path, "w") as f:
        json.dump({}, f)

def main() -> None:
    subprocess.run(["python", os.path.join(os.path.dirname(__file__), 'UI', 'interface.py')])

    config_path = os.path.join(os.path.dirname(__file__), 'config', 'user_inputs.json')
    # Verifica se o arquivo user_inputs.json foi criado
    if not os.path.exists(config_path) or os.stat(config_path).st_size == 0:
        print("User input was not provided. Exiting the program.")
        return

    run_operations()
    
    # Abrir a interface de resultados
    subprocess.run(["python", os.path.join(os.path.dirname(__file__), 'UI', 'results_interface.py')])

    # Limpar o conteúdo de user_inputs.json
    clear_user_inputs()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Necessário para compatibilidade no Windows
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


if __name__ == "__main__":
    main()
    # variacao_mut()
