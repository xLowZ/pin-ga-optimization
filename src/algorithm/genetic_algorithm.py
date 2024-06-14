#########################################################
#                                                       #
#    File containing the core of the Genetic Algorithm  #
#                                                       #
#    Functions for:                                     #
#                                                       #
#        Initialization;                                #
#        Selection;                                     #
#        Crossover;                                     #
#        Find best;                                     #
#        Create new generation;                         #
#                                                       #
#########################################################


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithm import Solution
from utils import *
import numpy as np


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Initialization ---

def initialization(pop_size: int, *, nDimensions: int, target_fnc: int) -> list[Solution]:
    """
        Genetic Algorithm initialization function.

    Args:
        pop_size (int): Size of the population
        nDimensions (int): Number of dimensions
        target_fnc (int): Aimed Function

    Returns:
        list[Solution]: Initial population of individuals (solutions | chromosomes)
    """

    initial_population: list[Solution] = []

    if target_fnc in (RASTRIGIN, ACKLEY, SPHERE, EASOM):
        # Create solutions in target function bounds range
        for _ in range(pop_size):

            new_solution = Solution(np.random.uniform(low=BOUNDS[target_fnc][LOWER], high=BOUNDS[target_fnc][HIGHER], size=nDimensions))

            new_solution.calc_fitness(target_fnc)

            initial_population.append(new_solution)

        # initial_population = [Solution(np.random.uniform(low=BOUNDS[target_fnc][LOWER], high=BOUNDS[target_fnc][HIGHER], size=nDimensions)) for _ in range(pop_size)]

    elif target_fnc == MCCORMICK:
        # Create solutions in McCormick function bounds range
        for _ in range(pop_size):
            
            x = np.random.uniform(low=BOUNDS[MCCORMICK][LOWER][LOWER], high=BOUNDS[MCCORMICK][LOWER][HIGHER])
            y = np.random.uniform(low=BOUNDS[MCCORMICK][HIGHER][LOWER], high=BOUNDS[MCCORMICK][HIGHER][HIGHER])
            
            new_solution = Solution([x, y])

            new_solution.calc_fitness(MCCORMICK) 

            initial_population.append(new_solution)

    else:
        raise ValueError("Could not find function")        

    return initial_population

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Selection ---

def selection(population: list[Solution], pop_size: int, *, mode=TOURNAMENT) -> Solution: 
    """
        Select parents for crossover.

        Tournament selection and Fitness Proportionate Selection ( AKA Roulette Wheel Selection).

    Args:
        population (list[Solution]): Population of individuals.
        pop_size (int): Population size.
        mode (_type_, optional): Selection of the selection method. Defaults to TOURNAMENT.

    Returns:
        Solution: Selected solution for crossover.
    """

    if mode == TOURNAMENT:
        
        # Should be 2 or 3
        nCandidates: int = 3
        
        # Random selection of indices
        selected: list[int] = np.random.choice(range(0, pop_size), nCandidates, replace=False) # Replace=False -> No repeated numbers

        # Separate the selected
        candidates: list[Solution] = [ population[i] for i in selected ]

        # Select the best candidate by fitness
        winner: Solution = min(candidates, key=lambda sol: sol.get_fitness())
        
        return winner

    elif mode == FPS: # bug com EASOM: divisão por 0
        
        # Get total fitness 
        total_fitness: float = sum( (1.0 / ind.get_fitness()) for ind in population ) 

        # Get each individual probability
        probabilities: list[float] = [ ((1.0 / ind.get_fitness()) / total_fitness) for ind in population ] 

        # Make the roulette 
        roulette: list[float] = np.cumsum(probabilities)

        # Select a radom number for wheel
        magic_num: float = np.random.rand()

        # Find and return the selected solution
        for i, chances in enumerate(roulette):
            if magic_num < chances:
                return population[i]


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Crossover ---

def crossover(first_parent: Solution, second_parent: Solution, *, nPoints=ONE) -> tuple[Solution, Solution]: 
    """
        Crossover function to generate new solutions.

    Args:
        first_parent (Solution): Parent for crossover
        second_parent (Solution): Parent for crossover
        nPoints (_type_, optional): Crossover function version. Defaults to ONE.

    Returns:
        Two child solutions.
    """

    first_parent_genes = first_parent.get_genes()
    second_parent_genes = second_parent.get_genes()

    if len(first_parent) == 2:

        part1: int = np.random.randint(0, 2)
        part2: int = np.random.randint(0, 2)

        first_child_genes: list[float] = [first_parent_genes[0] if part1 == 0 else first_parent_genes[1],
                                          second_parent_genes[0] if part2 == 0 else second_parent_genes[1]]
        
        second_child_genes: list[float] = [second_parent_genes[0] if part1 == 0 else second_parent_genes[1],
                                           first_parent_genes[0] if part2 == 0 else first_parent_genes[1]]


    # One point crossover
    elif nPoints == ONE:
            
        point = np.random.randint(1, len(first_parent))

        first_child_genes = np.concatenate([first_parent_genes[:point], second_parent_genes[point:]])
        second_child_genes = np.concatenate([second_parent_genes[:point], first_parent_genes[point:]])


    # Two points crossover
    elif nPoints == TWO:

        first_point, second_point = sorted(np.random.choice(range(1, len(first_parent)), 2, replace=False))

        first_child_genes = np.concatenate([first_parent_genes[:first_point],
                                            second_parent_genes[first_point:second_point],
                                            first_parent_genes[second_point:]])
                                           
        second_child_genes = np.concatenate([second_parent_genes[:first_point],
                                             first_parent_genes[first_point:second_point],
                                             second_parent_genes[second_point:]])


    else:
        raise ValueError("nPoints must be ONE or TWO")


    first_child  = Solution(first_child_genes)
    second_child = Solution(second_child_genes)

    return first_child, second_child


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Find Best ---

def find_best(population: list[Solution]) -> Solution: 
    """
        Find and return the best solution of the generation, for elitism. 

    Args:
        population (list[Solution]): Current population of solutions.

    Returns:
        Solution: Best solution of the generation.
    """

    # assert all(isinstance(ind, Solution) for ind in population), "Population contains non-Solution elements"
    # assert all(ind is not None for ind in population), "Population contains None elements"

    population = np.sort(population)

    return population[0]


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Create New Generation ---

def create_new_gen(prev_gen: list[Solution], mutation_rate: float, pop_size: int, target_fnc: int, ms: float = 0.2) -> list[Solution]: 
    """
        Generate new solutions an keeps the best solution from previous generation.

    Args:
        prev_gen (list[Solution]): Previous population of solutions.
        mutation_rate (float): Mutation rate.
        target_fnc (int): Current target function.
        ms (float): Mutation strength.

    Returns:
        list[Solution]: A new population of solutions.
    """

    # Elitism: Keep the best solution of the last generation
    new_generation = [find_best(prev_gen)] 
    

    if len(prev_gen[0]) > 1:
    # Uses two solutions to create two new solutions. Therefore, only half iterations will be required
        for _ in range ( len(prev_gen) // 2 ):
            first_parent  = selection(prev_gen, pop_size, mode=TOURNAMENT)
            second_parent = selection(prev_gen, pop_size, mode=TOURNAMENT)   

            first_child, second_child = crossover(first_parent, second_parent, nPoints=ONE)

            # Mutation
            if np.random.rand() < mutation_rate:
                first_child.mutate(mutation_strength=ms)
            if np.random.rand() < mutation_rate:
                second_child.mutate(mutation_strength=ms)

            # Evaluate the new solutions
            first_child.calc_fitness(target_fnc)
            second_child.calc_fitness(target_fnc)

            new_generation.append(first_child)        
            new_generation.append(second_child) 

    else:

        fitness = np.array([sol.get_fitness() for sol in prev_gen])

        selected_indices = np.argsort(fitness)[:pop_size // 2]
        selected = [prev_gen[i] for i in selected_indices]

        offspring = []
        for nInd in selected:
            new_solution = Solution(np.random.uniform(low=BOUNDS[target_fnc][LOWER], high=BOUNDS[target_fnc][HIGHER], size=1))

            new_solution.calc_fitness(target_fnc)

            offspring.append(new_solution)

        # Mutation
        if np.random.rand() < mutation_rate:
            inds = np.random.choice(range(len(offspring)), int(len(offspring) / 4), replace=False)
            for i in inds:
                offspring[i].mutate(mutation_strength=0.2)        
                offspring[i].calc_fitness(target_fnc)        

        new_generation = selected + offspring

        # Ensure that the new population has the correct size
        if len(new_generation) > pop_size:
            new_generation = new_generation[:pop_size]   


    return new_generation         


def VAR_create_new_gen(prev_gen: list[Solution], mutation_rate: float, pop_size: int, target_fnc: int, ms: float = 0.2, elite_fraction: float = 0.05) -> list[Solution]: 
    """
        Generate new solutions and keep the best solutions from the previous generation.

    Args:
        prev_gen (list[Solution]): Previous population of solutions.
        mutation_rate (float): Mutation rate.
        pop_size (int): Population size.
        target_fnc (int): Current target function.
        ms (float): Mutation strength.
        elite_fraction (float): Fraction of the best solutions to keep as elite.

    Returns:
        list[Solution]: A new population of solutions.
    """

    # Determine number of elite solutions to keep
    num_elites = max(1, int(elite_fraction * pop_size))
    
    # Sort previous generation by fitness
    sorted_prev_gen = sorted(prev_gen, key=lambda sol: sol.get_fitness())
    
    # Elitism: Keep the best solutions from the last generation
    new_generation = sorted_prev_gen[:num_elites]

    while len(new_generation) < pop_size:

        if len(prev_gen[0]) > 1:

            first_parent = selection(prev_gen, pop_size, mode=FPS)
            second_parent = selection(prev_gen, pop_size, mode=FPS)
            
            first_child, second_child = crossover(first_parent, second_parent, nPoints=TWO)

            # Mutation
            if np.random.rand() < mutation_rate:
                first_child.mutate(mutation_strength=ms)
            if np.random.rand() < mutation_rate:
                second_child.mutate(mutation_strength=ms)

            # Evaluate the new solutions
            first_child.calc_fitness(target_fnc)
            second_child.calc_fitness(target_fnc)

            new_generation.append(first_child)
            new_generation.append(second_child)
            
        else:
            fitness = np.array([sol.get_fitness() for sol in prev_gen])

            selected_indices = np.argsort(fitness)[:pop_size // 2]
            selected = [prev_gen[i] for i in selected_indices]

            offspring = []
            for nInd in selected:
                new_solution = Solution(np.random.uniform(low=BOUNDS[target_fnc][LOWER], high=BOUNDS[target_fnc][HIGHER], size=1))

                new_solution.calc_fitness(target_fnc)

                offspring.append(new_solution)

            if np.random.rand() < mutation_rate:
                inds = np.random.choice(range(len(offspring)), int(len(offspring) / 4), replace=False)
                for i in inds:
                    offspring[i].mutate(mutation_strength=ms)        
                    offspring[i].calc_fitness(target_fnc)     

            new_generation = selected + offspring

            # Ensure that the new population has the correct size
            if len(new_generation) > pop_size:
                new_generation = new_generation[:pop_size]

    return new_generation


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Test Functions ---

def tests() -> None:
    import time

    np.random.seed(10)

    population_size = 1_000

    start = time.time()
    new_gen = initialization(population_size, nDimensions=2, target_fnc=RASTRIGIN)
    end = time.time()

    print(sum(ind.get_genes() for ind in new_gen))
   
    print(f"Time: {end - start:.4f} seconds")


def test_selection():
    np.random.seed(10)

    population_size = 10

    pop = initialization(population_size, nDimensions=2, target_fnc=RASTRIGIN)

    # for ind in pop:
    #     ind.calc_fitness(RASTRIGIN)

    pai1 = selection(pop, population_size, mode=FPS)    
    pai2 = selection(pop, population_size)    

    print(pai1)
    print(pai2)

    pop.sort()

    # print([ind.get_genes() for ind in pop])

    print('\bMelhor solulção: \n')

    print(pop[0])


def test_find_best():
    np.random.seed(10)

    population_size = 10

    pop = initialization(population_size, nDimensions=2, target_fnc=RASTRIGIN)

    print(find_best(pop))


def test_cross():
    np.random.seed(10)

    population_size = 10

    pop = initialization(population_size, nDimensions=2, target_fnc=RASTRIGIN)

    pai1 = selection(pop, population_size, mode=FPS)    
    pai2 = selection(pop, population_size)

    filho1, filho2 = crossover(pai1, pai2)

    filho1 = Solution(filho1)


def easomT():
    np.random.seed(10)

    population_size = 10

    pop = initialization(population_size, nDimensions=2, target_fnc=EASOM)

    pop.pop()

    best = Solution([np.pi, np.pi])
    best.calc_fitness(EASOM)

    print(best.get_fitness())

    pop.append(best)

    melhor = find_best(pop, EASOM)

    pai1 = selection(pop, population_size)    
    pai2 = selection(pop, population_size)
    

if __name__ == "__main__":

    # tests()

    # test_selection()

    # test_find_best()

    # test_cross()

    easomT()
