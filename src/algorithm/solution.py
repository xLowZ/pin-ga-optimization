import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from benchmark.functions import TARGET_FUNCTIONS
# from utils import BOUNDS, LOWER, HIGHER

# def scaled_fitness(value):
#     return -np.log(value + 1e-12)  

class Solution:
    """
        Solution representation for the functions

        AKA Individual, Chromosome
    """


    def __init__(self, genes: list[float|int]) -> None:
        self.__chromosome = np.array(genes)
        self.__fitness_value = 10_000
        

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # --- Evaluation ---

    def calc_fitness(self, fnc_repr: int, *, fast: bool=True) -> None:
        """
            Evaluate Solution based on the target function.

        Args:
            fnc_repr (int): Target function representation.
            fast (bool, optional): Whether to perform the faster version or the safe version of the selected function. Defaults to True.
        """
        # if np.any((self.__chromosome <= BOUNDS[fnc_repr][LOWER]) | (self.__chromosome >=BOUNDS[fnc_repr][HIGHER])):
        #     self.__chromosome = np.array([np.random.uniform(low=BOUNDS[fnc_repr][LOWER], high=BOUNDS[fnc_repr][HIGHER])])
        # self.__fitness_value = scaled_fitness(value)
        self.__fitness_value = TARGET_FUNCTIONS[fnc_repr](self.__chromosome, fast=fast)
            

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # --- Fitness getter ---

    def get_fitness(self) -> float:
        """
            Return the fitness value of the Solution.

        """

        return self.__fitness_value


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # --- Genes getter ---

    def get_genes(self) -> list[float]:
        """
            Return the array of genes of the solution.

        Returns:
            Chromosome array.
        """
        return self.__chromosome
    
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # --- Mutation function ---

    def mutate(self, mutation_strength: float = 0.5) -> None:
        """
            Apply Gaussian mutation to the solution.

        Args:
            mutation_strength (float): Standard deviation of the Gaussian distribution.
        """
        
        for i in range(len(self.__chromosome)):
            self.__chromosome[i] += np.random.normal(0, mutation_strength)
        
        # Normal
        # for i in range(len(self.__chromosome)):
        #     self.__chromosome[i] += np.random.rand()
        # mutation_vector = np.random.rand(len(self.__chromosome)) < mutation_strength
        # self.__chromosome[mutation_vector] += np.random.normal(0, 1, sum(mutation_vector))


    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # --- Class info + Dunder methods ---

    @staticmethod
    def info() -> str:
        return "Class representation of the solution for the benchmark optimization problems"
    

    def __repr__(self) -> str:
        return f'Solution(Chromosome="{self.__chromosome}", fitness_value="{self.__fitness_value:.10f}")'
    

    def __str__(self) -> str:
        return f'Chromosome = {self.__chromosome}, Fitness = {self.__fitness_value:.15f}'
    

    def __lt__(self, other):
        return self.__fitness_value < other.get_fitness()
    

    def __len__(self) -> int:
        """
            Return the number of dimensions of a Solution.
        """
        return len(self.__chromosome)
    