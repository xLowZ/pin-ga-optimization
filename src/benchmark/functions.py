import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import numpy as np
import time
from utils import RASTRIGIN, ACKLEY, SPHERE, EASOM, MCCORMICK


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --------- Benchmark Functions Module --------
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


TARGET_FUNCTIONS: dict = {}


def reg_fnc(name: int):
    """
        Decorator for easier function calls

    Args:
        name (int [enum]): Function name
    """
    def decorator(func):
        TARGET_FUNCTIONS[name] = func
        return func
    return decorator


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Rastrigin benchmark function ---

@reg_fnc(RASTRIGIN)
def rastrigin_fnc(x: list[float | int], *, bounds: tuple[float | int]=(-5.12, 5.12), fast: bool=False) -> float:
    """
        Rastrigin benchmark function for n dimensions.

    Args:
        x (list): Values of the variables for n dimensions (domain)
        bounds (tuple): Bounds of the problem
        fast (bool, optional): Whether to perform the faster version or the safe version. Defaults to False.

    Returns:
        float: Value of y ( or f(x, ...) )
    """

    A: int = 10
    Ai: int = A * len(x)

    if not fast:
    
        # if isinstance(x, (int, float)):
        #     x = np.array([x])

        # x = np.array(x)

        # # notValid: bool = any((xi >= bounds[0] or xi <= bounds[1]) for xi in x)
        # notValid: bool = np.any((x <= bounds[0]) | (x >= bounds[1]))

        # if notValid:
        #     raise ValueError("Domain out of bounds")

        for i in range(len(x)):
            if not (bounds[0] <= x[i] <= bounds[1]):
                x[i] = max(bounds[0], min(x[i], bounds[1]))

    # return Ai + np.sum(np.fromiter(((xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x), dtype=float)) #-> Old

    return Ai + np.sum(x**2 - A * np.cos(2 * np.pi * x))


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Ackley benchmark function ---

@reg_fnc(ACKLEY)
def ackley_fnc(x: list[float | int], *, bounds: tuple[float | int]=(-5, 5), fast: bool=False) -> float:
    """
        Ackley benchmark function for 2 or more dimensions.
    
    Args:
        x (list): Values of the variables for 2 or more dimensions (domain)
        bounds (tuple): Bounds of the problem
        fast (bool, optional): Whether to perform the faster version or the safe version. Defaults to False.

    Returns:
        float: Value of y ( or f(x, ...) )
    """

    A: int = 20    
    B: float = 0.2
    C: float = 2 * np.pi
    nDim: int = len(x)

    if not fast:

        # x = np.array(x)

        # if nDim < 2:
        #     raise ValueError("The input array must have at least 2 dimensions")

        # notValid: bool = np.any((x <= bounds[0]) | (x >= bounds[1]))

        # if notValid:
        #     raise ValueError("Domain out of bounds")

        for i in range(len(x)):
            if not (bounds[0] <= x[i] <= bounds[1]):
                x[i] = max(bounds[0], min(x[i], bounds[1]))

    term1 = -A * np.exp(-B * np.sqrt(np.sum(x**2) / nDim))

    term2 = -np.exp(np.sum(np.cos(C * x) / nDim))
    
    return term1 + term2 + np.e + A


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Sphere benchmark function ---

@reg_fnc(SPHERE)
def sphere_fnc(x: list[float | int], *, bounds: tuple[float | int]=(-1_000, 1_000), fast: bool=False) -> float:
    """
        Sphere benchmark function for n dimensions.

    Args:
        x (list): Values of the variables for n dimensions (domain)
        fast (bool, optional): Whether to perform the faster version or the safe version. Defaults to False.

    Returns:
        float: Value of y ( or f(x, ...) )
    """

    if not fast:

        # if isinstance(x, (int, float)):
        #     x = np.array([x])

        # x = np.array(x)

        for i in range(len(x)):
            if not (bounds[0] <= x[i] <= bounds[1]):
                x[i] = max(bounds[0], min(x[i], bounds[1]))

    return np.sum(x**2)    


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Easom benchmark function ---

@reg_fnc(EASOM)
def easom_fnc(x: list[float | int], *, bounds: tuple[float | int]=(-100, 100), fast: bool=False) -> float:
    """
        Ackley benchmark function for 2 dimensions.
    
    Args:
        x (list): Values of the variables for 2 dimensions (domain)
        bounds (tuple): Bounds of the problem
        fast (bool, optional): Whether to perform the faster version or the safe version. Defaults to False.

    Returns:
        float: Value of y ( or f(x, ...) )
    """

    if not fast:

        # x = np.array(x)

        # if len(x) != 2:
        #     raise ValueError("The input array must have exactly 2 dimensions")

        # notValid: bool = np.any((x <= bounds[0]) | (x >= bounds[1]))

        # if notValid:
        #     raise ValueError("Domain out of bounds")

        for i in range(len(x)):
            if not (bounds[0] <= x[i] <= bounds[1]):
                x[i] = max(bounds[0], min(x[i], bounds[1]))
        
    x1, x2 = x[0], x[1]

    term1 = -np.cos(x1) * np.cos(x2)
    term2 = np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

    return term1 * term2 


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- McCormick benchmark function ---

@reg_fnc(MCCORMICK)
def mccormick_fnc(x: list[float | int], *, bounds: list[tuple[float | int]]=[(-1.5, 4), (-3, 4)], fast: bool=False) -> float:
    """
        McCormick benchmark function for 2 dimensions.
    
    Args:
        x (list): Values of the variables for 2 dimensions (domain)
        bounds (tuple): Bounds of the problem
        fast (bool, optional): Whether to perform the faster version or the safe version. Defaults to False.
        
    Returns:
        float: Value of y ( or f(x, ...) )
    """

    if not fast:

        # x = np.array(x)

        # if len(x) != 2:
        #     raise ValueError("The input array must have exactly 2 dimensions")
        
        for i in range(len(x)):
            if not (bounds[i][0] <= x[i] <= bounds[i][1]):
                x[i] = max(bounds[i][0], min(x[i], bounds[i][1]))

    x1, x2 = x[0], x[1]

    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# --- Test Functions ---

def testes() -> None:

    # x = [-2, 3, -0.3, 4]
    # x = -5.11
    # print(f"{rastrigin_fnc(x) = }\n")

    # Benchmark
    x = np.array(np.random.uniform(-5.12, 5.12, size=3))

    iterations = 500_000

    # Warm-up to stabilize performance measurements
    for _ in range(100):
        rastrigin_fnc(x, fast=True)
        rastrigin_fnc(x, fast=False)

    start = time.time()
    for _ in range(iterations):
        y = rastrigin_fnc(x, fast=True)
    end = time.time()
    print(f"Fast=True: {end - start:.4f} seconds")
    print(y)

    start = time.time()
    for _ in range(iterations):
        y = rastrigin_fnc(x, fast=False)
    end = time.time()
    print(f"Fast=False: {end - start:.4f} seconds")
    print(y)


def test_easomT() -> None:
    x = np.array([np.pi, np.pi])

    print(easom_fnc(x, fast=True))


if __name__ == "__main__":

    np.random.seed(10)

    # testes()

    # test_easomT()
    # print(easom_fnc([np.pi, np.pi]))
    