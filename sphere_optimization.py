"""
Local Search, Heuristics, and Simulated Annealing Homework.

Minimize the Sphere function f(x) = sum(x_i^2) over x_i in [-5,5]
using:
  1) Hill Climbing
  2) Random Local Search
  3) Simulated Annealing
"""

import random
import math
from typing import List, Tuple
from tabulate import tabulate

def sphere_function(x: List[float]) -> float:
    return sum(xi ** 2 for xi in x)

def random_neighbor(point: List[float],
                    bounds: List[Tuple[float,float]],
                    step_size: float = 0.1) -> List[float]:
    """Generate a neighbor by random perturbation within bounds."""
    neighbor = []
    for xi, (low, high) in zip(point, bounds):
        delta = random.uniform(-step_size, step_size) * (high - low)
        x_new = max(min(xi + delta, high), low)
        neighbor.append(x_new)
    return neighbor

def hill_climbing(func, bounds: List[Tuple[float,float]],
                  iterations=1000, epsilon=1e-6) -> Tuple[List[float], float]:
    point = [random.uniform(low, high) for low, high in bounds]
    value = func(point)
    step_size = 0.1
    for _ in range(iterations):
        neighbor = random_neighbor(point, bounds, step_size)
        nv = func(neighbor)
        if value - nv > epsilon:
            point, value = neighbor, nv
        else:
            step_size *= 0.99
        if abs(value) < epsilon:
            break
    return point, value

def random_local_search(func, bounds: List[Tuple[float,float]],
                        iterations=1000, epsilon=1e-6) -> Tuple[List[float], float]:
    best = [random.uniform(low, high) for low, high in bounds]
    bv = func(best)
    for _ in range(iterations):
        candidate = [random.uniform(low, high) for low, high in bounds]
        cv = func(candidate)
        if bv - cv > epsilon:
            best, bv = candidate, cv
        if abs(bv) < epsilon:
            break
    return best, bv

def simulated_annealing(func, bounds: List[Tuple[float,float]],
                        iterations=1000, temp=1000.0,
                        cooling_rate=0.95, epsilon=1e-6) -> Tuple[List[float], float]:
    point = [random.uniform(low, high) for low, high in bounds]
    value = func(point)
    best, best_val = point[:], value
    T = temp
    for _ in range(iterations):
        neighbor = random_neighbor(point, bounds, step_size=0.1)
        nv = func(neighbor)
        delta = nv - value
        if delta < 0 or random.random() < math.exp(-delta / T):
            point, value = neighbor, nv
            if value < best_val:
                best, best_val = point[:], value
        T *= cooling_rate
        if T < epsilon:
            break
    return best, best_val

if __name__ == "__main__":
    bounds = [(-5,5), (-5,5)]
    # Run each algorithm
    hc_sol, hc_val = hill_climbing(sphere_function, bounds)
    rls_sol, rls_val = random_local_search(sphere_function, bounds)
    sa_sol, sa_val = simulated_annealing(sphere_function, bounds)

    table = [
        ["Hill Climbing",           hc_sol, f"{hc_val:.6e}"],
        ["Random Local Search",     rls_sol, f"{rls_val:.6e}"],
        ["Simulated Annealing",     sa_sol, f"{sa_val:.6e}"],
    ]

    print("\nResults:\n")
    print(tabulate(
        table,
        headers=["Method", "Solution (x)", "Value f(x)"],
        tablefmt="pretty",
        stralign="center"
    ), "\n")
