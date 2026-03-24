import itertools
import numpy as np


def solve_tsp_bruteforce(tsp):

    N = tsp.N
    start = tsp.start_city

    cities = list(range(N))
    cities.remove(start)

    best_cost = np.inf
    best_route = None

    # iterate over all permutations
    for perm in itertools.permutations(cities):

        route = [start] + list(perm)

        cost = tsp.route_cost(route)

        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route, best_cost

