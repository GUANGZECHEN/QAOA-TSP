import itertools
import numpy as np

def solve_tsp_bruteforce(tsp):

    N = tsp.N
    cities = list(range(N))

    best_cost = np.inf
    best_route = None

    # -------------------------
    # CASE 1: return_to_start = True (cycle)
    # fix start to avoid duplicates
    # -------------------------
    if tsp.return_to_start:
        start = tsp.start_city
        cities.remove(start)

        for perm in itertools.permutations(cities):
            route = [start] + list(perm)
            cost = tsp.route_cost(route)

            if cost < best_cost:
                best_cost = cost
                best_route = route

    # -------------------------
    # CASE 2: open path (your case)
    # DO NOT fix start
    # -------------------------
    else:
        for perm in itertools.permutations(cities):
            route = list(perm)
            cost = tsp.route_cost(route)

            if cost < best_cost:
                best_cost = cost
                best_route = route

    return best_route, best_cost

