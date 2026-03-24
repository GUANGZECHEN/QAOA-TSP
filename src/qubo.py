import numpy as np
import itertools

from tsp_generator import TSP
from classical_solver import solve_tsp_bruteforce


# =========================================================
# Solver
# =========================================================

def solve_qubo_bruteforce(tsp):   # best for N < 5

    tsp.ensure_qubo()

    Q = tsp.Q
    N = tsp.N
    num_vars = N * N

    best_E = np.inf
    best_x = None

    for bits in itertools.product([0, 1], repeat=num_vars):

        x = np.array(bits)

        E = x @ Q @ x

        if E < best_E:
            best_E = E
            best_x = x.copy()

    route = tsp.x_to_route(best_x)

    return route, best_E


# =========================================================
# Utilities
# =========================================================

def normalize_route(route, start):
    idx = route.index(start)
    return route[idx:] + route[:idx]


# =========================================================
# Sanity check for QUBO builder (TSP class)
# =========================================================

def sanity_test_qubo(
    num_tests=100,
    N=4,
    seed_start=0,
    verbose=False,
    stop_on_error=True
):
    """
    Run multiple random TSP instances and compare QUBO vs classical solution.
    """

    failures = 0

    for k in range(num_tests):

        seed = seed_start + k

        tsp = TSP.random_asymmetric(N=N, seed=seed)
        start = tsp.start_city

        # --- QUBO solution ---
        qubo_route, qubo_energy = solve_qubo_bruteforce(tsp)
        qubo_route = normalize_route(qubo_route, start)

        # --- Ground truth ---
        true_route, true_cost = solve_tsp_bruteforce(tsp)

        # --- Evaluate ---
        qubo_cost = tsp.route_cost(qubo_route)

        if not np.isclose(qubo_cost, true_cost):

            failures += 1

            print("\n❌ ERROR at test", k)
            print("Seed:", seed)
            print("Distance matrix:\n", tsp.distance_matrix)

            print("\nQUBO route:", qubo_route)
            print("QUBO cost:", qubo_cost)
            print("QUBO energy:", qubo_energy)

            print("\nTrue route:", true_route)
            print("True cost:", true_cost)

            try:
                tsp.plot(true_route)
                tsp.plot(qubo_route)
            except:
                pass

            if stop_on_error:
                raise ValueError("Sanity test failed")

        elif verbose:
            print(f"✅ Test {k} passed")

    print("\n==============================")
    print(f"Total tests: {num_tests}")
    print(f"Failures: {failures}")
    print("==============================")

    if failures == 0:
        print("🎉 All tests passed!")
