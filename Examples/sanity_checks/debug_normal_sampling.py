import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math

sys.path.append(os.path.abspath("../../src"))

from tsp_generator import TSP
from classical_solver import solve_tsp_bruteforce
from qaoa_qiskit_solver import solve_ising_qaoa_qiskit
from qubo import normalize_route


# =========================================================
# p = 0 baseline: uniform random sampling
# =========================================================
def sample_uniform_tsp(tsp, shots):

    N = tsp.N
    n = tsp.ising.h.shape[0]

    best_route = None
    best_cost = np.inf

    for _ in range(shots):

        z = np.random.choice([-1, 1], size=n)
        x = tsp.z_to_x(z)

        route = tsp.x_to_route(x)

        if route==None:
            continue

        cost = tsp.route_cost(route)

        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route, best_cost
    
def sanity_check_route_distribution(tsp, shots=10000, seed=None):

    import numpy as np
    from collections import Counter

    # Create local RNG (independent of global np.random)
    rng = np.random.default_rng(seed)

    n = tsp.ising.h.shape[0]

    bit_counter = Counter()
    route_counter = Counter()

    for _ in range(shots):

        # use local RNG instead of np.random
        z = rng.choice([-1, 1], size=n)

        bit_counter[tuple(z)] += 1

        x = ((1 - z) // 2).astype(int)
        route = tsp.x_to_route(x)

        if route is None:
            route_counter["None"] += 1
        else:
            route_counter[tuple(route)] += 1

    # -------------------------
    # Route distribution
    # -------------------------
    print("\n===== ROUTE DISTRIBUTION =====\n")

    for key, count in route_counter.items():
        prob = count / shots
        print(f"{key}: P = {prob:.4f}")

    # -------------------------
    # Summary
    # -------------------------
    valid_prob = 1 - route_counter["None"] / shots if "None" in route_counter else 1.0

    print("\nTotal valid probability:", valid_prob)
    print("Total invalid probability:", 1 - valid_prob)

    # -------------------------
    # (Optional) expected valid fraction
    # -------------------------
    N = tsp.N
    n_valid = math.factorial(N)
    total_states = 2**n

    print("\nExpected valid fraction (uniform):", n_valid / total_states)
    
N = 2
tsp = TSP.random_asymmetric(N=N, seed=100)
tsp.return_to_start = False
tsp.build_qubo(A=2 * np.max(tsp.distance_matrix))
tsp.ensure_ising()

sanity_check_route_distribution(tsp, shots=10)

