import sys
import os

sys.path.append(os.path.abspath("../src"))

from tsp_generator import TSP
from qubo import solve_qubo_bruteforce, normalize_route
from classical_solver import solve_tsp_bruteforce


# -------------------------
# Generate instance
# -------------------------
tsp = TSP.random_asymmetric(N=3, seed=1)
start = tsp.start_city

# -------------------------
# Solve using QUBO
# -------------------------
qubo_route, qubo_energy = solve_qubo_bruteforce(tsp)
qubo_route = normalize_route(qubo_route, start)

# -------------------------
# Ground truth
# -------------------------
true_route, true_cost = solve_tsp_bruteforce(tsp)

# -------------------------
# Print results
# -------------------------
print("QUBO route:", qubo_route)
print("QUBO energy:", qubo_energy)

print("True route:", true_route)
print("True cost:", true_cost)

# -------------------------
# Plot
# -------------------------
tsp.plot(true_route)
tsp.plot(qubo_route)
