import sys
import os

sys.path.append(os.path.abspath("../src"))

from tsp_generator import TSP
from qubo import solve_qubo_bruteforce, normalize_route
from Ising import solve_ising_bruteforce
from classical_solver import solve_tsp_bruteforce


# -------------------------
# Generate instance
# -------------------------
N = 3
tsp = TSP.random_asymmetric(N=N, seed=9)
start = tsp.start_city

# -------------------------
# Solve using Ising
# -------------------------
ising_route, ising_energy = solve_ising_bruteforce(tsp)
ising_route = normalize_route(ising_route, start)

# -------------------------
# Ground truth
# -------------------------
true_route, true_cost = solve_tsp_bruteforce(tsp)

# -------------------------
# Print results
# -------------------------
print("Ising route:", ising_route)
print("Ising energy:", ising_energy)

print("True route:", true_route)
print("True cost:", true_cost)

# -------------------------
# Plot
# -------------------------
tsp.plot(true_route)
tsp.plot(ising_route)
