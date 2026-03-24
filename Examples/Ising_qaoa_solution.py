import sys
import os

sys.path.append(os.path.abspath("../src"))

from tsp_generator import TSP
from Ising import solve_ising_qaoa
from classical_solver import solve_tsp_bruteforce
from qubo import normalize_route


# -------------------------
# Generate instance
# -------------------------
N = 3
tsp = TSP.random_asymmetric(N=N, seed=9)
start = tsp.start_city

# -------------------------
# Solve using QAOA (new API)
# -------------------------
ising_route, ising_energy = solve_ising_qaoa(
    tsp,
    p=2,
    samples=100,
    debug=True
)

# handle possible failure
if ising_route is None:
    print("⚠️ QAOA did not find a valid solution")
else:
    ising_route = normalize_route(ising_route, start)

# -------------------------
# Ground truth
# -------------------------
true_route, true_cost = solve_tsp_bruteforce(tsp)

# -------------------------
# Print results
# -------------------------
print("QAOA route:", ising_route)
print("QAOA energy:", ising_energy)

print("True route:", true_route)
print("True cost:", true_cost)

# -------------------------
# Plot
# -------------------------
tsp.plot(true_route)

if ising_route is not None:
    tsp.plot(ising_route)
