import sys
import os

sys.path.append(os.path.abspath("../src"))

from tsp_generator import TSP
from qaoa_qiskit_solver import solve_ising_qaoa_qiskit
from classical_solver import solve_tsp_bruteforce
from qubo import normalize_route

# -------------------------
# Generate instance
# -------------------------
N = 3
tsp = TSP.random_asymmetric(N=N, seed=0)
start = tsp.start_city

# -------------------------
# Solve using Qiskit
# -------------------------
ising_route, ising_energy = solve_ising_qaoa_qiskit(tsp, p=1, shots=100, simulator=False)
qaoa_cost = tsp.route_cost(ising_route)

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
print("QAOA cost:", qaoa_cost)

print("True route:", true_route)
print("True cost:", true_cost)

import json
from datetime import datetime

def save_result(route, energy, tsp, filename="result.txt"):
    data = {
        "timestamp": str(datetime.now()),
        "route": route,
        "energy": float(energy),
        "cost": tsp.route_cost(route) if route is not None else None,
        "distance_matrix": tsp.distance_matrix.tolist()
    }

    with open(filename, "a") as f:
        f.write(json.dumps(data) + "\n")

save_result(ising_route, ising_energy, tsp)
        
# -------------------------
# Plot
# -------------------------
tsp.plot(true_route)

if ising_route is not None:
    tsp.plot(ising_route)
