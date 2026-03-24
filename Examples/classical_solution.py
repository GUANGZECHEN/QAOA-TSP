import sys
import os

sys.path.append(os.path.abspath("../src"))

from tsp_generator import *
from classical_solver import *

tsp = TSP.random_asymmetric(N=5, seed=9)
print(tsp.distance_matrix)
route, cost = solve_tsp_bruteforce(tsp)

print("Best route:", route)
print("Cost:", cost)

tsp.plot(route)

