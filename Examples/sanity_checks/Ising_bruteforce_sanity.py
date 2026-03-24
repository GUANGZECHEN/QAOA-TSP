import sys
import os

sys.path.append(os.path.abspath("../../src"))

from Ising import sanity_test_ising_solver

sanity_test_ising_solver(
    num_tests=20,
    N=3,
    seed_start=0,
    verbose=True,
    stop_on_error=True
)
