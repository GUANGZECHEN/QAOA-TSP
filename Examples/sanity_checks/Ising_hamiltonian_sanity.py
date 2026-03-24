import sys
import os

sys.path.append(os.path.abspath("../../src"))

from Ising import sanity_test_cost_hamiltonian

sanity_test_cost_hamiltonian(
    num_tests=50,
    N=3,
    seed_start=0,
    verbose=True,
    stop_on_error=True
)
