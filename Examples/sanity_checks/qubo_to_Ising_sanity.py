import sys
import os

sys.path.append(os.path.abspath("../../src"))

from Ising import sanity_test_qubo_to_ising

sanity_test_qubo_to_ising(
    num_tests=20,
    N=4,
    verbose=True
)

