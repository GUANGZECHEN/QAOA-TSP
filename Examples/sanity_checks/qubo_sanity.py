import sys
import os

sys.path.append(os.path.abspath("../../src"))

from qubo import sanity_test_qubo

sanity_test_qubo(num_tests=20, N=4, verbose=False)

