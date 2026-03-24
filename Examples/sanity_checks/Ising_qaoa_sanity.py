import sys
import os

sys.path.append(os.path.abspath("../../src"))

from Ising import sanity_test_qaoa

sanity_test_qaoa(
    num_tests=20,
    N=3,
    p=2,
    seed_start=9,
    samples=500,
    verbose=True
)

