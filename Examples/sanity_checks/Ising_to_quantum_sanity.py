import sys
import os

sys.path.append(os.path.abspath("../../src"))

from qaoa_qiskit_solver import sanity_check_qiskit_vs_true_from_tsp

sanity_check_qiskit_vs_true_from_tsp(
    num_tests=5,
    N=3,
    seed_start=0,
    num_z_samples=10,
    atol=1e-8
)

