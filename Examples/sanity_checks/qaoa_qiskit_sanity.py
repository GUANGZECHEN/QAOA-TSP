import sys
import os

sys.path.append(os.path.abspath("../../src"))

from qaoa_qiskit_solver import sanity_test_qaoa_qiskit

sanity_test_qaoa_qiskit(
    num_tests=20,
    N=3,
    p=1,
    shots=100,
    constrained=False,
    verbose=True
)
