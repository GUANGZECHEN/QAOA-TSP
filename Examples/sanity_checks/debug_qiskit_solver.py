import sys
import os
import numpy as np
from itertools import product

# adjust path if needed
sys.path.append(os.path.abspath("../../src"))

from tsp_generator import TSP
from qaoa_qiskit_solver import (
    build_qaoa_circuit,
    run_qaoa_qiskit,
    is_valid_assignment,
    _get_backend
)

from qiskit import transpile
from qiskit.quantum_info import Statevector


# =========================================================
# Debug function (STATEVECTOR, no sampling)
# =========================================================

def debug_qaoa_qiskit_statevector(tsp, p_values=[1, 2, 3], tol=1e-9):

    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    N = tsp.N
    n_qubits = len(h)

    # enumerate all z ∈ {-1, +1}^n
    all_z = np.array(list(product([-1, 1], repeat=n_qubits)))

    print("\n===== QISKIT STATEVECTOR DEBUG =====\n")

    backend = _get_backend(simulator=True)

    for p in p_values:

        print(f"\n==============================")
        print(f"Running QAOA with p = {p}")
        print(f"==============================")

        # -------------------------
        # Optimize parameters (still sampling-based)
        # -------------------------
        params = run_qaoa_qiskit(
            h, J,
            backend=backend,
            p=p,
            N=N,
            constrained=False,
            shots=500,
            simulator=True,
            optimizer="BFGS",
        )

        gammas = params[:p]
        betas  = params[p:]

        # -------------------------
        # Build circuit
        # -------------------------
        qc = build_qaoa_circuit(
            h, J,
            gammas, betas,
            N=N,
            constrained=False
        )

        # -------------------------
        # Get statevector
        # -------------------------
        qc_sv = qc.copy()
        qc_sv.save_statevector()

        qc_trans = transpile(qc_sv, backend)

        result = backend.run(qc_trans).result()
        psi = result.get_statevector()

        probs = np.abs(psi) ** 2
        probs /= probs.sum()

        best_valid_E = np.inf
        best_valid_prob = 0
        best_valid_z = None

        for idx, z in enumerate(all_z):

            prob = probs[idx]

            if prob < tol:
                continue

            x = ((1 - z) // 2).astype(int)
            valid = is_valid_assignment(x, N)

            E = z @ J @ z + h @ z + const

            if valid and E < best_valid_E:
                best_valid_E = E
                best_valid_prob = prob
                best_valid_z = z

            print(
                f"z={z}, x={x}, "
                f"P={prob:.6f}, "
                f"E={E:.4f}, "
                f"{'VALID' if valid else 'INVALID'}"
            )

        print("Probability of best valid state:", best_valid_prob)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    np.random.seed(0)

    # Smallest nontrivial ATSP
    N = 2
    tsp = TSP.random_asymmetric(N=N, seed=10)
    tsp.return_to_start=False
    
    tsp.build_qubo(A=2 * np.max(tsp.distance_matrix))
    tsp.return_to_start = True

    print("Distance matrix:")
    print(tsp.distance_matrix)

    # run debug
    debug_qaoa_qiskit_statevector(tsp, p_values=[1, 2, 3, 4, 5])
