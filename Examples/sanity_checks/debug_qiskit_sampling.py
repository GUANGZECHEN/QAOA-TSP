import sys
import os
import numpy as np
from itertools import product

sys.path.append(os.path.abspath("../../src"))

from tsp_generator import TSP
from qaoa_qiskit_solver import (
    build_qaoa_circuit,
    run_qaoa_qiskit,
    _get_backend,
    _sample_from_backend,
    is_valid_assignment
)

from qiskit.quantum_info import Statevector


def debug_qaoa_valid_only(tsp, p_values=[3], shots=1000, tol=1e-9):

    import numpy as np
    from itertools import product
    from qiskit.quantum_info import Statevector

    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    N = tsp.N
    n = len(h)

    backend = _get_backend(simulator=True)

    print("\n===== VALID-ONLY DEBUG =====\n")

    for p in p_values:

        print(f"\n==============================")
        print(f"QAOA p = {p}")
        print(f"==============================")

        # -------------------------
        # Optimize
        # -------------------------
        params = run_qaoa_qiskit(
            h, J,
            backend=backend,
            p=p,
            N=N,
            constrained=False,
            shots=16,
            simulator=True,
            optimizer="BFGS",
        )

        gammas = params[:p]
        betas  = params[p:]

        qc = build_qaoa_circuit(
            h, J,
            gammas, betas,
            N=N,
            constrained=False
        )

        # -------------------------
        # STATEVECTOR
        # -------------------------
        psi = Statevector.from_instruction(qc).data
        probs_sv = np.abs(psi)**2

        # -------------------------
        # SAMPLING
        # -------------------------
        samples = _sample_from_backend(
            qc, backend,
            shots=shots,
            simulator=True
        )

        counts = {}
        for z in samples:
            bitstring = ''.join(['0' if zi == 1 else '1' for zi in z])
            counts[bitstring] = counts.get(bitstring, 0) + 1

        # -------------------------
        # Collect VALID states only
        # -------------------------
        results = []

        for idx in range(2**n):

            # consistent mapping
            bitstring = format(idx, f"0{n}b")
            bitstring_correct = bitstring[::-1]
            z = np.array([1 if b == '0' else -1 for b in bitstring_correct])

            x = ((1 - z) // 2).astype(int)
            valid = is_valid_assignment(x, N)

            if not valid:
                continue

            p_sv = probs_sv[idx]

            key = ''.join(['0' if zi == 1 else '1' for zi in z])
            p_sample = counts.get(key, 0) / shots

            E = z @ J @ z + h @ z + const

            results.append((z, p_sv, p_sample, E))

        # -------------------------
        # Sort by statevector prob
        # -------------------------
        results.sort(key=lambda x: -x[1])

        # -------------------------
        # Print
        # -------------------------
        total_sv = 0
        total_sample = 0

        for z, p_sv, p_sample, E in results:

            total_sv += p_sv
            total_sample += p_sample

            print(
                f"z={tuple(z)}, "
                f"P_sv={p_sv:.4f}, "
                f"P_sample={p_sample:.4f}, "
                f"diff={abs(p_sv - p_sample):.4f}, "
                f"E={E:.2f}"
            )

        print("\n--- Totals over VALID subspace ---")
        print(f"Total P_sv     = {total_sv:.4f}")
        print(f"Total P_sample = {total_sample:.4f}")


N=2
tsp = TSP.random_asymmetric(N=N, seed=20)
tsp.return_to_start = False
    
debug_qaoa_valid_only(tsp)    
