import sys
import os

sys.path.append(os.path.abspath("../../src"))

import numpy as np
import itertools

from tsp_generator import TSP
from Ising import build_cost_hamiltonian, sample_bitstrings, index_to_z, z_to_index   # adjust import


import numpy as np

def sanity_check_tsp_hamiltonian(
    N=3,
    seed=0,
    atol=1e-8
):

    import numpy as np

    # -------------------------
    # Generate TSP
    # -------------------------
    tsp = TSP.random_asymmetric(N=N, seed=seed)
    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    
    print(h,J,const)

    n = len(h)

    print(f"\n=== TSP Hamiltonian check (N={N}, seed={seed}) ===")
    print(f"Number of qubits: {n}")
    print(f"Hilbert size: {2**n}")

    # -------------------------
    # Build Hamiltonian
    # -------------------------
    H = build_cost_hamiltonian(h, J, const)

    # -------------------------
    # Check ALL basis states
    # -------------------------
    for s in range(2**n):

        z = index_to_z(s, n)

        # matrix energy
        psi = np.zeros(2**n)
        psi[s] = 1.0
        print(psi)
        print(H)
        E_matrix = np.real(np.conj(psi) @ (H @ psi)) + const

        # direct energy
        E_direct = z @ J @ z + h @ z + const

        if not np.isclose(E_matrix, E_direct, atol=atol):

            print("\n❌ MISMATCH FOUND")
            print("index:", s)
            print("z:", z)

            print("E_matrix :", E_matrix)
            print("E_direct :", E_direct)
            print("diff     :", E_matrix - E_direct)

            return

    print("\n🎉 All basis states match!")

def sanity_check_sampling(
    num_tests=5,
    N=3,
    seed_start=0,
    num_samples=20,
    atol=1e-8
):

    import numpy as np

    failures = 0

    for k in range(num_tests):

        seed = seed_start + k
        tsp = TSP.random_asymmetric(N=N, seed=seed)
        tsp.ensure_ising()

        h = tsp.ising.h
        J = tsp.ising.J
        const = tsp.ising.const

        n = len(h)

        H = build_cost_hamiltonian(h, J)

        # random quantum state (proper complex state)
        psi = np.random.randn(2**n) + 1j * np.random.randn(2**n)
        psi /= np.linalg.norm(psi)

        probs = np.abs(psi)**2

        print(f"\n=== Test {k} (seed={seed}) ===")

        # sample indices
        indices = np.random.choice(len(probs), size=num_samples, p=probs)

        for idx, s in enumerate(indices):

            # -------------------------
            # index -> spin
            # -------------------------
            z = index_to_z(s, n)

            # -------------------------
            # direct energy
            # -------------------------
            E_direct = z @ J @ z + h @ z + const

            # -------------------------
            # matrix energy
            # -------------------------
            psi_basis = np.zeros(2**n)
            psi_basis[s] = 1.0

            E_matrix = np.real(np.conj(psi_basis) @ (H @ psi_basis)) + const

            # -------------------------
            # compare
            # -------------------------
            if not np.isclose(E_direct, E_matrix, atol=atol):

                failures += 1

                print("\n❌ MISMATCH")
                print("sample:", idx)
                print("index :", s)
                print("z     :", z)

                print("E_direct :", E_direct)
                print("E_matrix :", E_matrix)
                print("diff     :", E_direct - E_matrix)

                return

            else:
                print(f"✅ sample {idx} OK")

    print("\n==============================")
    print(f"Total failures: {failures}")
    print("==============================")

    if failures == 0:
        print("🎉 Sampling is fully consistent!")
            
if __name__ == "__main__":
    #sanity_check_tsp_hamiltonian(N=3)
    sanity_check_sampling(N=3)
