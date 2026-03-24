import sys
import os
import numpy as np
from itertools import product

# adjust path if needed
sys.path.append(os.path.abspath("../../src"))

from tsp_generator import TSP
from Ising import run_qaoa, qaoa_state   # adjust if different file
from Ising import is_valid_assignment          # adjust import if needed
import matplotlib.pyplot as plt


# -------------------------
# Debug function
# -------------------------
def debug_qaoa_distribution(tsp, p_values=[1, 2, 3], tol=1e-9):
    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    N = tsp.N
    n_qubits = len(h)

    #assert n_qubits == 4, "This debug is intended for N=2 (4 qubits)"

    # enumerate all z ∈ {-1, +1}^n
    all_z = np.array(list(product([-1, 1], repeat=n_qubits)))

    print("\n===== FULL STATE ENUMERATION (N=2 ATSP) =====\n")

    for p in p_values:
        #print(f"\n==============================")
        #print(f"Running QAOA with p = {p}")
        #print(f"==============================")

        # optimize parameters
        params = run_qaoa(h, J, p)
        #print("Optimized params:", params)

        # statevector
        psi = qaoa_state(h, J, params, p)

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

            #print(
            #    f"z={z}, x={x}, "
            #    f"P={prob:.6f}, "
            #    f"E={E:.4f}, "
            #    f"{'VALID' if valid else 'INVALID'}"
            #)

        #print("\n--- Summary ---")
        #print("Best valid z:", best_valid_z)
        #print("Best valid energy:", best_valid_E)
        print("Probability of best valid state:", best_valid_prob)

def plot_energy_spectrum(tsp):
    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    N = tsp.N
    n_qubits = len(h)

    print(f"Number of qubits: {n_qubits} (total states = {2**n_qubits})")

    # enumerate all configurations
    all_z = np.array(list(product([-1, 1], repeat=n_qubits)))

    energies = []
    valid_flags = []

    for z in all_z:
        x = ((1 - z) // 2).astype(int)
        valid = is_valid_assignment(x, N)

        E = z @ J @ z + h @ z + const

        energies.append(E)
        valid_flags.append(valid)

    energies = np.array(energies)
    valid_flags = np.array(valid_flags)

    print("Number of valid states:", valid_flags.sum())
    print("Min valid energy:", energies[valid_flags].min())
    print("Min invalid energy:", energies[~valid_flags].min())

    # sort by energy
    sorted_idx = np.argsort(energies)
    energies_sorted = energies[sorted_idx]
    valid_sorted = valid_flags[sorted_idx]

    x_axis = np.arange(1, len(energies_sorted) + 1)

    # plot
    plt.figure()

    # invalid states (black)
    plt.scatter(
        x_axis[~valid_sorted],
        energies_sorted[~valid_sorted],
        s=10,
        label="Invalid"
    )

    # valid states (red)
    plt.scatter(
        x_axis[valid_sorted],
        energies_sorted[valid_sorted],
        s=20,
        label="Valid"
    )

    plt.xlabel("Configuration index (sorted by energy)")
    plt.ylabel("Energy")
    plt.title(f"Energy spectrum (N={N})")
    plt.legend()

    plt.show()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    np.random.seed(0)

    # Smallest nontrivial ATSP
    N = 2
    tsp = TSP.random_asymmetric(N=N, seed=0)
    tsp.build_qubo(A = 2 * np.max(tsp.distance_matrix))
    tsp.return_to_start=True
    plot_energy_spectrum(tsp)

    print("Distance matrix:")
    print(tsp.distance_matrix)

    # run debug
    debug_qaoa_distribution(tsp, p_values=[1, 2, 3, 4])
