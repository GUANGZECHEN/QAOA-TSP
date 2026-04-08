import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../../src"))

import matplotlib.pyplot as plt

from tsp_generator import TSP
from classical_solver import solve_tsp_bruteforce
from qaoa_qiskit_solver import run_qaoa_qiskit, build_qaoa_circuit
from qiskit.quantum_info import Statevector


def average_success_probability_vs_p(
    N=2,
    num_instances=100,
    p_max=5,
    lam=2.0,
    seed=0,
):
    np.random.seed(seed)

    # store raw data: rows = instances, cols = p
    raw_data = np.zeros((num_instances, p_max + 1))

    for inst in range(num_instances):

        # -------------------------
        # Generate instance
        # -------------------------
        tsp = TSP.random_asymmetric(N=N, seed=seed + inst)

        tsp.return_to_start = False
        A = lam * np.max(tsp.distance_matrix)
        tsp.build_qubo(A=A)

        tsp.ensure_ising()

        h = tsp.ising.h
        J = tsp.ising.J
        const = tsp.ising.const

        n = len(h)

        # -------------------------
        # True optimal solution
        # -------------------------
        true_route, _ = solve_tsp_bruteforce(tsp)

        x_opt = tsp.route_to_x(true_route)
        z_opt = 1 - 2 * x_opt

        bitstring = ''.join(['0' if zi == 1 else '1' for zi in z_opt])
        idx_opt = int(bitstring, 2)

        # -------------------------
        # Loop over p
        # -------------------------
        for p in range(p_max + 1):

            if p == 0:
                gammas, betas = [], []
            else:
                params = run_qaoa_qiskit(
                    h, J,
                    backend=None,
                    p=p,
                    N=N,
                    constrained=False,
                    simulator=True,
                    optimizer="BFGS",
                    shots=4,
                    maxiter=100
                )
                gammas = params[:p]
                betas  = params[p:]

            qc = build_qaoa_circuit(
                h, J,
                gammas, betas,
                N=N,
                constrained=False
            )

            psi = Statevector.from_instruction(qc).data
            probs = np.abs(psi) ** 2

            raw_data[inst, p] = probs[idx_opt]

    # -------------------------
    # Average
    # -------------------------
    avg_data = np.mean(raw_data, axis=0)

    p_values = np.arange(p_max + 1)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure()
    plt.plot(p_values, avg_data, marker='o')
    plt.xlabel("QAOA depth p")
    plt.ylabel("Average success probability")
    plt.title(f"N={N}, ATSP, lambda={lam}")
    plt.grid()

    fig_name = f"QAOA_performance_N={N}_ATSP_lam={lam}.png"
    plt.savefig(fig_name, dpi=300)
    plt.show()

    print(f"Figure saved as: {fig_name}")

    # -------------------------
    # Save raw data
    # -------------------------
    raw_filename = f"raw_data_N={N}_ATSP_lam={lam}.txt"
    np.savetxt(raw_filename, raw_data)

    print(f"Raw data saved as: {raw_filename}")

    # -------------------------
    # Save averaged data
    # -------------------------
    ave_filename = f"ave_data_N={N}_ATSP_lam={lam}.txt"
    np.savetxt(ave_filename, avg_data)

    print(f"Averaged data saved as: {ave_filename}")

    return p_values, avg_data, raw_data
    
average_success_probability_vs_p(N=2, lam=2.2, p_max=2, num_instances=50)    
