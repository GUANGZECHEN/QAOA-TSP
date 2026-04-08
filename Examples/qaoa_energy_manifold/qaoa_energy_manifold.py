import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../../src"))

import numpy as np
import matplotlib.pyplot as plt

from tsp_generator import TSP
from classical_solver import solve_tsp_bruteforce
from qaoa_qiskit_solver import run_qaoa_qiskit, build_qaoa_circuit
from qiskit.quantum_info import Statevector

def plot_energy_landscape_p1_dual(
    tsp,
    lam=2.0,
    gamma_points=100,
    beta_points=100
):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from qiskit.quantum_info import Statevector
    from qaoa_qiskit_solver import build_qaoa_circuit
    from itertools import product

    # -------------------------
    # Build problem
    # -------------------------
    A = lam * np.max(tsp.distance_matrix)
    tsp.build_qubo(A=A)

    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const

    n = len(h)

    # -------------------------
    # Grid
    # -------------------------
    gammas = np.linspace(0, np.pi, gamma_points)
    betas  = np.linspace(0, np.pi, beta_points)

    E_grid = np.zeros((gamma_points, beta_points))

    all_z = np.array(list(product([-1, 1], repeat=n)))

    # -------------------------
    # Evaluate landscape
    # -------------------------
    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):

            qc = build_qaoa_circuit(
                h, J,
                gammas=[gamma],
                betas=[beta],
                N=tsp.N,
                constrained=False
            )

            psi = Statevector.from_instruction(qc).data
            probs = np.abs(psi) ** 2

            E = 0.0
            for idx, z in enumerate(all_z):
                E += probs[idx] * (z @ J @ z + h @ z)

            E_grid[i, j] = E

    # -------------------------
    # Create figure
    # -------------------------
    fig = plt.figure(figsize=(12, 5))

    # =========================
    # (a) 3D surface
    # =========================
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    B, G = np.meshgrid(betas / np.pi, gammas / np.pi)

    ax1.plot_surface(B, G, E_grid, cmap='viridis')

    ax1.set_xlabel(r'$\beta / \pi$')
    ax1.set_ylabel(r'$\gamma / \pi$')
    ax1.set_zlabel('Energy')
    ax1.set_title('(a) Energy surface')

    # =========================
    # (b) 2D heatmap
    # =========================
    ax2 = fig.add_subplot(1, 2, 2)

    im = ax2.imshow(
        E_grid,
        extent=[0, 1, 0, 1],
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )

    ax2.set_xlabel(r'$\beta / \pi$')
    ax2.set_ylabel(r'$\gamma / \pi$')
    ax2.set_title('(b) Energy heatmap')

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Energy')

    # -------------------------
    # Save
    # -------------------------
    filename = f"energy_landscape_dual_N={tsp.N}_lam={lam}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved: {filename}")
    
tsp = TSP.random_asymmetric(N=3, seed=0)
#tsp.return_to_start=False
plot_energy_landscape_p1_dual(tsp, lam=5)
