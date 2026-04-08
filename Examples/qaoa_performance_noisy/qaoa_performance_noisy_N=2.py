import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../../src"))

from tsp_generator import TSP
from classical_solver import solve_tsp_bruteforce
from qaoa_qiskit_solver import solve_ising_qaoa_qiskit
from qubo import normalize_route


# =========================================================
# p = 0 baseline: uniform random sampling
# =========================================================
def sample_uniform_tsp(tsp, shots, seed=None):

    N = tsp.N
    n = tsp.ising.h.shape[0]

    best_route = None
    best_cost = np.inf
    
    rng = np.random.default_rng(seed)

    for _ in range(shots):

        z = rng.choice([-1, 1], size=n)
        x = tsp.z_to_x(z)

        route = tsp.x_to_route(x)

        if route==None:
            continue

        cost = tsp.route_cost(route)

        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route, best_cost


# =========================================================
# Main benchmark
# =========================================================
def success_rate_vs_shots(
    N=3,
    num_instances=50,
    p_max=5,
    shots_list=(50, 100, 200, 500),
    lam=5,
    seed=0,
    use_cost=True,   # ← switch metric here
):

    np.random.seed(seed)

    num_shots = len(shots_list)
    raw_data = np.zeros((num_instances, p_max + 1, num_shots))

    for inst in range(num_instances):

        # -------------------------
        # Generate instance
        # -------------------------
        tsp = TSP.random_asymmetric(N=N, seed=seed + inst)

        tsp.return_to_start = False
        start = tsp.start_city

        A = lam * np.max(tsp.distance_matrix)
        tsp.build_qubo(A=A)
        tsp.ensure_ising()

        # -------------------------
        # Ground truth
        # -------------------------
        true_route, true_cost = solve_tsp_bruteforce(tsp)

        if tsp.return_to_start:
            true_route_cmp = normalize_route(true_route, start)
        else:
            true_route_cmp = true_route

        #print(true_cost)
        # -------------------------
        # Loop over p and shots
        # -------------------------
        
        ###
        for p in range(p_max + 1):
        ###

            for s_idx, shots in enumerate(shots_list):

                try:

                    # -------------------------
                    # p = 0 baseline
                    # -------------------------
                    if p == 0:
                        route_qaoa, cost_qaoa = sample_uniform_tsp(tsp, shots)

                    # -------------------------
                    # QAOA
                    # -------------------------
                    else:
                        route_qaoa, energy_qaoa = solve_ising_qaoa_qiskit(
                            tsp,
                            p=p,
                            shots=shots,
                            constrained=False,
                            simulator=True,
                            noise=True,
                            optimizer="BFGS"   # ← important improvement
                        )

                    # -------------------------
                    # Evaluate success
                    # -------------------------
                    if route_qaoa is not None:
                    
                        cost_qaoa = tsp.route_cost(route_qaoa)
                        #print(f"p={p}:",cost_qaoa)

                        # ---------- COST-based (recommended) ----------
                        if use_cost:
                            success = abs(cost_qaoa - true_cost) < 1e-6

                        # ---------- ROUTE-based ----------
                        else:
                            if tsp.return_to_start:
                                route_qaoa_cmp = normalize_route(route_qaoa, start)
                            else:
                                route_qaoa_cmp = route_qaoa

                            success = (route_qaoa_cmp == true_route_cmp)

                        raw_data[inst, p, s_idx] = int(success)

                    else:
                        raw_data[inst, p, s_idx] = 0

                except Exception as e:
                    print(f"Error at inst={inst}, p={p}, shots={shots}: {e}")
                    raw_data[inst, p, s_idx] = 0

    # =========================================================
    # Average
    # =========================================================
    avg_data = np.mean(raw_data, axis=0)
    
    ###
    p_values = np.arange(p_max + 1)
    ###


    # =========================================================
    # Plot
    # =========================================================
    plt.figure()

    for s_idx, shots in enumerate(shots_list):
        plt.plot(
            p_values,
            avg_data[:, s_idx],
            marker='o',
            label=f"shots={shots}"
        )

    plt.xlabel("QAOA depth p")
    plt.ylabel("Success rate")
    plt.title(f"N={N}, ATSP, lambda={lam}")
    plt.legend()
    plt.grid()

    fig_name = f"QAOA_success_vs_p_N={N}_lam={lam}.png"
    plt.savefig(fig_name, dpi=300)
    plt.show()

    print(f"Figure saved as: {fig_name}")

    # =========================================================
    # Save data
    # =========================================================
    np.savetxt(
        f"raw_success_N={N}_lam={lam}.txt",
        raw_data.reshape(num_instances, -1)
    )

    np.savetxt(
        f"ave_success_N={N}_lam={lam}.txt",
        avg_data
    )

    return p_values, shots_list, avg_data, raw_data

# -------------------------
# Run experiment
# ------------------------

N=2
p_max=7
shots_list=[2, 4, 8, 16]
num_instances=100
    
success_rate_vs_shots(
    N=N,
    num_instances=num_instances,
    p_max=p_max,
    shots_list=shots_list,
    lam=5,
    seed=0,
)
