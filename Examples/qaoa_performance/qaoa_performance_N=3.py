import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../../src"))

from tsp_generator import TSP
from classical_solver import solve_tsp_bruteforce
from qaoa_qiskit_solver import solve_ising_qaoa_qiskit
from qubo import normalize_route


# =========================================================
# p = 0 baseline
# =========================================================
def sample_uniform_tsp(tsp, shots, seed=None):

    n = tsp.ising.h.shape[0]

    best_route = None
    best_cost = np.inf
    
    rng = np.random.default_rng(seed)

    for _ in range(shots):

        z = rng.choice([-1, 1], size=n)
        x = tsp.z_to_x(z)

        route = tsp.x_to_route(x)

        if route is None:
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
    p_list=(0, 1, 2, 3, 4, 5),
    shots_list=(50, 100, 200, 500),
    lam=5,
    seed=0,
    use_cost=True,
    verbose=True,
):

    np.random.seed(seed)

    num_shots = len(shots_list)
    num_p = len(p_list)

    raw_data = np.zeros((num_instances, num_p, num_shots))
    runtime_data = np.zeros((num_instances, num_p, num_shots))

    total_jobs = num_instances * num_p * num_shots
    job_counter = 0
    start_time_global = time.time()

    for inst in range(num_instances):

        tsp = TSP.random_asymmetric(N=N, seed=seed + inst)
        start = tsp.start_city

        A = lam * np.max(tsp.distance_matrix)
        tsp.build_qubo(A=A)
        tsp.ensure_ising()

        true_route, true_cost = solve_tsp_bruteforce(tsp)

        if tsp.return_to_start:
            true_route_cmp = normalize_route(true_route, start)
        else:
            true_route_cmp = true_route

        for p_idx, p in enumerate(p_list):

            for s_idx, shots in enumerate(shots_list):

                job_counter += 1
                start_time_job = time.time()

                if verbose:
                    print(f"[{job_counter}/{total_jobs}] Start: inst={inst}, p={p}, shots={shots}")

                try:

                    if p == 0:
                        route_qaoa, cost_qaoa = sample_uniform_tsp(tsp, shots)
                    else:
                        route_qaoa, energy_qaoa = solve_ising_qaoa_qiskit(
                            tsp,
                            p=p,
                            shots=shots,
                            constrained=False,
                            simulator=True,
                            optimizer="BFGS"
                        )

                    elapsed_job = time.time() - start_time_job
                    runtime_data[inst, p_idx, s_idx] = elapsed_job

                    if route_qaoa is not None:

                        cost_qaoa = tsp.route_cost(route_qaoa)

                        if use_cost:
                            success = abs(cost_qaoa - true_cost) < 1e-6
                        else:
                            if tsp.return_to_start:
                                route_qaoa_cmp = normalize_route(route_qaoa, start)
                            else:
                                route_qaoa_cmp = route_qaoa

                            success = (route_qaoa_cmp == true_route_cmp)

                        raw_data[inst, p_idx, s_idx] = int(success)
                    else:
                        raw_data[inst, p_idx, s_idx] = 0

                    elapsed_total = time.time() - start_time_global
                    avg_time = elapsed_total / job_counter
                    remaining = avg_time * (total_jobs - job_counter)

                    if verbose:
                        print(
                            f"[{job_counter}/{total_jobs}] Done: "
                            f"inst={inst}, p={p}, shots={shots} | "
                            f"time={elapsed_job:.2f}s | total={elapsed_total:.1f}s | "
                            f"ETA={remaining/60:.1f} min"
                        )

                except Exception as e:

                    elapsed_job = time.time() - start_time_job
                    runtime_data[inst, p_idx, s_idx] = elapsed_job

                    if verbose:
                        print(
                            f"[{job_counter}/{total_jobs}] ERROR: "
                            f"inst={inst}, p={p}, shots={shots} | "
                            f"time={elapsed_job:.2f}s | error={e}"
                        )

                    raw_data[inst, p_idx, s_idx] = 0

    # =========================================================
    # Averages
    # =========================================================
    avg_data = np.mean(raw_data, axis=0)
    avg_runtime = np.mean(runtime_data, axis=0)

    p_values = np.array(p_list)

    # =========================================================
    # Save data
    # =========================================================
    np.savetxt(f"ave_success_N={N}_lam={lam}.txt", avg_data)

    runtime_filename = f"running_time_N={N}_lam={lam}_num_instance={num_instances}.txt"
    np.savetxt(runtime_filename, avg_runtime)

    print(f"Runtime saved to: {runtime_filename}")

    # =========================================================
    # Plot success
    # =========================================================
    plt.figure()

    for s_idx, shots in enumerate(shots_list):
        plt.plot(p_values, avg_data[:, s_idx], marker='o', label=f"shots={shots}")

    plt.xlabel("QAOA depth p")
    plt.ylabel("Success rate")
    plt.title(f"N={N}, ATSP, lambda={lam}")
    plt.legend()
    plt.grid()

    fig_name = f"QAOA_success_vs_p_custom_N={N}_lam={lam}.png"
    plt.savefig(fig_name, dpi=300)
    plt.show()

    print(f"Figure saved as: {fig_name}")

    # =========================================================
    # NEW: Runtime vs p plot (averaged over shots)
    # =========================================================
    runtime_vs_p = np.mean(avg_runtime, axis=1)

    plt.figure()
    plt.plot(p_values, runtime_vs_p, marker='o')

    plt.xlabel("QAOA depth p")
    plt.ylabel("Average runtime (s)")
    plt.title(f"Runtime vs p (averaged over shots), N={N}")

    plt.grid()

    runtime_fig_name = f"runtime_vs_p_N={N}_lam={lam}.png"
    plt.savefig(runtime_fig_name, dpi=300)
    plt.show()

    print(f"Runtime figure saved as: {runtime_fig_name}")

    # =========================================================
    # Optional print
    # =========================================================
    print("\nAverage runtime (seconds):")
    for p_idx, p in enumerate(p_list):
        print(f"p={p} -> {runtime_vs_p[p_idx]:.3f}s")

    return p_values, shots_list, avg_data, avg_runtime, runtime_vs_p, raw_data


# =========================================================
# Run experiment
# =========================================================
if __name__ == "__main__":

    N = 3
    p_list = [0, 2, 4, 6, 8, 10, 12]
    shots_list = [128]
    num_instances = 10

    success_rate_vs_shots(
        N=N,
        num_instances=num_instances,
        p_list=p_list,
        shots_list=shots_list,
        lam=5,
        seed=0,
        verbose=True,
    )
