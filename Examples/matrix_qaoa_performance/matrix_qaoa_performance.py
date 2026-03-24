import sys
import os

sys.path.append(os.path.abspath("../../src"))

import numpy as np
import matplotlib.pyplot as plt

from tsp_generator import TSP
from classical_solver import solve_tsp_bruteforce
from Ising import solve_ising_qaoa   # <-- your matrix solver
from qubo import normalize_route


# =========================================================
# Experiment
# =========================================================

def run_experiment_matrix(N):

    A_scales = [10]
    p_values = [1, 2, 3, 4, 5]
    samples_values = [100]

    num_instances = 10

    # -----------------------------------------------------
    # Generate dataset ONCE
    # -----------------------------------------------------
    tsps = []
    ground_truth = []

    print("Generating TSP instances...")

    for seed in range(num_instances):
        tsp = TSP.random_asymmetric(N=N, seed=seed)
        tsps.append(tsp)

        true_route, true_cost = solve_tsp_bruteforce(tsp)
        true_route = normalize_route(true_route, tsp.start_city)
        ground_truth.append((true_route, true_cost))

    print("Done.\n")

    results = []

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------
    for A_scale in A_scales:

        print(f"\n===== A scale = {A_scale} =====")

        for p in p_values:
            for samples in samples_values:

                print(f"N={N}, A={A_scale}, p={p}, samples={samples}")

                success_count = 0
                valid_counts = []

                for idx, tsp in enumerate(tsps):

                    start = tsp.start_city
                    true_route, true_cost = ground_truth[idx]

                    # -------------------------
                    # rebuild QUBO with A
                    # -------------------------
                    max_dist = np.max(tsp.distance_matrix)
                    A = A_scale * max_dist

                    tsp.Q = None
                    tsp.ising = None
                    tsp.build_qubo(A=A)
                    tsp.to_ising()

                    # -------------------------
                    # QAOA (matrix version)
                    # -------------------------
                    try:
                        route, _ = solve_ising_qaoa(
                            tsp,
                            p=p,
                            samples=samples,
                            debug=False
                        )
                    except Exception as e:
                        print("Error:", e)
                        continue

                    if route is None:
                        valid_counts.append(0)
                        continue

                    route = normalize_route(route, start)
                    qaoa_cost = tsp.route_cost(route)

                    # count success
                    if np.isclose(qaoa_cost, true_cost):
                        success_count += 1

                success_rate = success_count / num_instances
                print(f"  -> success rate: {success_rate:.3f}")

                results.append((A_scale, p, samples, success_rate))

    # -----------------------------------------------------
    # Save raw data
    # -----------------------------------------------------
    filename = f"matrix_QAOA_N={N}.txt"
    with open(filename, "w") as f:
        for row in results:
            f.write(f"{row[0]} {row[1]} {row[2]} {row[3]}\n")

    print(f"\nSaved raw data to {filename}")

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    for A_scale in A_scales:

        plt.figure()

        for (A_s, p, samples, success) in results:
            if A_s == A_scale:
                cost = p * samples
                label = f"(p={p}, s={samples})"
                plt.scatter(cost, success)
                plt.text(cost, success, label)

        plt.xlabel("Cost (p * samples)")
        plt.ylabel("Success rate")
        plt.title(f"Matrix QAOA | N={N}, A={A_scale} * max(distance)")

        plot_name = f"matrix_QAOA_N={N}_A={A_scale}.png"
        plt.savefig(plot_name)
        plt.close()

        print(f"Saved plot {plot_name}")

    return results


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    run_experiment_matrix(N=3)
