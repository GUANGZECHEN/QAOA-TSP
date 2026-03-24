import numpy as np
import itertools

from tsp_generator import TSP
from qubo import solve_qubo_bruteforce, normalize_route
from classical_solver import solve_tsp_bruteforce


# =========================================================
# Energy
# =========================================================

def ising_energy(z, h, J, const=0.0):
    return float(z @ J @ z + h @ z + const)


# =========================================================
# Sanity: QUBO → Ising
# =========================================================

def sanity_test_qubo_to_ising(
    num_tests=20,
    N=4,
    seed_start=0,
    atol=1e-8,
    verbose=False,
    stop_on_error=True
):

    failures = 0
    total_checked = 0

    for k in range(num_tests):

        seed = seed_start + k

        tsp = TSP.random_asymmetric(N=N, seed=seed)
        tsp.ensure_ising()

        Q = tsp.Q
        h, J, const = tsp.ising.h, tsp.ising.J, tsp.ising.const

        n = Q.shape[0]

        for bits in itertools.product([0, 1], repeat=n):

            x = np.array(bits)
            z = 1 - 2 * x

            E_qubo = x @ Q @ x
            E_ising = z @ J @ z + h @ z + const

            total_checked += 1

            if not np.isclose(E_qubo, E_ising, atol=atol):

                failures += 1

                print("\n❌ ENERGY MISMATCH")
                print("Seed:", seed)
                print("x:", x)
                print("E_qubo :", E_qubo)
                print("E_ising:", E_ising)

                if stop_on_error:
                    raise ValueError("QUBO → Ising mismatch")

        if verbose:
            print(f"✅ Test {k} passed")

    print("\n==============================")
    print(f"TSP instances tested: {num_tests}")
    print(f"Total states checked: {total_checked}")
    print(f"Failures: {failures}")
    print("==============================")

    if failures == 0:
        print("🎉 QUBO → Ising transformation is EXACT!")


# =========================================================
# Classical Ising solver
# =========================================================

def solve_ising_bruteforce(tsp):   # best for N < 5

    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const

    N = tsp.N
    num_vars = N * N

    best_E = np.inf
    best_z = None

    for spins in itertools.product([-1, 1], repeat=num_vars):

        z = np.array(spins)
        E = ising_energy(z, h, J, const)

        if E < best_E:
            best_E = E
            best_z = z.copy()

    # map back
    x = ((1 - best_z) // 2).astype(int)
    route = tsp.x_to_route(x)

    return route, best_E


# =========================================================
# Sanity: solver consistency
# =========================================================

def sanity_test_ising_solver(
    num_tests=20,
    N=4,
    seed_start=0,
    verbose=False,
    stop_on_error=True
):

    failures = 0

    for k in range(num_tests):

        seed = seed_start + k

        tsp = TSP.random_asymmetric(N=N, seed=seed)
        start = tsp.start_city

        # -------------------------
        # QUBO solver
        # -------------------------
        qubo_route, _ = solve_qubo_bruteforce(tsp)
        qubo_route = normalize_route(qubo_route, start)
        qubo_cost = tsp.route_cost(qubo_route)

        # -------------------------
        # Ising solver
        # -------------------------
        ising_route, _ = solve_ising_bruteforce(tsp)
        ising_route = normalize_route(ising_route, start)
        ising_cost = tsp.route_cost(ising_route)

        # -------------------------
        # Ground truth (classical TSP)
        # -------------------------
        true_route, true_cost = solve_tsp_bruteforce(tsp)
        true_route = normalize_route(true_route, start)

        # -------------------------
        # Compare all
        # -------------------------
        ok_qubo_ising = np.isclose(qubo_cost, ising_cost)
        ok_qubo_true  = np.isclose(qubo_cost, true_cost)
        ok_ising_true = np.isclose(ising_cost, true_cost)

        if not (ok_qubo_ising and ok_qubo_true and ok_ising_true):

            failures += 1

            print("\n❌ SOLVER MISMATCH")
            print("Seed:", seed)
            print("Distance matrix:\n", tsp.distance_matrix)

            print("\nQUBO route:", qubo_route)
            print("QUBO cost :", qubo_cost)

            print("\nIsing route:", ising_route)
            print("Ising cost :", ising_cost)

            print("\nTrue route:", true_route)
            print("True cost :", true_cost)

            # Optional: show energy-level mismatch too
            tsp.ensure_ising()
            h, J, const = tsp.ising.h, tsp.ising.J, tsp.ising.const

            z_qubo  = 1 - 2 * tsp.route_to_x(qubo_route)
            z_ising = 1 - 2 * tsp.route_to_x(ising_route)

            E_qubo  = z_qubo @ J @ z_qubo + h @ z_qubo + const
            E_ising = z_ising @ J @ z_ising + h @ z_ising + const

            print("\nEnergy diagnostics:")
            print("E(qubo) :", E_qubo)
            print("E(ising):", E_ising)

            if stop_on_error:
                raise ValueError("Full solver mismatch")

        elif verbose:
            print(f"✅ Test {k} passed")

    # -------------------------
    # Summary
    # -------------------------
    print("\n==============================")
    print(f"Total tests: {num_tests}")
    print(f"Failures: {failures}")
    print(f"Success rate: {(num_tests - failures)/num_tests:.2%}")
    print("==============================")

    if failures == 0:
        print("🎉 All solvers match perfectly!")


# =========================================================
# Valid assignment
# =========================================================

def is_valid_assignment(x, N):
    X = x.reshape(N, N)
    return np.all(X.sum(axis=0) == 1) and np.all(X.sum(axis=1) == 1)



########## Quantum solver

import numpy as np
from functools import reduce
from scipy.linalg import expm
from scipy.optimize import minimize

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# =========================================================
# Utilities
# =========================================================

def kron_n(ops):
    return reduce(np.kron, ops)


def initial_state(n):
    plus = np.array([1, 1]) / np.sqrt(2)
    state = plus
    for _ in range(n - 1):
        state = np.kron(state, plus)
    return state.astype(complex)
    
# =========================================================
# Hamiltonians
# =========================================================

def build_cost_hamiltonian(h, J, const=0.0):
    n = len(h)
    H = np.zeros((2**n, 2**n), dtype=complex)

    # linear terms
    for i in range(n):
        ops = [I] * n
        ops[i] = Z
        H += h[i] * kron_n(ops)

    # quadratic terms
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if J[i, j] != 0:
                ops = [I] * n
                ops[i] = Z
                ops[j] = Z
                H += J[i, j] * kron_n(ops)
                
    # add constant shift
    diag_sum = np.sum(np.diag(J))
    H += (diag_sum) * np.eye(2**n)

    return H


def build_mixer_hamiltonian(n):
    H = np.zeros((2**n, 2**n), dtype=complex)

    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H += kron_n(ops)

    return H        

def solve_ising_via_hamiltonian(tsp):
    """
    Solve Ising by brute-force using the full Hamiltonian matrix H
    """

    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    N = tsp.N

    H = build_cost_hamiltonian(h, J)

    n = len(h)

    best_E = np.inf
    best_z = None

    # iterate over all spin configurations
    for spins in itertools.product([-1, 1], repeat=n):

        z = np.array(spins)

        # build computational basis state |z>
        # map: z=+1 -> |0>, z=-1 -> |1>
        bits = [(0 if zi == 1 else 1) for zi in z]
        index = int("".join(map(str, bits)), 2)

        psi = np.zeros(2**n)
        psi[index] = 1.0

        # energy = <z|H|z>
        E = np.real(np.conj(psi) @ (H @ psi)) + const

        if E < best_E:
            best_E = E
            best_z = z.copy()

    # map back to route
    x = ((1 - best_z) // 2).astype(int)
    route = tsp.x_to_route(x)

    return route, best_E

def sanity_test_cost_hamiltonian(
    num_tests=10,
    N=4,
    seed_start=0,
    verbose=False,
    stop_on_error=True
):

    failures = 0

    for k in range(num_tests):

        seed = seed_start + k

        tsp = TSP.random_asymmetric(N=N, seed=seed)
        start = tsp.start_city

        # -------------------------
        # Solve via Hamiltonian
        # -------------------------
        ham_route, ham_energy = solve_ising_via_hamiltonian(tsp)
        ham_route = normalize_route(ham_route, start)
        ham_cost = tsp.route_cost(ham_route)

        # -------------------------
        # Ground truth
        # -------------------------
        true_route, true_cost = solve_tsp_bruteforce(tsp)
        true_route = normalize_route(true_route, start)

        # -------------------------
        # Compare
        # -------------------------
        if not np.isclose(ham_cost, true_cost):

            failures += 1

            print("\n❌ HAMILTONIAN MISMATCH")
            print("Seed:", seed)
            print("Distance matrix:\n", tsp.distance_matrix)

            print("\nHamiltonian route:", ham_route)
            print("Hamiltonian cost :", ham_cost)

            print("\nTrue route:", true_route)
            print("True cost :", true_cost)

            # extra diagnostics
            tsp.ensure_ising()
            h, J, const = tsp.ising.h, tsp.ising.J, tsp.ising.const

            z = 1 - 2 * tsp.route_to_x(ham_route)
            E_direct = z @ J @ z + h @ z + const

            print("\nEnergy diagnostics:")
            print("From H      :", ham_energy)
            print("Direct Ising:", E_direct)

            if stop_on_error:
                raise ValueError("Cost Hamiltonian mismatch")

        elif verbose:
            print(f"✅ Test {k} passed")

    # -------------------------
    # Summary
    # -------------------------
    print("\n==============================")
    print(f"Total tests: {num_tests}")
    print(f"Failures: {failures}")
    print(f"Success rate: {(num_tests - failures)/num_tests:.2%}")
    print("==============================")

    if failures == 0:
        print("🎉 Cost Hamiltonian is correct!")
                
# =========================================================
# QAOA core
# =========================================================

import numpy as np
from scipy.optimize import minimize


def run_qaoa(h, J, p, n_starts=1, init_params=None, seed=None):
    """
    Optimize QAOA parameters.

    Args:
        h, J: Ising Hamiltonian
        p: QAOA depth
        n_starts: number of random restarts
        init_params: optional initial guess (array of length 2p)
        seed: random seed

    Returns:
        best_params: optimal parameters found
    """

    if seed is not None:
        np.random.seed(seed)

    def objective(params):
        psi = qaoa_state(h, J, params, p)
        probs = np.abs(psi) ** 2

        # expected energy
        E = 0.0
        for idx, z in enumerate(all_z):
            E += probs[idx] * (z @ J @ z + h @ z)

        return E

    # precompute all bitstrings once (important for speed)
    n = len(h)
    all_z = np.array(list(product([-1, 1], repeat=n)))

    best_E = np.inf
    best_params = None

    # ---------
    # helper: one optimization run
    # ---------
    def optimize_once(x0):
        res = minimize(
            objective,
            x0,
            method="BFGS",   # or "Nelder-Mead", "BFGS"
            options={"maxiter": 200}
        )
        return res.x, res.fun

    # ---------
    # main loop
    # ---------
    for i in range(n_starts):

        if i == 0 and init_params is not None:
            x0 = np.array(init_params)
        else:
            # random initialization in [0, 2π]
            x0 = np.random.uniform(0, 2*np.pi, size=2*p)

        params, E = optimize_once(x0)

        if E < best_E:
            best_E = E
            best_params = params

    return best_params


def qaoa_state(h, J, params, p):

    n = len(h)

    Hc = build_cost_hamiltonian(h, J)
    Hm = build_mixer_hamiltonian(n)

    gammas = params[:p]
    betas  = params[p:]

    psi = initial_state(n)

    for gamma, beta in zip(gammas, betas):
        psi = expm(-1j * beta * Hm) @ (expm(-1j * gamma * Hc) @ psi)

    return psi

# =========================================================
# Sampling
# =========================================================

def index_to_z(s, n):
    """
    index -> spin vector (consistent with kron ordering)
    """
    z = np.zeros(n, dtype=int)

    for i in range(n):
        bit = (s >> (n - 1 - i)) & 1  # MSB-first
        z[i] = 1 if bit == 0 else -1

    return z


def z_to_index(z):
    """
    spin vector -> basis index (inverse of index_to_z)
    """
    n = len(z)
    index = 0

    for i in range(n):
        bit = 0 if z[i] == 1 else 1
        index = (index << 1) | bit

    return index
    
def sample_bitstrings(psi, num_samples=100):

    probs = np.abs(psi)**2
    probs = probs / np.sum(probs)

    n = int(np.log2(len(psi)))

    samples = np.random.choice(len(probs), size=num_samples, p=probs)

    z_samples = []
    for s in samples:
        z = index_to_z(s, n)   # ✅ ONLY THIS
        z_samples.append(z)

    return z_samples
        
def is_valid_assignment(x, N):
    X = x.reshape(N, N)
    return np.all(X.sum(axis=0) == 1) and np.all(X.sum(axis=1) == 1)


# =========================================================
# Main solver qaoa
# =========================================================

def solve_ising_qaoa(tsp, p=1, samples=500, debug=False):

    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    N = tsp.N

    # optimize parameters
    params = run_qaoa(h, J, p)

    # build final state
    psi = qaoa_state(h, J, params, p)

    if debug:
        probs = np.abs(psi)**2
        probs = probs / np.sum(probs)
        print("Max probability:", probs.max())
        print("Top 5 probabilities:", np.sort(probs)[-5:])

    # sampling
    z_samples = sample_bitstrings(psi, samples)

    best_E = np.inf
    best_route = None
    valid_count = 0

    for z in z_samples:

        x = ((1 - z) // 2).astype(int)

        if not is_valid_assignment(x, N):
            continue

        valid_count += 1

        route = tsp.x_to_route(x)
        E = z @ J @ z + h @ z + const

        if E < best_E:
            best_E = E
            best_route = route

    print("Valid samples:", valid_count)

    return best_route, best_E

############## Debug qaoa

from itertools import product

def debug_qaoa_distribution(tsp, p_values=[1, 2, 3], tol=1e-9):
    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    N = tsp.N
    n_qubits = len(h)

    assert n_qubits == 4, "This debug is intended for N=2 (4 qubits)"

    # enumerate all bitstrings in z ∈ {-1, +1}^n
    all_z = np.array(list(product([-1, 1], repeat=n_qubits)))

    print("\n===== FULL STATE ENUMERATION (N=2 ATSP) =====\n")

    for p in p_values:
        print(f"\n--- p = {p} ---")

        # optimize parameters
        params = run_qaoa(h, J, p)

        # get full statevector
        psi = qaoa_state(h, J, params, p)

        probs = np.abs(psi) ** 2
        probs /= probs.sum()

        # track best valid solution probability
        best_valid_E = np.inf
        best_valid_prob = 0

        for idx, z in enumerate(all_z):
            prob = probs[idx]

            if prob < tol:
                continue

            # map to x ∈ {0,1}
            x = ((1 - z) // 2).astype(int)

            valid = is_valid_assignment(x, N)

            E = z @ J @ z + h @ z + const

            if valid and E < best_valid_E:
                best_valid_E = E
                best_valid_prob = prob

            print(
                f"z={z}, x={x}, "
                f"P={prob:.4f}, "
                f"E={E:.2f}, "
                f"{'VALID' if valid else 'INVALID'}"
            )

        print(f"\nBest valid energy: {best_valid_E}")
        print(f"Probability of best valid state: {best_valid_prob:.4f}")


    
# =========================================================
# Sanity check qaoa solver
# =========================================================    
    
def sanity_test_qaoa(
    num_tests=50,
    N=3,
    seed_start=0,
    p=1,
    samples=300,
    verbose=False,
    stop_on_error=False
):

    failures = 0

    for k in range(num_tests):

        seed = seed_start + k

        tsp = TSP.random_asymmetric(N=N, seed=seed)
        start = tsp.start_city

        tsp.ensure_ising()
        h, J, const = tsp.ising.h, tsp.ising.J, tsp.ising.const

        # --- Solve with QAOA ---
        try:
            ising_route, ising_energy = solve_ising_qaoa(
                tsp,
                p=p,
                samples=samples,
                debug=False
            )
        except Exception as e:
            print(f"\n❌ QAOA crashed at test {k}, seed={seed}")
            print(e)
            failures += 1
            continue

        # --- Handle no valid solution ---
        if ising_route is None:
            print(f"\n⚠️ No valid solution at test {k}, seed={seed}")
            failures += 1
            continue

        ising_route = normalize_route(ising_route, start)

        # --- True solution ---
        true_route, true_cost = solve_tsp_bruteforce(tsp)

        # --- Evaluate ---
        ising_cost = tsp.route_cost(ising_route)

        if not np.isclose(ising_cost, true_cost):

            failures += 1

            print("\n❌ FAILURE at test", k)
            print("Seed:", seed)
            print("Distance matrix:\n", tsp.distance_matrix)

            print("\nQAOA route:", ising_route)
            print("QAOA cost:", ising_cost)

            print("\nTrue route:", true_route)
            print("True cost:", true_cost)

            # --- Diagnostics ---
            try:
                params = run_qaoa(h, J, p)
                psi = qaoa_state(h, J, params, p)
                probs = np.abs(psi)**2

                print("\nMax probability:", probs.max())
                print("Top 5 probs:", np.sort(probs)[-5:])
            except:
                pass

            try:
                tsp.plot(true_route)
                tsp.plot(ising_route)
            except:
                pass

            if stop_on_error:
                raise ValueError("QAOA sanity test failed")

        elif verbose:
            print(f"✅ Test {k} passed")

    print("\n==============================")
    print(f"Total tests: {num_tests}")
    print(f"Failures: {failures}")
    print(f"Success rate: {(num_tests - failures)/num_tests:.2%}")
    print("==============================")

    if failures == 0:
        print("🎉 QAOA passed all tests!")


