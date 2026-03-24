import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

######## test functions

def true_ising_energy_from_tsp(tsp, z):
    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    print(J)

    return z @ J @ z + h @ z + const
    
def qiskit_ising_energy_from_tsp(tsp, z):
    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const

    E = const
    n = len(h)

    # linear terms
    for i in range(n):
        E += h[i] * z[i]

    # quadratic terms (IMPORTANT: i < j only)
    for i in range(n):
        for j in range(n):
            if J[i, j] != 0:
                E += J[i, j] * z[i] * z[j]

    return E
    

def sanity_check_qiskit_vs_true_from_tsp(
    num_tests=5,
    N=4,
    seed_start=0,
    num_z_samples=10,
    atol=1e-8
):

    import numpy as np

    for k in range(num_tests):

        seed = seed_start + k
        tsp = TSP.random_asymmetric(N=N, seed=seed)

        tsp.ensure_ising()
        n = len(tsp.ising.h)

        print(f"\n=== Test {k} (seed={seed}) ===")

        for s in range(num_z_samples):

            z = np.random.choice([-1, 1], size=n)

            E_true = true_ising_energy_from_tsp(tsp, z)
            E_qiskit = qiskit_ising_energy_from_tsp(tsp, z)

            diff = E_true - E_qiskit

            print(f"z sample {s}")
            print("E_true   :", E_true)
            print("E_qiskit :", E_qiskit)
            print("diff     :", diff)
            print("-" * 30)

            if np.isclose(E_true, E_qiskit, atol=atol):
                continue

        print("⚠️ If differences are systematic → scaling bug confirmed")
            
# =========================================================
# Helper functions
# =========================================================

def is_valid_assignment(x, N):
    X = x.reshape(N, N)
    return np.all(X.sum(axis=0) == 1) and np.all(X.sum(axis=1) == 1)


def prepare_constrained_initial_state(qc, N):

    def idx(i, t):
        return i * N + t

    for t in range(N):
        qubits = [idx(i, t) for i in range(N)]
        state = np.zeros(2**N)

        for i in range(N):
            bit = ['0'] * N
            bit[i] = '1'
            index = int("".join(bit), 2)
            state[index] = 1

        state /= np.linalg.norm(state)
        qc.initialize(state, qubits)


def apply_xy_mixer(qc, beta, N):

    def idx(i, t):
        return i * N + t

    for t in range(N):
        for i in range(N):
            for j in range(i + 1, N):
                q1 = idx(i, t)
                q2 = idx(j, t)
                qc.rxx(2 * beta, q1, q2)
                qc.ryy(2 * beta, q1, q2)


# =========================================================
# Circuit construction
# =========================================================

def build_qaoa_circuit(h, J, gammas, betas, N, constrained=False):

    n = len(h)
    p = len(gammas)

    qc = QuantumCircuit(n)

    if constrained:
        prepare_constrained_initial_state(qc, N)
    else:
        qc.h(range(n))

    for layer in range(p):

        gamma = gammas[layer]
        beta  = betas[layer]

        # -------------------------
        # Linear terms
        # -------------------------
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * gamma * h[i], i)

        # -------------------------
        # Quadratic terms (FULL J)
        # -------------------------
        for i in range(n):
            for j in range(n):

                if i == j:
                    # Z_i Z_i = I → global phase → ignore
                    continue

                if J[i, j] != 0:
                    qc.rzz(2 * gamma * J[i, j], i, j)

        # -------------------------
        # Mixer
        # -------------------------
        if constrained:
            apply_xy_mixer(qc, beta, N)
        else:
            for i in range(n):
                qc.rx(2 * beta, i)

    return qc


# =========================================================
# Backend
# =========================================================

def _get_backend(simulator=True):
    if simulator:
        return AerSimulator()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        return service.least_busy(simulator=False, operational=True)


# =========================================================
# Sampling (correct modern API)
# =========================================================

def _sample_from_backend(qc, backend, shots=500, simulator=True):

    qc_meas = qc.copy()
    qc_meas.measure_all()

    qc_trans = transpile(qc_meas, backend)

    samples = []

    if simulator:
        # Local simulator path
        job = backend.run(qc_trans, shots=shots)
        
        result = job.result()
        counts = result.get_counts()

    else:
        # IBM Runtime path (Sampler V2)
        from qiskit_ibm_runtime import Sampler, Session

        with Session(backend=backend):
            sampler = Sampler()
            job = sampler.run([qc_trans], shots=shots)
            result = job.result()
            counts = result[0].data.meas.get_counts()

    for bitstring, count in counts.items():
        bitstring = bitstring[::-1]

        for _ in range(count):
            z = np.array([1 if b == '0' else -1 for b in bitstring])
            samples.append(z)

    return samples


# =========================================================
# QAOA optimizer
# =========================================================

def _spsa_optimize(objective, init, maxiter=100, a=0.2, c=0.1):

    theta = init.copy()
    dim = len(theta)

    for k in range(maxiter):

        delta = np.random.choice([-1, 1], size=dim)

        theta_plus  = theta + c * delta
        theta_minus = theta - c * delta

        y_plus  = objective(theta_plus)
        y_minus = objective(theta_minus)

        # gradient estimate
        g = (y_plus - y_minus) / (2 * c * delta)

        # learning rate decay (optional but helpful)
        ak = a / (k + 1)**0.602
        ck = c / (k + 1)**0.101

        theta = theta - ak * g

    return theta
    
def run_qaoa_qiskit(
    h, J,
    backend,
    p=1,
    N=None,
    constrained=False,
    shots=200,
    simulator=True,
    optimizer="COBYLA",
    maxiter=100
):

    n = len(h)

    def objective_statevector(params):

        gammas = params[:p]
        betas  = params[p:]

        qc = build_qaoa_circuit(
            h, J, gammas, betas,
            N=N,
            constrained=constrained
        )

        psi = Statevector.from_instruction(qc)

        probs = np.abs(psi.data) ** 2

        E = 0.0

        for idx in range(2**n):
            bitstring = format(idx, f"0{n}b")[::-1]
            z = np.array([1 if b == '0' else -1 for b in bitstring])

            E += probs[idx] * (z @ J @ z + h @ z)

        return E

    def objective_sampling(params):

        gammas = params[:p]
        betas  = params[p:]

        qc = build_qaoa_circuit(
            h, J, gammas, betas,
            N=N,
            constrained=constrained
        )

        samples = _sample_from_backend(
            qc, backend,
            shots=shots,
            simulator=simulator
        )

        energies = [z @ J @ z + h @ z for z in samples]

        return np.mean(energies)

    # =====================================================
    # Initialization
    # =====================================================

    init = np.random.uniform(0, np.pi, size=2 * p)

    # =====================================================
    # Optimizer selection
    # =====================================================

    if optimizer.upper() == "BFGS":

        # ### NEW: use statevector objective
        result = minimize(
            objective_statevector,
            init,
            method="BFGS",
            options={"maxiter": maxiter}
        )
        return result.x

    elif optimizer.upper() == "COBYLA":

        # ### CHANGED: keep sampling-based
        result = minimize(
            objective_sampling,
            init,
            method="COBYLA",
            options={"maxiter": maxiter}
        )
        return result.x

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

# =========================================================
# Main solver
# =========================================================

def solve_ising_qaoa_qiskit(
    tsp,
    p=1,
    shots=500,
    constrained=False,
    debug=False,
    simulator=True,
    optimizer="COBYLA"
):

    tsp.ensure_ising()

    h = tsp.ising.h
    J = tsp.ising.J
    const = tsp.ising.const
    N = tsp.N

    backend = _get_backend(simulator=simulator)

    # -------------------------
    # Optimize
    # -------------------------
    params = run_qaoa_qiskit(
    h, J,
    backend=backend,
    p=p,
    N=N,
    constrained=constrained,
    shots=shots,
    simulator=simulator,
    optimizer=optimizer,
    maxiter=100
    )

    gammas = params[:p]
    betas  = params[p:]

    # -------------------------
    # Final circuit
    # -------------------------
    qc = build_qaoa_circuit(
        h, J,
        gammas, betas,
        N=N,
        constrained=constrained
    )

    # -------------------------
    # Sampling
    # -------------------------
    z_samples = _sample_from_backend(
        qc, backend,
        shots=shots,
        simulator=simulator
    )

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

    if debug:
        print(f"Valid samples: {valid_count}/{shots}")

    return best_route, best_E
    
from tsp_generator import TSP
from classical_solver import solve_tsp_bruteforce
from qubo import normalize_route

# =========================================================
# Sanity test: QAOA (Qiskit) vs exact solver
# =========================================================

def sanity_test_qaoa_qiskit(
    num_tests=20,
    N=3,
    seed_start=0,
    p=1,
    shots=500,
    constrained=False,
    verbose=False,
    stop_on_error=False,
    optimizer="COBYLA"
):
    """
    Compare QAOA (Qiskit) solver with exact brute-force TSP solver

    Parameters
    ----------
    constrained : bool
        If True, use constrained mixer + initial state
    """

    failures = 0
    no_solution = 0

    mode = "CONSTRAINED" if constrained else "STANDARD"

    for k in range(num_tests):

        seed = seed_start + k

        tsp = TSP.random_asymmetric(N=N, seed=seed)
        start = tsp.start_city

        # -------------------------
        # QAOA (Qiskit)
        # -------------------------
        try:
            qaoa_route, qaoa_energy = solve_ising_qaoa_qiskit(
                tsp,
                p=p,
                shots=shots,
                constrained=constrained,
                debug=False,
                optimizer=optimizer
            )
        except Exception as e:
            print(f"\n❌ QAOA crashed at test {k}, seed={seed}")
            print(e)
            failures += 1
            continue

        # -------------------------
        # Handle no valid solution
        # -------------------------
        if qaoa_route is None:
            print(f"\n⚠️ No valid solution at test {k}, seed={seed}")
            no_solution += 1
            failures += 1
            continue

        # normalize (same as your pipeline)
        qaoa_route = normalize_route(qaoa_route, start)

        # -------------------------
        # Ground truth
        # -------------------------
        true_route, true_cost = solve_tsp_bruteforce(tsp)

        # -------------------------
        # Compare
        # -------------------------
        qaoa_cost = tsp.route_cost(qaoa_route)

        if not np.isclose(qaoa_cost, true_cost):

            failures += 1

            print("\n❌ FAILURE at test", k, f"[{mode}]")
            print("Seed:", seed)
            print("Distance matrix:\n", tsp.distance_matrix)

            print("\nQAOA route:", qaoa_route)
            print("QAOA cost:", qaoa_cost)

            print("\nTrue route:", true_route)
            print("True cost:", true_cost)

            # optional visualization
            try:
                tsp.plot(true_route)
                tsp.plot(qaoa_route)
            except:
                pass

            if stop_on_error:
                raise ValueError("QAOA Qiskit sanity test failed")

        elif verbose:
            print(f"✅ Test {k} passed [{mode}]")

    # -------------------------
    # Summary
    # -------------------------
    print("\n==============================")
    print(f"Mode: {mode}")
    print(f"Total tests: {num_tests}")
    print(f"Failures: {failures}")
    print(f"No valid solutions: {no_solution}")
    print(f"Success rate: {(num_tests - failures)/num_tests:.2%}")
    print("==============================")

    if failures == 0:
        print(f"🎉 QAOA ({mode}) passed all tests!")
