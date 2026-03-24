# QAOA for Traveling Salesman Problem (TSP)

This repository implements a full pipeline for solving the Traveling Salesman Problem (TSP) using the Quantum Approximate Optimization Algorithm (QAOA), including:

- TSP → QUBO → Ising mapping
- Exact classical solvers for benchmarking
- Statevector QAOA simulation
- Qiskit-based circuit implementation
- Tools to analyze success probability vs QAOA depth

---

## Features

-  Generate random **asymmetric TSP (ATSP)** instances  
-  Exact **brute-force solvers** for validation  
-  Automatic **QUBO → Ising transformation**  
-  Full **Hamiltonian construction**  
-  **Statevector QAOA** (exact simulation)  
-  **Qiskit-based QAOA circuits** (sampling + hardware-ready)  
-  Evaluation of **success probability vs circuit depth**
