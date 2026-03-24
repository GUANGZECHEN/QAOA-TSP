from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session

# -------------------------
# Backend
# -------------------------
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True)

print("Using backend:", backend.name)
print("Pending jobs:", backend.status().pending_jobs)

# -------------------------
# Simple circuit
# -------------------------
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

qc = transpile(qc, backend)

# -------------------------
# Run ONE job
# -------------------------
with Session(backend=backend):
    sampler = Sampler()

    print("Submitting job...")
    job = sampler.run([qc], shots=100)

    print("Job ID:", job.job_id())   # ADD THIS

    print("Waiting for result...")
    result = job.result()

# -------------------------
# Result
# -------------------------
counts = result[0].data.meas.get_counts()

print("Counts:", counts)
