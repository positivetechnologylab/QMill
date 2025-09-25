from __future__ import annotations
import os, csv
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

REAL_DATA_DIR = "ce_high_data"
SIM_DATA_DIR  = "ce_high_sim_data"
SHOTS         = 2048

N_QUBITS  = 3
N_ANCILLA = 3

def build_sensor_circuit(phi_soil: float, phi_free: float) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    data_qubits = list(range(N_QUBITS - 1))
    meas_qubit  = N_QUBITS - 1

    qc.h(data_qubits[0])
    qc.x(data_qubits[1])
    for i in range(len(data_qubits)-1):
        qc.cx(data_qubits[i], data_qubits[i+1])

    # soil phase on qubit 0, free phase on the rest
    qc.p(phi_soil, data_qubits[0])
    for q in data_qubits[1:]:
        qc.p(phi_free, q)

    qc.barrier()
    qc.cx(data_qubits[0], meas_qubit)
    qc.barrier()
    qc.h(data_qubits)
    qc.barrier()
    for q in data_qubits:
        qc.cx(q, meas_qubit)
    qc.barrier()

    return qc

def build_swaptest_circuit(phi_s: float, phi_f: float) -> QuantumCircuit:
    total_qubits = N_ANCILLA + 2 * N_QUBITS
    qc = QuantumCircuit(total_qubits, N_ANCILLA)

    # two copies side by side
    qc.compose(build_sensor_circuit(phi_s, phi_f),
               qubits=range(N_ANCILLA, N_ANCILLA + N_QUBITS),
               inplace=True)
    qc.compose(build_sensor_circuit(phi_s, phi_f),
               qubits=range(N_ANCILLA + N_QUBITS,
                            N_ANCILLA + 2 * N_QUBITS),
               inplace=True)
    qc.barrier()

    # parallel SWAP-test
    for i in range(N_ANCILLA):
        qc.h(i)
        qc.cswap(i,
                 N_ANCILLA + i,
                 N_ANCILLA + N_QUBITS + i)
        qc.h(i)
    qc.barrier()
    qc.measure(range(N_ANCILLA), range(N_ANCILLA))
    return qc

service = QiskitRuntimeService()
backend = service.backend("ibm_sherbrooke")
pm      = generate_preset_pass_manager(backend=backend,
                                       optimization_level=3)

noise_model   = NoiseModel.from_backend(backend)
coupling_map  = backend.configuration().coupling_map
basis_gates   = noise_model.basis_gates
simulator     = AerSimulator(
    noise_model=noise_model,
    coupling_map=coupling_map,
    basis_gates=basis_gates,
)

os.makedirs(SIM_DATA_DIR, exist_ok=True)

data = np.load(os.path.join(REAL_DATA_DIR,
                            "ce_swaptest_high_data.npz"))
ps_arr = data["phi_soil"]
pf_arr = data["phi_free"]

p0_list, ce_list = [], []

for idx, (ps, pf) in enumerate(zip(ps_arr, pf_arr), start=1):
    circ = build_swaptest_circuit(ps, pf)
    transpiled = pm.run(circ)

    job = simulator.run(transpiled, shots=SHOTS)
    result = job.result()
    counts = result.get_counts()

    p0 = counts.get("0" * N_ANCILLA, 0) / SHOTS
    ce = 1.0 - p0

    p0_list.append(p0)
    ce_list.append(ce)
    print(f"[{idx}/{len(ps_arr)}] ps={ps:.4f}, pf={pf:.4f} → p0={p0:.4f}, CE={ce:.6f}")

ps_arr = np.array(ps_arr)
pf_arr = np.array(pf_arr)
p0_arr = np.array(p0_list)
ce_arr = np.array(ce_list)

np.savez(
    os.path.join(SIM_DATA_DIR, "ce_swaptest_high_data_sim.npz"),
    phi_soil=ps_arr, phi_free=pf_arr,
    p0=p0_arr, ce=ce_arr
)

with open(os.path.join(SIM_DATA_DIR, "ce_swaptest_high_data_sim.csv"),
          "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["phi_soil", "phi_free", "p0", "ce"])
    for ps, pf, p0, ce in zip(ps_arr, pf_arr, p0_arr, ce_arr):
        writer.writerow([f"{ps:.6f}", f"{pf:.6f}",
                         f"{p0:.6f}", f"{ce:.6f}"])

print("\nSimulation results saved in:", SIM_DATA_DIR)
