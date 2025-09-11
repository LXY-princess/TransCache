"""
Run a Bell circuit on a real IBM Quantum device and print the quasi-probability distribution.
Dependencies:
    pip install -U qiskit qiskit-ibm-runtime
Either set the environment variable QISKIT_IBM_TOKEN, or replace YOUR_API_TOKEN below.
"""

# file: bell_runner.py
from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import Sampler as LocalSampler
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as CloudSampler, Session
import json, pathlib, argparse, pickle
from collections import Counter
from pathlib import Path

SAVE_DIR = pathlib.Path("./savedRes")     # ← 缓存目录改到这里
SAVE_DIR.mkdir(exist_ok=True)

def make_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="Bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def run_cloud(backend_name=None, shots=1024):
    """在 IBM Runtime 上跑：需要 API token"""
    service = QiskitRuntimeService()
    print("You have", len(service.backends()), "backends available.")
    for b in service.backends():
        print(
            f"{b.name:12} qubits={b.num_qubits:<3} "
            f"simulator = {b.configuration().simulator} "
            # f"status={b.status}"
        )
    # backend = service.backend(backend_name) if backend_name else service.least_busy(min_qubits=2) # simulator=False,
    backend = service.least_busy(simulator=False) #
    print("Selected backend:", backend.name)

    bell_raw = make_bell()
    bell_qpu = transpile(
        bell_raw,
        backend=backend,
        optimization_level=3,  # 可调 0–3；3 最激进
        initial_layout=None  # 若想固定物理比特可在此指定
    )

    sampler = CloudSampler(backend)
    job = sampler.run([bell_qpu], shots=shots)
    results = job.result()
    save_primitive(results, tag = backend.name, shots=shots)
    return results

def save_primitive(result, tag, shots):
    path = SAVE_DIR / f"{tag}_{shots}.pkl"
    with path.open("wb") as f:
        pickle.dump(result, f)
    print(f"[CACHE] PrimitiveResult pickled to {path}")

def load_primitive(file:str):
    path = SAVE_DIR / f"{file}.pkl"
    if path.exists():
        with path.open("rb") as f:
            result = pickle.load(f)
        print(f"[CACHE] Loaded PrimitiveResult from {path}")
        return result
    print(f"[CACHE] No PrimitiveResult found for {file}")
    return None

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="save ibm hardware result to savedRes/")
parser.add_argument("--load", type=str, default="", help="specify loaded file under savedRes/")
args = parser.parse_args()

if args.save:
    res = run_cloud(backend_name=args.backend)
if args.load:
    res = load_primitive(args.load)
else:
    res = load_primitive("ibm_torino_124")


bitarr = res[0].data.c  # BitArray(num_shots, num_bits)
raw_bool = bitarr._array  # ← 直接拿底层 np.bool_ 矩阵
bitstrs = ["".join("1" if b else "0" for b in row) for row in raw_bool]
counts = Counter(bitstrs)  # {'00': 503, '11': 521}
probs = {b: c / raw_bool.shape[0] for b, c in counts.items()}

print("Integer counts:", counts)
print("Probabilities :", probs)

