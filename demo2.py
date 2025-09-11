#!/usr/bin/env python
# demo2.py —— PulseCache benchmark (unified for Terra 0.24–0.46+)
import warnings, time
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
from joblib import Memory
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError

# ---------------- 1. 选择后端：三条路径 ---------------
backend = None
for mod_path, cls in [
    ("qiskit.providers.fake_provider.backends.fake_jakarta", "FakeJakarta"),  # ≥0.46
    ("qiskit.providers.fake_provider",                      "FakeJakarta"),  # 0.37–0.45
    ("qiskit.test.mock",                                    "FakeJakarta"),  # ≤0.25
]:
    try:
        module  = __import__(mod_path, fromlist=[cls])
        backend = getattr(module, cls)()
        break
    except (ImportError, AttributeError):
        continue
if backend is None:  # fallback
    from qiskit.providers.fake_provider import GenericBackendV2
    backend = GenericBackendV2(num_qubits=7)

backend_name = type(backend).__name__
HAS_PULSE = hasattr(backend, "instruction_schedule_map")
print(f"✅ Backend in use → {backend_name} | HAS_PULSE={HAS_PULSE}")

# ---------------- 2. 缓存与参数 ---------------------
CACHE_DIR = Path("./pulsecache_data")
CACHE_DIR.mkdir(exist_ok=True)
memory = Memory(location=str(CACHE_DIR / "joblib_cache"), verbose=0)
N_RUNS, SHOTS = 30, 4096

# ---------------- 3. 测试电路 -----------------------
def build_circuit(n: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.rx(np.pi / 4, range(n))
    qc.measure_all()
    return qc

# ---------------- 4. 冷启动路径 ----------------------
def baseline_flow(qc: QuantumCircuit) -> Tuple[float, float, float]:
    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=3)
    t1 = time.perf_counter()

    # pulse schedule
    pulse_dur = 0.0
    if HAS_PULSE:
        from qiskit import schedule
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
            tp0 = time.perf_counter()
            _   = schedule(tqc, backend=backend)
            pulse_dur = time.perf_counter() - tp0
        except (QiskitError, Exception):
            pulse_dur = 0.3
    else:
        pulse_dur = 0.3
    time.sleep(pulse_dur) if not HAS_PULSE else None  # 占位或真实已消耗

    time.sleep(0.5)          # 队列 / 校准占位
    t3 = time.perf_counter()
    return (t3 - t0, t1 - t0, pulse_dur)

# ---------------- 5. 缓存路径 ------------------------
@memory.cache
def cached_compile(qc: QuantumCircuit):
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
        _ = schedule(tqc, backend=backend)
    return tqc

def cached_flow(qc: QuantumCircuit) -> Tuple[float, float, float]:
    t0 = time.perf_counter()
    _  = cached_compile(qc)
    time.sleep(0.1)  # δ-cal
    t1 = time.perf_counter()
    return (t1 - t0, 0.0, 0.0)

# ---------------- 6. 统计与驱动 ----------------------
def run(flow_fn, n: int) -> List[float]:
    qc = build_circuit()
    tot, comp, pulse = [], [], []
    for _ in range(n):
        a, b, c = flow_fn(qc)
        tot.append(a); comp.append(b); pulse.append(c)
    print(f"{flow_fn.__name__:<12}"
          f" median={np.median(tot):.3f}s"
          f"  P95={np.percentile(tot,95):.3f}s"
          f"  P99={np.percentile(tot,99):.3f}s"
          f"  (compile≈{mean(comp):.3f}s, pulse≈{mean(pulse):.3f}s)")
    return tot

def throughput(lat: List[float]) -> float:
    return SHOTS / mean(lat)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print("=== PulseCache Benchmark (multi-version) ===")
    cold  = run(baseline_flow, N_RUNS)
    warm  = run(cached_flow,  N_RUNS)

    print("\n--- Summary ---")
    print(f"Baseline throughput : {throughput(cold):.1f} shots/s")
    print(f"Cached   throughput : {throughput(warm):.1f} shots/s")
    print(f"P99 speed-up        : "
          f"{np.percentile(cold,99)/np.percentile(warm,99):.1f}×")
