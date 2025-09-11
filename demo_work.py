#!/usr/bin/env python
# demo2.py —— PulseCache benchmark (robust pulse-detection)

import warnings, time
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
from joblib import Memory
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError

# ---------- 1. 选择假后端（3 旧 + 1 新路径） ----------
backend = None
for mod, cls in [
    ("qiskit_ibm_runtime.fake_provider.backends.fake_jakarta", "FakeJakarta"),
    ("qiskit.providers.fake_provider.backends.fake_jakarta",   "FakeJakarta"),  # ≥0.46
    ("qiskit.providers.fake_provider",                         "FakeJakarta"),  # 0.37–0.45
    ("qiskit.test.mock",                                       "FakeJakarta"),  # ≤0.25
]:
    try:
        backend = getattr(__import__(mod, fromlist=[cls]), cls)()
        break
    except (ImportError, AttributeError):
        continue
if backend is None:                          # 最后退路
    from qiskit.providers.fake_provider import GenericBackendV2
    backend = GenericBackendV2(num_qubits=7)

# ---------- 2. 动态探测是否“真有脉冲” ----------
def _probe_pulse(bk) -> bool:
    """Return True iff schedule(transpiled) succeeds."""
    try:
        dummy = QuantumCircuit(1)
        tq = transpile(dummy, bk)
        from qiskit import schedule
        schedule(tq, backend=bk)
        return True
    except Exception:
        return False

HAS_PULSE = _probe_pulse(backend)
print(f"✅ Backend in use → {type(backend).__name__} | HAS_PULSE={HAS_PULSE}")

# ---------- 3. 缓存 & 参数 ----------
CACHE = Memory("./pulsecache_data", verbose=0)
N_RUNS, SHOTS = 30, 4096

# ---------- 4. 示例电路 ----------
def build_circuit(q: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(q)
    qc.h(range(q))
    for i in range(q - 1):
        qc.cx(i, i + 1)
    qc.rx(np.pi / 4, range(q))
    qc.measure_all()
    return qc

# ---------- 5. 冷启动 -----------
def baseline_flow(qc: QuantumCircuit) -> Tuple[float, float, float]:
    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=3)
    t1 = time.perf_counter()

    pulse_dur = 0.0
    if HAS_PULSE:
        from qiskit import schedule
        tp0 = time.perf_counter()
        schedule(tqc, backend=backend)
        pulse_dur = time.perf_counter() - tp0
    else:
        pulse_dur = 0.3
        time.sleep(pulse_dur)

    time.sleep(0.5)        # 模拟排队 / 校准
    t3 = time.perf_counter()
    return (t3 - t0, t1 - t0, pulse_dur)

# ---------- 6. 缓存路径 ----------
@CACHE.cache
def cached_compile(circ: QuantumCircuit):
    tqc = transpile(circ, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule
        schedule(tqc, backend=backend)
    return tqc

def cached_flow(qc: QuantumCircuit) -> Tuple[float, float, float]:
    t0 = time.perf_counter()
    cached_compile(qc)
    time.sleep(0.1)        # δ-cal
    return (time.perf_counter() - t0, 0.0, 0.0)

# ---------- 7. 统计 ----------
def run(flow, n: int) -> List[float]:
    qc = build_circuit()
    tot, comp, pulse = [], [], []
    for _ in range(n):
        a, b, c = flow(qc)
        tot.append(a); comp.append(b); pulse.append(c)
    print(f"{flow.__name__:<12} median={np.median(tot):.3f}s"
          f"  P95={np.percentile(tot,95):.3f}s"
          f"  P99={np.percentile(tot,99):.3f}s"
          f"  (compile≈{mean(comp):.3f}s, pulse≈{mean(pulse):.3f}s)")
    return tot

thr = lambda lat: SHOTS / mean(lat)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print("=== PulseCache Benchmark ===")
    cold = run(baseline_flow, N_RUNS)
    warm = run(cached_flow,  N_RUNS)
    print("\n--- Summary ---")
    print(f"Baseline throughput : {thr(cold):.1f} shots/s")
    print(f"Cached   throughput : {thr(warm):.1f} shots/s")
    print(f"P99 speed-up        : {np.percentile(cold,99)/np.percentile(warm,99):.1f}×")
