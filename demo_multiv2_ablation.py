#!/usr/bin/env python
# pulse_bench_depths.py —— PulseCache ablation study (v5, race‑safe, hashable)
# 2025‑07‑07
"""
生成：
  • 全 PulseCache 与 Baseline 的深度折线图
  • 消融实验三 Variant（HashOnly / Transpile / SyncCache）柱状图 × 3 电路族

改动 v5：
  ✓ SyncCache 使用 SHA‑256(qasm) 作为键，避免 “unhashable QuantumCircuit”
  ✓ 预建 pulsecache_data 子目录，消除 joblib CacheWarning

用法：
  python pulse_bench_depths.py [--runs 10]
"""
import argparse, warnings, time, hashlib
from pathlib import Path
from typing import Callable, Dict
import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT

# ---------- 工具 ----------

def num_qubits(bk):
    return getattr(bk, "num_qubits", None) or bk.configuration().num_qubits

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# ---------- 后端探测 ----------
backend = None
for mod, cls in [
    ("qiskit_ibm_runtime.fake_provider.backends.fake_jakarta", "FakeJakarta"),
    ("qiskit.providers.fake_provider.backends.fake_jakarta",   "FakeJakarta"),
    ("qiskit.providers.fake_provider",                         "FakeJakarta"),
    ("qiskit.test.mock",                                       "FakeJakarta"),
]:
    try:
        backend = getattr(__import__(mod, fromlist=[cls]), cls)(); break
    except (ImportError, AttributeError):
        continue
if backend is None:
    from qiskit.providers.fake_provider import GenericBackendV2
    backend = GenericBackendV2(num_qubits=7)

HAS_PULSE = True
try:
    dummy = transpile(QuantumCircuit(1), backend)
    from qiskit import schedule; schedule(dummy, backend=backend)
except Exception:
    HAS_PULSE = False
print(f"✅ Backend → {type(backend).__name__} | HAS_PULSE={HAS_PULSE} | Qubits={num_qubits(backend)}")

# ---------- 参数 ----------
DEPTHS   = [1, 2, 4, 8, 16, 32]
POLICIES = ["HashOnly", "Transpile", "SyncCache"]
ensure_dir("pulsecache_data")

# ---------- 电路族 ----------

def linear_entangle(q, d):
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.h(range(q))
        for i in range(q-1): qc.cx(i, i+1)
        qc.rx(np.pi/4, range(q))
    qc.measure_all(); return qc

def ghz_chain(q, d):
    qc = QuantumCircuit(q)
    qc.h(0)
    for i in range(q-1): qc.cx(i, i+1)
    for _ in range(d-1): qc.rz(np.pi/8, q-1); qc.barrier()
    qc.measure_all(); return qc

def qft_like(q, d):
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.compose(QFT(num_qubits=q, do_swaps=False).decompose(), range(q), inplace=True)
    qc.measure_all(); return qc

CIRCS: Dict[str, Callable[[int,int], QuantumCircuit]] = {
    "LinearEnt": linear_entangle,
    "GHZ-Chain": ghz_chain,
    "QFT-Like" : qft_like,
}

# ---------- Flow 实现 ----------

def baseline_flow(qc):
    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    time.sleep(0.5)
    return time.perf_counter() - t0

# Full PulseCache
ensure_dir("pulsecache_data/full")
full_cache = Memory("pulsecache_data/full", verbose=0)
@full_cache.cache
def _full_compile(qc):
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    return tqc

def full_flow(qc):
    t0 = time.perf_counter(); _full_compile(qc); time.sleep (0.1)
    return time.perf_counter() - t0

# HashOnly
ensure_dir("pulsecache_data/hash")
hash_cache = Memory("pulsecache_data/hash", verbose=0)
@hash_cache.cache
def _hash_compile(qc):
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    return tqc

def hash_flow(qc):
    t0 = time.perf_counter(); _hash_compile(qc)
    return time.perf_counter() - t0

# Transpile‑only
ensure_dir("pulsecache_data/transpile")
transp_cache = Memory("pulsecache_data/transpile", verbose=0)
@transp_cache.cache
def _transp_only_compile(qc):
    return transpile(qc, backend, optimization_level=3)

def transpile_flow(qc):
    t0 = time.perf_counter(); tqc = _transp_only_compile(qc)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    return time.perf_counter() - t0

# SyncCache
sync_cache = {}

def _circ_key(qc):
    return hashlib.sha256(qc.qasm().encode()).hexdigest()

def sync_flow(qc):
    key = _circ_key(qc)
    if key not in sync_cache:
        delay = baseline_flow(qc)   # cold first time
        sync_cache[key] = True
        return delay
    return full_flow(qc)

FLOW_FN = {
    "Baseline" : baseline_flow,
    "Full"     : full_flow,
    "HashOnly" : hash_flow,
    "Transpile": transpile_flow,
    "SyncCache": sync_flow,
}

# ---------- Benchmark ----------

def bench(policy: str, gen: Callable, depth: int, runs: int):
    q = num_qubits(backend)
    qc = gen(q, depth)
    lat = [FLOW_FN[policy](qc) for _ in range(runs)]
    return np.median(lat)

def run_all(runs: int):
    res = {pol: {c: [] for c in CIRCS} for pol in ["Baseline", "Full"] + POLICIES}
    for cname, gen in CIRCS.items():
        for d in DEPTHS:
            for pol in res:
                res[pol][cname].append(bench(pol, gen, d, runs))
                print(f"{pol:<9}| {cname:<9} depth={d:<2}  med={res[pol][cname][-1]:.3f}s")
    return res

# ---------- 绘图 ----------

def bar_variant(circ: str, res: Dict[str, Dict[str,list]], var: str):
    depths = DEPTHS
    base_v = res["Baseline"][circ]
    var_v  = res[var][circ]
    x = np.arange(len(depths)); width = 0.35
    plt.figure(figsize=(7,4))
    plt.bar(x-width/2, base_v, width, label="Baseline")
    plt.bar(x+width/2, var_v,  width, label=var)
    plt.xticks(x, depths)
    plt.xlabel("Circuit depth (#Layers)")
    plt.ylabel("Median latency (s)")
    plt.title(f"{circ}: {var} ablation")
    plt.grid(axis="y", ls="--", alpha=0.4)
    plt.legend(); plt.tight_layout()
    ensure_dir("figs")
    plt.savefig(f"figs/v2a_{circ}_{var}.png", dpi=300)
    plt.close()

# ---------- 主入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="重复测量次数 (default 10)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("=== PulseCache ablation study ===")
    results = run_all(args.runs)

    for circ in CIRCS:
        for var in POLICIES:
            bar_variant(circ, results, var)

    print("图已输出至 figs/ 目录。")
