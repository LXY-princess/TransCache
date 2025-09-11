#!/usr/bin/env python
"""
PulseCache-on-Qiskit v5 (IBM-only)
==================================
Ablation benchmark of PulseCache strategies:
  - Baseline, Full, HashOnly, Transpile, SyncCache
  - Only runs on IBMQ real hardware (e.g. ibmq_lima)

To run:
    python demo_multiv5_ibmqonly.py --runs 3
"""

import argparse, warnings, time, hashlib
from pathlib import Path
from typing import Callable, Dict, List
import numpy as np
import matplotlib.pyplot as plt

from qiskit import IBMQ, QuantumCircuit, transpile
from qiskit.pulse import Schedule
from qiskit.compiler import schedule as q_schedule
from qiskit.circuit.library import QFT
from joblib import Memory
from joblib._store_backends import CacheWarning

# ────────────────────── Backend ──────────────────────
warnings.filterwarnings("ignore", category=CacheWarning)

def get_backend():
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub="ibm-q")
    return provider.get_backend("ibmq_lima")  # 可修改为其它可用后端

backend = get_backend()

def num_qubits(bk):
    return getattr(bk, "num_qubits", None) or bk.configuration().num_qubits

print(f"✅ Backend → {backend.name()} | Qubits={num_qubits(backend)}")

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

# ─────────────────── 电路族 ───────────────────────
def make_linear_ent(q: int, d: int):
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.h(range(q))
        for i in range(q - 1): qc.cx(i, i + 1)
        qc.rx(np.pi / 4, range(q))
    qc.measure_all(); return qc

def make_ghz_chain(q: int, d: int):
    qc = QuantumCircuit(q); qc.h(0)
    for i in range(q - 1): qc.cx(i, i + 1)
    for _ in range(d - 1): qc.rz(np.pi / 8, q - 1); qc.barrier()
    qc.measure_all(); return qc

def make_qft_like(q: int, d: int):
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.compose(QFT(q, do_swaps=False).decompose(), inplace=True)
    qc.measure_all(); return qc

CIRCS: Dict[str, Callable[[int, int], QuantumCircuit]] = {
    "LinearEnt": make_linear_ent,
    "GHZ-Chain": make_ghz_chain,
    "QFT-Like": make_qft_like,
}

# ────────────────────── 参数 ──────────────────────
DEPTHS   = [1, 2, 4, 8, 16, 32]
POLICIES = ["HashOnly", "Transpile", "SyncCache"]
CONC_POL = ["Baseline", "Full"] + POLICIES
COLORS   = {"Baseline": "tab:gray", "Full": "tab:blue",
            "HashOnly": "tab:orange", "Transpile": "tab:green", "SyncCache": "tab:red"}

# ────────────────────── 缓存 ──────────────────────
full_cache   = Memory("ibmq_full", verbose=0)
hash_cache   = Memory("ibmq_hash", verbose=0)
transp_cache = Memory("ibmq_trans", verbose=0)
sync_seen: set[str] = set()

def qasm_key(qc: QuantumCircuit) -> str:
    qasm = qc.qasm()
    return f"{hashlib.md5(qasm.encode()).hexdigest()}:{len(qasm)}"

@full_cache.cache
def _full_compile(qasm: str):
    qc = QuantumCircuit.from_qasm_str(qasm)
    tqc = transpile(qc, backend, optimization_level=3)
    q_schedule(tqc, backend=backend)
    time.sleep(0.1)
    return tqc

@hash_cache.cache
def _hash_compile(qasm: str):
    return _full_compile(qasm)

@transp_cache.cache
def _tr_compile(qasm: str):
    qc = QuantumCircuit.from_qasm_str(qasm)
    return transpile(qc, backend, optimization_level=3)

# ────────────────────── Flows ──────────────────────
def flow_baseline(qc):
    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=3)
    q_schedule(tqc, backend=backend)
    time.sleep(0.5)
    return time.perf_counter() - t0

def flow_full(qc):
    t0 = time.perf_counter(); _full_compile(qc.qasm()); return time.perf_counter() - t0

def flow_hash(qc):
    t0 = time.perf_counter(); _hash_compile(qc.qasm()); return time.perf_counter() - t0

def flow_transpile(qc):
    t0 = time.perf_counter()
    tqc = _tr_compile(qc.qasm())
    q_schedule(tqc, backend=backend)
    return time.perf_counter() - t0

def flow_sync(qc):
    k = qasm_key(qc)
    if k not in sync_seen:
        sync_seen.add(k)
        return flow_baseline(qc)
    return flow_full(qc)

FLOW_FN = {
    "Baseline": flow_baseline, "Full": flow_full,
    "HashOnly": flow_hash,     "Transpile": flow_transpile,
    "SyncCache": flow_sync,
}

# ────────────────────── Benchmark ──────────────────────
def bench(policy: str, gen: Callable, depth: int, runs: int):
    qc = gen(num_qubits(backend), depth)
    return np.median([FLOW_FN[policy](qc) for _ in range(runs)])

def run_all(runs: int):
    res = {p: {c: [] for c in CIRCS} for p in CONC_POL}
    for cname, gen in CIRCS.items():
        for d in DEPTHS:
            for p in res:
                med = bench(p, gen, d, runs)
                res[p][cname].append(med)
                print(f"{p:<9}| {cname:<9} depth={d:<2}  med={med:.3f}s")
    return res

# ────────────────────── 绘图 ──────────────────────
def bar_variant(circ: str, res: Dict[str, Dict[str, List[float]]], var: str):
    x = np.arange(len(DEPTHS)); w = 0.35
    plt.figure(figsize=(7,4))
    plt.bar(x - w/2, res["Baseline"][circ], w, label="Baseline")
    plt.bar(x + w/2, res[var][circ],      w, label=var)
    plt.xticks(x, DEPTHS); plt.grid(axis="y", ls="--", alpha=0.4)
    plt.xlabel("Depth"); plt.ylabel("Latency (s)")
    plt.title(f"{circ}: {var}"); plt.legend(); plt.tight_layout()
    ensure_dir("figs"); plt.savefig(f"figs/v6_ibmq_{circ}_{var}.png", dpi=300)
    plt.close()

def multi_variant_bar(circ: str, res: Dict[str, Dict[str, List[float]]]):
    x = np.arange(len(DEPTHS)); w = 0.15
    plt.figure(figsize=(8,4))
    for i, lbl in enumerate(CONC_POL):
        plt.bar(x + (i-2)*w, res[lbl][circ], w, label=lbl, color=COLORS[lbl])
    plt.xticks(x, DEPTHS); plt.grid(axis="y", ls="--", alpha=0.4)
    plt.xlabel("Depth"); plt.ylabel("Latency (s)")
    plt.title(f"{circ}: PulseCache Variants")
    plt.legend(ncol=5); plt.tight_layout()
    ensure_dir("figs"); plt.savefig(f"figs/v6_ibmq_{circ}_AllVariants.png", dpi=300)
    plt.close()

# ────────────────────── CLI 入口 ──────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    print("=== PulseCache v5 IBMQ Only ===")
    results = run_all(args.runs)

    for circ in CIRCS:
        multi_variant_bar(circ, results)
        for var in POLICIES:
            bar_variant(circ, results, var)

    print("✅ PNG 已输出至 figs/")
