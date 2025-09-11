#!/usr/bin/env python
# pulse_bench_breakdown.py —— PulseCache time-breakdown (v7)
# 2025-07-30  Author: ChatGPT

import argparse, warnings, time, hashlib
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory
from joblib._store_backends import CacheWarning
from qiskit import QuantumCircuit, transpile, schedule
from qiskit.circuit.library import QFT

warnings.filterwarnings("ignore", category=CacheWarning)

# ───────────── backend autodetect (同 v5) ─────────────
def num_qubits(bk):
    return getattr(bk, "num_qubits", None) or bk.configuration().num_qubits

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
if backend is None:                       # fallback
    from qiskit.providers.fake_provider import GenericBackendV2
    backend = GenericBackendV2(num_qubits=7)

HAS_PULSE = True
try:
    schedule(transpile(QuantumCircuit(1), backend), backend=backend)
except Exception:
    HAS_PULSE = False

print(f"✅ Backend → {type(backend).__name__} | HAS_PULSE={HAS_PULSE} | Qubits={num_qubits(backend)}")

# ───────────── parameters ─────────────
DEPTHS      = [1, 2, 4, 8, 16, 32]
# DEPTHS      = [1, 2, 4, 8, 32]
POLICIES    = ["Baseline", "Full", "HashOnly", "Transpile", "SyncCache"]
COMPONENTS  = ["lookup", "transpile", "schedule", "calibration"]
COL_STAGE   = dict(lookup="tab:gray", transpile="tab:orange",
                   schedule="tab:blue",  calibration="tab:green")

# ───────────── circuit generators ─────────────
def make_linear_ent(q: int, d: int):
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.h(range(q))
        for i in range(q - 1):
            qc.cx(i, i + 1)
        qc.rx(np.pi / 4, range(q))
    qc.measure_all()
    return qc

def make_ghz_chain(q: int, d: int):
    qc = QuantumCircuit(q)
    qc.h(0)
    for i in range(q - 1):
        qc.cx(i, i + 1)
    for _ in range(d - 1):
        qc.rz(np.pi / 8, q - 1)
        qc.barrier()
    qc.measure_all()
    return qc

def make_qft_like(q: int, d: int):
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.compose(QFT(num_qubits=q, do_swaps=False).decompose(), range(q), inplace=True)
    qc.measure_all()
    return qc

CIRCS = {
    "LinearEnt": make_linear_ent,
    "GHZ-Chain": make_ghz_chain,
    "QFT-Like": make_qft_like,
}


# ───────────── helpers ─────────────
def ensure_dir(p: str | Path): Path(p).mkdir(parents=True, exist_ok=True)

@contextmanager
def timer(buf: Dict[str,float], key: str):
    t0 = time.perf_counter(); yield
    buf[key] += time.perf_counter() - t0

def qasm_key(qc: QuantumCircuit) -> str:
    """Canonical QASM string作为缓存键(含长度防碰撞)。"""
    qasm = qc.qasm()
    return f"{hashlib.md5(qasm.encode()).hexdigest()}:{len(qasm)}"

# ───────────── caches ─────────────
full_cache   = Memory("pc_full",   verbose=0)
hash_cache   = Memory("pc_hash",   verbose=0)
tr_cache     = Memory("pc_trans",  verbose=0)

def _full_compile_helper(qasm_str: str):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    return tqc

@full_cache.cache
def _full_compile(qasm_str: str):
    return _full_compile_helper(qasm_str)

@hash_cache.cache
def _hash_compile(qasm_str: str):
    return _full_compile_helper(qasm_str)

@tr_cache.cache
def _tr_compile(qasm_str: str):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    return transpile(qc, backend, optimization_level=3)

# ───────────── flow functions (time-dict) ─────────────
sync_seen: Dict[str,bool] = {}

def flow_baseline(qc):
    s = dict.fromkeys(COMPONENTS, 0.0)
    with timer(s,"transpile"):
        tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        with timer(s,"schedule"): schedule(tqc, backend=backend)
    else:
        with timer(s,"schedule"): time.sleep(0.3)
    with timer(s,"calibration"):  time.sleep(0.05)
    return s

def flow_full(qc):
    s = dict.fromkeys(COMPONENTS, 0.0)
    with timer(s,"lookup"): _full_compile(qc.qasm())
    with timer(s,"calibration"):  time.sleep(0.05)
    return s

def flow_hash(qc):
    s = dict.fromkeys(COMPONENTS, 0.0)
    with timer(s,"lookup"): _hash_compile(qc.qasm())
    return s

def flow_transpile(qc):
    s = dict.fromkeys(COMPONENTS, 0.0)
    with timer(s,"lookup"): tqc = _tr_compile(qc.qasm())
    if HAS_PULSE:
        with timer(s,"schedule"): schedule(tqc, backend=backend)
    else:
        with timer(s,"schedule"): time.sleep(0.3)
    return s

def flow_sync(qc):
    k = qasm_key(qc)
    if k not in sync_seen:          # 第一次出现 → 走 Baseline
        sync_seen[k] = True
        return flow_baseline(qc)
    else:                           # 之后命中 → 直接用缓存
        return flow_full(qc)

FLOW = {"Baseline": flow_baseline, "Full": flow_full,
        "HashOnly": flow_hash,    "Transpile": flow_transpile,
        "SyncCache": flow_sync}

# ───────────── benchmark helpers ─────────────
def bench_breakdown(pol:str, gen:Callable, depth:int, runs:int):
    qc = gen(num_qubits(backend), depth)
    trials = [FLOW[pol](qc) for _ in range(runs)]
    return {c: np.median([t[c] for t in trials]) for c in COMPONENTS}

def run_all(runs:int):
    res: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        p:{c:{comp:[] for comp in COMPONENTS} for c in CIRCS} for p in POLICIES}
    for cname, g in CIRCS.items():
        for d in DEPTHS:
            for p in POLICIES:
                med = bench_breakdown(p, g, d, runs)
                for comp in COMPONENTS: res[p][cname][comp].append(med[comp])
                print(f"{p:<9}| {cname:<9} depth={d:<2}  -> "
                      + ", ".join(f"{k}:{v:.3f}" for k,v in med.items()))
    return res

# ───────────── plotting ─────────────
def plot_stacked_breakdown(circ:str, pol:str, data):
    y_stack = np.array([data[pol][circ][c] for c in COMPONENTS])  # (4, len(depths))
    totals  = y_stack.sum(axis=0); pct = y_stack / totals * 100
    x = np.arange(len(DEPTHS))
    plt.figure(figsize=(8,4))
    bottom = np.zeros_like(DEPTHS, dtype=float)
    for i,c in enumerate(COMPONENTS):
        plt.bar(x, pct[i], bottom=bottom, color=COL_STAGE[c], label=c)
        bottom += pct[i]
    plt.xticks(x, DEPTHS)
    plt.ylim(0,100); plt.ylabel("Percentage of total latency (%)")
    plt.xlabel("Circuit depth (#Layers)")
    plt.title(f"{circ} – {pol} time-breakdown")
    plt.legend(loc="upper center", ncol=len(COMPONENTS))
    plt.tight_layout(); ensure_dir("figs")
    plt.savefig(f"figs/v5abd_{pol}_{circ}.png", dpi=300); plt.close()


# ───────────── new plotting (horizontal stacked bars) ─────────────
def plot_depth_breakdown(circ:str, depth_idx:int, depth_val:int, data):
    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos   = np.arange(len(POLICIES))
    for i, pol in enumerate(POLICIES):
        bottom = 0.0
        for comp in COMPONENTS:
            width = data[pol][circ][comp][depth_idx]
            ax.barh(i, width, left=bottom, color=COL_STAGE[comp], height=0.6,
                    label=comp if (i==0) else None)
            bottom += width
    ax.set_yticks(y_pos, POLICIES)
    ax.set_xlabel("Latency (s)")
    ax.set_title(f"{circ} – depth {depth_val}")
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout(); ensure_dir("figs")
    plt.savefig(f"figs/v5abd_d{depth_val}_{circ}.png", dpi=300)
    plt.close()

import pickle
def save_results(obj, path: Path):
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    print(f"📦 results saved → {path}")

def load_results(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f)
    print(f"📂 results loaded ← {path}")
    return data

# ───────────── main ─────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10,
                    help="重复测量次数 (default=10)")
    ap.add_argument("--save", action="store_true",
                    help="运行基准并把结果存到 savedRes/ 下")
    ap.add_argument("--load", type=str, default="",
                    help="直接从指定 .pkl 文件载入结果并绘图")
    args = ap.parse_args()

    print("=== PulseCache time-breakdown benchmark ===")
    # results = run_all(args.runs)

    save_path = Path(f"savedRes/results_demo5abd2.pkl")

    if args.load:
        results = load_results(Path(args.load))
    else:
        results = run_all(args.runs)
        if args.save:
            save_results(results, save_path)

    # 绘图：每个方案 × 每个电路
    for pol in POLICIES:
        for circ in CIRCS:
            plot_stacked_breakdown(circ, pol, results)

    for i, d in enumerate(DEPTHS):
        for circ in CIRCS:
            plot_depth_breakdown(circ, i, d, results)

    print("✅ 完成！堆叠柱状图保存在 figs/ 目录。")
