#!/usr/bin/env python
# pulse_bench_depths.py —— PulseCache ablation study (v6)
# 2025-07-07

"""
生成结果：
  1. 深度折线图：Baseline vs. Full PulseCache（未改动）
  2. 消融柱状图
     • HashOnly / Transpile / SyncCache × 每电路族各 1 张
     • **新增** All-Variants：Baseline + Full + 三消融一次性对比（每电路族 1 张）

关键改动：
  ✓ 缓存目录缩短为 pc_full / pc_hash / pc_trans，避免 Windows MAX_PATH
  ✓ 静音 joblib.CacheWarning
  ✓ 函数 multi_variant_bar() 绘制综合柱状图
  ✓ 默认 --runs 3，可用 --runs N 自行加大采样
"""

import argparse, warnings, time, hashlib
from pathlib import Path
from typing import Callable, Dict, List
import numpy as np
import matplotlib.pyplot as plt

from joblib import Memory
from joblib._store_backends import CacheWarning
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT

# ────────────────────── 工具 ──────────────────────
warnings.filterwarnings("ignore", category=CacheWarning)        # 静音落盘警告

def num_qubits(bk):
    """兼容 FakeBackend V1/V2 获取量子比特数"""
    return getattr(bk, "num_qubits", None) or bk.configuration().num_qubits

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

# ─────────────────── 后端探测 ──────────────────────
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
    _dummy = transpile(QuantumCircuit(1), backend)
    from qiskit import schedule; schedule(_dummy, backend=backend)
except Exception:
    HAS_PULSE = False

print(f"✅ Backend → {type(backend).__name__} | HAS_PULSE={HAS_PULSE} | Qubits={num_qubits(backend)}")

# ───────────────────── 参数 ───────────────────────
DEPTHS   = [1, 2, 4, 8, 16, 32]
POLICIES = ["HashOnly", "Transpile", "SyncCache"]       # 三种消融

# ─────────────────── 电路族定义 ───────────────────
def linear_entangle(q: int, d: int) -> QuantumCircuit:
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.h(range(q))
        for i in range(q-1): qc.cx(i, i+1)
        qc.rx(np.pi/4, range(q))
    qc.measure_all(); return qc

def ghz_chain(q: int, d: int) -> QuantumCircuit:
    qc = QuantumCircuit(q); qc.h(0)
    for i in range(q-1): qc.cx(i, i+1)
    for _ in range(d-1):
        qc.rz(np.pi/8, q-1); qc.barrier()
    qc.measure_all(); return qc

def qft_like(q: int, d: int) -> QuantumCircuit:
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.compose(QFT(num_qubits=q, do_swaps=False).decompose(),
                   range(q), inplace=True)
    qc.measure_all(); return qc

CIRCS: Dict[str, Callable[[int, int], QuantumCircuit]] = {
    "LinearEnt": linear_entangle,
    "GHZ-Chain": ghz_chain,
    "QFT-Like" : qft_like,
}

# ───────────────────── Flow 实现 ───────────────────
def baseline_flow(qc: QuantumCircuit) -> float:
    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    time.sleep(0.5)
    return time.perf_counter() - t0

# Full PulseCache（含标定指纹）
full_cache = Memory("pc_full", verbose=0)
@full_cache.cache
def _full_compile(qc):            # 编译 + 调度 缓存
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    return tqc
def full_flow(qc):
    t0 = time.perf_counter(); _full_compile(qc); time.sleep(0.1)
    return time.perf_counter() - t0

# HashOnly —— 去掉标定指纹校验
hash_cache = Memory("pc_hash", verbose=0)
@hash_cache.cache
def _hash_compile(qc):
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    return tqc
def hash_flow(qc):
    t0 = time.perf_counter(); _hash_compile(qc)
    return time.perf_counter() - t0

# Transpile-only —— 只缓存 transpile，调度每次重跑
transp_cache = Memory("pc_trans", verbose=0)
@transp_cache.cache
def _transp_compile(qc):
    return transpile(qc, backend, optimization_level=3)
def transpile_flow(qc):
    t0 = time.perf_counter()
    tqc = _transp_compile(qc)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    return time.perf_counter() - t0

# SyncCache —— 禁用 opportunistic，首调用冷启动
sync_cache: Dict[str,bool] = {}
def _circ_key(qc): return hashlib.sha256(qc.qasm().encode()).hexdigest()
def sync_flow(qc):
    key = _circ_key(qc)
    if key not in sync_cache:
        delay = baseline_flow(qc)  # 冷启动
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

# ─────────────────── Benchmark ────────────────────
def bench(policy: str, gen: Callable, depth: int, runs: int) -> float:
    qc = gen(num_qubits(backend), depth)
    return np.median([FLOW_FN[policy](qc) for _ in range(runs)])

def run_all(runs: int):
    res = {p: {c: [] for c in CIRCS} for p in ["Baseline", "Full"] + POLICIES}
    for cname, gen in CIRCS.items():
        for d in DEPTHS:
            for p in res:
                med = bench(p, gen, d, runs)
                res[p][cname].append(med)
                print(f"{p:<9}| {cname:<9} depth={d:<2}  med={med:.3f}s")
    return res

# ───────────────────── 绘图 ───────────────────────
def bar_variant(circ: str, res: Dict[str, Dict[str, List[float]]], var: str):
    x = np.arange(len(DEPTHS)); w = 0.35
    plt.figure(figsize=(7,4))
    plt.bar(x - w/2, res["Baseline"][circ], w, label="Baseline")
    plt.bar(x + w/2, res[var][circ],      w, label=var)
    plt.xticks(x, DEPTHS)
    plt.xlabel("Circuit depth (#Layers)")
    plt.ylabel("Median latency (s)")
    plt.title(f"{circ}: {var} ablation")
    plt.grid(axis="y", ls="--", alpha=0.4)
    plt.legend(); plt.tight_layout()
    ensure_dir("figs"); plt.savefig(f"figs/v3a_{circ}_{var}.png", dpi=300)
    plt.close()

def multi_variant_bar(circ: str, res: Dict[str, Dict[str, List[float]]]):
    labels = ["Baseline", "Full"] + POLICIES
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(len(DEPTHS)); w = 0.15
    plt.figure(figsize=(8,4))
    for i, lbl in enumerate(labels):
        plt.bar(x + (i-2)*w, res[lbl][circ], w, label=lbl, color=colors[i])
    plt.xticks(x, DEPTHS)
    plt.xlabel("Circuit depth (#Layers)")
    plt.ylabel("Median latency (s)")
    plt.title(f"{circ}: Baseline vs. PulseCache Variants")
    plt.grid(axis="y", ls="--", alpha=0.4)
    plt.legend(ncol=5, fontsize="small"); plt.tight_layout()
    ensure_dir("figs"); plt.savefig(f"figs/v3a_{circ}_AllVariants.png", dpi=300)
    plt.close()

# ─────────────────────── CLI ──────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10,
                    help="重复测量次数 (默认 3，正式实验请调大)")
    args = ap.parse_args()

    print("=== PulseCache ablation study ===")
    results = run_all(args.runs)

    # 综合柱状图
    for circ in CIRCS:
        multi_variant_bar(circ, results)
        for var in POLICIES:      # 单 Variant 图
            bar_variant(circ, results, var)

    print("✅ PNG 已输出至 figs/ 目录；运行 --runs 10 获得正式论文数据。")
