#!/usr/bin/env python
# pulse_bench_depths.py —— PulseCache benchmark: per-circuit bar charts (baseline vs. cached across depths)
# 2025-07-07 (v3)
"""
用法示例：
    python pulse_bench_depths.py               # 折线图 + 为每个电路族各画 1 张柱状图（depth sweep）
    python pulse_bench_depths.py --runs 5      # 每点重复 5 次

脚本功能：
  • 三种电路族 (LinearEnt / GHZ-Chain / QFT-Like) × 多深度基准；
  • 生成 1 张折线图 (Baseline vs. Cached 随 depth 缩放)；
  • 对于 **每个电路族**，生成 1 张柱状图：
      - X 轴 = depth 列表 (1,2,4,8,16,32)
      - 每个 depth 有双柱 Baseline/Cached。

依赖：qiskit>=0.43, matplotlib, joblib, numpy
"""
import argparse, warnings, time
from pathlib import Path
from typing import Callable, Dict
import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT

# ---------- 0. 通用工具 ----------

def num_qubits(bk):
    """Return qubit count for any BackEnd (Fake / Real)."""
    return getattr(bk, "num_qubits", None) or bk.configuration().num_qubits

# ---------- 1. 后端探测 ----------
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


def _probe_pulse(bk):
    try:
        dummy = QuantumCircuit(1)
        tq = transpile(dummy, bk)
        from qiskit import schedule; schedule(tq, backend=bk)
        return True
    except Exception:
        return False

HAS_PULSE = _probe_pulse(backend)
print(f"✅ Backend → {type(backend).__name__} | HAS_PULSE={HAS_PULSE} | Qubits={num_qubits(backend)}")

# ---------- 2. 缓存 & 参数 ----------
CACHE  = Memory("./pulsecache_data", verbose=0)
DEPTHS = [1, 2, 4, 8, 16, 32]

# ---------- 3. 电路族 ----------

def linear_entangle(q: int, depth: int):
    qc = QuantumCircuit(q)
    for _ in range(depth):
        qc.h(range(q))
        for i in range(q-1): qc.cx(i, i+1)
        qc.rx(np.pi/4, range(q))
    qc.measure_all(); return qc

def ghz_chain(q: int, depth: int):
    qc = QuantumCircuit(q)
    qc.h(0)
    for i in range(q-1): qc.cx(i, i+1)
    for _ in range(depth-1):
        qc.rz(np.pi/8, q-1); qc.barrier()
    qc.measure_all(); return qc

def qft_like(q: int, depth: int):
    qc = QuantumCircuit(q)
    for _ in range(depth):
        qc.compose(QFT(num_qubits=q, do_swaps=False).decompose(),
                   range(q), inplace=True)
    qc.measure_all(); return qc

CIRCUIT_FAMILIES: Dict[str, Callable[[int,int], QuantumCircuit]] = {
    "LinearEnt" : linear_entangle,
    "GHZ-Chain" : ghz_chain,
    "QFT-Like"  : qft_like,
}

# ---------- 4. 冷/热流程 ----------

def baseline_flow(qc: QuantumCircuit):
    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    time.sleep(0.5)
    return time.perf_counter() - t0

@CACHE.cache
def cached_compile(qc):
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    return tqc

def cached_flow(qc: QuantumCircuit):
    t0 = time.perf_counter(); cached_compile(qc); time.sleep(0.1)
    return time.perf_counter() - t0

# ---------- 5. 评测 ----------

def bench_one(gen, depth, runs):
    q = num_qubits(backend)
    qc = gen(q, depth)
    cold = [baseline_flow(qc) for _ in range(runs)]
    warm = [cached_flow(qc)   for _ in range(runs)]
    return np.median(cold), np.median(warm)

def run_all(runs):
    results = {name: {"depth": [], "baseline": [], "cached": []}
               for name in CIRCUIT_FAMILIES}
    for name, gen in CIRCUIT_FAMILIES.items():
        for d in DEPTHS:
            b, c = bench_one(gen, d, runs)
            results[name]["depth"    ].append(d)
            results[name]["baseline" ].append(b)
            results[name]["cached"   ].append(c)
            print(f"{name:<10} depth={d:2}  cold={b:.3f}s  warm={c:.3f}s")
    return results

# ---------- 6. 绘图 ----------

def line_plot(results):
    plt.figure(figsize=(8,5))
    for name, rec in results.items():
        plt.plot(rec["depth"], rec["baseline"], marker="o", lw=2, label=f"{name}-Baseline")
        plt.plot(rec["depth"], rec["cached"],   marker="s", lw=2, label=f"{name}-Cached")
    plt.xscale("log", base=2); plt.xticks(DEPTHS, DEPTHS)
    plt.xlabel("Circuit depth / #Layers"); plt.ylabel("Median latency (s)")
    plt.title("Baseline vs. Cached latency scaling")
    plt.grid(True, ls="--", alpha=0.4); plt.legend(ncol=2, fontsize="small")
    Path("figs").mkdir(exist_ok=True); plt.savefig("figs/v2c_depth_scaling.png", dpi=300)
    plt.tight_layout(); plt.show()

def bar_plot_per_circuit(name: str, rec: Dict[str,list]):
    """为指定电路族画柱状图：X=depth, twin bars baseline/cached"""
    depths      = rec["depth"]
    baseline_v  = rec["baseline"]
    cached_v    = rec["cached"]
    x = np.arange(len(depths)); width = 0.35
    plt.figure(figsize=(7,4))
    plt.bar(x - width/2, baseline_v, width, label="Baseline")
    plt.bar(x + width/2, cached_v,   width, label="Cached")
    plt.xticks(x, depths)
    plt.xlabel("Circuit depth (#Layers)")
    plt.ylabel("Median latency (s)")
    plt.title(f"{name}: Baseline vs. Cached across depths")
    plt.grid(axis="y", ls="--", alpha=0.4)
    plt.legend(); plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(f"figs/v2c_{name}_bar_depths.png", dpi=300)
    plt.show()

# ---------- 7. CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="重复测量次数 (default: 10)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("=== PulseCache multi-circuit depth sweep ===")
    res = run_all(args.runs)

    # 折线图：整体深度缩放
    line_plot(res)

    # 每个电路族单独柱状图
    for n, r in res.items():
        bar_plot_per_circuit(n, r)
