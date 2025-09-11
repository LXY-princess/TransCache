#!/usr/bin/env python
# pulse_bench_depths.py —— PulseCache benchmark: multi-circuit depth sweep ▲2025-07-07
#
# 用法:
#   python pulse_bench_depths.py          # 跑完自动弹出折线图并把 PNG 存到 figs/
#   python pulse_bench_depths.py --runs 5 # 每个点重复 5 次（默认 10）
#
# 依赖: qiskit >=0.43, matplotlib, joblib, numpy

import argparse, warnings, time
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
# ------------------------------------------------------------
# 1.  后端探测 —— 复用 demo_work 里的“多版本 FakeJakarta fallback”逻辑
# ------------------------------------------------------------
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

def _probe_pulse(bk) -> bool:
    try:
        dummy = QuantumCircuit(1)
        tq = transpile(dummy, bk)
        from qiskit import schedule; schedule(tq, backend=bk)
        return True
    except Exception:
        return False

HAS_PULSE = _probe_pulse(backend)
print(f"✅ Backend → {type(backend).__name__} | HAS_PULSE={HAS_PULSE}")

# ------------------------------------------------------------
# 2.  缓存 & 全局参数
# ------------------------------------------------------------
CACHE  = Memory("./pulsecache_data", verbose=0)
SHOTS  = 4096                           # 吞吐量统计时使用
DEPTHS = [1, 2, 4, 8, 16, 32]           # 可自行扩充
# ------------------------------------------------------------
# 3.  电路族生成器
# ------------------------------------------------------------
def linear_entangle(q: int, depth: int) -> QuantumCircuit:
    qc = QuantumCircuit(q)
    for _ in range(depth):
        qc.h(range(q))
        for i in range(q-1): qc.cx(i, i+1)
        qc.rx(np.pi/4, range(q))
    qc.measure_all(); return qc

def ghz_chain(q: int, depth: int) -> QuantumCircuit:
    qc = QuantumCircuit(q)
    qc.h(0)
    for i in range(q-1): qc.cx(i, i+1)
    for _ in range(depth-1):
        qc.rz(np.pi/8, q-1); qc.barrier()
    qc.measure_all(); return qc

def qft_like(q: int, depth: int) -> QuantumCircuit:
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

# ------------------------------------------------------------
# 4.  baseline_flow / cached_compile / cached_flow
#     —— 与 demo_work 中实现保持完全一致
# ------------------------------------------------------------
def baseline_flow(qc: QuantumCircuit) -> float:
    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)                 # 假调度
    time.sleep(0.5)                     # 模拟 queue & 校准
    return time.perf_counter() - t0

@CACHE.cache
def cached_compile(qc):
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule; schedule(tqc, backend=backend)
    return tqc

def cached_flow(qc: QuantumCircuit) -> float:
    t0 = time.perf_counter()
    if cached_compile.check_call_in_cache(qc):
        print("Cache hit ✓")
    else:
        print("Cache miss → run transpile+schedule")
    cached_compile(qc)
    time.sleep(0.1)                     # δ-cal
    return time.perf_counter() - t0

# ------------------------------------------------------------
# 5.  评测主逻辑
# ------------------------------------------------------------
def bench_one(gen: Callable[[int,int], QuantumCircuit],
              depth: int, runs: int) -> Tuple[float,float]:
    qnum = getattr(backend, "num_qubits", None)
    if qnum is None:  # FakeJakarta V2 等
        qnum = backend.configuration().num_qubits
    qc = gen(qnum, depth)
    cold  = [baseline_flow(qc) for _ in range(runs)]
    warm  = [cached_flow(qc)   for _ in range(runs)]
    return np.median(cold), np.median(warm)

def run_all(runs: int) -> Dict[str, Dict[str, List[float]]]:
    results = {name: {"depth": [], "baseline": [], "cached": []}
               for name in CIRCUIT_FAMILIES}
    for name, gen in CIRCUIT_FAMILIES.items():
        for d in DEPTHS:
            b, c = bench_one(gen, d, runs)
            results[name]["depth"].append(d)
            results[name]["baseline"].append(b)
            results[name]["cached"].append(c)
            print(f"{name:<10} depth={d:2} median cold={b:.3f}s  warm={c:.3f}s")
    return results

# ------------------------------------------------------------
# 6.  绘图
# ------------------------------------------------------------
def plot(results: Dict[str, Dict[str, List[float]]]):
    plt.figure(figsize=(8,5))
    for name, rec in results.items():
        plt.plot(rec["depth"], rec["baseline"],
                 marker="o", linewidth=2, label=f"{name}-Baseline")
        plt.plot(rec["depth"], rec["cached"],
                 marker="s", linewidth=2, label=f"{name}-Cached")
    plt.xscale("log", base=2); plt.xticks(DEPTHS, DEPTHS)
    plt.xlabel("Circuit depth / #Layers")
    plt.ylabel("Median E2E latency (s)")
    plt.title("Latency scaling across circuit families")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(ncol=2, fontsize="small"); plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig("figs/v2_depth_scaling.png", dpi=300)
    plt.show()

# ------------------------------------------------------------
# 7.  CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10,
                        help="重复测量次数 (default: 10)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("=== PulseCache multi-circuit depth sweep ===")
    res = run_all(args.runs)
    plot(res)

    # 额外打印整体吞吐 & P99 加速比
    for name, rec in res.items():
        thr_cold  = SHOTS / mean(rec["baseline"])
        thr_warm  = SHOTS / mean(rec["cached"])
        p99_speed = (np.percentile(rec["baseline"],99) /
                     np.percentile(rec["cached"], 99))
        print(f"[{name}]  avg-throughput: cold {thr_cold:.1f}  warm {thr_warm:.1f} "
              f"| P99 speed-up ≈ {p99_speed:.1f}×")
