#!/usr/bin/env python
"""
Demo ‑ multiv5 *QASM‑key* edition
================================
在保持 **demo_multiv4.py** 其余逻辑、CLI、Circuit 家族、深度扫描
和 Poisson 并发流程完全不动的前提下，将 Joblib 缓存的“键”从
`QuantumCircuit` 对象改为 **canonical QASM 字符串**。

改动范围只限三处：
1. 统一的 `qasm_key(qc)` 帮助函数。
2. 每个 `@Memory.cache` 装饰的编译函数参数改为 `qasm_str`，并在
   内部 `QuantumCircuit.from_qasm_str()` 反序列化后继续原逻辑。
3. 各 *flow_* 包装器把 `qc.qasm()` 传给缓存函数，其余 sleep / 调用
   顺序完全保持原样。
"""
import argparse, warnings, time, hashlib, platform, os, concurrent.futures as cf, multiprocessing as mp
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory, _store_backends
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT

warnings.filterwarnings("ignore", category=_store_backends.CacheWarning)

num_qubits = lambda bk: getattr(bk, "num_qubits", None) or bk.configuration().num_qubits
ensure_dir  = lambda p: Path(p).mkdir(parents=True, exist_ok=True)

# ───────────────── backend probe ─────────────────
backend = None
for mod, cls in [("qiskit.test.mock", "FakeJakarta")]:
    try:
        backend = getattr(__import__(mod, fromlist=[cls]), cls)(); break
    except Exception:
        pass
from qiskit.providers.fake_provider import GenericBackendV2
if backend is None:
    backend = GenericBackendV2(num_qubits=7)
try:
    from qiskit import schedule
    _ = schedule(QuantumCircuit(1), backend=backend)
    HAS_PULSE = True
except Exception:
    HAS_PULSE = False
print(f"✅ Backend → {type(backend).__name__} | HAS_PULSE={HAS_PULSE}")

# ───────────────── parameters ────────────────────
DEPTHS   = [1, 2, 4, 8, 16, 32]
POLICIES = ["HashOnly", "Transpile", "SyncCache"]
CONC_POL = ["Baseline", "Full", "HashOnly", "Transpile"]
COLORS   = {"Baseline": "tab:gray", "Full": "tab:blue", "HashOnly": "tab:orange", "Transpile": "tab:green"}

# ───────────────── circuit families ──────────────

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

# ───────────────── helpers ───────────────────────

def qasm_key(qc: QuantumCircuit) -> str:
    """Canonical QASM string作为缓存键(含长度防碰撞)。"""
    qasm = qc.qasm()
    return f"{hashlib.md5(qasm.encode()).hexdigest()}:{len(qasm)}"

# ───────────────── flows ─────────────────────────
full_cache      = Memory("pc_full", verbose=0)
hash_cache      = Memory("pc_hash", verbose=0)
transp_cache    = Memory("pc_trans", verbose=0)

@full_cache.cache
def _full_compile_qasm(qasm_str: str):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    return tqc

@hash_cache.cache
def _hash_compile_qasm(qasm_str: str):
    return _full_compile_qasm(qasm_str)

@transp_cache.cache
def _tr_compile_qasm(qasm_str: str):
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    return transpile(qc, backend, optimization_level=3)

# ── flow wrappers: 逻辑保持不变，只改传参 ─────────

def flow_baseline(qc):
    t0 = time.perf_counter()
    tqc = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        from qiskit import schedule;
        schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    time.sleep(0.1)
    return time.perf_counter() - t0

def flow_full(qc):
    t0 = time.perf_counter(); _full_compile_qasm(qc.qasm()); time.sleep(0.1)
    return time.perf_counter() - t0

def flow_hash(qc):
    t0 = time.perf_counter(); _hash_compile_qasm(qc.qasm())
    return time.perf_counter() - t0

def flow_transpile(qc):
    t0 = time.perf_counter(); tqc = _tr_compile_qasm(qc.qasm())
    if HAS_PULSE:
        schedule(tqc, backend=backend)
    else:
        time.sleep(0.3)
    return time.perf_counter() - t0

sync_cache: Dict[str, bool] = {}

def flow_sync(qc):
    k = qasm_key(qc)
    if k not in sync_cache:
        delay = flow_baseline(qc); sync_cache[k] = True; return delay
    return flow_full(qc)

FLOW = {"Baseline": flow_baseline, "Full": flow_full, "HashOnly": flow_hash, "Transpile": flow_transpile, "SyncCache": flow_sync}

# ───────────────── poisson concurrent ─────────────

def poisson_arr(rate, n):
    return np.cumsum(np.random.exponential(1 / rate, n))

def rand_circ():
    name = np.random.choice(list(CIRCS)); depth = int(np.random.choice(DEPTHS))
    return CIRCS[name](num_qubits(backend), depth)

def task(policy):
    return policy, FLOW[policy](rand_circ())

def run_concurrent(rate=20, total=5000):
    if platform.system() == "Windows":
        workers = min(12, os.cpu_count() * 2)
        print(f"⚠️  Windows thread pool ×{workers}")
        executor = cf.ThreadPoolExecutor(max_workers=workers)
        submit = executor.submit
    else:
        workers = min(64, os.cpu_count())
        pool = mp.Pool(workers)
        submit = lambda fn, *a: pool.apply_async(fn, a)
    arrivals = poisson_arr(rate, total)
    policies = np.random.choice(CONC_POL, total)
    lat = {p: [] for p in CONC_POL}
    t0 = time.time(); futures = []
    for at, pol in zip(arrivals, policies):
        time.sleep(max(t0 + at - time.time(), 0))
        futures.append(submit(task, pol))
    for fu in futures:
        p, l = fu.result() if platform.system() == "Windows" else fu.get()
        lat[p].append(l)
    if platform.system() == "Windows":
        executor.shutdown()
    else:
        pool.close(); pool.join()
    return lat

def plot_cdf(lat):
    plt.figure(figsize=(6, 4))
    for p, data in lat.items():
        if not data: continue
        s = np.sort(data); cdf = np.linspace(0, 1, len(s))
        plt.step(s, cdf, where='post', label=p, color=COLORS[p])
    plt.xscale('log'); plt.xlabel('Latency (s)'); plt.ylabel('CDF'); plt.title('Poisson workload')
    plt.grid(ls='--', alpha=0.4); plt.legend(); ensure_dir('figs'); plt.tight_layout()
    plt.savefig('figs/v5_cdf_concurrent.png', dpi=600)

# ───────────────────────── CLI ────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument('--concurrent', action='store_true'); args = ap.parse_args()
    if args.concurrent:
        print('\n=== Concurrent workload CDF (QASM‑key) ===')
        lat = run_concurrent(); plot_cdf(lat); print('→ figs/v5_cdf_concurrent.png')
    else:
        print('Add --concurrent to run Poisson CDF experiment (QASM‑key version).')
