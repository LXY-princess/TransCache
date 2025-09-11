"""
demo_9_pulseCache_op.py
~~~~~~~~~~~~~~~~~~~~~~~
• 发现 backend.open_pulse == True 时，把编译好的 **脉冲 ScheduleBlock** 缓存下来，
  以后同一线路 (同一校准时间戳) 可直接复用，跳过编译 / scheduling。
• 非 open-pulse 设备沿用门级 QuantumCircuit 缓存。
"""

import time, json, argparse, pathlib, pickle, hashlib, sys
from datetime import datetime, timezone
from typing import Union

from qiskit import QuantumCircuit, transpile
from qiskit.compiler import schedule  # Qiskit ≥0.32
from qiskit.transpiler import Target
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.pulse import ScheduleBlock   # new pulse object in Terra 0.22+
from qiskit import qasm3
from qiskit.circuit.library import QFT

# ----------------------------- GLOBALS ------------------------------------
plt_rc = {
    "font.family": "Times New Roman",
    "font.size":   15,
}
FIGDIR   = pathlib.Path("./figs"); FIGDIR.mkdir(exist_ok=True)
CACHE_P  = pathlib.Path("./figs/v9_pulsecache.pkl")     # pickle file
# --------------------------------------------------------------------------

# ──────────────────────────── util funcs ──────────────────────────────────
def now_perf(): return time.perf_counter()
def dsec(start): return round(time.perf_counter()-start, 6)

def md5_qasm(circ: QuantumCircuit) -> str:
    """Stable MD5 hash for a circuit (QASM2 or QASM3)."""
    try:
        txt = circ.qasm()              # Terra 1.x
    except AttributeError:             # Terra 2.x
        txt = qasm3.dumps(circ)
    return hashlib.md5(txt.encode()).hexdigest()

def load_cache() -> dict:
    return pickle.loads(CACHE_P.read_bytes()) if CACHE_P.exists() else {}

def save_cache(c: dict) -> None:
    CACHE_P.write_bytes(pickle.dumps(c))

def calib_ts(backend) -> str:
    """Return backend’s latest calibration time (ISO str).
       If properties() unavailable, fall back to '0'."""
    prop = backend.properties()
    if prop is None or not hasattr(prop, "last_update_date"):
        return "0"
    ts = prop.last_update_date
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc).isoformat()
    return str(ts)

# ─────────────────────── sample quantum circuits ──────────────────────────
def make_bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    return qc

def make_linear_ent(q=12, d=5):
    qc = QuantumCircuit(q); import numpy as np
    for _ in range(d):
        qc.h(range(q))
        for i in range(q-1):  qc.cx(i,i+1)
        qc.rx(np.pi/4, range(q))
    qc.measure_all(); return qc

def make_ghz_chain(q=12, d=5):
    qc = QuantumCircuit(q); import numpy as np
    qc.h(0);  [qc.cx(i,i+1) for i in range(q-1)]
    for _ in range(d-1):
        qc.rz(np.pi/8, q-1); qc.barrier()
    qc.measure_all(); return qc

def make_qft_like(q=12, d=5):
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.compose(QFT(num_qubits=q, do_swaps=False).decompose(), range(q), inplace=True)
    qc.measure_all(); return qc


CIRCUITS = {"bell": make_bell}

# CIRCUITS = {"bell": make_bell,
#             "LinearEnt": make_linear_ent,
#             "GHZ-Chain": make_ghz_chain,
#             "QFT-Like":  make_qft_like}

def parse_iso(ts):
    """Accept either ISO-8601 string or datetime; return tz-aware UTC datetime."""
    if isinstance(ts, datetime):
        # already datetime: ensure it’s timezone-aware (UTC)
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    # otherwise treat as str
    ts = ts.replace("Z", "+00:00")
    if "+" not in ts[-6:]:
        ts += "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        head, frac_tz = ts.split(".")
        frac, tz = frac_tz[:frac_tz.find("+")], frac_tz[frac_tz.find("+"):]
        return datetime.fromisoformat(f"{head}.{frac.ljust(6,'0')[:6]}{tz}")

# ────────────────────────── 核心执行逻辑 ─────────────────────────────────
def compile_or_reuse(qc_raw: QuantumCircuit,
                     backend,
                     cache_dict: dict,
                     opt_level: int = 3) -> Union[QuantumCircuit, ScheduleBlock]:
    """
    若缓存命中直接返回编译产物（QC 或 Schedule），否则重新编译并写入缓存。
    """
    mode = "pulse" if backend.configuration().open_pulse else "gate"
    key  = f"{backend.name}:{calib_ts(backend)}:{md5_qasm(qc_raw)}:{mode}"

    if key in cache_dict:
        print(f"Cache Hit, Mode - {mode}")
        return cache_dict[key]

    print(f"Cache Miss, Mode - {mode}")

    # --- 重新编译 ---
    qc_comp = transpile(qc_raw, backend=backend, optimization_level=opt_level)

    if mode == "pulse":
        compiled_obj = schedule(qc_comp, backend=backend, method="asap")
    else:
        compiled_obj = qc_comp

    cache_dict[key] = compiled_obj
    save_cache(cache_dict)
    return compiled_obj

def run_once(qc_func,
             backend_name: str,
             shots: int = 1024,
             use_cache: bool = True) -> dict:
    """编译（或复用缓存）+ 提交 + 收集时延切片"""
    laps, _tic = {}, None
    tic = lambda k: laps.setdefault(k, -now_perf())
    toc = lambda k: laps.__setitem__(k, dsec(-laps[k]))

    # 1) 连接 & 拉 backend
    tic("02a_tls_auth")
    svc     = QiskitRuntimeService()
    backend = svc.backend(backend_name)
    toc("02a_tls_auth")

    qc_raw = qc_func()

    # 2) 编译 or 复用缓存
    tic("03_compile")
    cache = load_cache()
    compiled_obj = (compile_or_reuse(qc_raw, backend, cache)
                    if use_cache else
                    compile_or_reuse(qc_raw, backend, cache_dict={}))
    toc("03_compile")

    # 3) 提交并等待
    tic("04_submit")
    # job = backend.run(compiled_obj, shots=shots)  # 统一接口
    job_id = "d2782po56fsc73e5n7cg"
    job = svc.job(job_id)
    toc("04_submit")

    result = job.result()

    # server stamps
    met = job.metrics();
    ts = met["timestamps"]
    t_created = parse_iso(ts["created"])
    t_running = parse_iso(ts["running"])
    t_finished = parse_iso(ts["finished"])

    spans_obj = result.metadata["execution"]["execution_spans"]
    span = getattr(spans_obj, "spans", [spans_obj])[0]
    span_start = parse_iso(span.start)
    span_stop = parse_iso(span.stop)

    # 细粒度切片
    laps.update({
        "05_09_prep": round((t_running - t_created).total_seconds(), 4),
        "10_11_bind_load": round((span_start - t_running).total_seconds(), 4),
        "12_qpu_exec": round((span_stop - span_start).total_seconds(), 4),
        "13_14_read_post": round((t_finished - span_stop).total_seconds(), 4),
        # "15_return":       round(laps["09_to_15_wait"] -
        #                          (t_finished - t_running ).total_seconds(),4),
    })

    laps["total"] = round(sum(v for v in laps.values()), 4)
    return laps

# ────────────────────────────── CLI ───────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_brisbane") # "ibm_brisbane"
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--runs",  type=int, default=1)
    args = ap.parse_args()

    for name, fn in CIRCUITS.items():
        print(f"\n=== {name} – baseline ===")
        for _ in range(args.runs):
            print(run_once(fn, args.backend, args.shots, use_cache=False))

        print(f"=== {name} – PulseCache ===")
        for _ in range(args.runs):
            print(run_once(fn, args.backend, args.shots, use_cache=True))
