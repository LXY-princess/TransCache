# bell_real_backend_compare.py  – baseline vs PulseCache (real QPU)
# -----------------------------------------------------------------
import time, json, argparse, pathlib, pickle, hashlib, sys
from datetime import datetime, timezone
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import matplotlib.pyplot as plt
from qiskit import qasm3
import matplotlib.patches as mpatches
from qiskit.circuit.library import QFT
import numpy as np


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size":   15,          # 视论文排版可再调
})
FIGDIR   = pathlib.Path("./figs"); FIGDIR.mkdir(exist_ok=True)
CACHE_P  = pathlib.Path("./figs/v8m_pulsecache.pkl")  # simple pickle {hash: (backend, qc_sched)}
COLOR_MAP = {
    "02a_tls_auth"       : "#9ecae1",
    "03_transpile"          : "#6baed6",
    "04_submit"          : "#4292c6",
    "05_09_prep"         : "#fedcb2",
    "10_11_bind_load"    : "#fdae6b",
    "12_qpu_exec"        : "#fd8d3c",
    "13_14_read_post"    : "#e6550d",
    # "15_return"          : "#8c6bb1",
    # "16_cache_write"     : "#6e559c",
}

# ── helpers ──────────────────────────────────────────────────────
def now_perf(): return time.perf_counter()
def dsec(start): return round(time.perf_counter()-start, 6)

# ---- build three depth-2 circuits -------------------------------------------
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

def make_bell():
    qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1]); return qc

# CIRCUITS = {"bell": make_bell,
#             "LinearEnt": make_linear_ent,
#             "GHZ-Chain": make_ghz_chain,
#             "QFT-Like":  make_qft_like}

CIRCUITS = {"LinearEnt": make_linear_ent,
            "GHZ-Chain": make_ghz_chain,
            "QFT-Like":  make_qft_like}

# CIRCUITS = {"LinearEnt": make_bell,
#             "GHZ-Chain": make_bell,
#             "QFT-Like":  make_bell}

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

def md5_qasm(circ):
    """
    Return a stable MD5 hash of the circuit text.
    Works on both Terra 1.x (qasm v2) and ≥2.0 (qasm3).
    """
    try:
        txt = circ.qasm()         # Terra 1.x 仍然存在
    except AttributeError:        # Terra 2.x 走这里
        txt = qasm3.dumps(circ)   # QASM 3.0 text
    return hashlib.md5(txt.encode()).hexdigest()

def load_cache():  return pickle.loads(CACHE_P.read_bytes()) if CACHE_P.exists() else {}
def save_cache(c): CACHE_P.write_bytes(pickle.dumps(c))

# ── single run (mode = baseline | cache) ─────────────────────────
def run_once(qc_func, mode:str, backend_name:str, shots:int):
    laps, _tic = {}, None
    def tic(k): nonlocal _tic; _tic = now_perf(); laps[k] = -1
    def toc(k): laps[k] = dsec(_tic)

    tic("02a_tls_auth")
    svc = QiskitRuntimeService()
    backend = svc.backend(backend_name)
    sampler = Sampler(backend)
    toc("02a_tls_auth")

    qc_raw = qc_func()

    # client-side
    tic("03_transpile")
    if mode=="cache":
        hkey = f"{backend.name}:{md5_qasm(qc_raw)}"
        cache = load_cache()
        if hkey in cache:
            qc_sched = cache[hkey]               #    ↖ warm-hit
        else:
            qc_sched = transpile(qc_raw, backend, optimization_level=3)
            cache[hkey] = qc_sched
            save_cache(cache)
        toc("03_transpile")
    else:
        qc_sched = transpile(qc_raw, backend, optimization_level=3)
        toc("03_transpile")

    # submit & wait
    tic("04_submit")
    job = sampler.run([qc_sched], shots=shots)
    # job_id = "d2782po56fsc73e5n7cg"
    # job = svc.job(job_id)
    toc("04_submit")

    result = job.result()

    # server stamps
    met = job.metrics(); ts = met["timestamps"]
    t_created  = parse_iso(ts["created"])
    t_running  = parse_iso(ts["running"])
    t_finished = parse_iso(ts["finished"])

    spans_obj = result.metadata["execution"]["execution_spans"]
    span      = getattr(spans_obj, "spans", [spans_obj])[0]
    span_start = parse_iso(span.start)
    span_stop = parse_iso(span.stop)

    # 细粒度切片
    laps.update({
        "05_09_prep": round((t_running - t_created ).total_seconds(),4),
        "10_11_bind_load": round((span_start - t_running ).total_seconds(),4),
        "12_qpu_exec":     round((span_stop - span_start).total_seconds(),4),
        "13_14_read_post": round((t_finished - span_stop ).total_seconds(),4),
        # "15_return":       round(laps["09_to_15_wait"] -
        #                          (t_finished - t_running ).total_seconds(),4),
    })
    laps["total"] = round(sum(v for k,v in laps.items() if k!="total"),4)
    return laps

def run_circuits(runs:int, backend:str, shots:int) -> None:
    """run all circuits in CIRCUITS over baseline and cache, """
    results = {}
    for name, fn in CIRCUITS.items():
        print(f"▶ {name} baseline")
        base_runs = []
        for r in range(runs):
            base = run_once(fn, "baseline", backend, shots)
            base_runs.append(base)
            (FIGDIR / f"v8m_{name}_baseline_{r}.json").write_text(json.dumps(base, indent=2))
        base_sum = {k: round(sum(r[k] for r in base_runs), 4) for k in base_runs[0]}
        (FIGDIR / f"v8m_{name}_baseline_sum.json").write_text(json.dumps(base_sum, indent=2))

        print(f"▶ {name} PulseCache")
        pc_runs = []
        for r in range(runs):
            pc = run_once(fn, "cache", backend, shots)
            pc_runs.append(pc)
            (FIGDIR / f"v8m_{name}_pulsecache{r}.json").write_text(json.dumps(pc, indent=2))
        cache_sum = {k: round(sum(r[k] for r in pc_runs), 4) for k in pc_runs[0]}
        (FIGDIR / f"v8m_{name}_pulsecache_sum.json").write_text(json.dumps(cache_sum, indent=2))
        results[name] = (base_sum, cache_sum)

    plot_all(results)

def run_compare(cirName:str, runs:int, backend:str, shots:int) -> None:
    """run circit over baseline and cache, """
    results = {}

    print(f"▶ {cirName} baseline")
    circ_fn = CIRCUITS[cirName]

    base_runs = []
    for r in range(runs):
        base = run_once(circ_fn, "baseline", backend, shots)
        base_runs.append(base)
        (FIGDIR / f"v8m_{cirName}_baseline_{r}.json").write_text(json.dumps(base, indent=2))
    base_sum = {k: round(sum(r[k] for r in base_runs), 4) for k in base_runs[0]}
    (FIGDIR / f"v8m_{cirName}_baseline_sum.json").write_text(json.dumps(base_sum, indent=2))

    print(f"▶ {cirName} PulseCache")
    pc_runs = []
    for r in range(runs):
        pc = run_once(circ_fn, "cache", backend, shots)
        pc_runs.append(pc)
        (FIGDIR / f"v8m_{cirName}_pulsecache{r}.json").write_text(json.dumps(pc, indent=2))
    cache_sum = {k: round(sum(r[k] for r in pc_runs), 4) for k in pc_runs[0]}
    (FIGDIR / f"v8m_{cirName}_pulsecache_sum.json").write_text(json.dumps(cache_sum, indent=2))
    results[cirName] = (base_sum, cache_sum)

    plot_compare(base_sum, cache_sum)
    # plot_all(results)

# ── plotting comparison ------------------------------------------------------
def plot_compare(baseline, cache):
    order = [k for k in baseline if k not in ("total")]
    fig, ax = plt.subplots(figsize=(11, 3))
    left = {"Baseline": 0.0, "PulseCache": 0.0}

    for key in order:
        for label, data in [("Baseline", baseline), ("PulseCache", cache)]:
            width = data.get(key, 0.0)
            ax.barh(label, width, left=left[label],
                    color=COLOR_MAP[key], edgecolor="black", height=0.35)
            left[label] += width

    ax.set_xlabel("Latency (s)")
    ax.set_xlim(0, max(baseline["total"], cache["total"]) * 1.2)
    ax.margins(x=0.05)

    # -------- legend right side --------------
    handles = [mpatches.Patch(color=COLOR_MAP[k], label=k) for k in order]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = FIGDIR / "v8m_lat_compare.png"
    plt.savefig(out, dpi=600)
    plt.show()
    print("figure saved ->", out)

def plot_all(res_dict):
    fig,ax=plt.subplots(figsize=(12,2+1.4*len(res_dict)))
    ytick, ylbl = [],[]
    order = list(COLOR_MAP.keys())
    for i,(name,(base,cache)) in enumerate(res_dict.items()):
        for j,(mode,laps) in enumerate([("Baseline",base),("PulseCache",cache)]):
            y = i*1.4 + 0.4*j
            left=0
            for k in order:
                w=laps.get(k,0)
                ax.barh(y,w,left=left,color=COLOR_MAP[k],edgecolor="black",height=0.2, zorder=2)
                left+=w
            # ax.text(left+0.05,y,f"{left:.2f}s",va="center",fontsize=8)
            ytick.append(y); ylbl.append(f"{name} – {mode}")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_yticks(ytick); ax.set_yticklabels(ylbl, fontweight="bold")
    ax.set_xlabel("Latency (s)", fontweight="bold"); ax.margins(x=0.05)
    handles=[mpatches.Patch(color=COLOR_MAP[k],label=k) for k in order]
    # ★ 把四条外围框线加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.legend(handles=handles,bbox_to_anchor=(1.02,1),loc="upper left")
    plt.tight_layout()
    out=FIGDIR/"v8m_compare_multi3.png"
    plt.savefig(out,dpi=600)
    plt.show()
    print("figure saved ->",out)

# ── CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_torino") # "ibm_brisbane"
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    # pre warm control electronics
    # run_once(make_bell, "baseline", args.backend, args.shots)

    # run one circuit
    # run_compare("GHZ-Chain", args.runs, args.backend, args.shots)
    # run_compare("GHZ-Chain", 1, args.backend, args.shots)

    # run all circuits
    run_circuits(args.runs, args.backend, args.shots)