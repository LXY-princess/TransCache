# bell_real_backend_compare.py  – baseline vs PulseCache (real QPU)
# -----------------------------------------------------------------
import time, json, argparse, pathlib, pickle, hashlib, sys
from datetime import datetime, timezone
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import matplotlib.pyplot as plt
from qiskit import qasm3
import matplotlib.patches as mpatches

FIGDIR   = pathlib.Path("./figs"); FIGDIR.mkdir(exist_ok=True)
CACHE_P  = pathlib.Path("./figs/v8_pulsecache.pkl")  # simple pickle {hash: (backend, qc_sched)}

# ── helpers ──────────────────────────────────────────────────────
def now_perf(): return time.perf_counter()
def dsec(start): return round(time.perf_counter()-start, 6)
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

def make_bell():
    qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1]); return qc

def load_cache():  return pickle.loads(CACHE_P.read_bytes()) if CACHE_P.exists() else {}
def save_cache(c): CACHE_P.write_bytes(pickle.dumps(c))

# ── single run (mode = baseline | cache) ─────────────────────────
def run_once(mode:str, backend_name:str, shots:int):
    laps, _tic = {}, None
    def tic(k): nonlocal _tic; _tic = now_perf(); laps[k] = -1
    def toc(k): laps[k] = dsec(_tic)

    tic("02a_tls_auth")
    svc = QiskitRuntimeService()
    backend = svc.backend(backend_name)
    sampler = Sampler(backend)
    toc("02a_tls_auth")
    qc_raw  = make_bell()

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
    # job = sampler.run(qc_sched, shots=shots)
    job_id = "d2782po56fsc73e5n7cg"
    job = svc.job(job_id)
    toc("04_submit")

    # tic("09_to_15_wait")
    result = job.result()
    # toc("09_to_15_wait")

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
        "10_11_bind_load": round((span_start- t_running ).total_seconds(),4),
        "12_qpu_exec":     round((span_stop  - span_start).total_seconds(),4),
        "13_14_read_post": round((t_finished- span_stop ).total_seconds(),4),
        # "15_return":       round(laps["09_to_15_wait"] -
        #                          (t_finished - t_running ).total_seconds(),4),
    })
    laps["total"] = round(sum(v for k,v in laps.items() if k!="total"),4)
    return laps

COLOR_MAP = {
    "02a_tls_auth"       : "#9ecae1",
    "03_transpile"          : "#6baed6",
    "04_submit"          : "#4292c6",
    "05_09_prep"         : "#fedcb2",
    "10_11_bind_load"    : "#fdae6b",
    "12_qpu_exec"        : "#fd8d3c",
    "13_14_read_post"    : "#e6550d",
    "15_return"          : "#8c6bb1",
    "16_cache_write"     : "#6e559c",
}

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
    out = FIGDIR / "v8_lat_compare.png"
    plt.savefig(out, dpi=600)
    plt.show()
    print("figure saved ->", out)

# ── CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_torino")
    ap.add_argument("--shots", type=int, default=1024)
    args = ap.parse_args()

    print("▶ baseline run...")
    laps_base = run_once("baseline", args.backend, args.shots)
    (FIGDIR/"v8_lat_baseline.json").write_text(json.dumps(laps_base,indent=2))

    print("▶ PulseCache warm-hit run...")
    laps_cache = run_once("cache", args.backend, args.shots)
    (FIGDIR/"v8_lat_pulsecache.json").write_text(json.dumps(laps_cache,indent=2))



    plot_compare(laps_base, laps_cache)
