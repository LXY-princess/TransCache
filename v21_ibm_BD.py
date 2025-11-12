# v21_ibm_BD.py  — reverted to "v17-style" Sampler run path (no svc.job(job_id))
# make a circuit and submit to ibm real backend, monitor each time durations.
# This is a basic job submission to ibm bkd, no acceleration applied
# Output: events_times.csv, step_durations.csv, latency.png

import time as _bt; _BOOT_START = _bt.perf_counter()

import time, json, pathlib, argparse, csv, sys
from datetime import datetime, timezone, timedelta

from qiskit import QuantumCircuit, transpile
# from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
# If you prefer session-style in your env, you can later switch to:
# from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
from v11_quasa_bench_circuits import CIRCUITS_QUASA

# ------------------- paths -------------------
VNUM = 21
SAVE_DIR = pathlib.Path("./figs")/f"v{VNUM}"; SAVE_DIR.mkdir(exist_ok=True)

EVENTS_file_name = "events_times.csv"
BREAKDOWN_file_name = "step_durations.csv"
JSON_LAT_file_name = "v21_latencies.json"
PNG_PATH_file_name = "v21_latencies.png"

# ------------------- global state -------------------
laps = {}               # step_name -> duration (seconds)
events = []             # list of {"event","time_utc","source","note"}

# ---- unified local time base: map perf_counter to absolute UTC ----
_T0_PERF = time.perf_counter()
_T0_UTC  = datetime.now(timezone.utc)  # reference wall clock (UTC)
def utc_now_from_perf():
    """Map current perf_counter to an absolute UTC timestamp."""
    delta = time.perf_counter() - _T0_PERF
    return _T0_UTC + timedelta(seconds=delta)

# ------------------- event helpers -------------------
_phase_stack = {}  # label -> start_perf

def event_point(name, source="local", note=""):
    """Record an instantaneous event (single timestamp)."""
    ts = utc_now_from_perf() if source == "local" else source  # if source is datetime
    if isinstance(ts, datetime):
        ts_utc = ts.astimezone(timezone.utc)
        ts_iso = ts_utc.isoformat()
    else:
        ts_iso = str(ts)  # fallback
    events.append({"event": name, "time_utc": ts_iso, "source": ("server" if isinstance(source, datetime) else source), "note": note})

def tic(label):
    _phase_stack[label] = time.perf_counter()
    event_point(f"{label}_start", source="local")

def toc(label):
    t1 = time.perf_counter()
    t0 = _phase_stack.get(label, t1)
    laps[label] = t1 - t0
    event_point(f"{label}_end", source="local")


# ------------------- ISO parsing -------------------
def parse_iso(ts):
    """Accept either ISO-8601 string or datetime; return tz-aware UTC datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    s = ts.replace("Z", "+00:00")
    if "+" not in s[-6:]:
        s += "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        head, frac_tz = s.split(".")
        plus = frac_tz.rfind("+")
        frac, tz = (frac_tz[:plus], frac_tz[plus:]) if plus != -1 else (frac_tz, "+00:00")
        return datetime.fromisoformat(f"{head}.{frac.ljust(6,'0')[:6]}{tz}")

# ------------------- save/load helpers -------------------
def write_events_csv(path: pathlib.Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["event","time_utc","source","note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

def read_events_csv(path: pathlib.Path):
    out = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out

def save_lat_json(d, JSON_LAT: pathlib.Path):
    d2 = {k: round(float(v), 6) for k, v in d.items()}
    JSON_LAT.write_text(json.dumps(d2, indent=2))
    print("[✔] latency JSON saved to", JSON_LAT)
    return d2

# ------------------- main measurement path -------------------
def tick_time(args, qc_raw):
    # 1) auth
    tic("02a_tls_auth")
    svc = QiskitRuntimeService()  # 构造 QiskitRuntimeService() 的本地时长（多为本地凭证读取，真正握手常发生在下一步）
    toc("02a_tls_auth")

    # 2) backend
    tic("02b_backend_query")
    backend = (svc.backend(args.backend) if args.backend else svc.least_busy(simulator=False))
    sampler = Sampler(mode=backend)
    toc("02b_backend_query")
    print(f"backend: {backend}")

    # 3) transpile (local)
    tic("03_transpile")
    qc_qpu = transpile(qc_raw, backend=backend, optimization_level=3)
    toc("03_transpile")

    # 4) submit a NEW job via Sampler (v17-style) — DO NOT fetch historical jobs
    tic("04_upload_to_cloud")
    # v17 style: directly run; this avoids reading any prior QPY payloads.
    job = sampler.run([qc_qpu], shots=args.shots)
    # job_id = "d3rp6m5q5lhs73be12v0"
    # job = svc.job(job_id)
    toc("04_upload_to_cloud")

    # --- client blocks waiting for result (local band) ---
    event_point("client_block_start", source="local")
    # t_block_start_perf = time.perf_counter()
    result = job.result()
    # t_client_rcv_perf = time.perf_counter()
    event_point("client_result_received", source="local")
    # laps["client_wait"] = t_client_rcv_perf - t_block_start_perf

    # --- server band: timestamps & spans ---
    met = job.metrics()  # {'timestamps': {...}, 'usage': {...}, ...}
    ts = met.get("timestamps", {}) if isinstance(met, dict) else {}
    # timestamps keys may vary slightly among releases; guard accesses:
    t_created  = parse_iso(ts.get("created"))   if ts.get("created")  else None
    t_running  = parse_iso(ts.get("running"))   if ts.get("running")  else None
    t_finished = parse_iso(ts.get("finished"))  if ts.get("finished") else None

    # record raw server instants (when available)
    if t_created:  event_point("srv_created",  source=t_created,  note="server job CREATED")
    if t_running:  event_point("srv_running",  source=t_running,  note="server job RUNNING")
    if t_finished: event_point("srv_finished", source=t_finished, note="server job FINISHED")

    # execution spans (optional — depends on runtime version)
    span_start = span_stop = None
    try:
        exec_meta = result.metadata.get("execution", {}) if hasattr(result, "metadata") else {}
        spans_obj = exec_meta.get("execution_spans")

        if spans_obj is not None:
            # 统一成 list
            items = getattr(spans_obj, "spans", spans_obj)
            if not isinstance(items, (list, tuple)):
                items = [items]
            first = items[0] if items else None

            # 取 start/stop，先 getattr 再（如是 dict 才）.get
            start_raw = getattr(first, "start", None)
            if start_raw is None and isinstance(first, dict):
                start_raw = first.get("start")

            stop_raw = getattr(first, "stop", None)
            if stop_raw is None and isinstance(first, dict):
                stop_raw = first.get("stop")

            if start_raw is not None and stop_raw is not None:
                span_start = parse_iso(start_raw)
                span_stop = parse_iso(stop_raw)
                event_point("srv_span_start", source=span_start, note="server exec span START")
                event_point("srv_span_stop", source=span_stop, note="server exec span STOP")
            else:
                print("[i] execution_spans present but missing start/stop; first item =", type(first).__name__)
        else:
            print("[i] no execution_spans in result.metadata['execution']")
    except Exception as e:
        print("[i] spans parse skipped:", repr(e))

    # --- server band durations ---
    def _sec(a, b):
        return (b - a).total_seconds() if (a and b) else None

    return events

# ------------------- offline analyzer -------------------
def analyze_from_events_csv(path: pathlib.Path, BREAKDOWN_CSV: pathlib.Path):
    rows = read_events_csv(path)
    def to_dt(ev):
        return parse_iso(ev["time_utc"]).astimezone(timezone.utc)
    E = {r["event"]: r for r in rows}

    def dur_local(start_ev, end_ev):
        if start_ev in E and end_ev in E:
            return (to_dt(E[end_ev]) - to_dt(E[start_ev])).total_seconds()
        return None

    local_steps = {
        "02a_tls_auth":           dur_local("02a_tls_auth_start", "02a_tls_auth_end"),
        "02b_backend_query":      dur_local("02b_backend_query_start", "02b_backend_query_end"),
        "03_transpile":           dur_local("03_transpile_start", "03_transpile_end"),
        "04_upload_to_cloud":     dur_local("04_upload_to_cloud_start", "04_upload_to_cloud_end"),
        # "client_wait":            dur_local("client_block_start", "srv_created"),
    }
    for k, v in local_steps.items():
        laps[k] = v

    def dur_server(a, b):
        if a in E and b in E:
            if a == "srv_running":
                print(to_dt(E[a]),to_dt(E[b]))
            return (to_dt(E[b]) - to_dt(E[a])).total_seconds()
        return None

    server_steps = {
        "05_to_09_queue":           dur_server("srv_created", "srv_running"),
        # "10_11_bindElec_loadPulse": dur_server("srv_running", "srv_span_start"),
        "server_run":              dur_server("srv_running", "srv_finished"),
        "downlink_return":          dur_server("srv_finished", "client_result_received"),
        # "server_E2E":               dur_server("srv_created", "srv_finished"),
    }
    for k, v in server_steps.items():
        laps[k] = v

    with open(BREAKDOWN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "duration_sec", "band"])
        for k, v in local_steps.items():
            if v is not None:
                w.writerow([k, f"{v:.6f}", "client/local"])
        for k, v in server_steps.items():
            if v is not None:
                w.writerow([k, f'{v:.6f}', "server/utc"])
    print("[✔] breakdown CSV written to", BREAKDOWN_CSV)

    print("== Local (client) ==")
    for k, v in local_steps.items():
        if v is not None:
            print(f"{k:26s}: {v:.6f} s")
    print("== Server (cloud) ==")
    for k, v in server_steps.items():
        if v is not None:
            print(f"{k:26s}: {v:.6f} s")

# ------------------- small plot (optional) -------------------
def plot_bar(lat_json, PNG_PATH: pathlib.Path):
    manual_colors = {
        "01_bootstrap":             "#c6dbef",
        "02a_tls_auth":             "#9ecae1",
        "02b_backend_query":        "#6baed6",
        "03_transpile":             "#3182bd",
        "04_upload_to_cloud":       "#7CCD7C",
        # "client_wait": "#fef3ce",
        "05_to_09_queue":           "#fee6ce",
        # "10_11_bindElec_loadPulse": "#f9c28b",
        "server_run":              "#fd8d3c",
        # "downlink_return":          "#EE6A50",
        # "client_wait":              "#8B7D6B",
    }
    import matplotlib.pyplot as plt
    labels, times = zip(*sorted(lat_json.items()))
    cum = 0
    plt.figure(figsize=(10, 3))
    for l, t in zip(labels, times):
        if l not in manual_colors:
            continue
        plt.barh([0.5], t, left=cum, color=manual_colors[l],
                 height=0.25, edgecolor="black", linewidth=0.6, label=l.replace("_", " "))
        cum += t
    ax = plt.gca()
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.margins(x=0.05)
    ax.set_ylim(0, 1)
    plt.xlabel("Time Latency (s)", fontweight="bold")
    plt.ylabel("Bell Circuit", fontweight="bold")
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=600)
    # plt.show()
    print("[✔] stacked-bar saved to", PNG_PATH)

# ------------------- main -------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_brisbane",help="e.g., ibm_brisbane / ibm_torino / ibm_marrakesh / ibm_fez /etc.")
    ap.add_argument("--shots", type=int, default=128)
    ap.add_argument("--analyze", type=str, help="path to events_times.csv for offline analysis")
    args = ap.parse_args()

    if args.analyze:
        analyze_from_events_csv(pathlib.Path(args.analyze))
        sys.exit(0)

    cir_name = ["QFT"]#"GHZ", "QFT", "QAOA", "VQE", "Bell"
    qc_raw_list = {}
    for k in cir_name:
        qc_raw_list[k] = CIRCUITS_QUASA.get(k)(113,1)

    for k, qc_raw in qc_raw_list.items():
        events.clear()
        laps.clear()
        events = tick_time(args, qc_raw)
        CIRC_DIR = pathlib.Path("./figs") / f"v{VNUM}" / k
        CIRC_DIR.mkdir(exist_ok=True)

        EVENTS_CSV = CIRC_DIR / EVENTS_file_name
        BREAKDOWN_CSV = CIRC_DIR / BREAKDOWN_file_name
        JSON_LAT = CIRC_DIR / JSON_LAT_file_name
        PNG_PATH = CIRC_DIR / PNG_PATH_file_name

        write_events_csv(EVENTS_CSV, events)
        print("[✔] events CSV written to", EVENTS_CSV)


        analyze_from_events_csv(EVENTS_CSV, BREAKDOWN_CSV)

        PNG_PATH_file_name = "v21_latencies.png"

        lat_json = save_lat_json(laps, JSON_LAT)
        try:
            plot_bar(lat_json, PNG_PATH)
        except Exception as e:
            print("[i] plot skipped:", e)
