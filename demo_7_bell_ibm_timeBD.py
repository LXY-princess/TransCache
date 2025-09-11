import time as _bt; _BOOT_START = _bt.perf_counter()
import time, json, pathlib, argparse, pickle, sys
from collections import Counter
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from datetime import datetime, timezone
# --------- simple timer helpers ----------
_t0, laps = None, {}

laps["01_bootstrap"] = time.perf_counter() - _BOOT_START

SAVE_DIR = pathlib.Path("./figs")
SAVE_DIR.mkdir(exist_ok=True)

def tic(label):   global _t0; _t0 = time.perf_counter(); laps[label] = -1
def toc(label):   laps[label] = time.perf_counter() - _t0

# --------- make circuit ----------
def make_bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    return qc

# ── 2. 安全地解析 ISO 时间戳 → datetime ───────────────
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

def saveTimeBreakdown():
    # --------- 写 JSON & plot ---------
    lat_json = {k: round(v, 4) for k, v in laps.items()}
    json_path = SAVE_DIR / "v7_bell_latencies.json"
    json_path.write_text(json.dumps(lat_json, indent=2))
    print("[✔] latency JSON saved to", json_path)
    return lat_json

def tickTime():
    tic("02a_tls_auth")
    svc = QiskitRuntimeService()  # 建立 TLS, 完成 Auth
    toc("02a_tls_auth")

    tic("02b_backend_query")
    backend = (svc.backend(args.backend)
               if args.backend else
               svc.least_busy(simulator=False))
    sampler = Sampler(backend)
    toc("02b_backend_query")

    # ② Transpile
    tic("03_transpile")
    qc_qpu = transpile(qc_raw, backend=backend, optimization_level=3)
    toc("03_transpile")

    # # ③~④: TLS + Front-end
    tic("04_upload_to_cloud")
    # job = sampler.run([qc_qpu], shots=args.shots)
    job_id = "d2782po56fsc73e5n7cg"
    job = svc.job(job_id)
    toc("04_upload_to_cloud")  # 上传完成时即 Front-end 收到

    # ⑨-⑭ 真机阶段：用 SDK 回调 block 住，同时计时
    t_block_start = time.perf_counter()
    result = job.result()  # 阻塞直至 PrimitiveResult 就绪
    t_client_rcv = time.perf_counter()

    # ── 1. 取出 metrics 字典 ───────────────────────────────
    met = job.metrics()  # {'timestamps': {...}, 'usage': {...}, ...}
    ts = met.get("timestamps", {})
    t_created = parse_iso(ts["created"])
    t_running = parse_iso(ts["running"])
    t_finished = parse_iso(ts["finished"])

    """ Granularity """
    # ------ 取 spans，抓 start/stop ------
    spans_obj = result.metadata["execution"]["execution_spans"]
    span = getattr(spans_obj, "spans", [spans_obj])[0]  # 兼容旧 SDK
    span_start = parse_iso(span.start)
    span_stop = parse_iso(span.stop)

    # print(t_created, t_running, span_start, span_stop, t_finished)

    # ------ 四段细粒度时间 (秒) ------
    queue_prep = (t_running - t_created).total_seconds()  # ①
    bind_load = (span_start - t_running).total_seconds()  # ②
    qpu_exec = (span_stop - span_start).total_seconds()  # ③
    read_post = (t_finished - span_stop).total_seconds()  # ④

    # client_wait = t_client_rcv - t_block_start  # seconds (perf_counter)
    # server_work = (t_finished - t_running).total_seconds()  # all server after RUNNING
    # return_net = round(client_wait - server_work, 6)  # may be 0.02-0.08 s

    # # ── 3. 计算粗粒度区段 ───────────────────────────────
    # queue_s    = (t_running  - t_created ).total_seconds()   # 图⑨
    # cloud_exec = (t_finished - t_running ).total_seconds()   # ⑪-⑭ 整体
    # # wall_s     = (t_finished - t_created).total_seconds()    # 整个云端
    #
    # # ── 4. usage 里的量子秒数（若缺键则 0.0） ────────────
    # quantum_s  = met.get("usage", {}).get("quantum_seconds",
    #               met.get("usage", {}).get("seconds", 0.0))   # 图⑫
    #
    # ── 5. 写入 laps ────────────────────────────────────
    laps.update({
        "05_to_09_queue": round(queue_prep, 4),
        "10_11_bindElec_loadPulse": round(bind_load, 4),
        "12_qpu_exec": round(qpu_exec, 4),
        "13_14_read_post": round(read_post, 4),  # 控电加载+读出+M3
        # "15_return_to_client": round(return_net if return_net > 0 else 0.0, 4)  # ⑤–⑮ 总和
    })

def plotBar(lat_json):
    manual_colors = {
        # Client-side (blue palette suggestion)
        "01_bootstrap":       "#c6dbef",
        "02a_tls_auth":       "#9ecae1",
        "02b_backend_query":  "#6baed6",
        "03_transpile":   "#3182bd",
        "04_upload_to_cloud": "#7CCD7C",
        # Cloud-side (orange palette suggestion)
        "05_to_09_queue":           "#fee6ce",
        "10_11_bindElec_loadPulse": "#f9c28b",
        "12_qpu_exec":           "#fd8d3c",
        "13_14_read_post": "#EE6A50",
        # Back to client
        # "15_return_to_client": "#8B7D6B",
    }

    import matplotlib.pyplot as plt

    labels, times = zip(*sorted(lat_json.items()))
    cum = 0
    plt.figure(figsize=(10, 3))
    for l, t in zip(labels, times):
        plt.barh([0.5], t,             # y 设 0.5 可以让条居中
                 left=cum,
                 color=manual_colors[l],
                 height=0.25,
                 edgecolor="black",
                 linewidth=0.6,
                 label=l.replace("_", " "))
        cum += t

    ax = plt.gca()
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.margins(x=0.05)                 # ← 关键：两侧各留 5% 空白
    ax.set_ylim(0, 1)                  # 上下也留一点空白
    plt.xlabel("Time Latency (s)", fontweight="bold")
    plt.ylabel("Bell Circuit", fontweight="bold")
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    png_path = SAVE_DIR / "v7_bell_latencies.png"
    plt.savefig(png_path, dpi=600)
    plt.show()
    print("[✔] stacked-bar saved to", png_path)


if __name__ == "__main__":
    # --------- CLI ---------
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend")        # 手动指定后端
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    qc_raw = make_bell()
    tickTime()
    lat_json = saveTimeBreakdown()
    plotBar(lat_json)