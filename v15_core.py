# v15_core.py
import time, math, hashlib, json, pickle, pathlib, warnings, random
from typing import Any, Callable, Dict, List, Tuple, Optional
from collections import defaultdict, deque, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from qiskit import QuantumCircuit, transpile, qasm3
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler

# circuits
from v11_quasa_bench_circuits import CIRCUITS_QUASA as CIRCUITS

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------- paths / cache ----------
VNUM = 15
ROOT = pathlib.Path("./figs")/f"v{VNUM}"
EVENTS_DIR = ROOT/"events"
PLOT_DIR = ROOT/"plots"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOAD_ROOT = pathlib.Path("./figs")/f"v{VNUM}_7_scale_on_workloads"
LOAD_EVENTS_DIR = LOAD_ROOT/"events"

def md5_qasm(circ: QuantumCircuit) -> str:
    try:
        txt = circ.qasm()
    except Exception:
        txt = qasm3.dumps(circ)
    return hashlib.md5(txt.encode()).hexdigest()

def _prepare_kwargs():
    aer = AerSimulator()
    try:
        return {"target": aer.target, "optimization_level": 3, "seed_transpiler": 42}
    except Exception:
        cfg = aer.configuration()
        return {"basis_gates": cfg.basis_gates, "optimization_level": 3, "seed_transpiler": 42}

# --------- recent calls (for predictor) ----------
SLIDING_MAXLEN = 256
RECENT_CALLS: Dict[str, deque] = defaultdict(lambda: deque(maxlen=SLIDING_MAXLEN))
def record_arrival(key: str, ts: Optional[float] = None) -> None:
    RECENT_CALLS[key].append(float(ts) if ts is not None else time.time())
def clear_recent(): RECENT_CALLS.clear()

# --------- predictor ----------
class PoissonPredictor:
    """ λ MLE on sliding window;  p(≤τ)=1-exp(-λτ) """
    def __init__(self, sliding_window_sec: float = 60.0, min_samples: int = 2):
        self.sliding_window_sec = float(sliding_window_sec)
        self.min_samples = int(min_samples)

    def _recent(self, key: str, now: Optional[float] = None) -> List[float]:
        q = RECENT_CALLS.get(key, deque())
        if not q: return []
        now = float(now) if now is not None else time.time()
        lo = now - self.sliding_window_sec
        return [t for t in q if lo <= t <= now]

    def est_lambda(self, key: str, now: Optional[float] = None) -> float:
        ts = self._recent(key, now)
        if len(ts) < self.min_samples: return 0.0
        T = max(1e-9, max(ts) - min(ts)); N = len(ts)
        return N / T if T > 0 else 0.0

    @staticmethod
    def prob_within(lam: float, tau: float) -> float:
        if lam <= 0 or tau <= 0: return 0.0
        return 1.0 - math.exp(-lam * tau)

    def score_candidates(self, candidates, lookahead_sec: float, prob_threshold: float = 0.5):
        now = time.time(); out = []
        for maker in candidates:
            qc, key, info = maker()
            lam = self.est_lambda(key, now)
            p = self.prob_within(lam, lookahead_sec)
            if p >= prob_threshold:
                out.append({"key": key, "prob": p, "lambda": lam,
                            "qc_raw": qc, "info": info})
        out.sort(key=lambda r: r["prob"], reverse=True)
        return out

# --------- catalog & workload (with timestamps) ----------
def build_catalog(q_list: List[int], d_list: List[int]):
    makers = []
    meta = []
    for q in q_list:
        for d in d_list:
            for name, make in CIRCUITS.items():
                def _mk(name_=name, make_=make, q_=q, d_=d):
                    def _call():
                        qc = make_(q_, d_)
                        key = f"AerSV:{md5_qasm(qc)}"
                        info = {"circ": name_, "q": q_, "d": d_,
                                "n_qubits": qc.num_qubits, "depth": qc.depth()}
                        return qc, key, info
                    return _call
                makers.append(_mk())
                def _mk_run(m_=make, q_=q, d_=d):
                    return lambda: m_(q_, d_)
                meta.append({"name": name, "q": q, "d": d, "maker_run": _mk_run()})
    return makers, meta

def build_workload_poisson_superposition(meta, workload_len: int,
                                         hot_fraction: float = 0.25, hot_boost: float = 8.0,
                                         rps: float = 1.0, rng_seed: int = 123,
                                         return_timestamps: bool = True):
    assert workload_len > 0
    rng = random.Random(rng_seed)
    H = workload_len / max(1e-9, rps)
    m = len(meta); weights = np.ones(m, dtype=float)
    hot_k = max(1, int(round(hot_fraction * m)))
    for i in rng.sample(range(m), hot_k): weights[i] *= max(1.0, hot_boost)
    w_sum = float(weights.sum()); lambdas = (weights / w_sum) * rps
    events: List[Tuple[float, int]] = []
    for i, lam in enumerate(lambdas):
        if lam <= 0: continue
        t = 0.0
        while t < H and len(events) < workload_len * 5:
            t += rng.expovariate(lam)
            if t <= H: events.append((t, i))
    events.sort(key=lambda x: x[0])
    if len(events) > workload_len:
        events = events[:workload_len]
    elif len(events) < workload_len:
        probs = (weights / w_sum).tolist()
        extra = rng.choices(range(m), weights=probs, k=workload_len - len(events))
        for j, i in enumerate(extra):
            events.append((H + (j+1)*1e-3, i))
        events.sort(key=lambda x: x[0])

    out = []
    for t, i in events:
        it = {"name": meta[i]["name"], "q": meta[i]["q"], "d": meta[i]["d"],
              "maker_run": meta[i]["maker_run"]}
        if return_timestamps: it["t_arr"] = float(t)
        out.append(it)
    return out

# --------- seeding predictor ----------
def seed_recent_calls_for_predictor(predictor_window_sec: float, makers_all, workload,
                                    seed_keys: int = 4, per_key_samples: int = 2, spacing_sec: float = 3.0):
    prefix = workload[:max(1, min(64, len(workload)//2))]
    counts: Dict[Tuple[str,int,int], int] = defaultdict(int)
    for it in prefix:
        counts[(it["name"], it["q"], it["d"])] += 1
    popular = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:seed_keys]
    now = time.time(); horizon_lo = now - predictor_window_sec * 0.8
    for (name, q, d), _ in popular:
        hit_key = None
        for mk in makers_all:
            qc, key, info = mk()
            if info["circ"] == name and info["q"] == q and info["d"] == d:
                hit_key = key; break
        if hit_key is None: continue
        for j in range(max(2, per_key_samples)):
            record_arrival(hit_key, ts=horizon_lo + (j+1)*spacing_sec)

# --------- compile/run helpers ----------
def compile_with_idle_cache(qc_raw: QuantumCircuit, bk_name: str,
                            cache: Dict[str, QuantumCircuit]) -> Tuple[QuantumCircuit, str, bool, float]:
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
    qc_exec = cache.get(key); hit = qc_exec is not None
    compile_sec = 0.0
    if not hit:
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        compile_sec = time.perf_counter() - t0
        cache[key] = qc_exec
    return qc_exec, key, hit, compile_sec

def run_once_with_cache(qc_func: Callable[[], QuantumCircuit], cache: Dict[str, QuantumCircuit],
                        shots: int = 256) -> Dict[str, Any]:
    qc_raw = qc_func()
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk = "AerSV"
    qc_exec, key, hit, compile_sec = compile_with_idle_cache(qc_raw, bk, cache)
    t0 = time.perf_counter()
    _ = sampler.run([qc_exec], shots=shots).result()
    exec_sec = time.perf_counter() - t0
    record_arrival(key)
    return {"key": key, "cache_hit": hit, "compile_sec": compile_sec, "exec_sec": exec_sec,
            "n_qubits": qc_raw.num_qubits, "depth_in": qc_raw.depth(), "depthT": qc_exec.depth()}

def run_once_nocache(qc_func: Callable[[], QuantumCircuit], shots: int = 256) -> Dict[str, Any]:
    qc_raw = qc_func()
    t0 = time.perf_counter()
    qc_exec = transpile(qc_raw, **_prepare_kwargs()); compile_sec = time.perf_counter() - t0
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    t1 = time.perf_counter()
    _ = sampler.run([qc_exec], shots=shots).result(); exec_sec = time.perf_counter() - t1
    return {"compile_sec": compile_sec, "exec_sec": exec_sec,
            "n_qubits": qc_raw.num_qubits, "depth_in": qc_raw.depth(), "depthT": qc_exec.depth()}

# ---------- events IO ----------
def save_events_json(method_name: str, events: List[Dict[str, Any]]) -> pathlib.Path:
    path = EVENTS_DIR/f"{method_name}.json"
    path.write_text(json.dumps(events, ensure_ascii=False, indent=2))
    return path

def load_events_json(path: pathlib.Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())


# ---------- labels / bars ----------
def label_of(name: str, q: int, d: int) -> str:
    return f"{name}_q{q}_d{d}"

def compute_freq_and_hits(workload, hit_keys: Dict[str,int]) -> Tuple[Dict[str,int], Dict[str,float], float]:
    freq_by_label: Dict[str,int] = Counter([label_of(it["name"], it["q"], it["d"]) for it in workload])
    hits_by_label: Dict[str,int] = Counter()
    total_hits = 0
    for L, h in hit_keys.items():
        hits_by_label[L] += h; total_hits += h
    hitrate_by_label = {L: (hits_by_label.get(L,0)/freq) for L, freq in freq_by_label.items()}
    overall = total_hits / max(1, len(workload))
    return freq_by_label, hitrate_by_label, overall

def plot_freq_hitrate_bars(freq_by_label, hitrate_by_label, overall_hit_rate,
                           out_png: pathlib.Path, title: str, top_k: Optional[int] = None):
    labels = list(freq_by_label.keys())
    labels.sort(key=lambda L: freq_by_label[L], reverse=True)
    if top_k and len(labels) > top_k: labels = labels[:top_k]
    freq_vals = [freq_by_label[L] for L in labels]
    hit_vals = [100.0*hitrate_by_label.get(L,0.0) for L in labels]
    x = np.arange(len(labels)); w = 0.4
    fig, ax1 = plt.subplots(figsize=(max(10, 0.5*len(labels)+6), 6))
    b1 = ax1.bar(x - w/2, freq_vals, width=w, label="Frequency (count)")
    ax1.set_ylabel("Frequency (count)"); ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w/2, hit_vals, width=w, label="Hit rate (%)", color="#ff7f0e")
    ax2.set_ylabel("Hit rate (%)"); ax2.set_ylim(0, 105)
    ax1.legend(handles=[b1,b2], labels=["Frequency (count)","Hit rate (%)"], loc="upper right")
    ax2.legend([plt.Line2D([0],[0], color='none')], [f"Overall hit rate: {overall_hit_rate*100:.1f}%"],
               loc="upper left", frameon=False)
    plt.title(title, loc="left"); plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight"); print(f"[save] bars -> {out_png}")

# ---------- multi-method timeline ----------
def draw_timeline_multi(method_events: Dict[str, List[Dict[str,Any]]],
                        out_png: pathlib.Path, title: str, legend_topk: int = 16):
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})
    PRED_LABEL = "__predictor__"
    # collect circuit labels
    labels = []
    for evs in method_events.values():
        labels += [e["label"] for e in evs if "label" in e and e["label"] != PRED_LABEL and e["kind"]!="queue_wait"]
    freq = Counter(labels)
    uniq = list(freq.keys())
    cmap = plt.cm.get_cmap("tab20", max(20, len(uniq)))
    label2color = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq)}

    # size
    n = len(method_events)
    # fig, ax = plt.subplots(figsize=(14, 2.2*n + 1.2))
    fig, ax = plt.subplots(figsize=(14, 7))
    T = 0.0
    for evs in method_events.values():
        T = max(T, max((e["start"]+e["dur"] for e in evs), default=0.0))
    ax.set_xlim(0, T*1.05)
    ax.set_ylim(-0.6, n-0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(method_events.keys())[::-1], fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    # ax.set_title(title, loc="left")
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.5, zorder=0)
    for spine in ax.spines.values(): spine.set_linewidth(2)

    def draw_row(evs, y):
        h = 0.3
        for e in evs:
            x0, w = e["start"], e["dur"]
            if e["kind"] == "predict":
                rect = patches.Rectangle((x0, y - h/2), w, h, facecolor="white", edgecolor="k", hatch="xx", alpha=0.9)
            elif e["kind"] == "prewarm":
                color = label2color.get(e["label"], (0.7,0.7,0.7,1.0))
                rect = patches.Rectangle((x0, y - h/2), w, h, facecolor=color, edgecolor="k", hatch="//", alpha=0.85)
            elif e["kind"] == "queue_wait":
                rect = patches.Rectangle((x0, y - h/2), w, h, facecolor="none", edgecolor="k", linestyle="--", alpha=0.6)
            else:  # run
                color = label2color.get(e["label"], (0.7,0.7,0.7,1.0))
                rect = patches.Rectangle((x0, y - h/2), w, h, facecolor=color, edgecolor="k", alpha=0.95)
            ax.add_patch(rect)

    for idx, (name, evs) in enumerate(method_events.items()):
        draw_row(evs, y=(n-1-idx))

    handles = []
    for lab,_ in freq.most_common(legend_topk):
        handles.append(patches.Patch(facecolor=label2color[lab], edgecolor="k", label=lab))
    handles += [patches.Patch(facecolor="white", edgecolor="k", hatch="//", label="prewarm compile"),
                patches.Patch(facecolor="white", edgecolor="k", hatch="xx", label="predictor scoring"),
                patches.Patch(facecolor="none", edgecolor="k", linestyle="--", label="queue wait")]
    # ax.legend(handles=handles, loc="lower right", ncol=2, fontsize=12, frameon=False)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),  # (横向, 纵向) 偏移
        ncol=4,
        fontsize=12,
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    print(f"[save] timeline -> {out_png}")


def plot_cache_size_change(cache_size_cahnges: Dict[str, List[Dict[str,Any]]],):
    # 画图
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})

    def _to_xy(series):
        if not series: return [], []
        ts = [float(p["t"]) for p in series]
        sz = [int(p["size"]) for p in series]
        # 为了阶梯图效果，插入前一点
        return ts, sz

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, series in cache_size_cahnges.items():
        ts, sz = _to_xy(series)
        ax.step(ts, sz, where="post", label=label)

    # ts4, sz4 = _to_xy(s4_series)
    # ts5, sz5 = _to_xy(s5_series)
    # ts3, sz3 = _to_xy(s3_series)
    #
    #
    # # 用阶梯图表现“事件后大小变更”的感觉
    # ax.step(ts4, sz4, where="post", label="TransCache_no_cache_management")
    # ax.step(ts5, sz5, where="post", label="TransCache")
    # ax.step(ts3, sz3, where="post", label="FirstSeen")

    ax.set_xlabel("Timeline (s)")
    ax.set_ylabel("Cache size (#circuits)")
    ax.set_title("Cache size over time")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    out_png = PLOT_DIR / "cache_size.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    print(f"[save] cache-size lines -> {out_png}")
