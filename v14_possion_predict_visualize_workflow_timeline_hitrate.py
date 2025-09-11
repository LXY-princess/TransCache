# v19_predictor_timeline_plus_bars.py
# ------------------------------------------------------------
# Keep the existing timeline visualization, and:
#   (1) Add predictor scoring time to the timeline (Method A track)
#   (2) Draw a grouped bar chart for per-circuit Frequency & Hit rate
#       and show Overall hit rate in the legend
# ------------------------------------------------------------

import time, math, hashlib, pickle, pathlib, argparse, warnings, random
from collections import defaultdict, deque, Counter
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from networkx.algorithms.bipartite import color

from qiskit import QuantumCircuit, transpile, qasm3
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer import AerSimulator

# ---- circuits library (v11) ----
from v11_quasa_bench_circuits import CIRCUITS_QUASA as CIRCUITS

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------ 路径/缓存（用于方法 A） -----------
VNUM = 14
FIGDIR = pathlib.Path(f"./figs/v{VNUM}/"); FIGDIR.mkdir(parents=True, exist_ok=True)
P_IDLE = FIGDIR / f"v{VNUM}_idle_cache.pkl"
_IDLE_MEM: Optional[Dict[str, QuantumCircuit]] = None

def load_idle() -> Dict[str, QuantumCircuit]:
    global _IDLE_MEM
    if _IDLE_MEM is not None:
        return _IDLE_MEM
    _IDLE_MEM = pickle.loads(P_IDLE.read_bytes()) if P_IDLE.exists() else {}
    return _IDLE_MEM

def save_idle(c: Dict[str, QuantumCircuit]) -> None:
    global _IDLE_MEM
    _IDLE_MEM = c
    P_IDLE.write_bytes(pickle.dumps(c))

def clear_idle() -> None:
    global _IDLE_MEM
    _IDLE_MEM = {}
    if P_IDLE.exists():
        P_IDLE.unlink()

# ------------------------ 工具函数 -------------------------
def md5_qasm(circ: QuantumCircuit) -> str:
    try:
        txt = circ.qasm()
    except AttributeError:
        txt = qasm3.dumps(circ)
    return hashlib.md5(txt.encode()).hexdigest()

def _prepare_kwargs():
    aer = AerSimulator()
    try:
        return {"target": aer.target, "optimization_level": 3, "seed_transpiler": 42}
    except Exception:
        cfg = aer.configuration()
        return {"basis_gates": cfg.basis_gates, "optimization_level": 3, "seed_transpiler": 42}

# ------------------------ RECENT CALLS ---------------------
SLIDING_MAXLEN = 256
RECENT_CALLS: Dict[str, deque] = defaultdict(lambda: deque(maxlen=SLIDING_MAXLEN))
def record_arrival(key: str, ts: Optional[float] = None) -> None:
    RECENT_CALLS[key].append(float(ts) if ts is not None else time.time())

# ------------------------ 预测器（沿用 v14/v17 思路） -------
class PoissonPredictor:
    """
    λ: 最近窗口内 MLE 估计；p(≤τ)=1-exp(-λτ)
    """
    def __init__(self, sliding_window_sec: float = 60.0, min_samples: int = 2):
        self.sliding_window_sec = float(sliding_window_sec)
        self.min_samples = int(min_samples)

    def _recent_in_window(self, key: str, now: Optional[float] = None) -> List[float]:
        q = RECENT_CALLS.get(key, deque())
        if not q:
            return []
        now = float(now) if now is not None else time.time()
        lo = now - self.sliding_window_sec
        return [t for t in q if lo <= t <= now]

    def est_lambda(self, key: str, now: Optional[float] = None) -> float:
        ts = self._recent_in_window(key, now)
        if len(ts) < self.min_samples:
            return 0.0
        T = max(1e-9, max(ts) - min(ts)); N = len(ts)
        return N / T if T > 0 else 0.0

    @staticmethod
    def prob_within(lam: float, tau: float) -> float:
        if lam <= 0 or tau <= 0: return 0.0
        return 1.0 - math.exp(-lam * tau)

    def score_candidates(
        self,
        candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
        lookahead_sec: float, prob_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        now = time.time()
        scored = []
        for maker in candidates:
            qc, key, info = maker()  # 仅生成电路拿 key（不编译）
            lam = self.est_lambda(key, now)
            p = self.prob_within(lam, lookahead_sec)
            if p >= prob_threshold:
                scored.append({
                    "key": key, "prob": p, "lambda": lam,
                    "qc_raw": qc, "info": info
                })
        scored.sort(key=lambda r: r["prob"], reverse=True)
        return scored

# ------------------------ 预热（两种：原版 & 带测时） -------
def prewarm_from_predictions(
    predictor: PoissonPredictor,
    candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    lookahead_sec: float, prob_threshold: float, max_compile: int = 1
) -> List[Dict[str, Any]]:
    """
    原版（与旧脚本兼容）：不返回预测器耗时。
    """
    cache = load_idle()
    decided = predictor.score_candidates(candidates, lookahead_sec, prob_threshold)
    done: List[Dict[str, Any]] = []
    compiled = 0
    for item in decided:
        if compiled >= max_compile: break
        key = item["key"]
        if key in cache:
            done.append({"key": key, "action": "skip_in_idle", **item})
            continue
        qc_raw = item["qc_raw"]
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        cost = time.perf_counter() - t0
        cache[key] = qc_exec
        compiled += 1
        done.append({
            "key": key, "action": "compile",
            "prob": item["prob"], "lambda": item["lambda"],
            "compile_sec": cost,
            "n_qubits": qc_raw.num_qubits, "depth": qc_raw.depth(),
            "circ": item["info"].get("circ"), "q": item["info"].get("q"), "d": item["info"].get("d"),
        })
    if compiled > 0: save_idle(cache)
    return done

def prewarm_with_measured_predictor(
    predictor: PoissonPredictor,
    candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    lookahead_sec: float, prob_threshold: float, max_compile: int = 1
) -> Tuple[List[Dict[str, Any]], float]:
    """
    新增：测量 predictor.score_candidates 的耗时并复用其结果编译。
    返回： (decisions_list, predict_sec)
    """
    t_pred0 = time.perf_counter()
    decided = predictor.score_candidates(candidates, lookahead_sec, prob_threshold)
    predict_sec = time.perf_counter() - t_pred0

    cache = load_idle()
    done: List[Dict[str, Any]] = []
    compiled = 0
    for item in decided:
        if compiled >= max_compile: break
        key = item["key"]
        if key in cache:
            done.append({"key": key, "action": "skip_in_idle", **item})
            continue
        qc_raw = item["qc_raw"]
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        cost = time.perf_counter() - t0
        cache[key] = qc_exec
        compiled += 1
        done.append({
            "key": key, "action": "compile",
            "prob": item["prob"], "lambda": item["lambda"],
            "compile_sec": cost,
            "n_qubits": qc_raw.num_qubits, "depth": qc_raw.depth(),
            "circ": item["info"].get("circ"), "q": item["info"].get("q"), "d": item["info"].get("d"),
        })
    if compiled > 0: save_idle(cache)
    return done, predict_sec

# ------------------------ 运行路径 --------------------------
def compile_with_idle_cache(qc_raw: QuantumCircuit, bk_name: str
) -> Tuple[QuantumCircuit, str, bool, float]:
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
    cache = load_idle()
    qc_exec = cache.get(key); hit = qc_exec is not None
    compile_sec = 0.0
    if not hit:
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        compile_sec = time.perf_counter() - t0
        cache[key] = qc_exec; save_idle(cache)
    return qc_exec, key, hit, compile_sec

def run_once_with_cache(qc_func: Callable[[], QuantumCircuit], shots: int = 256) -> Dict[str, Any]:
    qc_raw = qc_func()
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk = "AerSV"
    qc_exec, key, hit, compile_sec = compile_with_idle_cache(qc_raw, bk)
    t0 = time.perf_counter()
    _ = sampler.run([qc_exec], shots=shots).result()
    exec_sec = time.perf_counter() - t0
    record_arrival(key)
    return {
        "key": key,
        "cache_hit": hit, "compile_sec": compile_sec, "exec_sec": exec_sec,
        "n_qubits": qc_raw.num_qubits, "depth_in": qc_raw.depth(), "depthT": qc_exec.depth()
    }

# ------------------------ 运行路径（方法 B：无缓存） ---------
def run_once_nocache(qc_func: Callable[[], QuantumCircuit], shots: int = 256) -> Dict[str, Any]:
    qc_raw = qc_func()
    # 始终编译，不保存
    t0 = time.perf_counter()
    qc_exec = transpile(qc_raw, **_prepare_kwargs())
    compile_sec = time.perf_counter() - t0
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    t1 = time.perf_counter()
    _ = sampler.run([qc_exec], shots=shots).result()
    exec_sec = time.perf_counter() - t1
    return {
        "label": f"{qc_raw.name if qc_raw.name else 'circ'}_{qc_raw.num_qubits}_{qc_raw.depth()}",
        "compile_sec": compile_sec, "exec_sec": exec_sec,
        "n_qubits": qc_raw.num_qubits, "depth_in": qc_raw.depth(), "depthT": qc_exec.depth()
    }

# ------------------------ 候选全集 & workload ----------------
def build_catalog(q_list: List[int], d_list: List[int]):
    makers: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]] = []
    meta: List[Dict[str, Any]] = []
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

def build_workload_poisson_superposition(
    meta: List[Dict[str, Any]], workload_len: int,
    hot_fraction: float = 0.25, hot_boost: float = 8.0, rps: float = 1.0, rng_seed: int = 123
) -> List[Dict[str, Any]]:
    assert workload_len > 0
    rng = random.Random(rng_seed)
    H = workload_len / max(1e-9, rps)
    m = len(meta); weights = np.ones(m, dtype=float)
    hot_k = max(1, int(round(hot_fraction * m)))
    for i in rng.sample(range(m), hot_k):
        weights[i] *= max(1.0, hot_boost)
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
        for j,i in enumerate(extra):
            events.append((H + (j+1)*1e-3, i))
        events.sort(key=lambda x: x[0])
    return [{"name": meta[i]["name"], "q": meta[i]["q"], "d": meta[i]["d"], "maker_run": meta[i]["maker_run"]}
            for _,i in events]

def seed_recent_calls_for_predictor(
    predictor_window_sec: float,
    makers_all: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    workload: List[Dict[str, Any]], seed_keys: int = 4, per_key_samples: int = 2, spacing_sec: float = 3.0
) -> None:
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

# ------------------------ 绘图 1：堆叠 latency 时间线 -------
def draw_timeline_png(events_A: List[Dict[str, Any]], events_B: List[Dict[str, Any]],
                      out_png: str, title: str, legend_topk: int = 14) -> None:
    """
    events_*: 每个元素：
      { "kind": "predict"|"prewarm"|"run", "label": "circ_q_d" or "__predictor__", "start": s, "dur": t }
    """
    PRED_LABEL = "__predictor__"

    # 颜色映射仅对“电路”生效；predictor 用独立样式
    all_labels = [e["label"] for e in events_A + events_B
                  if "label" in e and e["label"] != PRED_LABEL]
    freq = Counter(all_labels)
    uniq_labels = list(freq.keys())
    cmap = plt.cm.get_cmap("tab20", max(20, len(uniq_labels)))
    label2color = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq_labels)}

    # 时间范围
    T_A = max((e["start"] + e["dur"] for e in events_A), default=0.0)
    T_B = max((e["start"] + e["dur"] for e in events_B), default=0.0)
    T = max(T_A, T_B) * 1.05

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, T)
    ax.set_ylim(-0.8, 1.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Method B: No prewarm & no cache",
                        "Method A: Predictor prewarm + idle cache"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")

    def draw_events(events: List[Dict[str, Any]], y: float):
        h = 0.28
        for e in events:
            x0, w = e["start"], e["dur"]
            if e["kind"] == "predict":  # predictor 时间
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor="white", edgecolor="k",
                                         hatch="xx", alpha=0.9)
            else:
                color = label2color.get(e["label"], (0.7,0.7,0.7,1.0))
                if e["kind"] == "prewarm":
                    rect = patches.Rectangle((x0, y - h/2), w, h,
                                             facecolor=color, edgecolor="k",
                                             hatch="//", alpha=0.85)
                else:  # run
                    rect = patches.Rectangle((x0, y - h/2), w, h,
                                             facecolor=color, edgecolor="k",
                                             alpha=0.95)
            ax.add_patch(rect)

    draw_events(events_A, y=1.0)
    draw_events(events_B, y=0.0)

    # 图例（电路 Top-K + 两种纹理）
    handles = []
    for lab, _cnt in freq.most_common(legend_topk):
        handles.append(patches.Patch(facecolor=label2color[lab], edgecolor="k", label=lab))
    handles.append(patches.Patch(facecolor="white", edgecolor="k", hatch="//", label="prewarm compile"))
    handles.append(patches.Patch(facecolor="white", edgecolor="k", hatch="xx", label="predictor scoring"))
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=8, frameon=True)

    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"[save] timeline -> {out_png}")

# ------------------------ 绘图 2：Frequency & Hit rate Bars --
def plot_freq_hitrate_bars(freq_by_label: Dict[str,int],
                           hitrate_by_label: Dict[str,float],
                           overall_hit_rate: float,
                           out_png: str,
                           title: str,
                           top_k: Optional[int] = None) -> None:
    labels = list(freq_by_label.keys())
    labels.sort(key=lambda L: freq_by_label[L], reverse=True)
    if top_k is not None and top_k > 0 and len(labels) > top_k:
        labels = labels[:top_k]

    freq_vals = [freq_by_label[L] for L in labels]
    hit_vals = [100.0 * hitrate_by_label.get(L, 0.0) for L in labels]  # %
    x = np.arange(len(labels)); w = 0.4

    fig, ax1 = plt.subplots(figsize=(max(10, 0.5*len(labels)+6), 6))
    bars1 = ax1.bar(x - w/2, freq_vals, width=w, label="Frequency (count)")
    ax1.set_ylabel("Frequency (count)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, hit_vals, width=w, label="Hit rate (%)", color="#ff7f0e")
    ax2.set_ylabel("Hit rate (%)"); ax2.set_ylim(0, 105)

    overall_lbl = f"Overall hit rate: {overall_hit_rate*100:.1f}%"
    ax1.legend(handles=[bars1, bars2], labels=["Frequency (count)", "Hit rate (%)"],
               loc="upper right", frameon=True)
    ax2.legend([plt.Line2D([0],[0], color='none')], [overall_lbl],
               loc="upper left", frameon=False)

    plt.title(title, loc="left", fontsize=12, fontweight="bold")
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"[save] bar chart -> {out_png}")

# ------------------------ 主流程 ---------------------------
def label_of(name: str, q: int, d: int) -> str:
    return f"{name}_q{q}_d{d}"

def drive_and_plot(
    q_list: List[int], d_list: List[int], workload_len: int, prewarm_every: int,
    shots: int, lookahead_sec: float, prob_th: float, max_compile: int,
    sliding_window_sec: float, min_samples: int,
    hot_fraction: float, hot_boost: float, rps: float, rng_seed: int,
    out_timeline: str, out_bars: str, top_k: Optional[int] = None
):
    # 1) 候选与 workload
    makers_all, meta = build_catalog(q_list, d_list)
    workload = build_workload_poisson_superposition(meta, workload_len, hot_fraction, hot_boost, rps, rng_seed)

    # 2) 预喂 recent_calls，确保首轮能预测
    predictor = PoissonPredictor(sliding_window_sec=sliding_window_sec, min_samples=min_samples)
    seed_recent_calls_for_predictor(sliding_window_sec, makers_all, workload,
                                    seed_keys=4, per_key_samples=max(2, min_samples))

    # 3) 统计容器（frequency & hit）
    lbl = lambda it: label_of(it["name"], it["q"], it["d"])
    freq_by_label: Dict[str,int] = Counter([lbl(it) for it in workload])
    hit_by_label: Dict[str,int] = Counter()
    total_hits = 0

    # 4) 时间线事件 & 运行
    clear_idle()
    events_A: List[Dict[str, Any]] = []
    events_B: List[Dict[str, Any]] = []

    tA = 0.0
    run_count = 0
    while run_count < len(workload):
        # --- 方法 A：预热阶段（包含 predictor 计时） ---
        if (run_count % prewarm_every) == 0:
            decisions, predict_sec = prewarm_with_measured_predictor(
                predictor=predictor, candidates=makers_all,
                lookahead_sec=lookahead_sec, prob_threshold=prob_th, max_compile=max_compile
            )
            # predictor scoring 时间条
            if predict_sec > 0:
                events_A.append({"kind":"predict","label":"__predictor__","start":tA,"dur":predict_sec})
                tA += predict_sec
            # 编译时间条
            for it in decisions:
                if it.get("action") == "compile":
                    plabel = label_of(it["circ"], it["q"], it["d"])
                    dur = float(it.get("compile_sec", 0.0))
                    if dur > 0:
                        events_A.append({"kind":"prewarm","label":plabel,"start":tA,"dur":dur})
                        tA += dur

        # --- 方法 A：run 单条 ---
        item = workload[run_count]
        meta_run = run_once_with_cache(item["maker_run"], shots=shots)
        run_dur = float(meta_run["compile_sec"]) + float(meta_run["exec_sec"])
        events_A.append({"kind":"run", "label": lbl(item), "start": tA, "dur": run_dur})
        tA += run_dur

        # 命中统计
        if meta_run["cache_hit"]:
            total_hits += 1
            hit_by_label[lbl(item)] += 1

        # --- 方法 B：基线（同步推进一条） ---
        meta_base = run_once_nocache(item["maker_run"], shots=shots)
        base_dur = float(meta_base["compile_sec"]) + float(meta_base["exec_sec"])
        startB = (events_B[-1]["start"] + events_B[-1]["dur"]) if events_B else 0.0
        events_B.append({"kind":"run","label":lbl(item),"start":startB,"dur":base_dur})

        run_count += 1

    # 5) 命中率统计与作图
    hitrate_by_label: Dict[str,float] = {}
    for L, freq in freq_by_label.items():
        hits = hit_by_label.get(L, 0)
        hitrate_by_label[L] = (hits / freq) if freq > 0 else 0.0
    overall_hit_rate = total_hits / max(1, len(workload))

    title_tl = (f"Timeline — prewarm_every={prewarm_every}, lookahead={lookahead_sec}s, "
                f"prob_th={prob_th}, max_compile={max_compile}, sliding_window={sliding_window_sec}s; "
                f"q={q_list}, d={d_list}, N={workload_len}")
    draw_timeline_png(events_A, events_B, out_png=out_timeline, title=title_tl)

    title_bar = (f"Workload Frequency & Hit Rate (prewarm method) — "
                 f"Overall hit rate={overall_hit_rate*100:.1f}%")
    plot_freq_hitrate_bars(freq_by_label, hitrate_by_label, overall_hit_rate,
                           out_png=out_bars, title=title_bar, top_k=top_k)

# ------------------------ CLI ------------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 参数空间与 workload
    ap.add_argument("--q_list", type=str, default="5,11")
    ap.add_argument("--d_list", type=str, default="4,8")
    ap.add_argument("--workload_len", type=int, default=80)
    ap.add_argument("--prewarm_every", type=int, default=5)
    ap.add_argument("--shots", type=int, default=256)
    # predictor / prewarm
    ap.add_argument("--lookahead", type=float, default=8.0)
    ap.add_argument("--prob_th", type=float, default=0.45)
    ap.add_argument("--max_compile", type=int, default=3)
    ap.add_argument("--sliding_window_sec", type=float, default=60.0)
    ap.add_argument("--min_samples", type=int, default=2)
    # workload 形状
    ap.add_argument("--hot_fraction", type=float, default=0.25)
    ap.add_argument("--hot_boost", type=float, default=8.0)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--rng_seed", type=int, default=123)
    # 输出
    ap.add_argument("--out_timeline", type=str, default=str(FIGDIR / "timeline.png"))
    ap.add_argument("--out_bars", type=str, default=str(FIGDIR / "freq_hitrate.png"))
    ap.add_argument("--top_k", type=int, default=0, help="Show top-K circuits by frequency (0=all)")
    args = ap.parse_args()

    drive_and_plot(
        q_list=parse_int_list(args.q_list),
        d_list=parse_int_list(args.d_list),
        workload_len=args.workload_len,
        prewarm_every=args.prewarm_every,
        shots=args.shots,
        lookahead_sec=args.lookahead,
        prob_th=args.prob_th,
        max_compile=args.max_compile,
        sliding_window_sec=args.sliding_window_sec,
        min_samples=args.min_samples,
        hot_fraction=args.hot_fraction,
        hot_boost=args.hot_boost,
        rps=args.rps,
        rng_seed=args.rng_seed,
        out_timeline=args.out_timeline,
        out_bars=args.out_bars,
        top_k=(None if args.top_k is None or args.top_k <= 0 else args.top_k),
    )
