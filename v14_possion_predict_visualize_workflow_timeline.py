# v17_latency_timeline.py
# ------------------------------------------------------------
# Stacked-latency timeline PNG comparing:
#   A) Prewarm + idle-cache (predictor-driven)
#   B) No prewarm, no caching (always transpile)
#
# - Build Poisson-style workload across multiple (q,d) × circuits
# - Seed RECENT_CALLS before first prewarm
# - Prewarm every K runs, run full workload
# - Measure:
#     * prewarm compile time (per compiled circuit)
#     * per-run total latency = (compile_if_any + execute_sampling)
# - Draw a two-row timeline (y: methods; x: cumulative time), bars colored by circuit.
# ------------------------------------------------------------

import time, math, hashlib, pickle, pathlib, argparse, warnings, random
from collections import defaultdict, deque, Counter
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# ------------------------ 预测器（沿用 v14/v15 思路） -------
class PoissonPredictor:
    """
    λ: 最近 T 秒内计数 N 与跨度 ΔT 的比值（MLE）；p(≤τ)=1-exp(-λτ)
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
            qc, key, info = maker()  # 生成电路以拿 md5 key（不做编译）
            lam = self.est_lambda(key, now)
            p = self.prob_within(lam, lookahead_sec)
            if p >= prob_threshold:
                scored.append({
                    "key": key, "prob": p, "lambda": lam,
                    "qc_raw": qc, "info": info
                })
        scored.sort(key=lambda r: r["prob"], reverse=True)
        return scored

# ------------------------ 预热（方法 A） --------------------
def prewarm_from_predictions(
    predictor: PoissonPredictor,
    candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    lookahead_sec: float, prob_threshold: float, max_compile: int = 1
) -> List[Dict[str, Any]]:
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

# ------------------------ 运行路径（方法 A） ----------------
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
        "key": key, "label": f"{qc_raw.name if qc_raw.name else 'circ'}_{qc_raw.num_qubits}_{qc_raw.depth()}",
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
) -> List[Dict[str, Any]]:
    prefix = workload[:max(1, min(64, len(workload)//2))]
    counts: Dict[Tuple[str,int,int], int] = defaultdict(int)
    for it in prefix:
        counts[(it["name"], it["q"], it["d"])] += 1
    popular = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:seed_keys]
    seeded: List[Dict[str, Any]] = []
    now = time.time(); horizon_lo = now - predictor_window_sec * 0.8
    for (name, q, d), _ in popular:
        hit_key = None
        for mk in makers_all:
            qc, key, info = mk()
            if info["circ"] == name and info["q"] == q and info["d"] == d:
                hit_key = key; break
        if hit_key is None: continue
        for j in range(per_key_samples):
            record_arrival(hit_key, ts=horizon_lo + (j+1)*spacing_sec)
        seeded.append({"key": hit_key, "name": name, "q": q, "d": d})
    return seeded

# ------------------------ 绘图：堆叠 latency 时间线 ----------
def draw_timeline_png(events_A: List[Dict[str, Any]], events_B: List[Dict[str, Any]],
                      out_png: str, title: str, legend_topk: int = 14) -> None:
    """
    events_*: 序列中的每个元素包含
      { "kind": "prewarm"|"run", "label": "circ_q_d", "start": s, "dur": t }
    """
    # 颜色映射（按出现频率取前 legend_topk 放图例）
    all_labels = [e["label"] for e in events_A + events_B if "label" in e]
    freq = Counter(all_labels)
    uniq_labels = list(freq.keys())
    cmap = plt.cm.get_cmap("tab20", max(20, len(uniq_labels)))
    label2color = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq_labels)}

    # 计算时间范围
    T_A = max((e["start"] + e["dur"] for e in events_A), default=0.0)
    T_B = max((e["start"] + e["dur"] for e in events_B), default=0.0)
    T = max(T_A, T_B) * 1.05

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, T)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Method B: No prewarm & no cache", "Method A: Predictor prewarm + idle cache"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")

    # 绘制辅助函数
    def draw_events(events: List[Dict[str, Any]], y: float):
        h = 0.32
        for e in events:
            x0, w = e["start"], e["dur"]
            color = label2color.get(e["label"], (0.7,0.7,0.7,1.0))
            if e["kind"] == "prewarm":
                rect = patches.Rectangle((x0, y - h/2), w, h, facecolor=color, edgecolor="k", hatch="//", alpha=0.8)
            else:  # run
                rect = patches.Rectangle((x0, y - h/2), w, h, facecolor=color, edgecolor="k", alpha=0.9)
            ax.add_patch(rect)

    draw_events(events_A, y=1.0)
    draw_events(events_B, y=0.0)

    # 图例（前 legend_topk 个电路）
    handles = []
    for lab, _cnt in freq.most_common(legend_topk):
        patch = patches.Patch(facecolor=label2color[lab], edgecolor="k", label=lab)
        handles.append(patch)
    # 再加一个预热纹理示例
    handles.append(patches.Patch(facecolor="white", edgecolor="k", hatch="//", label="prewarm (hatch)"))
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=8, frameon=True)

    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"[save] timeline -> {out_png}")

# ------------------------ 主流程 ---------------------------
def drive_and_plot(
    q_list: List[int], d_list: List[int], workload_len: int, prewarm_every: int,
    shots: int, lookahead_sec: float, prob_th: float, max_compile: int,
    sliding_window_sec: float, min_samples: int,
    hot_fraction: float, hot_boost: float, rps: float, rng_seed: int,
    out_png: str
):
    # 准备候选与 workload
    makers_all, meta = build_catalog(q_list, d_list)
    predictor = PoissonPredictor(sliding_window_sec=sliding_window_sec, min_samples=min_samples)
    workload = build_workload_poisson_superposition(meta, workload_len, hot_fraction, hot_boost, rps, rng_seed)

    # 预喂 recent_calls（方法 A 用）
    seed_recent_calls_for_predictor(sliding_window_sec, makers_all, workload,
                                    seed_keys=4, per_key_samples=max(2, min_samples))

    # 构造 label 生成器（统一颜色）
    def label_of(name: str, q: int, d: int) -> str:
        return f"{name}_q{q}_d{d}"

    # 方法 A：时间轴事件
    clear_idle()  # 确保干净的 idle cache
    events_A: List[Dict[str, Any]] = []
    tA = 0.0
    run_count = 0
    while run_count < len(workload):
        if (run_count % prewarm_every) == 0:
            # 预热（前置在当前时刻），每个编译生成一个 prewarm 事件
            decisions = prewarm_from_predictions(
                predictor=predictor, candidates=makers_all,
                lookahead_sec=lookahead_sec, prob_threshold=prob_th, max_compile=max_compile
            )
            for it in decisions:
                if it.get("action") == "compile":
                    lbl = label_of(it["circ"], it["q"], it["d"])
                    dur = float(it.get("compile_sec", 0.0))
                    if dur > 0:
                        events_A.append({"kind":"prewarm","label":lbl,"start":tA,"dur":dur})
                        tA += dur

        # run（一条）
        item = workload[run_count]
        lbl = label_of(item["name"], item["q"], item["d"])
        meta_run = run_once_with_cache(item["maker_run"], shots=shots)
        # run 的条形长度 = compile_if_miss + 执行时间
        dur = float(meta_run["compile_sec"]) + float(meta_run["exec_sec"])
        events_A.append({"kind":"run","label":lbl,"start":tA,"dur":dur})
        tA += dur
        run_count += 1

    # 方法 B：基线（无预热/缓存）
    events_B: List[Dict[str, Any]] = []
    tB = 0.0
    for item in workload:
        lbl = label_of(item["name"], item["q"], item["d"])
        meta_run = run_once_nocache(item["maker_run"], shots=shots)
        dur = float(meta_run["compile_sec"]) + float(meta_run["exec_sec"])
        events_B.append({"kind":"run","label":lbl,"start":tB,"dur":dur})
        tB += dur

    # 绘图
    title = (f"Stacked Latency Timeline — prewarm_every={prewarm_every}, "
             f"lookahead={lookahead_sec}s, prob_th={prob_th}, max_compile={max_compile}, "
             f"sliding_window={sliding_window_sec}s; q={q_list}, d={d_list}, N={workload_len}")
    draw_timeline_png(events_A, events_B, out_png=out_png, title=title)

# ------------------------ CLI ------------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 参数空间与 workload
    ap.add_argument("--q_list", type=str, default="5,11")
    ap.add_argument("--d_list", type=str, default="4,8")
    ap.add_argument("--workload_len", type=int, default=50)
    ap.add_argument("--prewarm_every", type=int, default=2)
    ap.add_argument("--shots", type=int, default=256)
    # predictor / prewarm
    ap.add_argument("--lookahead", type=float, default=8.0)
    ap.add_argument("--prob_th", type=float, default=0.45)
    ap.add_argument("--max_compile", type=int, default=2)
    ap.add_argument("--sliding_window_sec", type=float, default=60.0)
    ap.add_argument("--min_samples", type=int, default=2)
    # workload 形状
    ap.add_argument("--hot_fraction", type=float, default=0.25)
    ap.add_argument("--hot_boost", type=float, default=8.0)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--rng_seed", type=int, default=123)
    # 输出
    ap.add_argument("--out", type=str, default=str(FIGDIR / "latency_timeline.png"))
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
        out_png=args.out,
    )
