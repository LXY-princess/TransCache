# v16_fourpanel_viz.py
# ------------------------------------------------------------
# Build a Poisson-style workload, seed predictor, prewarm every K runs,
# run the workload, collect detailed logs, and draw a four-panel PNG:
# A) flowchart, B) timeline, C) heatmap (p by circuit×batch),
# D) aggregated funnel (candidates -> passed -> compiled -> later hits).
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

# ------------------------ 路径/缓存 ------------------------
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

# ------------------------ 预测器 ---------------------------
class PoissonPredictor:
    """
    λ: 最近 T 秒内计数 N 与跨度 ΔT 的比值（MLE）。
    p(≤τ) = 1 - exp(-λ τ)
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
        T = max(1e-9, max(ts) - min(ts))
        N = len(ts)
        return N / T if T > 0 else 0.0

    @staticmethod
    def prob_within(lam: float, tau: float) -> float:
        if lam <= 0 or tau <= 0:
            return 0.0
        return 1.0 - math.exp(-lam * tau)

    def score_candidates(
        self,
        candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
        lookahead_sec: float,
        prob_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        now = time.time()
        scored = []
        for maker in candidates:
            qc, key, info = maker()  # 生成电路拿到 md5 key（不编译）
            lam = self.est_lambda(key, now)
            p = self.prob_within(lam, lookahead_sec)
            if p >= prob_threshold:
                scored.append({
                    "key": key, "prob": p, "lambda": lam,
                    "qc_raw": qc, "info": info
                })
        scored.sort(key=lambda r: r["prob"], reverse=True)
        return scored

# ------------------------ 预热 ------------------------------
def prewarm_from_predictions(
    predictor: PoissonPredictor,
    candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    lookahead_sec: float,
    prob_threshold: float,
    max_compile: int = 1,
) -> List[Dict[str, Any]]:
    cache = load_idle()
    decided = predictor.score_candidates(candidates, lookahead_sec, prob_threshold)
    done: List[Dict[str, Any]] = []
    compiled = 0
    for item in decided:
        if compiled >= max_compile:
            break
        key = item["key"]
        if key in cache:
            done.append({"key": key, "action": "skip_in_idle", **item})
            continue
        qc_raw = item["qc_raw"]
        tp_kwargs = _prepare_kwargs()
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **tp_kwargs)
        cost = time.perf_counter() - t0
        cache[key] = qc_exec
        compiled += 1
        done.append({
            "key": key, "action": "compile",
            "prob": item["prob"], "lambda": item["lambda"],
            "compile_sec": cost,
            "n_qubits": qc_raw.num_qubits,
            "depth": qc_raw.depth(),
            "circ": item["info"].get("circ"),
            "q": item["info"].get("q"),
            "d": item["info"].get("d"),
        })
    if compiled > 0:
        save_idle(cache)
    return done

# ------------------------ 运行路径 --------------------------
def compile_with_idle_cache(qc_raw: QuantumCircuit, bk_name: str
) -> Tuple[QuantumCircuit, str, bool, float]:
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
    cache = load_idle()
    qc_exec = cache.get(key)
    was_hit = qc_exec is not None
    compile_sec = 0.0
    if not was_hit:
        tp_kwargs = _prepare_kwargs()
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **tp_kwargs)
        compile_sec = time.perf_counter() - t0
        cache[key] = qc_exec
        save_idle(cache)
    return qc_exec, key, was_hit, compile_sec

def run_once(qc_func: Callable[[], QuantumCircuit], shots: int = 512) -> Dict[str, Any]:
    qc_raw = qc_func()
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk = "AerSV"
    qc_exec, key, was_hit, compile_sec = compile_with_idle_cache(qc_raw, bk)

    job = sampler.run([qc_exec], shots=shots)
    _ = job.result()  # 结果在本可视化中非关键

    record_arrival(key)
    return {
        "key": key, "cache_hit": was_hit, "compile_sec": compile_sec,
        "n_qubits": qc_raw.num_qubits, "depth_in": qc_raw.depth(),
        "depth_transpiled": qc_exec.depth(), "size_transpiled": qc_exec.size()
    }

# ------------------------ 候选构建（多 q/d） ----------------
def build_catalog(q_list: List[int], d_list: List[int]):
    makers: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]] = []
    meta: List[Dict[str, Any]] = []
    key2label: Dict[str, str] = {}

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
                # 运行时 maker（只返回 qc）
                def _mk_run(make_=make, q_=q, d_=d):
                    return lambda: make_(q_, d_)
                meta.append({"name": name, "q": q, "d": d, "maker_run": _mk_run()})
    # 预先生成一次 key2label（仅用于标注，不编译）
    for mk in makers:
        qc, key, info = mk()
        key2label[key] = f"{info['circ']}_q{info['q']}_d{info['d']}"
    return makers, meta, key2label

# ------------------------ Workload（泊松叠加） ---------------
def build_workload_poisson_superposition(
    meta: List[Dict[str, Any]],
    workload_len: int,
    hot_fraction: float = 0.25,
    hot_boost: float = 8.0,
    rps: float = 1.0,
    rng_seed: int = 123,
) -> List[Dict[str, Any]]:
    assert workload_len > 0
    rng = random.Random(rng_seed)
    H = workload_len / max(1e-9, rps)  # 时间地平线
    m = len(meta)
    if m == 0:
        return []
    weights = np.ones(m, dtype=float)
    hot_k = max(1, int(round(hot_fraction * m)))
    for i in rng.sample(range(m), hot_k):
        weights[i] *= max(1.0, hot_boost)
    w_sum = float(weights.sum())
    lambdas = (weights / w_sum) * rps

    events: List[Tuple[float, int]] = []
    for i, lam in enumerate(lambdas):
        if lam <= 0: continue
        t = 0.0
        while t < H and len(events) < workload_len * 5:
            dt = rng.expovariate(lam)
            t += dt
            if t <= H:
                events.append((t, i))
    events.sort(key=lambda x: x[0])
    if len(events) > workload_len:
        events = events[:workload_len]
    elif len(events) < workload_len:
        probs = (weights / w_sum).tolist()
        need = workload_len - len(events)
        extra = rng.choices(range(m), weights=probs, k=need)
        for j, i in enumerate(extra):
            events.append((H + (j+1)*1e-3, i))
        events.sort(key=lambda x: x[0])

    workload = []
    for _, i in events:
        workload.append({
            "name": meta[i]["name"],
            "q": meta[i]["q"], "d": meta[i]["d"],
            "maker_run": meta[i]["maker_run"]
        })
    return workload

# ------------------------ 预喂 recent_calls ----------------
def seed_recent_calls_for_predictor(
    predictor_window_sec: float,
    makers_all: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    workload: List[Dict[str, Any]],
    seed_keys: int = 4, per_key_samples: int = 2, spacing_sec: float = 3.0
) -> List[Dict[str, Any]]:
    prefix = workload[:max(1, min(64, len(workload)//2))]
    counts: Dict[Tuple[str,int,int], int] = defaultdict(int)
    for it in prefix:
        counts[(it["name"], it["q"], it["d"])] += 1
    popular = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:seed_keys]

    seeded: List[Dict[str, Any]] = []
    now = time.time()
    horizon_lo = now - predictor_window_sec * 0.8
    for (name, q, d), _cnt in popular:
        hit_key = None
        for mk in makers_all:
            qc, key, info = mk()
            if info["circ"] == name and info["q"] == q and info["d"] == d:
                hit_key = key; break
        if hit_key is None: continue
        for j in range(per_key_samples):
            ts = horizon_lo + (j + 1) * spacing_sec
            record_arrival(hit_key, ts=ts)
        seeded.append({"key": hit_key, "name": name, "q": q, "d": d,
                       "samples": per_key_samples, "spacing": spacing_sec})
    return seeded

# ------------------------ 计算 p(所有候选) -----------------
def compute_p_for_all_candidates(
    predictor: PoissonPredictor,
    candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    lookahead_sec: float
) -> Dict[str, Tuple[float, Dict[str, Any]]]:
    """返回 {key: (p, info)}，用于热力图。"""
    now = time.time()
    out = {}
    for mk in candidates:
        qc, key, info = mk()
        lam = predictor.est_lambda(key, now)
        p = predictor.prob_within(lam, lookahead_sec)
        out[key] = (p, info)
    return out

# ------------------------ 日志容器 --------------------------
class VizLogger:
    def __init__(self):
        self.prewarm_batches: List[Dict[str, Any]] = []
        self.run_events: List[Dict[str, Any]] = []
        self.event_idx = 0   # 时间轴 x

    def log_prewarm_batch(self, batch_id: int, p_all: Dict[str, Tuple[float, Dict[str, Any]]],
                          passed_keys: List[str], decisions: List[Dict[str, Any]]):
        self.prewarm_batches.append({
            "batch_id": batch_id, "p_all": p_all,
            "passed_keys": passed_keys, "decisions": decisions,
            "event_idx": self.event_idx
        })
        self.event_idx += 1

    def log_run(self, run_id: int, name: str, q: int, d: int,
                meta: Dict[str, Any]):
        evt = {
            "run_id": run_id, "name": name, "q": q, "d": d,
            "key": meta["key"], "cache_hit": bool(meta["cache_hit"]),
            "compile_sec": float(meta["compile_sec"]),
            "depth_in": int(meta["depth_in"]),
            "depth_transpiled": int(meta["depth_transpiled"]),
            "event_idx": self.event_idx
        }
        self.run_events.append(evt)
        self.event_idx += 1

# ------------------------ 主流程（含绘图） -------------------
def drive_and_plot(
    q_list: List[int], d_list: List[int], workload_len: int, prewarm_every: int,
    shots: int, lookahead_sec: float, prob_th: float, max_compile: int,
    sliding_window_sec: float, min_samples: int,
    hot_fraction: float, hot_boost: float, rps: float, rng_seed: int,
    out_png: str, heat_topk: int = 15
):
    # 0) 准备
    makers_all, meta, key2label = build_catalog(q_list, d_list)
    predictor = PoissonPredictor(sliding_window_sec=sliding_window_sec, min_samples=min_samples)
    workload = build_workload_poisson_superposition(
        meta=meta, workload_len=workload_len,
        hot_fraction=hot_fraction, hot_boost=hot_boost, rps=rps, rng_seed=rng_seed
    )
    seeded = seed_recent_calls_for_predictor(sliding_window_sec, makers_all, workload,
                                             seed_keys=4, per_key_samples=max(2, min_samples))
    logger = VizLogger()

    # 1) 运行：每 K 次 run 之前 prewarm 一次（第 0 次 run 前也 prewarm）
    run_count = 0
    batch_id = 0
    while run_count < len(workload):
        if (run_count % prewarm_every) == 0:
            # 先计算 p_all（用于热力图与统计）
            p_all = compute_p_for_all_candidates(predictor, makers_all, lookahead_sec)
            passed_keys = [k for k,(p,_info) in p_all.items() if p >= prob_th]
            decisions = prewarm_from_predictions(
                predictor, makers_all, lookahead_sec, prob_th, max_compile=max_compile
            )
            logger.log_prewarm_batch(batch_id, p_all, passed_keys, decisions)
            batch_id += 1

        # run 单条
        item = workload[run_count]
        name, q, d = item["name"], item["q"], item["d"]
        meta_run = run_once(item["maker_run"], shots=shots)
        logger.log_run(run_count, name, q, d, meta_run)
        run_count += 1

    # 2) 汇总统计（用于 D 面板）
    num_batches = len(logger.prewarm_batches)
    candidates_per_batch = len(makers_all)
    total_candidates = candidates_per_batch * num_batches
    total_passed = sum(len(b["passed_keys"]) for b in logger.prewarm_batches)
    total_compiled = sum(sum(1 for d in b["decisions"] if d.get("action") == "compile")
                         for b in logger.prewarm_batches)

    # 对“后续命中”：找出被编译的 key，在随后的 run 中第一次命中计入
    compiled_batches: List[Tuple[int, str]] = []
    for b in logger.prewarm_batches:
        for d in b["decisions"]:
            if d.get("action") == "compile":
                compiled_batches.append((b["batch_id"], d["key"]))
    compiled_keys = [k for _bid, k in compiled_batches]
    later_hits = set()
    seen_hit_pair = set()
    for evt in logger.run_events:
        if evt["cache_hit"] and evt["key"] in compiled_keys:
            # 只计一次
            pair = evt["key"]
            if pair not in seen_hit_pair:
                later_hits.add(pair)
                seen_hit_pair.add(pair)
    total_later_hits = len(later_hits)

    # 3) 画图
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1], width_ratios=[1,1], hspace=0.3, wspace=0.25)

    # --- A) Flowchart ---
    axA = fig.add_subplot(gs[0,0])
    axA.axis('off')
    # 位置参数
    boxes = {
        "workload": (0.05, 0.65, 0.35, 0.25, "Workload Driver\n(Poisson Superposition)"),
        "predictor": (0.60, 0.70, 0.35, 0.20, "PoissonPredictor\n(λ MLE & p=1-e^{-λτ})"),
        "prewarm": (0.60, 0.40, 0.35, 0.20, "prewarm_from_predictions\n(filter / sort / compile)"),
        "idle": (0.35, 0.05, 0.30, 0.20, "Idle Cache\n(compiled circuits)"),
        "run": (0.05, 0.25, 0.35, 0.25, "run_once\n(hit / miss→compile&write)"),
        "recent": (0.70, 0.05, 0.25, 0.20, "RECENT_CALLS\n(update arrivals)"),
    }
    rects = {}
    for k,(x,y,w,h,label) in boxes.items():
        rect = patches.FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.02", linewidth=1.5)
        axA.add_patch(rect)
        axA.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
        rects[k] = rect
    def center(k):
        x,y,w,h,_ = boxes[k]; return (x+w/2, y+h/2)
    def arrow(p, q):
        axA.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.5))
    arrow(center("workload"), center("predictor"))
    arrow(center("predictor"), center("prewarm"))
    arrow(center("prewarm"), center("idle"))
    arrow(center("workload"), center("run"))
    arrow(center("idle"), center("run"))
    arrow(center("run"), center("recent"))
    arrow(center("recent"), center("predictor"))
    axA.set_title("A｜Flowchart Overview", loc="left", fontsize=12, fontweight='bold')

    # --- B) Timeline ---
    axB = fig.add_subplot(gs[0,1])
    # 预热事件（compile）
    pw_x, pw_y, pw_p = [], [], []
    for b in logger.prewarm_batches:
        x0 = b["event_idx"]
        for d in b["decisions"]:
            if d.get("action") == "compile":
                pw_x.append(x0)
                pw_y.append(1.0)
                pw_p.append(d.get("prob", 0.0))
    sc1 = axB.scatter(pw_x, pw_y, marker="^", s=80, c=pw_p, cmap="viridis", label="prewarm compile (p)", zorder=3)

    # 运行事件
    run_x_hit, run_y_hit = [], []
    run_x_miss, run_y_miss, miss_text = [], [], []
    for evt in logger.run_events:
        if evt["cache_hit"]:
            run_x_hit.append(evt["event_idx"]); run_y_hit.append(0.0)
        else:
            run_x_miss.append(evt["event_idx"]); run_y_miss.append(0.0)
            miss_text.append(f"{evt['compile_sec']:.2f}s")
    axB.scatter(run_x_hit, run_y_hit, marker="o", s=40, label="run hit", zorder=2)
    axB.scatter(run_x_miss, run_y_miss, marker="o", s=40, facecolors='none', edgecolors='k', label="run miss", zorder=2)
    # 标注 miss 的编译时长
    for x,y,txt in zip(run_x_miss, run_y_miss, miss_text):
        axB.text(x, y-0.08, txt, ha='center', va='top', fontsize=8, rotation=0)

    axB.set_yticks([0, 1]); axB.set_yticklabels(["run", "prewarm"])
    axB.set_xlabel("event index (prewarm / run sequence)")
    axB.set_title("B｜Timeline of Prewarm & Run", loc="left", fontsize=12, fontweight='bold')
    cb = plt.colorbar(sc1, ax=axB, fraction=0.05, pad=0.02); cb.set_label("p")

    # --- C) Heatmap (Top-K circuits × batches) ---
    axC = fig.add_subplot(gs[1,0])
    # 选择 Top-K（按 workload 频次）
    freq = Counter([f"{e['name']}_q{e['q']}_d{e['d']}" for e in workload])
    top_labels = [lbl for lbl,_cnt in freq.most_common(heat_topk)]
    # 建立 label->keys（有的 label 可能对应多个 q/d 组合的 md5，但这里 label 定义唯一）
    label2key = {}
    for k,label in key2label.items():
        if label in top_labels and label not in label2key:
            label2key[label] = k
    # 构建矩阵（行=label，列=batch），默认 NaN
    B = len(logger.prewarm_batches)
    R = len(top_labels)
    mat = np.full((R, B), np.nan, dtype=float)
    compiled_marks: Dict[Tuple[int,int], bool] = {}
    for j,b in enumerate(logger.prewarm_batches):
        p_all: Dict[str, Tuple[float, Dict[str, Any]]] = b["p_all"]
        # p 值填入
        for i,label in enumerate(top_labels):
            k = label2key.get(label)
            if k is None: continue
            if k in p_all:
                mat[i, j] = p_all[k][0]
        # 编译标记
        for d in b["decisions"]:
            if d.get("action") == "compile":
                lbl = f"{d['circ']}_q{d['q']}_d{d['d']}"
                if lbl in top_labels:
                    i = top_labels.index(lbl)
                    compiled_marks[(i,j)] = True

    im = axC.imshow(mat, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    axC.set_yticks(range(R)); axC.set_yticklabels(top_labels, fontsize=8)
    axC.set_xticks(range(B)); axC.set_xticklabels([f"batch{j}" for j in range(B)], fontsize=8)
    axC.set_title("C｜Heatmap of p by Circuit × Prewarm Batch", loc="left", fontsize=12, fontweight='bold')
    cb2 = plt.colorbar(im, ax=axC, fraction=0.046, pad=0.02); cb2.set_label("p")
    # 叠加 ✓ 标记编译
    for (i,j),_ in compiled_marks.items():
        axC.text(j, i, "✓", ha='center', va='center', fontsize=10, fontweight='bold')

    # --- D) Aggregated Funnel (Bars) ---
    axD = fig.add_subplot(gs[1,1])
    stages = ["candidates", "passed", "compiled", "later_hit"]
    vals = [total_candidates, total_passed, total_compiled, total_later_hits]
    bars = axD.bar(range(len(stages)), vals)
    axD.set_xticks(range(len(stages))); axD.set_xticklabels(stages)
    axD.set_title("D｜Aggregated Funnel Across Prewarm Batches", loc="left", fontsize=12, fontweight='bold')
    # 在柱上标数值与相对占比
    def pct(a,b):
        return f"{(100.0*a/max(1,b)):.1f}%" if b>0 else "-"
    for i,b in enumerate(bars):
        h = b.get_height()
        # 相对上一阶段的比例
        ratio = pct(h, vals[i-1]) if i>0 else ""
        axD.text(b.get_x()+b.get_width()/2, h, f"{int(h)} {ratio}", ha='center', va='bottom', fontsize=10)

    # 全局标题
    fig.suptitle(
        "Prewarm & Run Visualization (Poisson-driven workload)\n"
        f"lookahead={lookahead_sec}s, prob_th={prob_th}, max_compile={max_compile}, "
        f"sliding_window={sliding_window_sec}s, prewarm_every={prewarm_every}, "
        f"q={q_list}, d={d_list}, workload_len={workload_len}",
        fontsize=12, y=0.99
    )
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    print(f"[save] {out_png}")

# ------------------------ CLI ------------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 参数空间 / workload
    ap.add_argument("--q_list", type=str, default="5,11")
    ap.add_argument("--d_list", type=str, default="4,8")
    ap.add_argument("--workload_len", type=int, default=60)
    ap.add_argument("--prewarm_every", type=int, default=2)
    ap.add_argument("--shots", type=int, default=256)

    # predictor / prewarm
    ap.add_argument("--lookahead", type=float, default=8.0)
    ap.add_argument("--prob_th", type=float, default=0.45)
    ap.add_argument("--max_compile", type=int, default=2)
    ap.add_argument("--sliding_window_sec", type=float, default=60.0)
    ap.add_argument("--min_samples", type=int, default=2)

    # workload 形状（泊松叠加）
    ap.add_argument("--hot_fraction", type=float, default=0.25)
    ap.add_argument("--hot_boost", type=float, default=8.0)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--rng_seed", type=int, default=123)

    # 输出
    ap.add_argument("--out", type=str, default=str(FIGDIR / "fourpanel.png"))
    ap.add_argument("--heat_topk", type=int, default=15)

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
        heat_topk=args.heat_topk,
    )
