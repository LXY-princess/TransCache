# v15_core.py (sim-time ready)
import time, math, hashlib, json, pickle, pathlib, warnings, random
from typing import Any, Callable, Dict, List, Tuple, Optional
from collections import defaultdict, deque, Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from qiskit import QuantumCircuit, transpile, qasm3
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from matplotlib import cm, colors as mcolors
# circuits
from v11_quasa_bench_circuits import CIRCUITS_QUASA as CIRCUITS

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------- paths / cache ----------
VNUM = 18
ROOT = pathlib.Path("./figs")/f"v{VNUM}"
# EVENTS_DIR = ROOT/"events"
PLOT_DIR = ROOT/"plots"
# EVENTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOAD_ROOT = pathlib.Path("./figs")/f"v{VNUM}_e2e_latency_breakdown_wl150"
LOAD_EVENTS_DIR = LOAD_ROOT/"events"

def md5_qasm(circ: QuantumCircuit) -> str:
    try:
        txt = circ.qasm()
    except Exception:
        txt = qasm3.dumps(circ)
    return hashlib.md5(txt.encode()).hexdigest()

def _prepare_kwargs():
    aer = AerSimulator()
    # try:
    #     return {"target": aer.target, "optimization_level": 2, "seed_transpiler": 42}
    # except Exception:
    cfg = aer.configuration()
    return {"basis_gates": list(sorted(set((cfg.basis_gates or []) + ["measure"]))), "optimization_level": 2, "seed_transpiler": 42}

# --------- recent calls (for predictor) ----------
SLIDING_MAXLEN = 256
RECENT_CALLS: Dict[str, deque] = defaultdict(lambda: deque(maxlen=SLIDING_MAXLEN))
def record_arrival(key: str, ts: Optional[float] = None) -> None:
    """Append an arrival timestamp for `key` into RECENT_CALLS.

    If `ts` is provided, it is taken as the timestamp **in the caller's time base**
    (e.g., the simulated timeline `t`). Otherwise, wall clock (`time.time()`) is used.
    """
    RECENT_CALLS[key].append(float(ts) if ts is not None else time.time())
def clear_recent(): RECENT_CALLS.clear()

# --------- predictor ----------
class PoissonPredictor:
    """λ MLE on sliding window;  p(≤τ)=1-exp(-λτ)"""
    def __init__(self, sliding_window_sec: float = 60.0, min_samples: int = 2):
        # the sliding window length of RECENT_CALLS base on which to predict
        self.sliding_window_sec = float(sliding_window_sec)
        # the minimal number of samples neede to calculate lamda and prob
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

    def score_candidates(self, candidates, lookahead_sec: float,
                         prob_threshold: float = 0.5, now: Optional[float] = None):
        """IMPORTANT: pass `now` in the **same time base** as `record_arrival`."""
        now = float(now) if now is not None else time.time()
        out = []
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

def build_workload_poisson_superposition_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    hot_fraction: float = 0.25,
    hot_boost: float = 8.0,
    rps: float = 1.0,
    rng_seed: int = 123,
    return_timestamps: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    叠加泊松（superposition）精确生成：
      - 设总率 Λ = rps；按热点权重分解为各类 λ_i，∑λ_i = Λ；
      - 逐个事件：Δ ~ Exp(Λ), t += Δ；类别 ~ Categorical(p_i=λ_i/Λ)；
      - 重复 N=workload_len 次，得到恰好 N 个事件；时间窗 T_end 随机，更贴近真实。

    返回：
      workload: List[dict] 与你现有策略直接兼容（含 maker_run / 可选 t_arr）
      info: 记录 Λ、λ_i、类别概率、最终 T_end 等，便于可视化与检验
    """
    assert workload_len > 0 and len(meta) > 0
    py_rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    # 1) 构造热点权重 -> 归一化 -> λ_i
    m = len(meta)
    weights = np.ones(m, dtype=float)
    hot_k = max(1, int(round(hot_fraction * m)))
    for i in py_rng.sample(range(m), hot_k):
        weights[i] *= max(1.0, float(hot_boost))
    w_sum = float(weights.sum())
    if w_sum <= 0:
        weights[:] = 1.0
        w_sum = float(weights.sum())

    Λ = float(rps)
    lambdas = (weights / w_sum) * Λ           # 每类速率 λ_i
    probs   = (lambdas / max(1e-12, Λ))       # 分类用 p_i = λ_i / Λ

    # 2) 逐事件模拟：Δ ~ Exp(Λ)，类别 ~ Categorical(probs)
    t = 0.0
    events: List[Tuple[float, int]] = []
    for _ in range(workload_len):
        u = 1.0 - py_rng.random()
        delta = -math.log(max(1e-12, u)) / max(1e-12, Λ)  # Exp(Λ)
        t += delta
        i = int(np_rng.choice(m, p=probs))                 # 类别归属
        events.append((t, i))

    # 3) 组装 workload
    out: List[Dict[str, Any]] = []
    for t_i, idx in events:
        rec = {
            "name": meta[idx]["name"],
            "q":    meta[idx]["q"],
            "d":    meta[idx]["d"],
            "maker_run": meta[idx]["maker_run"],
        }
        if return_timestamps:
            rec["t_arr"] = float(t_i)  # 模拟到达时刻
        out.append(rec)

    info = {
        "Lambda": float(Λ),
        "lambdas": lambdas.tolist(),
        "probs": probs.tolist(),
        "weights": weights.tolist(),
        "hot_fraction": float(hot_fraction),
        "hot_boost": float(hot_boost),
        "rps": float(rps),
        "T_end": float(events[-1][0]) if events else 0.0,
        "N": int(workload_len),
    }
    return out, info


def visualize_superposed_poisson_exact(
    workload: List[Dict[str, Any]],
    info: Dict[str, Any],
    meta: List[Dict[str, Any]],
    bins: int = 40,
    topk_classes: int = 10
):
    """
    针对“superposition 精确生成”的可视化/健康检查：
      1) 全局到达间隔 Δt 直方图 ≈ Exp(Λ)（检验 total 过程）；
      2) N(t) 与期望 Λ t 的对比；
      3) 类别计数与 Multinomial(N; p_i=λ_i/Λ) 期望的对比（固定 N 条的条件下）；
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    Λ = float(info.get("Lambda", 1.0))
    probs = np.array(info.get("probs", []), dtype=float)
    N = int(info.get("N", len(workload)))

    # 取到达时刻
    ts = np.array([w["t_arr"] for w in workload], dtype=float)
    ts.sort()
    inter = np.diff(ts)  # Δt

    # 1) Δt 直方图 + 指数拟合曲线
    fig1, ax1 = plt.subplots(figsize=(6,4))
    if len(inter) > 0:
        ax1.hist(inter, bins=bins, density=True, alpha=0.7, label="empirical Δt")
        x = np.linspace(0, max(inter.max(), 1e-6), 200)
        ax1.plot(x, Λ*np.exp(-Λ*x), "r--", lw=2, label=f"Exp(Λ={Λ:.2f})")
    ax1.set_title("Inter-arrival histogram (should be ~Exp(Λ))")
    ax1.set_xlabel("Δt"); ax1.set_ylabel("density")
    ax1.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR/f"wl_{N}_inter_arrival.png", dpi=600)

    # 2) 计数过程 N(t) 与 E[N(t)]=Λ t
    fig2, ax2 = plt.subplots(figsize=(6,4))
    if len(ts) > 0:
        ax2.step(ts, np.arange(1, len(ts)+1), where="post", label="N(t)")
        tline = np.linspace(0, ts[-1], 200)
        ax2.plot(tline, Λ*tline, "r--", lw=2, label="E[N(t)] = Λ t")
    ax2.set_title("Counting process vs expectation")
    ax2.set_xlabel("t"); ax2.set_ylabel("N(t)")
    ax2.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR/f"wl_{N}_counting_process.png", dpi=600)

    # 3) 类别计数 vs Multinomial 期望
    if probs.size > 0:
        # 统计观测
        keys = [f'{w["name"]}|q{w["q"]}|d{w["d"]}' for w in workload]
        # 将 keys 映射回 meta 序号的方法：这里我们直接利用 meta 顺序的 probs 向量；
        # 统计每个 meta 索引的次数：用 name|q|d 定位到 meta 下标
        label2idx = {f'{m["name"]}|q{m["q"]}|d{m["d"]}': i for i, m in enumerate(meta)}
        counts = np.zeros(len(meta), dtype=int)
        for k in keys:
            i = label2idx.get(k, None)
            if i is not None:
                counts[i] += 1

        expect = N * probs
        # 展示 top-k
        idx_top = np.argsort(-counts)[:min(topk_classes, len(counts))]
        fig3, ax3 = plt.subplots(figsize=(max(7, 0.6*len(idx_top)+4), 4))
        x = np.arange(len(idx_top))
        ax3.bar(x - 0.2, counts[idx_top], width=0.4, label="observed")
        ax3.bar(x + 0.2, expect[idx_top], width=0.4, label="expected (N·p_i)")
        xt = [f'{meta[i]["name"]}|q{meta[i]["q"]}|d{meta[i]["d"]}' for i in idx_top]
        ax3.set_xticks(x); ax3.set_xticklabels(xt, rotation=45, ha="right")
        ax3.set_title("Per-class counts vs Multinomial expectation (fixed N)")
        ax3.set_ylabel("#events")
        ax3.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR/f"wl_{N}_counts_predict.png", dpi=600)
    # plt.show()

# --------- seeding predictor ----------
def seed_recent_calls_for_predictor(predictor_window_sec: float, makers_all, workload,
                                    seed_keys: int = 4, per_key_samples: int = 2, spacing_sec: float = 3.0,
                                    use_sim_time: bool = False, base_now: Optional[float] = None):
    """Seed RECENT_CALLS for most popular circuits. Set use_sim_time=True to seed on sim-time."""
    prefix = workload[:max(1, min(64, len(workload)//2))]
    counts: Dict[Tuple[str,int,int], int] = defaultdict(int)
    for it in prefix:
        counts[(it["name"], it["q"], it["d"])] += 1
    popular = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:seed_keys]
    now = float(base_now if (use_sim_time and base_now is not None) else (0.0 if use_sim_time else time.time()))
    horizon_lo = now - predictor_window_sec * 0.8
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
    t0 = time.perf_counter()
    qc_exec = cache.get(key)
    hit = qc_exec is not None
    compile_sec = time.perf_counter() - t0
    if not hit:
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        compile_sec = time.perf_counter() - t0
        cache[key] = qc_exec
    return qc_exec, key, hit, compile_sec

def run_once_with_cache(qc_func: Callable[[], QuantumCircuit],
                        cache: Dict[str, QuantumCircuit],
                        shots: int = 256,
                        ts: Optional[float] = None,
                        include_exec: bool = True) -> Dict[str, Any]:
    """Compile & run once. `ts` lets you record arrival in **sim-time** if provided."""
    qc_raw = qc_func()
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk = "AerSV"
    qc_exec, key, hit, compile_sec = compile_with_idle_cache(qc_raw, bk, cache)
    if include_exec:
        t0 = time.perf_counter()
        _ = sampler.run([qc_exec], shots=shots).result()
        exec_sec = time.perf_counter() - t0
    else:
        exec_sec = 0.0  # ← 不执行，置零
    record_arrival(key, ts=ts)
    return {"key": key, "cache_hit": hit, "compile_sec": compile_sec, "exec_sec": exec_sec,
            "n_qubits": qc_raw.num_qubits, "depth_in": qc_raw.depth(), "depthT": qc_exec.depth()}

def run_once_nocache(qc_func: Callable[[], QuantumCircuit], shots: int = 256, include_exec: bool = True) -> Dict[str, Any]:
    qc_raw = qc_func()
    t0 = time.perf_counter()
    qc_exec = transpile(qc_raw, **_prepare_kwargs())
    compile_sec = time.perf_counter() - t0

    if include_exec:
        method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
        sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
        t1 = time.perf_counter()
        _ = sampler.run([qc_exec], shots=shots).result()
        exec_sec = time.perf_counter() - t1
    else:
        exec_sec = 0.0
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
                           out_png: pathlib.Path, top_k: Optional[int] = None):
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
    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight"); print(f"[save] bars -> {out_png}")

def draw_timeline_run_only(method_events: Dict[str, List[Dict[str,Any]]],
                           out_png: pathlib.Path,
                           legend_topk: int = 16):
    """
    仅绘制 kind == 'run' 的事件块；y 轴方法标签用映射名；严格按 method_order 的顺序绘制。
    - method_order: 明确指定方法的上下顺序（从下到上绘制的顺序）。如果为 None，则使用 method_events 的键当前顺序。
    """
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})
    PRED_LABEL = "__predictor__"

    method_order = ["FullCompilation", "PR","FS", "FS+Pre+ttl+SE+ema",]
    # ----------- 仅统计 run 标签用于配色/图例 -----------
    run_labels = []
    for k in method_order:
        for e in method_events[k]:
            if e.get("kind", "run") == "run" and e.get("label") and e["label"] != PRED_LABEL:
                run_labels.append(e["label"])
    freq = Counter(run_labels)
    uniq = list(freq.keys())

    cmap = plt.cm.get_cmap("tab20", max(20, len(uniq)))
    label2color = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq)}

    # ----------- 画布 & 轴范围 -----------
    n = len(method_order)
    method_labels = {
        "FullCompilation": "FullComp",
        "PR": "Braket",
        "FS": "CCache",
        "FS+Pre+ttl+SE+ema": "TransCache",
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    # 全局时间上限
    T = 0.0
    for k in method_order:
        evs = method_events[k]
        T = max(T, max((e["start"] + e["dur"] for e in evs), default=0.0))
    ax.set_xlim(0, T * 1.05)
    ax.set_ylim(-0.6, n - 0.4)

    # y 轴刻度与标签：从下到上按 method_order 绘制，因此从上到下显示时要反转
    top_to_bottom_labels = [method_labels.get(k, k) for k in method_order][::-1]
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_to_bottom_labels, fontweight="bold")

    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # ----------- 逐行绘制（严格按 method_order） -----------
    def draw_row(evs, y):
        h = 0.3
        for e in evs:
            x0, w = e["start"], e["dur"]
            kind = e.get("kind", "run")
            if kind == "predict":
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor="white", edgecolor="k",
                                         hatch="xx", alpha=0.9)
            elif kind == "prewarm":
                color = label2color.get(e.get("label"), (0.7, 0.7, 0.7, 1.0))
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor=color, edgecolor="k",
                                         hatch="//", alpha=0.85)
            elif kind == "queue_wait":
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor="none", edgecolor="k",
                                         linestyle="--", alpha=0.6)
            else:  # run
                color = label2color.get(e.get("label"), (0.7, 0.7, 0.7, 1.0))
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor=color, edgecolor="none",
                                         alpha=0.95)
            ax.add_patch(rect)

    for idx, k in enumerate(method_order):
        draw_row(method_events[k], y=(n - 1 - idx))  # 使 method_order[0] 在最下方

    # ----------- 图例：仅展示 run 的 label，按频次取前 legend_topk -----------
    handles = [patches.Patch(facecolor=label2color[lab], edgecolor="k", label=lab)
               for lab, _ in freq.most_common(legend_topk)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.3),
              ncol=4, fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    print(f"[save] timeline -> {out_png}")

def draw_timeline_run_only_colorbar(method_events: Dict[str, List[Dict[str,Any]]],
                           out_png: pathlib.Path,
                           legend_topk: int = 16):  # legend_topk 已无用，但保留签名兼容
    """
    仅绘制 kind == 'run' 的事件块；
    颜色用连续 colormap 根据“电路索引”映射；
    colorbar 显示索引号（不展示电路具体名称）。
    """
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 45})
    PRED_LABEL = "__predictor__"

    # ----------- 明确绘图顺序（从下到上） -----------
    method_order = ["FullCompilation", "PR", "FS", "FS+Pre+ttl+SE+ema"]

    # ----------- 为 run 事件建立“电路索引”映射（按出现顺序稳定编号） -----------
    label2idx = {}
    for k in method_order:
        if k not in method_events:
            continue
        for e in method_events[k]:
            if e.get("kind", "run") == "run" and e.get("label") and e["label"] != PRED_LABEL:
                lab = e["label"]
                if lab not in label2idx:
                    label2idx[lab] = len(label2idx)  # 0..K-1

    K = max(len(label2idx), 1)
    cmap = cm.get_cmap("twilight")
    norm = mcolors.Normalize(vmin=0, vmax=K-1)

    # def color_for_label(lab):
    #     idx = label2idx.get(lab, None)
    #     return cmap(norm(idx)) if idx is not None else (0.7, 0.7, 0.7, 1.0)

    def color_for_label(lab):
        idx = label2idx.get(lab, None)
        if idx is None:
            return (0.7, 0.7, 0.7, 1.0)
        p = idx2perm[idx]  # 乱序后的离散位置 0..K-1
        pos = (p + 0.5) / K  # 放在每个色段的中点，视觉更均匀
        return cmap(pos)

    # 方式 A：固定随机种子（全局一致且可复现）
    rng = np.random.default_rng(seed=2025)
    perm = rng.permutation(K)
    # 反查：原始 idx -> 乱序后 idx
    idx2perm = {i: p for i, p in enumerate(perm)}

    # ----------- 画布 & 轴范围 -----------
    n = len(method_order)
    method_labels = {
        "FullCompilation": "FullComp",
        "PR": "Braket",
        "FS": "CCache",
        "FS+Pre+ttl+SE+ema": "TransCache",
    }

    fig, ax = plt.subplots(figsize=(20, 7))

    # 全局时间上限
    T = 0.0
    for k in method_order:
        if k not in method_events:
            continue
        evs = method_events[k]
        T = max(T, max((e["start"] + e["dur"] for e in evs), default=0.0))
    ax.set_xlim(-0.02, T * 1.05)
    ax.set_ylim(-0.6, n - 0.4)

    # y 轴刻度与标签（从上到下显示反序）
    top_to_bottom_labels = [method_labels.get(k, k) for k in method_order][::-1]
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_to_bottom_labels, fontweight="bold")

    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=4, alpha=0.5, zorder=0)


    # ----------- 逐行绘制（严格按 method_order） -----------
    def draw_row(evs, y):
        h = 0.4
        for e in evs:
            x0, w = e["start"], e["dur"]
            kind = e.get("kind", "run")
            lab  = e.get("label")
            if kind == "predict":
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor="white", edgecolor="k",
                                         hatch="xx", alpha=0.9)
            elif kind == "prewarm":
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor=color_for_label(lab), edgecolor="k",
                                         hatch="//", alpha=0.85)
            elif kind == "queue_wait":
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor="none", edgecolor="k",
                                         linestyle="--", alpha=0.6)
            else:  # run
                rect = patches.Rectangle((x0, y - h/2), w, h,
                                         facecolor=color_for_label(lab), edgecolor="none",
                                         alpha=0.95, zorder=3)
            ax.add_patch(rect)

    for idx, k in enumerate(method_order):
        if k not in method_events:
            continue
        draw_row(method_events[k], y=(n - 1 - idx))  # 使 method_order[0] 在最下方

    # ----------- 连续 colorbar（显示电路索引号，不显示电路名称） -----------
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for spine in ax.spines.values():
        spine.set_linewidth(5)

    # 横向 colorbar；把刻度放到顶部
    cbar = fig.colorbar(
            sm, ax=ax,
            orientation='vertical',   # ← 改为竖向
            pad=0.02,                 # 与主图的间距
            fraction=0.06,            # colorbar 占用的宽度比例
            shrink=0.95,               # 适当缩短一点高度（可按需调整/删除）
        )
    cbar.set_label("Circuit Index", fontweight="bold")
    cbar.ax.tick_params(labelsize=28, width=2, length=6)
    for t in cbar.ax.get_yticklabels():  # 横向时改为 get_xticklabels()
        t.set_fontweight("bold")

    # 设定整数刻度（索引号）
    if K <= 8:
        ticks = np.arange(K)
    else:
        # 若电路过多，稀疏一些刻度（最多 ~20 个）
        step = int(np.ceil(K / 8))
        ticks = np.arange(0, K, step)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(t)) for t in ticks])

    # —— 必须逐轴处理的部分（rcParams 无法直接控制粗体） ——
    # 1) 刻度标签加粗（rcParams 没有 tick label weight）
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    print(f"[save] timeline -> {out_png}")

# ---------- multi-method timeline ----------
def draw_timeline_multi(method_events: Dict[str, List[Dict[str,Any]]],
                        out_png: pathlib.Path, legend_topk: int = 16):
    # draw timeline, each top 16 circuits has a color legend
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
    fig, ax = plt.subplots(figsize=(14, 7))
    T = 0.0
    for evs in method_events.values():
        T = max(T, max((e["start"]+e["dur"] for e in evs), default=0.0))
    ax.set_xlim(0, T*1.05)
    ax.set_ylim(-0.6, n-0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(method_events.keys())[::-1], fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontweight="bold")
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
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
                rect = patches.Rectangle((x0, y - h/2), w, h, facecolor=color, edgecolor="none", alpha=0.95)
            ax.add_patch(rect)

    for idx, (name, evs) in enumerate(method_events.items()):
        draw_row(evs, y=(n-1-idx))

    handles = []
    for lab,_ in freq.most_common(legend_topk):
        handles.append(patches.Patch(facecolor=label2color[lab], edgecolor="k", label=lab))
    handles += [patches.Patch(facecolor="white", edgecolor="k", hatch="//", label="prewarm compile"),
                patches.Patch(facecolor="white", edgecolor="k", hatch="xx", label="predictor scoring"),
                patches.Patch(facecolor="none", edgecolor="k", linestyle="--", label="queue wait")]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=4,
        fontsize=12,
        frameon=False
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    print(f"[save] timeline -> {out_png}")

def plot_cache_size_change(cache_size_cahnges: Dict[str, List[Dict[str,Any]]],out_png: pathlib.Path, ):
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18})

    def _to_xy(series):
        if not series: return [], []
        ts = [float(p["t"]) for p in series]
        sz = [int(p["size"]) for p in series]
        return ts, sz

    fig, ax = plt.subplots(figsize=(10, 4))
    for label, series in cache_size_cahnges.items():
        ts, sz = _to_xy(series)
        ax.step(ts, sz, where="post", label=label)

    ax.set_xlabel("Timeline (s)")
    ax.set_ylabel("Cache size (#circuits)")
    ax.set_title("Cache size over time")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    print(f"[save] cache-size lines -> {out_png}")
