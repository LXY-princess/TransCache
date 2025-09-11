"""
TransCache (v13) — Opportunistic prewarm timeline (prewarm vs. cacheIdle run)
-------------------------------------------------------------------------------
• 记录并绘制时间轴：
  - prewarm 事件：每次 opportunistic_prewarm 调用内真正“预编译成功”的电路
  - run 事件：每次 run_once(..., mode="cacheIdle") 调用的电路（绿色=命中，红色=未命中）
• 输出：
  - 图：figs/v13sd/timeline_prewarm_vs_run.png
  - 事件：figs/v13sd/timeline_events.jsonl

依赖：Qiskit Aer；基准电路请使用 v11_quasa_bench_circuits.py 的 CIRCUITS_QUASA。
"""

import time, json, argparse, pathlib, pickle, hashlib, math, warnings
from typing import Dict, Callable, Tuple, List, Any
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile, qasm3
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer import AerSimulator

# 你的电路库（保持一致）
from v11_quasa_bench_circuits import CIRCUITS_QUASA as CIRCUITS

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============ 路径与全局 ============
v_num = 13
FIGDIR = pathlib.Path(f"./figs/v{v_num}/"); FIGDIR.mkdir(parents=True, exist_ok=True)

# 两份缓存：常规缓存 vs idle 预编译缓存（互不混用）
CACHE_P      = pathlib.Path(f"./figs/v{v_num}/v{v_num}_sim_transpile_cache.pkl")
CACHE_P_IDLE = pathlib.Path(f"./figs/v{v_num}/v{v_num}_sim_transpile_cache_idle.pkl")
CACHE_MEM, CACHE_MEM_IDLE = None, None

# 时间轴事件日志
TIMELINE_JSONL = FIGDIR / "timeline_events.jsonl"
EVENT_LOG: List[Dict[str, Any]] = []   # 内存中的事件列表

# ============ 基础工具 ============
def now_perf(): return time.perf_counter()
def dsec(t0):   return round(time.perf_counter() - t0, 6)

def md5_qasm(circ: QuantumCircuit) -> str:
    """兼容 Terra 1.x/2.x：优先 qasm2，不可用时用 qasm3"""
    try:
        txt = circ.qasm()
    except AttributeError:
        txt = qasm3.dumps(circ)
    return hashlib.md5(txt.encode()).hexdigest()

def load_cache() -> Dict[str, QuantumCircuit]:
    global CACHE_MEM
    if CACHE_MEM is not None: return CACHE_MEM
    CACHE_MEM = pickle.loads(CACHE_P.read_bytes()) if CACHE_P.exists() else {}
    return CACHE_MEM

def load_cache_idle() -> Dict[str, QuantumCircuit]:
    global CACHE_MEM_IDLE
    if CACHE_MEM_IDLE is not None: return CACHE_MEM_IDLE
    CACHE_MEM_IDLE = pickle.loads(CACHE_P_IDLE.read_bytes()) if CACHE_P_IDLE.exists() else {}
    return CACHE_MEM_IDLE

def save_cache(c: Dict[str, QuantumCircuit]) -> None:
    global CACHE_MEM
    CACHE_MEM = c
    CACHE_P.write_bytes(pickle.dumps(c))

def save_cache_idle(c: Dict[str, QuantumCircuit]) -> None:
    global CACHE_MEM_IDLE
    CACHE_MEM_IDLE = c
    CACHE_P_IDLE.write_bytes(pickle.dumps(c))

def _prepare_kwargs():
    """Aer 模拟器上的 transpile 参数（优先 target，回退 basis_gates）"""
    aer = AerSimulator()
    try:
        return {"target": aer.target, "optimization_level": 3, "seed_transpiler": 42}
    except Exception:
        cfg = aer.configuration()
        return {"basis_gates": cfg.basis_gates, "optimization_level": 3, "seed_transpiler": 42}

# ============ 计时器 ============
class PhaseTimer:
    def __init__(self): self.laps, self._t0 = {}, None
    def tic(self, name): self._t0 = now_perf(); self.laps[name] = -1.0
    def toc(self, name): self.laps[name] = dsec(self._t0)

# ============ 预测/空闲期预编译（InstaInfer 风格阈值） ============
SLIDING_W = 32
PLOAD, POFF = 0.06, 0.94
RECENT_CALLS = defaultdict(lambda: deque(maxlen=SLIDING_W))  # key -> 最近到达时间戳

def record_arrival(key: str):
    RECENT_CALLS[key].append(time.time())

def arrival_rate(key: str) -> float:
    q = RECENT_CALLS[key]
    if len(q) < 2: return 0.0
    Tw = q[-1] - q[0]
    return 0.0 if Tw <= 0 else len(q) / Tw

def thresholds(lam: float) -> Tuple[float, float]:
    if lam <= 0: return float("inf"), float("inf")
    Tload = -1.0 / lam * math.log(1.0 - PLOAD)
    Toff  = -1.0 / lam * math.log(1.0 - POFF)
    return Tload, Toff

def idle_since(key: str) -> float:
    q = RECENT_CALLS[key]
    if not q: return float("inf")
    return time.time() - q[-1]

# ============ key -> 电路元信息 映射（用于时间轴标签） ============
KEY_META: Dict[str, Dict[str, Any]] = {}  # key -> {"circ","q","d","n_qubits","depth"}

def register_key_info(key: str, info: Dict[str, Any]):
    old = KEY_META.get(key, {})
    merged = dict(old)
    for k, v in info.items():
        if v is not None:
            merged[k] = v
    KEY_META[key] = merged

def key_info_fallback(key: str) -> Dict[str, Any]:
    info = KEY_META.get(key, {}).copy()
    if "n_qubits" not in info or "depth" not in info:
        qc = load_cache_idle().get(key) or load_cache().get(key)
        if qc is not None:
            info.setdefault("n_qubits", qc.num_qubits)
            info.setdefault("depth", qc.depth())
    info.setdefault("circ", "unknown")
    info.setdefault("q", None)
    info.setdefault("d", None)
    return info

def label_for_key(key: str) -> str:
    info = key_info_fallback(key)
    circ = info.get("circ")
    q = info.get("q"); d = info.get("d")
    if circ not in (None, "unknown") and q is not None and d is not None:
        return f"{circ}_q{q}_d{d}"
    return key[:10] + "…"

# ============ 记录/绘制：时间轴事件 ============
def append_event(ev: Dict[str, Any]):
    EVENT_LOG.append(ev)
    with open(TIMELINE_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

def plot_timeline(events: List[Dict[str, Any]], out_png: pathlib.Path):
    """
    三行子图 (sharex=True)：
      Row-1: 主时间轴（prewarm/offload/run(HIT/MISS)）
      Row-2: Cache Size（仅在 cache 变化事件处采样；阶梯线 + 散点）
      Row-3: Cache Contents（同变化时刻，打印当时 cache 内所有电路标签）
    """
    if not events:
        print("No timeline events to plot.")
        return

    events_sorted = sorted(events, key=lambda e: e["ts"])
    t0 = events_sorted[0]["ts"]
    t_end = events_sorted[-1]["ts"]

    # ---------------- Row-1 数据：原有散点 ----------------
    xs_pre, ys_pre, labs_pre = [], [], []
    xs_off, ys_off, labs_off = [], [], []
    xs_run_hit, ys_run_hit, labs_run_hit = [], [], []
    xs_run_miss, ys_run_miss, labs_run_miss = [], [], []

    for e in events_sorted:
        x = e["ts"] - t0
        if e["type"] == "prewarm":
            xs_pre.append(x); ys_pre.append(1.0); labs_pre.append(event_label_with_seq(e))
        elif e["type"] == "OFFLOAD":
            xs_off.append(x); ys_off.append(0.5); labs_off.append(event_label_with_seq(e))
        elif e["type"] == "run":
            lab = label_for_key(e["key"])
            if e.get("hit", False):
                xs_run_hit.append(x);  ys_run_hit.append(0.0); labs_run_hit.append(lab)
            else:
                xs_run_miss.append(x); ys_run_miss.append(0.0); labs_run_miss.append(lab)

    # ---------------- Row-2/3 数据：根据事件重建 cache 轨迹 ----------------
    cache_state: set[str] = set()
    change_times: List[float] = []      # 相对时间 (s)
    cache_sizes: List[int] = []         # 对应 cache 大小
    cache_snap_labels: List[str] = []   # 对应 “当时cache中都有哪些电路” 的文本

    def _labels_for_state(st: set) -> str:
        # 把当前 cache 中的 key 映射为 标签（电路名_q_d）；按字母序便于稳定复现
        labs = [label_for_key(k) for k in sorted(st)]
        # 文本较多时自动换行：每个标签单独换行，后续显示时再旋转 90 度
        return "\n".join(labs) if labs else "(empty)"

    for e in events_sorted:
        k = e.get("key")
        t_rel = e["ts"] - t0
        mutated = False

        if e["type"] == "prewarm":
            if k not in cache_state:
                cache_state.add(k); mutated = True
        elif e["type"] == "OFFLOAD":
            if k in cache_state:
                cache_state.remove(k); mutated = True
        elif e["type"] == "run" and not e.get("hit", False):
            # cache miss 后兜底编译，并写回 idle-cache → 视作加入
            if k not in cache_state:
                cache_state.add(k); mutated = True

        if mutated:
            change_times.append(t_rel)
            cache_sizes.append(len(cache_state))
            cache_snap_labels.append(_labels_for_state(cache_state))

    # 若没有发生任何 cache 变化，给出占位，避免子图空白崩溃
    if not change_times:
        change_times = [0.0]
        cache_sizes = [0]
        cache_snap_labels = ["(no cache changes)"]

    # ---------------- 作图：三行子图，共享 X 轴 ----------------
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(12, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.8]}
    )

    # Row-1：主时间轴
    pre = ax0.scatter(xs_pre, ys_pre, marker="^", s=70, label="prewarm (compiled)")
    off = ax0.scatter(xs_off, ys_off, marker="v", s=70, label="offload (evicted)", color="tab:purple")
    rh  = ax0.scatter(xs_run_hit,  ys_run_hit,  marker="o", s=70, label="run cacheIdle — HIT",  color="tab:green")
    rm  = ax0.scatter(xs_run_miss, ys_run_miss, marker="o", s=70, label="run cacheIdle — MISS", color="tab:red")

    def annotate(ax, xs, ys, labs, dy, rot=-90, fs=6):
        for xi, yi, lb in zip(xs, ys, labs):
            ax.annotate(lb, (xi, yi+dy), xytext=(0, 6 if dy>=0 else -8),
                        textcoords="offset points", rotation=rot,
                        ha="left", va="bottom" if dy>=0 else "top",
                        fontsize=fs, alpha=0.9)

    annotate(ax0, xs_pre, ys_pre, labs_pre, dy=+0.03)
    annotate(ax0, xs_off, ys_off, labs_off, dy=+0.03)
    annotate(ax0, xs_run_hit, ys_run_hit, labs_run_hit, dy=-0.03)
    annotate(ax0, xs_run_miss, ys_run_miss, labs_run_miss, dy=-0.03)

    ax0.set_yticks([0.0, 0.5, 1.0])
    ax0.set_yticklabels(["Run (cacheIdle)", "Offload", "Prewarm"])
    ax0.grid(axis="x", linestyle="--", alpha=0.5)
    ax0.set_ylim(-0.35, 1.35)

    # Legend 放到图外上方
    ax0.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=4,
        frameon=False
    )

    # Row-2：Cache Size（阶梯线 + 散点）
    x2 = change_times
    y2 = cache_sizes
    ax1.step(x2, y2, where="post", linewidth=1.8, alpha=0.9)
    ax1.plot(x2, y2, "o", ms=4)
    ax1.set_ylabel("Cache size")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.set_ylim(bottom=0)

    # Row-3：Cache Contents（文本快照）
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks([])
    ax2.set_ylabel("Cache contents @ changes", labelpad=6)
    for xi, txt in zip(change_times, cache_snap_labels):
        # 竖线辅助对齐
        ax2.axvline(x=xi, color="#dddddd", linewidth=0.8, zorder=0)
        ax2.text(xi, 0.02, txt, rotation=90, ha="left", va="bottom", fontsize=6)

    # 统一 X 轴
    ax2.set_xlabel("Time since start (s)")
    xmax = max(t_end - t0, max(change_times) if change_times else 0.0)
    ax2.set_xlim(left=-0.02, right=xmax * 1.03 if xmax > 0 else 1.0)

    # 留出上部空间给 legend
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print("Timeline saved ->", out_png)


def event_label_with_seq(e: Dict[str, Any]) -> str:
    """
    生成带序列号的事件标签：
      • prewarm / OFFLOAD:  '#<seq> <circ>_q<q>_d<d>'（若没有元信息则回退到 key hash）
      • 其他事件:            仍用原有标签
    """
    base = label_for_key(e["key"])
    if e.get("type") in ("prewarm", "OFFLOAD"):
        s = e.get("seq", None)
        if s is not None:
            try:
                s_str = f"{int(s):03d}"
            except Exception:
                s_str = str(s)
            return f"#{s_str} {base}"
    return base

from typing import Optional  # 若文件已有则忽略

OFFLOAD_TTL_SEC: Optional[float] = 30.0  # 秒；None 表示关闭 TTL 回收（演示可设小一些便于看到 offload）
CACHE_IDLE_MAX_ENTRIES: Optional[int] = None  # 例如 64；None 表示不限制
LAST_TOUCH_IDLE: Dict[str, float] = {}  # key -> 最近创建/命中的时间戳
PREWARM_CALL_SEQ: int = 0                  # 递增调用序号（保证时间顺序）

# ============ 机会式预编译（返回本轮决策用于日志） ============
def opportunistic_prewarm(candidates, max_compile: int = 1):
    """
    后台(空闲期)调度入口（InstaInfer 风格阈值）：
      • 当 key 未缓存 且 idle_since(key) ≥ T_load(λ) 时，执行“预编译”写入 idle-cache；
      • 当 key 已缓存 且满足 Toff/TTL/容量 等条件时，执行“卸载”；
    约束：
      • 严格执行编译预算：单次调用至多预编译 `max_compile` 条；达到预算后不再编译，
        但仍可对其它 key 执行 OFFLOAD（不占编译预算）。
      • 每次调用分配一个递增序号 `seq`，随事件写入时间轴（prewarm/offload）。
    返回：
      decisions: List[Dict] —— 本轮每个候选的决策快照（便于调试/可视化）。
    """
    global PREWARM_CALL_SEQ
    PREWARM_CALL_SEQ += 1
    call_seq  = PREWARM_CALL_SEQ
    # call_time = time.time()

    cache    = load_cache_idle()
    compiled = 0
    mutated  = False
    decisions = []

    # 简单 LRU 选择器（用于容量回收）；缺失触摸时间的视为“最老”
    def _lru_victim():
        if not cache:
            return None
        return sorted(cache.keys(), key=lambda k: LAST_TOUCH_IDLE.get(k, 0.0))[0]

    for make in candidates:
        # 解析候选：(qc_raw, key, info) 或 (qc_raw, key)
        ret = make()
        if isinstance(ret, tuple) and len(ret) == 3:
            qc_raw, key, info = ret
        else:
            qc_raw, key = ret
            info = {
                "circ": "unknown", "q": None, "d": None,
                "n_qubits": qc_raw.num_qubits, "depth": qc_raw.depth(),
            }
        register_key_info(key, info)  # 给 timeline/label 用

        lam = arrival_rate(key)
        Tload, Toff = thresholds(lam)
        idle = idle_since(key)
        in_cache = key in cache
        now_t = time.time()

        decision, reason, cost = "NONE", None, 0.0

        # ---------------- 预编译（受预算约束） ----------------
        if (not in_cache) and (idle >= Tload) and (compiled < max_compile):
            preCache_time = time.time()
            tp_kwargs = _prepare_kwargs()
            t0 = time.perf_counter()
            qc_exec = transpile(qc_raw, **tp_kwargs)  # 真正的编译在后台完成
            cost = time.perf_counter() - t0

            cache[key] = qc_exec
            LAST_TOUCH_IDLE[key] = now_t  # 视为“触摸”
            compiled += 1
            mutated  = True
            decision = "PREWARM"

            # —— 时间轴：记录 prewarm 事件（带调用序号）——
            append_event({
                "type": "prewarm", "ts": preCache_time, "seq": call_seq,
                "key": key, "circ": info.get("circ"), "q": info.get("q"), "d": info.get("d"),
                "n_qubits": info.get("n_qubits"), "depth": info.get("depth"),
                "compile_sec": cost,
            })

        # ---------------- 机会式卸载（不占编译预算） ----------------
        else:
            # Toff 触发
            toff_fire = in_cache and (idle >= Toff)
            # TTL 触发：从最近触摸起超时
            last_touch = LAST_TOUCH_IDLE.get(key, now_t)
            ttl_fire = in_cache and (OFFLOAD_TTL_SEC is not None) and ((now_t - last_touch) >= OFFLOAD_TTL_SEC)
            # 容量触发：本条不立即驱逐（统一在循环后 LRU 清理），这里只做记号
            cap_fire = False

            if toff_fire or ttl_fire:
                offload_time = time.time()
                cache.pop(key, None)
                mutated = True
                reason = "ttl" if ttl_fire else "toff"
                decision = "OFFLOAD"
                append_event({
                    "type": "OFFLOAD", "ts": offload_time, "seq": call_seq,
                    "key": key, "reason": reason,
                    "circ": info.get("circ"), "q": info.get("q"), "d": info.get("d"),
                    "n_qubits": info.get("n_qubits"), "depth": info.get("depth"),
                })

        # -------- 决策快照（便于调试/可视化 predictor） --------
        decisions.append({
            "ts": time.time(), "key": key,
            "lambda": lam, "Tload": Tload, "Toff": Toff, "idle_since": idle,
            "in_cache": in_cache, "decision": decision, "reason": reason,
            "cost_sec": cost, "seq": call_seq,
        })

        # 若已经达到编译预算，则**不再进行后续的编译**（但继续遍历以便触发/记录 OFFLOAD）
        # —— 这里不需要额外 `continue`，因为上面的编译分支已经受 (compiled < max_compile) 保护；
        # —— 其余候选只会进入“卸载/记录”路径。

    # ---------------- 容量 LRU 回收（统一处理） ----------------
    if (CACHE_IDLE_MAX_ENTRIES is not None) and (len(cache) > CACHE_IDLE_MAX_ENTRIES):
        while len(cache) > CACHE_IDLE_MAX_ENTRIES:
            victim = _lru_victim()
            if victim is None:
                break
            cache.pop(victim, None)
            mutated = True
            append_event({
                "type": "OFFLOAD", "ts": time.time(), "seq": call_seq,
                "key": victim, "reason": "capacity",
                # label 信息可按需补充：如果事先 register 过，可从 KEY_META 里取
            })

    if mutated:
        save_cache_idle(cache)

    return decisions


# ============ 三种编译路径 ============
def compile_baseline(qc_raw: QuantumCircuit, timer: PhaseTimer) -> QuantumCircuit:
    tp_kwargs = _prepare_kwargs()
    timer.tic("03_transpile")
    qc_exec = transpile(qc_raw, **tp_kwargs)
    timer.toc("03_transpile")
    return qc_exec

def compile_with_cache(qc_raw: QuantumCircuit, bk_name: str, timer: PhaseTimer) -> Tuple[QuantumCircuit, str]:
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
    register_key_info(key, {"n_qubits": qc_raw.num_qubits, "depth": qc_raw.depth()})
    timer.tic("02_cache_search")
    cache = load_cache()
    qc_exec = cache.get(key)
    timer.toc("02_cache_search")
    if qc_exec is None:
        tp_kwargs = _prepare_kwargs()
        timer.tic("03_transpile")
        qc_exec = transpile(qc_raw, **tp_kwargs)
        timer.toc("03_transpile")
        timer.tic("07_cache_write")
        cache[key] = qc_exec
        save_cache(cache)
        timer.toc("07_cache_write")
    return qc_exec, key

def compile_with_idle_cache(qc_raw: QuantumCircuit, bk_name: str, timer: PhaseTimer) -> Tuple[QuantumCircuit, str, bool]:
    """
    只读 idle-cache；若 miss 则兜底在线编译并写回。
    返回：(qc_exec, key, was_hit)
    """
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
    register_key_info(key, {"n_qubits": qc_raw.num_qubits, "depth": qc_raw.depth()})

    timer.tic("02_cache_search")
    cache = load_cache_idle()
    qc_exec = cache.get(key)
    timer.toc("02_cache_search")
    was_hit = qc_exec is not None

    now_t = time.time()  # <-- 新增
    if was_hit:
        LAST_TOUCH_IDLE[key] = now_t  # <-- 新增：命中即触摸
    else:
        tp_kwargs = _prepare_kwargs()
        timer.tic("03_transpile")
        qc_exec = transpile(qc_raw, **tp_kwargs)
        timer.toc("03_transpile")
        timer.tic("07_cache_write")
        cache[key] = qc_exec
        save_cache_idle(cache)
        timer.toc("07_cache_write")
        LAST_TOUCH_IDLE[key] = now_t  # <-- 新增：兜底写回也视为触摸

    return qc_exec, key, was_hit

# ============ 前台统一入口（仅用于 run/cacheIdle 计时） ============
def run_once(qc_func: Callable[[], QuantumCircuit], mode: str, shots: int):
    timer = PhaseTimer()
    qc_raw = qc_func()

    # 记录“调用时刻”（用于时间轴的 run 事件）
    ts_call = time.time()

    # 01 setup
    timer.tic("01_setup")
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk_name = "AerSV"
    timer.toc("01_setup")

    if mode == "baseline":
        qc_exec = compile_baseline(qc_raw, timer)
        key = f"{bk_name}:{md5_qasm(qc_raw)}"
        was_hit = None
    elif mode == "cache":
        qc_exec, key = compile_with_cache(qc_raw, bk_name, timer)
        was_hit = None
    elif mode == "cacheIdle":
        qc_exec, key, was_hit = compile_with_idle_cache(qc_raw, bk_name, timer)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # 运行与收集
    timer.tic("04_submit"); job = sampler.run([qc_exec], shots=shots); timer.toc("04_submit")
    timer.tic("05_run");    res = job.result();                        timer.toc("05_run")
    timer.tic("06_collect")
    quasi = res.quasi_dists[0] if hasattr(res, "quasi_dists") else None
    meta = {
        "backend": bk_name,
        "n_qubits": qc_raw.num_qubits,
        "depth_in": qc_raw.depth(),
        "depth_transpiled": qc_exec.depth(),
        "size_transpiled": qc_exec.size(),
        "quasi_dist": dict(quasi) if quasi is not None else None,
        "key": key,
        "cache_hit": was_hit if was_hit is not None else False,
        "ts_call": ts_call,
    }
    timer.toc("06_collect")

    # 到达记录（供后续预测）
    try:
        record_arrival(key)
    except Exception:
        pass

    timer.laps["total"] = round(sum(v for k, v in timer.laps.items() if k != "total"), 6)
    return timer.laps, meta

# ============ 候选生成（返回 (qc, key, info) 并登记 KEY_META） ============
def make_candidates_for(q: int, d: int):
    cands = []
    for name, make in CIRCUITS.items():
        def _mk(name_=name, make_=make, q_=q, d_=d):
            def _call():
                qc = make_(q_, d_)
                key = f"AerSV:{md5_qasm(qc)}"
                info = {"circ": name_, "q": q_, "d": d_,
                        "n_qubits": qc.num_qubits, "depth": qc.depth()}
                register_key_info(key, info)
                return qc, key, info
            return _call
        cands.append(_mk())
    return cands

# ============ 批量驱动：交替 “prewarm -> run(cacheIdle)” 并画时间轴 ============
def run_timeline(runs: int, shots: int, q: int, d: int):
    """
    为了只关注时间轴，这里不跑 baseline/cache，对每个电路进行多轮：
      [opportunistic_prewarm] -> [run_once(..., 'cacheIdle')]
    """
    for name, make in CIRCUITS.items():
        fn = (lambda q_=q, d_=d: (lambda: make(q_, d_)))()
        # 1) 空闲期预编译（可限制本次最多预编译多少条）
        decisions = opportunistic_prewarm(make_candidates_for(q, d), max_compile=2)
        for r in range(runs):
            # 2) 业务到达，走 cacheIdle 路径
            laps, meta = run_once(fn, "cacheIdle", shots)
            # —— 时间轴：run 事件（颜色编码：hit/ miss）——
            append_event({
                "type": "run",
                "ts": meta["ts_call"],
                "key": meta["key"],
                "hit": bool(meta.get("cache_hit", False)),
            })
            # 也可以按需把阶段时延写盘，本文聚焦时间轴，省略

    plot_timeline(EVENT_LOG, FIGDIR / "timeline_prewarm_vs_run.png")

# ============ CLI ============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3, help="每个电路重复 [prewarm->run] 的轮数")
    ap.add_argument("--q", type=int, default=9)
    ap.add_argument("--d", type=int, default=4)
    args = ap.parse_args()

    run_timeline(runs=args.runs, shots=args.shots, q=args.q, d=args.d)
    print(f"📄 events jsonl -> {TIMELINE_JSONL}")
