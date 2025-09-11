"""
TransCache (opportunistic pre-loading) — end-to-end demo
--------------------------------------------------------
前台(请求到达): run_once(..., mode in {"baseline","cache","cacheIdle"})
后台(空闲期):   opportunistic_prewarm(...)  预测并预编译，下次到达命中

本程序做了三件事：
  1) 记录每个电路键的“最近到达”滑窗，用 Poisson 近似得到到达率 λ；
  2) 在空闲 tick 内，若 idle_since ≥ T_load(λ) 且未缓存，则预编译入 idle-cache；
     若 idle_since ≥ T_off(λ) 且已缓存，则机会式卸载（可选）；
  3) 前台 cacheIdle 路径仅“读” idle-cache，真正编译已提前完成。

备注：保持与现有统计/绘图口径一致（阶段名不变），方便对比图复用。
"""

import time, json, argparse, pathlib, pickle, hashlib, math, warnings
from typing import Dict, Callable, Tuple
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from qiskit import QuantumCircuit, transpile, qasm3
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer import AerSimulator

# 你的电路库（保持一致）
from v11_quasa_bench_circuits import CIRCUITS_QUASA as CIRCUITS

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ================== 目录与缓存 ==================
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14})
v_num = 13
FIGDIR  = pathlib.Path(f"./figs/v{v_num}sd/"); FIGDIR.mkdir(parents=True, exist_ok=True)

# 两份缓存：常规缓存 vs idle 预编译缓存（**互不混用**）
CACHE_P       = pathlib.Path(f"./figs/v{v_num}sd/v{v_num}_sim_transpile_cache.pkl")
CACHE_P_IDLE  = pathlib.Path(f"./figs/v{v_num}sd/v{v_num}_sim_transpile_cache_idle.pkl")
CACHE_MEM, CACHE_MEM_IDLE = None, None  # 进程内热层（减少磁盘反序列化）

COLOR_MAP = {
    "01_setup"       : "#9ecae1",
    "02_cache_search": "#00E5EE",
    "03_transpile"   : "#4292c6",
    "04_submit"      : "#8FBC8F",
    "05_run"         : "#fedcb2",
    "06_collect"     : "#E9967A",
    "07_cache_write" : "#6e559c",
}

# ================== 基础工具 ==================
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
    """常规缓存（前台 cache 模式）"""
    global CACHE_MEM
    if CACHE_MEM is not None: return CACHE_MEM
    CACHE_MEM = pickle.loads(CACHE_P.read_bytes()) if CACHE_P.exists() else {}
    return CACHE_MEM

def load_cache_idle() -> Dict[str, QuantumCircuit]:
    """idle 预编译缓存（前台 cacheIdle 模式只读它）"""
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
    """Aer 模拟器上的 transpile 参数（target 优先，否则 basis_gates）"""
    aer = AerSimulator()
    try:
        return {"target": aer.target, "optimization_level": 3, "seed_transpiler": 42}
    except Exception:
        cfg = aer.configuration()
        return {"basis_gates": cfg.basis_gates, "optimization_level": 3, "seed_transpiler": 42}

# ================== 计时包装器（保持阶段口径） ==================
class PhaseTimer:
    def __init__(self): self.laps, self._t0 = {}, None
    def tic(self, name): self._t0 = now_perf(); self.laps[name] = -1.0
    def toc(self, name): self.laps[name] = dsec(self._t0)

# ================== 预测与空闲期预编译（后台） ==================
# 滑窗与阈值：InstaInfer 风格（P_load=6%, P_off=94%）
SLIDING_W = 32
PLOAD, POFF = 0.06, 0.94
RECENT_CALLS = defaultdict(lambda: deque(maxlen=SLIDING_W))  # key -> 最近到达时间戳

def record_arrival(key: str):
    """前台：每次任务结束时记录一次到达，用于估计到达率 λ"""
    RECENT_CALLS[key].append(time.time())

def arrival_rate(key: str) -> float:
    q = RECENT_CALLS[key]
    if len(q) < 2: return 0.0
    Tw = q[-1] - q[0]
    return 0.0 if Tw <= 0 else len(q) / Tw

def thresholds(lam: float) -> Tuple[float, float]:
    """根据 Poisson 近似把概率阈值转换成时间阈值"""
    if lam <= 0: return float("inf"), float("inf")
    Tload = -1.0 / lam * math.log(1.0 - PLOAD)
    Toff  = -1.0 / lam * math.log(1.0 - POFF)
    return Tload, Toff

def idle_since(key: str) -> float:
    q = RECENT_CALLS[key]
    if not q: return float("inf")
    return time.time() - q[-1]

def opportunistic_prewarm(candidates, max_compile: int = 1):
    """
    后台(空闲期)调度入口：只在“空闲 tick”调用
    candidates: 可调用列表，每个调用返回 (qc_raw, key)
    max_compile: 单次最多预编译多少个条目，避免占用前台资源
    """
    cache = load_cache_idle()
    compiled = 0
    for make in candidates:
        qc_raw, key = make()

        lam = arrival_rate(key)
        Tload, Toff = thresholds(lam)
        idle = idle_since(key)

        # 预编译：未缓存且空闲时间超过加载阈值
        if key not in cache and idle >= Tload:
            tp_kwargs = _prepare_kwargs()
            qc_exec = transpile(qc_raw, **tp_kwargs)   # 真正的编译发生在“后台”
            cache[key] = qc_exec
            compiled += 1
            if compiled >= max_compile:
                break

        # 可选：机会式卸载，避免 idle-cache 长期膨胀
        elif key in cache and idle >= Toff:
            cache.pop(key, None)

    if compiled > 0:
        save_cache_idle(cache)

# ================== 三种前台编译路径（解耦业务逻辑） ==================
def compile_baseline(qc_raw: QuantumCircuit, timer: PhaseTimer) -> QuantumCircuit:
    """Baseline：每次都在线 transpile"""
    tp_kwargs = _prepare_kwargs()
    timer.tic("03_transpile")
    qc_exec = transpile(qc_raw, **tp_kwargs)
    timer.toc("03_transpile")
    return qc_exec

def compile_with_cache(qc_raw: QuantumCircuit, bk_name: str, timer: PhaseTimer) -> Tuple[QuantumCircuit, str]:
    """常规缓存：命中→跳过编译；未命中→编译并写回 CACHE_P"""
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
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

def compile_with_idle_cache(qc_raw: QuantumCircuit, bk_name: str, timer: PhaseTimer) -> Tuple[QuantumCircuit, str]:
    """
    Idle-Cache（只读）：**不在这里编译**！
    如果命中 idle-cache，则完全跳过 transpile；
    若未命中，可选择兜底在线编译（下方实现保留兜底，必要时可关闭）。
    """
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
    timer.tic("02_cache_search")
    cache = load_cache_idle()
    qc_exec = cache.get(key)
    timer.toc("02_cache_search")
    if qc_exec is None:
        # 兜底：为了完整性，仍在线编译一次并写回 idle-cache（可按需改为“不写回”）
        tp_kwargs = _prepare_kwargs()
        timer.tic("03_transpile")
        qc_exec = transpile(qc_raw, **tp_kwargs)
        timer.toc("03_transpile")
        timer.tic("07_cache_write")
        cache[key] = qc_exec
        save_cache_idle(cache)
        timer.toc("07_cache_write")
    return qc_exec, key

# ================== 前台统一入口（只做编排与计时） ==================
def run_once(qc_func: Callable[[], QuantumCircuit], mode: str, shots: int):
    """
    mode ∈ {'baseline','cache','cacheIdle'}
    前台路径：只做查/编/执行/统计；真正的“机会式预编译”在后台 opportunistic_prewarm()
    """
    timer = PhaseTimer()
    qc_raw = qc_func()

    # 01 setup（与原脚本一致：Sampler 路径，跳过 sampler 内置 transpilation）
    timer.tic("01_setup")
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk_name = "AerSV"
    timer.toc("01_setup")

    # 选择编译策略
    if mode == "baseline":
        qc_exec = compile_baseline(qc_raw, timer)
        key = f"{bk_name}:{md5_qasm(qc_raw)}"
    elif mode == "cache":
        qc_exec, key = compile_with_cache(qc_raw, bk_name, timer)
    elif mode == "cacheIdle":
        qc_exec, key = compile_with_idle_cache(qc_raw, bk_name, timer)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # 运行与收集（04~06）
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
    }
    timer.toc("06_collect")

    # 记一次“到达”，供后台预测用（关键：把到达与预编译解耦）
    try:
        record_arrival(key)
    except Exception:
        pass

    timer.laps["total"] = round(sum(v for k, v in timer.laps.items() if k != "total"), 6)
    return timer.laps, meta

# ================== 绘图（三种模式并排） ==================
def plot_all(res_dict, out_png: pathlib.Path):
    fig, ax = plt.subplots(figsize=(12, 2 + 1.4 * len(res_dict)))
    ytick, ylbl = [], []
    order = list(COLOR_MAP.keys())
    for i, (name, (base, cache, cacheIdle)) in enumerate(res_dict.items()):
        for j, (mode, laps) in enumerate([("Baseline", base), ("TranspileCache", cache), ("IdleCache", cacheIdle)]):
            y = i * 1.4 + 0.4 * j
            left = 0.0
            for k in order:
                w = laps.get(k, 0.0)
                ax.barh(y, w, left=left, color=COLOR_MAP.get(k, "#999999"),
                        edgecolor="black", height=0.2, zorder=2)
                left += w
            ytick.append(y); ylbl.append(f"{name} – {mode}")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_yticks(ytick); ax.set_yticklabels(ylbl, fontweight="bold")
    ax.set_xlabel("Latency (s)", fontweight="bold"); ax.margins(x=0.05)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    handles = [mpatches.Patch(color=COLOR_MAP.get(k, "#999999"), label=k) for k in order]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_png, dpi=600)
    print("figure saved ->", out_png)

# ================== 批量驱动与“空闲 tick” ==================
def make_candidates_for(q: int, d: int):
    """
    生成候选集合（闭包）：每个元素返回 (qc_raw, key)
    说明：我们把“将要在本轮测试的 (q,d) 下所有电路”作为候选，模拟“很快就会到来”
    """
    cands = []
    for name, make in CIRCUITS.items():
        def _mk(make_=make, q_=q, d_=d):
            def _call():
                qc = make_(q_, d_)
                key = f"AerSV:{md5_qasm(qc)}"
                return qc, key
            return _call
        cands.append(_mk())
    return cands

def run_circuits(runs: int, shots: int, q: int, d: int):
    """
    批量驱动器（示例）：
      - 在每个电路真正执行前，调用 1 次 opportunistic_prewarm() 作为“空闲 tick”，
        后台完成预测+预编译；
      - 随后前台 run_once(..., mode="cacheIdle") 直接命中，从而验证效果。
    """
    results = {}
    qd_tag = f"q{q}_d{d}"

    for name, make in CIRCUITS.items():
        fn = (lambda q_=q, d_=d: (lambda: make(q_, d_)))()

        # ——【后台】空闲 tick：在该电路到来“之前”预编译最可能命中的条目 —— #
        opportunistic_prewarm(make_candidates_for(q, d), max_compile=2)

        # Baseline
        print(f"▶ {qd_tag} {name} baseline")
        base_runs = []
        for r in range(runs):
            laps, meta = run_once(fn, "baseline", shots)
            base_runs.append(laps)
            (FIGDIR / f"sim_{qd_tag}_{name}_baseline_{r}.json").write_text(
                json.dumps({"laps": laps, "meta": meta}, indent=2))
        base_sum = {k: round(sum(rr.get(k, 0.0) for rr in base_runs), 6) for k in base_runs[0]}

        # TranspileCache（常规缓存）
        print(f"▶ {qd_tag} {name} TranspileCache")
        pc_runs = []
        for r in range(runs):
            laps, meta = run_once(fn, "cache", shots)
            pc_runs.append(laps)
            (FIGDIR / f"sim_{qd_tag}_{name}_tcache_{r}.json").write_text(
                json.dumps({"laps": laps, "meta": meta}, indent=2))
        cache_sum = {k: round(sum(rr.get(k, 0.0) for rr in pc_runs), 6) for k in pc_runs[0]}

        # IdleTranspileCache（只读 idle-cache，命中则完全跳过编译）
        print(f"▶ {qd_tag} {name} IdleTranspileCache")
        tc_idle_runs = []
        for r in range(runs):
            laps, meta = run_once(fn, "cacheIdle", shots)
            tc_idle_runs.append(laps)
            (FIGDIR / f"sim_{qd_tag}_{name}_tcacheIdle_{r}.json").write_text(
                json.dumps({"laps": laps, "meta": meta}, indent=2))
        cacheIdle_sum = {k: round(sum(rr.get(k, 0.0) for rr in tc_idle_runs), 6) for k in tc_idle_runs[0]}

        results[name] = (base_sum, cache_sum, cacheIdle_sum)

    plot_all(results, FIGDIR / f"v{v_num}_sim_compare_q{q}_d{d}.png")

# ================== CLI ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--q", type=int, default=9)
    ap.add_argument("--d", type=int, default=4)
    args = ap.parse_args()

    # 可先试单个 (q,d)，更直观看到 idle 预编译带来的“首帧”下降
    run_circuits(runs=args.runs, shots=args.shots, q=args.q, d=args.d)
