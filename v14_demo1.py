# v14_poisson_predictor_prewarm.py
# ------------------------------------------------------------
# A Poisson-process based "real" predictor + pre-warm pipeline
# Offload & capacity control: placeholders only for now.
#
# 依赖：
#   - Qiskit Aer (Sampler/AerSimulator)
#   - v11_quasa_bench_circuits.py 里提供的 CIRCUITS_QUASA
#
# 用法示例：
#   python v14_poisson_predictor_prewarm.py --runs 3 --q 9 --d 4 \
#       --lookahead 8.0 --prob_th 0.55 --max_compile 2
# ------------------------------------------------------------

import time, math, hashlib, pickle, pathlib, argparse, warnings
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np

from qiskit import QuantumCircuit, transpile, qasm3
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_aer import AerSimulator

# ---- circuits library (v11) ----
from v11_quasa_bench_circuits import CIRCUITS_QUASA as CIRCUITS

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------ 路径/缓存 ------------------------
VNUM = 14
FIGDIR = pathlib.Path(f"./figs/v{VNUM}/"); FIGDIR.mkdir(parents=True, exist_ok=True)
CACHE_P_IDLE = pathlib.Path(f"./figs/v{VNUM}/v{VNUM}_sim_transpile_cache_idle.pkl")
CACHE_MEM_IDLE: Optional[Dict[str, QuantumCircuit]] = None

def load_cache_idle() -> Dict[str, QuantumCircuit]:
    global CACHE_MEM_IDLE
    if CACHE_MEM_IDLE is not None:
        return CACHE_MEM_IDLE
    CACHE_MEM_IDLE = pickle.loads(CACHE_P_IDLE.read_bytes()) if CACHE_P_IDLE.exists() else {}
    return CACHE_MEM_IDLE

def save_cache_idle(c: Dict[str, QuantumCircuit]) -> None:
    global CACHE_MEM_IDLE
    CACHE_MEM_IDLE = c
    CACHE_P_IDLE.write_bytes(pickle.dumps(c))

# ------------------------ 工具函数 -------------------------
def md5_qasm(circ: QuantumCircuit) -> str:
    """兼容 Terra 1.x/2.x：优先 qasm2，不可用时回退 qasm3。"""
    try:
        txt = circ.qasm()
    except AttributeError:
        txt = qasm3.dumps(circ)
    return hashlib.md5(txt.encode()).hexdigest()

def _prepare_kwargs():
    """Aer 模拟器上的 transpile 参数（优先 target，回退 basis_gates）"""
    aer = AerSimulator()
    try:
        return {"target": aer.target, "optimization_level": 3, "seed_transpiler": 42}
    except Exception:
        cfg = aer.configuration()
        return {"basis_gates": cfg.basis_gates, "optimization_level": 3, "seed_transpiler": 42}

# ------------------------ 调用记录 -------------------------
SLIDING_MAXLEN = 64  # 最近样本数量上限（环形队列）；时间窗另见 predictor 配置
RECENT_CALLS: Dict[str, deque] = defaultdict(lambda: deque(maxlen=SLIDING_MAXLEN))

def record_arrival(key: str, ts: Optional[float] = None) -> None:
    """记录一次到达（运行）用于后续到达率估计。"""
    RECENT_CALLS[key].append(float(ts) if ts is not None else time.time())

# ------------------------ 预测器主体 -----------------------
class PoissonPredictor:
    """
    基于泊松过程的 predictor：
      • 用最近调用时间戳（滑动时间窗）做 λ 的 MLE 估计；
      • 在 lookahead 时间窗 τ 内计算 P{≥1 次到来} = 1 - exp(-λ τ)；
      • 按概率排序并输出候选。
    """
    def __init__(
        self,
        sliding_window_sec: float = 60.0,  # 只用最近 T 秒的样本做估计
        min_samples: int = 2               # 少于该数目则认为 λ=0
    ):
        self.sliding_window_sec = float(sliding_window_sec)
        self.min_samples = int(min_samples)

    def _recent_in_window(self, key: str, now: Optional[float] = None) -> List[float]:
        q = RECENT_CALLS.get(key, deque())
        if not q:
            return []
        now = float(now) if now is not None else time.time()
        lo = now - self.sliding_window_sec
        return [t for t in q if t >= lo and t <= now]

    def est_lambda(self, key: str, now: Optional[float] = None) -> float:
        """
        λ 的极大似然估计（均匀观测窗 T 内的计数 N： λ_hat = N/T）。
        等价于指数分布等待时间的 MLE（1/平均间隔），在观测窗足够长时二者一致。
        """
        ts = self._recent_in_window(key, now)
        if len(ts) < self.min_samples:
            return 0.0
        T = max(1e-9, max(ts) - min(ts))  # 防止除零
        N = len(ts)
        return N / T if T > 0 else 0.0

    @staticmethod
    def prob_within(lam: float, tau: float) -> float:
        if lam <= 0 or tau <= 0:
            return 0.0
        # P{N(tau) >= 1} = 1 - e^{-lambda * tau}
        return 1.0 - math.exp(-lam * tau)

    def score_candidates(
        self,
        candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
        lookahead_sec: float,
        prob_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        返回满足概率阈值的候选清单（含 λ 与概率），按概率降序。
        注意：为拿到 key（和元信息），这里需要“生成一次电路”，但不会编译。
        """
        scored: List[Dict[str, Any]] = []
        now = time.time()
        for maker in candidates:
            qc, key, info = maker()
            lam = self.est_lambda(key, now)
            p = self.prob_within(lam, lookahead_sec)
            if p >= prob_threshold:
                scored.append({
                    "key": key,
                    "prob": p,
                    "lambda": lam,
                    "qc_raw": qc,
                    "info": info,
                })
        scored.sort(key=lambda r: r["prob"], reverse=True)
        return scored

# ------------------------ 预热（pre-warm） ------------------
def prewarm_from_predictions(
    predictor: PoissonPredictor,
    candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    lookahead_sec: float,
    prob_threshold: float,
    max_compile: int = 1
) -> List[Dict[str, Any]]:
    """
    用 predictor 对候选打分并预编译 top-K（不命中 idle cache 的）。
    返回每个实际预编译成功条目的快照（便于日志/调参）。
    """
    cache = load_cache_idle()
    decided = predictor.score_candidates(candidates, lookahead_sec, prob_threshold)
    done: List[Dict[str, Any]] = []

    compiled = 0
    for item in decided:
        if compiled >= max_compile:
            break
        key = item["key"]
        if key in cache:
            # 已在 idle-cache：跳过
            continue
        qc_raw = item["qc_raw"]
        # 真正编译
        tp_kwargs = _prepare_kwargs()
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **tp_kwargs)
        cost = time.perf_counter() - t0
        cache[key] = qc_exec
        compiled += 1
        done.append({
            "key": key,
            "prob": item["prob"],
            "lambda": item["lambda"],
            "compile_sec": cost,
            "n_qubits": qc_raw.num_qubits,
            "depth": qc_raw.depth(),
            "circ": item["info"].get("circ"),
            "q": item["info"].get("q"),
            "d": item["info"].get("d"),
        })

    if done:
        save_cache_idle(cache)

    # ——占位：offload & 容量控制（后续实现）——
    offload_placeholder(cache)
    capacity_control_placeholder(cache)
    return done

# ------------------------ 运行路径 -------------------------
def compile_with_idle_cache(
    qc_raw: QuantumCircuit, bk_name: str
) -> Tuple[QuantumCircuit, str, bool]:
    """
    只读 idle-cache；若 miss 则兜底在线编译并写回。
    返回：(qc_exec, key, was_hit)
    """
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
    cache = load_cache_idle()
    qc_exec = cache.get(key)
    was_hit = qc_exec is not None
    if not was_hit:
        tp_kwargs = _prepare_kwargs()
        qc_exec = transpile(qc_raw, **tp_kwargs)
        cache[key] = qc_exec
        save_cache_idle(cache)
    return qc_exec, key, was_hit

def run_once(qc_func: Callable[[], QuantumCircuit], shots: int = 512) -> Dict[str, Any]:
    """
    前台一次性执行：优先用 idle-cache，miss 则编译并写回。
    同时记录 arrival（供 predictor 使用）。
    """
    qc_raw = qc_func()
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk_name = "AerSV"

    qc_exec, key, was_hit = compile_with_idle_cache(qc_raw, bk_name)

    # 提交并运行
    job = sampler.run([qc_exec], shots=shots)
    res = job.result()
    quasi = res.quasi_dists[0] if hasattr(res, "quasi_dists") else None

    # 重要：记录一次“真实到达”
    record_arrival(key)

    return {
        "key": key,
        "cache_hit": was_hit,
        "n_qubits": qc_raw.num_qubits,
        "depth_in": qc_raw.depth(),
        "depth_transpiled": qc_exec.depth(),
        "size_transpiled": qc_exec.size(),
        "quasi_dist": dict(quasi) if quasi is not None else None,
    }

# ------------------------ 候选生成 -------------------------
def make_candidates_for(q: int, d: int):
    """
    返回一个列表，每个元素是无参 callable：
    调用后返回 (qc_raw, key, info)，其中 key 用 AerSV:md5(qasm)。
    """
    makers: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]] = []
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
    return makers

# ------------------------ 占位：回收/容量 -------------------
def offload_placeholder(cache: Dict[str, QuantumCircuit]) -> None:
    """
    预留位置：基于 Toff / TTL / LRU / 生存时间等策略进行回收。
    当前不做任何事，只是占位，便于后续无缝接入。
    """
    return

def capacity_control_placeholder(cache: Dict[str, QuantumCircuit]) -> None:
    """
    预留位置：基于 cache 上限进行 LRU 淘汰或分层写入/下放（offload）。
    当前不做任何事，只是占位。
    """
    return

# ------------------------ 驱动逻辑（演示） -----------------
def drive_pipeline(
    runs: int,
    q: int,
    d: int,
    shots: int,
    lookahead_sec: float,
    prob_th: float,
    max_compile: int,
    sliding_window_sec: float,
):
    """
    为每个电路执行多轮：
      [prewarm_by_predictor] -> [run_once]
    注意：刚开始没有历史时（λ≈0），预测器不会预热；随着真实到达累积，预热会逐步生效。
    """
    predictor = PoissonPredictor(sliding_window_sec=sliding_window_sec, min_samples=1)
    makers = make_candidates_for(q, d)

    for name, make in CIRCUITS.items():
        # for _ in range(runs):
        # 每次业务到来前，先用 predictor 做一次机会式预热（后台编译）
        pre_decisions = prewarm_from_predictions(
            predictor=predictor,
            candidates=makers,                 # 所有候选都让 predictor 排序
            lookahead_sec=lookahead_sec,
            prob_threshold=prob_th,
            max_compile=max_compile,
        )
        if pre_decisions:
            print(f"[prewarm] compiled {len(pre_decisions)} items:")
            for it in pre_decisions:
                print(f"  - {it['circ']}_q{it['q']}_d{it['d']}  "
                      f"p={it['prob']:.2f}, λ={it['lambda']:.3f}/s, "
                      f"compile={it['compile_sec']:.3f}s")
        else:
            print(f"[prewarm] nothing to prewarm")

        # 业务到达：执行一次（这里按库中的电路顺序循环）
        fn = (lambda q_=q, d_=d: (lambda: make(q_, d_)))()
        meta = run_once(fn, shots=shots)
        print(f"[run] Cir={name} key={meta['key'][:10]}… hit={meta['cache_hit']} "
              f"q={meta['n_qubits']} depth={meta['depth_in']} -> "
              f"depthT={meta['depth_transpiled']}")

    # 再做 (runs-1) 轮
    for _ in range(runs - 1):
        pre_decisions = prewarm_from_predictions(
            predictor=predictor,
            candidates=makers,
            lookahead_sec=lookahead_sec,
            prob_threshold=prob_th,
            max_compile=max_compile,
        )
        if pre_decisions:
            print(f"[prewarm] compiled {len(pre_decisions)} items (loop):")
            for it in pre_decisions:
                print(f"  - {it['circ']}_q{it['q']}_d{it['d']}  "
                      f"p={it['prob']:.2f}, λ={it['lambda']:.3f}/s, "
                      f"compile={it['compile_sec']:.3f}s")

        # 演示：简单地按相同顺序再跑一遍
        for name, make in CIRCUITS.items():
            fn = (lambda q_=q, d_=d: (lambda: make(q_, d_)))()
            meta = run_once(fn, shots=shots)
            print(f"[run] Cir={name} key={meta['key'][:10]}… hit={meta['cache_hit']} "
                  f"q={meta['n_qubits']} depth={meta['depth_in']} -> "
                  f"depthT={meta['depth_transpiled']}")

# ------------------------ CLI ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="每个电路重复 prewarm->run 的轮数")
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--q", type=int, default=9)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--lookahead", type=float, default=8.0, help="预测的时间窗 τ (seconds)")
    ap.add_argument("--prob_th", type=float, default=0.55, help="预热概率阈值")
    ap.add_argument("--max_compile", type=int, default=2, help="每轮最多预编译条数")
    ap.add_argument("--sliding_window_sec", type=float, default=60.0,
                    help="λ 估计的滑动时间窗（秒）")
    args = ap.parse_args()

    drive_pipeline(
        runs=args.runs,
        q=args.q,
        d=args.d,
        shots=args.shots,
        lookahead_sec=args.lookahead,
        prob_th=args.prob_th,
        max_compile=args.max_compile,
        sliding_window_sec=args.sliding_window_sec,
    )
