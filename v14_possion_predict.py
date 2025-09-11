# v15_workload_prewarm_driver.py
# ------------------------------------------------------------
# Workload driver with Poisson superposition + predictor seeding
#   (1) Build a synthetic workload across multiple (q, d) combos
#   (2) Seed RECENT_CALLS so predictor can succeed on first prewarm
#   (3) Prewarm every K runs until the workload finishes
#   (4) Print detailed traces of prewarm & run
#
# 基础组件沿用 v14：PoissonPredictor、prewarm_from_predictions、idle-cache 等。
# 参见你上传的 v14 代码（v14_poisson_predictor_prewarm.py）。  [引用见回答正文]
# ------------------------------------------------------------

import time, math, hashlib, pickle, pathlib, argparse, warnings, random
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
CACHE_P_IDLE = FIGDIR / f"v{VNUM}_idle_cache.pkl"
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

# ------------------------ 调用记录（沿用 v14） --------------
SLIDING_MAXLEN = 256
RECENT_CALLS: Dict[str, deque] = defaultdict(lambda: deque(maxlen=SLIDING_MAXLEN))

def record_arrival(key: str, ts: Optional[float] = None) -> None:
    RECENT_CALLS[key].append(float(ts) if ts is not None else time.time())

# ------------------------ 预测器主体（沿用 v14） ------------
class PoissonPredictor:
    """
    基于泊松过程的 predictor：
      • 用最近调用时间戳（滑动时间窗）做 λ 的 MLE 估计；
      • 在 lookahead 时间窗 τ 内计算 P{≥1 次到来} = 1 - exp(-λ τ)；
      • 按概率排序并输出候选。
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
        scored: List[Dict[str, Any]] = []
        now = time.time()
        for maker in candidates:
            qc, key, info = maker()  # 注意：这里仅生成电路以拿 key，不做编译
            lam = self.est_lambda(key, now)
            p = self.prob_within(lam, lookahead_sec)
            if p >= prob_threshold:
                scored.append({
                    "key": key, "prob": p, "lambda": lam,
                    "qc_raw": qc, "info": info,
                })
        scored.sort(key=lambda r: r["prob"], reverse=True)
        return scored

# ------------------------ 预热（沿用 v14，增强打印） --------
def prewarm_from_predictions(
    predictor: PoissonPredictor,
    candidates: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    lookahead_sec: float,
    prob_threshold: float,
    max_compile: int = 1
) -> List[Dict[str, Any]]:
    cache = load_cache_idle()
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
        # 真正编译
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
        save_cache_idle(cache)
    return done

# ------------------------ 运行路径（沿用 v14） --------------
def compile_with_idle_cache(
    qc_raw: QuantumCircuit, bk_name: str
) -> Tuple[QuantumCircuit, str, bool, float]:
    """
    只读 idle-cache；若 miss 则编译并写回。
    返回：(qc_exec, key, was_hit, compile_sec)
    """
    key = f"{bk_name}:{md5_qasm(qc_raw)}"
    cache = load_cache_idle()
    qc_exec = cache.get(key)
    was_hit = qc_exec is not None
    compile_sec = 0.0
    if not was_hit:
        tp_kwargs = _prepare_kwargs()
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, **tp_kwargs)
        compile_sec = time.perf_counter() - t0
        cache[key] = qc_exec
        save_cache_idle(cache)
    return qc_exec, key, was_hit, compile_sec

def run_once(qc_func: Callable[[], QuantumCircuit], shots: int = 512) -> Dict[str, Any]:
    """
    前台一次性执行：优先用 idle-cache，miss 则编译并写回。
    同时记录 arrival（供 predictor 使用）。
    """
    qc_raw = qc_func()
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk_name = "AerSV"

    qc_exec, key, was_hit, compile_sec = compile_with_idle_cache(qc_raw, bk_name)

    # 提交并运行
    job = sampler.run([qc_exec], shots=shots)
    res = job.result()
    quasi = res.quasi_dists[0] if hasattr(res, "quasi_dists") else None

    # 记录一次“真实到达”
    record_arrival(key)

    return {
        "key": key,
        "cache_hit": was_hit,
        "compile_sec": compile_sec,
        "n_qubits": qc_raw.num_qubits,
        "depth_in": qc_raw.depth(),
        "depth_transpiled": qc_exec.depth(),
        "size_transpiled": qc_exec.size(),
        "quasi_dist": dict(quasi) if quasi is not None else None,
    }

# ------------------------ 候选构建（多 q / d） --------------
def build_catalog(q_list: List[int], d_list: List[int]):
    """
    为多组 (q, d) × CIRCUITS 构造候选 maker：
      返回 (makers, meta_list)
      makers: List[无参 callable -> (qc_raw, key, info)]
      meta_list: [{'name', 'q', 'd', 'maker_run'}]，其中 maker_run: 无参 callable -> qc_raw
    """
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

                # maker_run: 运行时使用（只返回 qc，不计算 key）
                def _mk_run(name_=name, make_=make, q_=q, d_=d):
                    return lambda: make_(q_, d_)
                meta.append({
                    "name": name, "q": q, "d": d,
                    "maker_run": _mk_run(),
                    "idx": len(meta)
                })
    return makers, meta

# ------------------------ Workload: 泊松叠加 ----------------
def build_workload_poisson_superposition(
    meta: List[Dict[str, Any]],
    workload_len: int,
    hot_fraction: float = 0.2,
    hot_boost: float = 6.0,
    rps: float = 1.0,
    rng_seed: int = 123,
) -> List[Dict[str, Any]]:
    """
    把每个 (name,q,d) 看作一个“类别 i”，构造权重 -> 到达率 λ_i：
      • 先挑 hot 类别，提高权重（hot_boost）；
      • 令 λ_total = rps（每秒请求数），H = workload_len / rps；
      • λ_i = λ_total * w_i / sum(w)；
      • 对每个 i 用指数分布模拟到达时间序列，再合并排序 -> 事件序列；
      • 若事件数 > workload_len：截断；若不足：按 w_i 补采样至长度。
    返回的事件元素：{"name","q","d","maker_run"}
    """
    assert workload_len > 0
    rng = random.Random(rng_seed)
    H = workload_len / max(1e-9, rps)  # 时间地平线（秒）
    m = len(meta)
    if m == 0:
        return []

    # 基线权重
    weights = np.ones(m, dtype=float)
    hot_k = max(1, int(round(hot_fraction * m)))
    hot_idx = rng.sample(range(m), hot_k)
    for i in hot_idx:
        weights[i] *= max(1.0, hot_boost)

    # 归一并给出 λ_i
    w_sum = float(weights.sum())
    lambdas = (weights / w_sum) * rps  # 每秒到达率
    # 按类别分别生成到达时间
    events: List[Tuple[float, int]] = []
    for i, lam in enumerate(lambdas):
        if lam <= 0:
            continue
        t = 0.0
        # 泊松过程到达间隔 ~ Exp(lam)
        while t < H and len(events) < workload_len * 5:  # 防止极端情况膨胀
            dt = rng.expovariate(lam)
            t += dt
            if t <= H:
                events.append((t, i))
    # 合并排序
    events.sort(key=lambda x: x[0])
    # 截断或补齐
    if len(events) > workload_len:
        events = events[:workload_len]
    elif len(events) < workload_len:
        # 还不够的话，按权重再补
        probs = (weights / w_sum).tolist()
        idx_pool = list(range(m))
        need = workload_len - len(events)
        extra = rng.choices(idx_pool, weights=probs, k=need)
        # 用时间戳略晚于 H 的小抖动
        for j, i in enumerate(extra):
            events.append((H + (j+1)*1e-3, i))
        events.sort(key=lambda x: x[0])

    # 映射为运行事件
    workload: List[Dict[str, Any]] = []
    for _, i in events:
        item = meta[i]
        workload.append({
            "name": item["name"], "q": item["q"], "d": item["d"],
            "maker_run": item["maker_run"]
        })
    return workload

# ------------------------ 预喂 recent_calls ----------------
def seed_recent_calls_for_predictor(
    predictor_window_sec: float,
    makers_all: List[Callable[[], Tuple[QuantumCircuit, str, Dict[str, Any]]]],
    workload: List[Dict[str, Any]],
    seed_keys: int = 4,
    per_key_samples: int = 2,
    spacing_sec: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    在第一次 prewarm 前，为“即将出场的前若干类电路”预喂 recent_calls：
      • 选取 workload 前缀里出现频率较高的若干 (name,q,d)；
      • 为其生成 key（调用 maker 但不编译），按 spacing_sec 回填 2~3 个时间戳；
      • 保证时间戳落在 predictor 的滑动窗内（T = predictor_window_sec）。
    返回已喂的条目列表，便于打印。
    """
    # 统计前缀频率
    prefix = workload[:max(1, min(64, len(workload)//2))]  # 看前 1/2 或至多 64 条
    counts: Dict[Tuple[str,int,int], int] = defaultdict(int)
    for it in prefix:
        counts[(it["name"], it["q"], it["d"])] += 1
    # 选 top-K
    popular = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:seed_keys]
    # 在 makers_all 中找到对应项，并生成 key
    seeded: List[Dict[str, Any]] = []
    now = time.time()
    horizon_lo = now - predictor_window_sec * 0.8  # 放在窗里，略靠近现在
    for (name, q, d), _cnt in popular:
        # 找到 maker
        hit_maker = None
        for mk in makers_all:
            qc, key, info = mk()
            if info["circ"] == name and info["q"] == q and info["d"] == d:
                hit_maker = (qc, key, info)
                break
        if hit_maker is None:
            continue
        qc, key, info = hit_maker
        # 回填若干条时间戳
        for j in range(per_key_samples):
            ts = horizon_lo + (j + 1) * spacing_sec  # 递增，确保在窗内
            record_arrival(key, ts=ts)
        seeded.append({"key": key, "name": name, "q": q, "d": d,
                       "samples": per_key_samples, "spacing": spacing_sec})
    return seeded

# ------------------------ 驱动逻辑 -------------------------
def drive_pipeline(
    q_list: List[int],
    d_list: List[int],
    workload_len: int,
    prewarm_every: int,
    shots: int,
    lookahead_sec: float,
    prob_th: float,
    max_compile: int,
    sliding_window_sec: float,
    min_samples: int,
    hot_fraction: float,
    hot_boost: float,
    rps: float,
    rng_seed: int,
):
    # 1) 构造候选全集（跨多组 q/d）
    makers_all, meta = build_catalog(q_list, d_list)

    # 2) 构造泊松叠加式 workload
    workload = build_workload_poisson_superposition(
        meta=meta, workload_len=workload_len,
        hot_fraction=hot_fraction, hot_boost=hot_boost,
        rps=rps, rng_seed=rng_seed
    )
    print(f"=== Workload built: N={len(workload)} across {len(meta)} variants "
          f"(q in {q_list}, d in {d_list}) ===")
    w_preview = ", ".join([f"{it['name']}_q{it['q']}_d{it['d']}" for it in workload[:12]])
    print(f"  preview[12]: {w_preview}{' ...' if len(workload)>12 else ''}")

    # 3) 预测器 + 预喂 recent_calls（确保首轮 prewarm 能生效）
    predictor = PoissonPredictor(sliding_window_sec=sliding_window_sec, min_samples=min_samples)
    seeded = seed_recent_calls_for_predictor(
        predictor_window_sec=sliding_window_sec,
        makers_all=makers_all,
        workload=workload,
        seed_keys=4, per_key_samples=max(2, min_samples), spacing_sec=3.0
    )
    if seeded:
        print(f"=== Seeded {len(seeded)} keys into RECENT_CALLS (each {seeded[0]['samples']} samples) ===")
        for s in seeded:
            print(f"  - seed {s['name']}_q{s['q']}_d{s['d']}  -> {s['key'][:10]}…")

    # 4) 主循环：每 prewarm_every 次 run 之前做一次 prewarm
    #    为了“第一次调用就 predict 成功”，我们在第 0 次 run 前也会做一次 prewarm。
    run_count = 0
    total_hits = 0
    total_compiles = 0

    while run_count < len(workload):
        # ——prewarm——
        if (run_count % prewarm_every) == 0:
            decisions = prewarm_from_predictions(
                predictor=predictor,
                candidates=makers_all,          # 用全集做决策（可改为活跃集优化）
                lookahead_sec=lookahead_sec,
                prob_threshold=prob_th,
                max_compile=max_compile,
            )
            if not decisions:
                print(f"[prewarm @{run_count}] nothing to prewarm")
            else:
                cnt_comp = sum(1 for dct in decisions if dct.get("action") == "compile")
                print(f"[prewarm @{run_count}] compiled={cnt_comp}, total_decisions={len(decisions)}")
                for it in decisions:
                    if it.get("action") == "compile":
                        print(f"  - compile  {it['circ']}_q{it['q']}_d{it['d']}  "
                              f"p={it['prob']:.2f} λ={it['lambda']:.3f}/s  "
                              f"compile={it['compile_sec']:.3f}s")
                total_compiles += cnt_comp

        # ——run（一条）——
        item = workload[run_count]
        name, q, d = item["name"], item["q"], item["d"]
        maker_run = item["maker_run"]
        meta = run_once(maker_run, shots=shots)
        total_hits += int(meta["cache_hit"])
        print(f"[run  #{run_count:04d}] {name}_q{q}_d{d}  hit={meta['cache_hit']}  "
              f"compile={meta['compile_sec']:.3f}s  "
              f"depth_in={meta['depth_in']} -> depthT={meta['depth_transpiled']}")
        run_count += 1

    # 5) 总结
    print("\n=== Summary ===")
    print(f"  runs total:     {len(workload)}")
    print(f"  prewarm compiles: {total_compiles}")
    print(f"  run cache hits: {total_hits}  "
          f"({100.0*total_hits/max(1,len(workload)):.1f}% hit rate)")

# ------------------------ CLI ------------------------------
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # workload / 参数空间
    ap.add_argument("--q_list", type=str, default="5,11",
                    help="逗号分隔的 qubits 列表（如 9, 11）")
    ap.add_argument("--d_list", type=str, default="4,8",
                    help="逗号分隔的 depth 列表（如 4,8）")
    ap.add_argument("--workload_len", type=int, default=60, help="总请求数")
    ap.add_argument("--prewarm_every", type=int, default=2, help="每隔多少次 run 做一次 prewarm")
    ap.add_argument("--shots", type=int, default=256)

    # predictor 与预热
    ap.add_argument("--lookahead", type=float, default=8.0, help="预热预测时间窗 τ (秒)")
    ap.add_argument("--prob_th", type=float, default=0.45, help="预热概率阈值")
    ap.add_argument("--max_compile", type=int, default=2, help="每次 prewarm 的最大编译条数")
    ap.add_argument("--sliding_window_sec", type=float, default=60.0, help="λ 的滑动时间窗")
    ap.add_argument("--min_samples", type=int, default=2, help="估计 λ 所需最少样本数")

    # workload 形状（泊松叠加）
    ap.add_argument("--hot_fraction", type=float, default=0.25, help="热门类别占比")
    ap.add_argument("--hot_boost", type=float, default=8.0, help="热门类别权重放大倍数")
    ap.add_argument("--rps", type=float, default=1.0, help="总体到达率 λ_total（req/s）")
    ap.add_argument("--rng_seed", type=int, default=123)

    args = ap.parse_args()

    drive_pipeline(
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
    )
