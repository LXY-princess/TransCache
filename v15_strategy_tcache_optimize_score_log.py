# v15_strategy_tcache_optimize_score_log.py  (SIM-TIME CONSISTENT VERSION)
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import time
import math

from v15_core import (
    PoissonPredictor, run_once_with_cache, label_of,
    seed_recent_calls_for_predictor, _prepare_kwargs
)

from qiskit import transpile

def _prewarm_with_predictor_and_insert(
    predictor: PoissonPredictor, makers_all, lookahead_sec: float, prob_th: float,
    max_compile: int, cache: Dict[str, Any],
    # for eviction scoring / stats
    now_t: float, key2label: Dict[str, str],
    est_compile_ema: Dict[str, float], last_used_t: Dict[str, float],
    default_compile_est: float
) -> List[str]:
    """
    预热：调用 predictor 在“当前模拟时间 now_t”下评分候选，然后编译并放入 idle cache。
    预热编译的耗时不计入 timeline，但会回灌到 est_compile_ema。
    返回本轮新插入的 key（便于保护它们不被立刻淘汰）。
    """
    t0 = time.perf_counter()
    # ★ 关键修复：显式传 now=now_t，保证与 RECENT_CALLS 的时间基一致
    decided = predictor.score_candidates(makers_all, lookahead_sec, prob_th, now=now_t)
    _ = time.perf_counter() - t0  # 预热/预测时间不计入 timeline（S4 语义）

    inserted_keys: List[str] = []
    compiled = 0
    for item in decided:
        if compiled >= max_compile:
            break
        key = item["key"]
        if key in cache:
            # 已存在，更新新近性
            lbl = key2label.get(key, None)
            if lbl is not None:
                last_used_t[key] = now_t
            continue

        # 实际编译（放入 idle cache）
        qc_raw = item["qc_raw"]
        t1 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        cost = time.perf_counter() - t1

        cache[key] = qc_exec
        lbl = label_of(item["info"]["circ"], item["info"]["q"], item["info"]["d"])
        key2label[key] = lbl
        last_used_t[key] = now_t  # 预热视作“最近活动”
        # 更新 compile 估计（不推进 timeline）
        prev = est_compile_ema.get(lbl, default_compile_est)
        est_compile_ema[lbl] = 0.7 * prev + 0.3 * float(cost)
        compiled += 1
        inserted_keys.append(key)
    return inserted_keys


def _evict_if_needed(
    predictor: PoissonPredictor, lookahead_sec: float,
    cache: Dict[str, Any], capacity: int, now_t: float,
    key2label: Dict[str, str],
    est_compile_ema: Dict[str, float], last_used_t: Dict[str, float],
    protect: Optional[set] = None, default_compile_est: float = 0.08
) -> int:
    """
    当 cache 超出 capacity 时执行淘汰；评分越低越先淘汰。
    评分采用“对数加权的保留价值”，避免乘法模式下的零因子问题。
    关键点：所有概率计算都使用 now_t（模拟时间）。
    """
    if capacity <= 0:
        return 0

    alpha, beta, gamma = 1.0, 1.6, 1.0  # 编译节省权重大一些
    eps = 1e-12

    evicted = 0
    protect = protect or set()
    while len(cache) > capacity:
        worst_key, worst_score = None, float("inf")
        for k, qc in list(cache.items()):
            if k in protect:
                continue
            lam = predictor.est_lambda(k, now=now_t)  # ★ 使用模拟时间
            p = predictor.prob_within(lam, lookahead_sec)
            lbl = key2label.get(k, "")
            est_c = est_compile_ema.get(lbl, default_compile_est)
            age = max(0.0, now_t - last_used_t.get(k, now_t))
            freshness = 1.0 / (1.0 + age / max(1e-9, predictor.sliding_window_sec))
            score_keep = (
                alpha * math.log(max(p, eps))
                + beta  * math.log(max(est_c, eps))
                + gamma * math.log(max(freshness, eps))
            )
            if score_keep < worst_score:
                worst_score, worst_key = score_keep, k

        if worst_key is None:
            break

        cache.pop(worst_key, None)
        key2label.pop(worst_key, None)
        last_used_t.pop(worst_key, None)
        evicted += 1
    return evicted


def run_strategy(
    workload, makers_all, predictor_cfg,
    prewarm_every: int = 5,
    lookahead_sec: float = 8.0, prob_th: float = 0.45,
    max_compile: int = 3, shots: int = 256,
    cache_capacity: int = 25,
    default_compile_est: float = 0.08
):
    """
    TransCache（simulate-time consistent）：
      - 预热/淘汰/准入全部基于**同一条模拟时间轴 t**；
      - 预热编译不计入 timeline，但会更新 compile 估计；
      - 淘汰使用对数加权的“保留价值”（p × compile × freshness）。
    """
    predictor = PoissonPredictor(**predictor_cfg)

    # ★ 关键修复：用模拟时间对 predictor 做 seeding（与后续 now_t 一致）
    seed_recent_calls_for_predictor(
        predictor_cfg["sliding_window_sec"], makers_all, workload,
        seed_keys=4, per_key_samples=2, spacing_sec=3.0,
        use_sim_time=True, base_now=0.0
    )

    cache: Dict[str, Any] = {}  # idle cache
    events = []
    t = 0.0

    # 命中与状态统计
    hit_by_label: Dict[str, int] = Counter()
    total_hits = 0

    # 评分数据
    key2label: Dict[str, str] = {}
    est_compile_ema: Dict[str, float] = defaultdict(lambda: default_compile_est)
    last_used_t: Dict[str, float] = {}

    cache_size_series: List[Tuple[float, int]] = []

    for idx, it in enumerate(workload):
        # ---- 预热（S4：不计入 timeline） ----
        if (idx % prewarm_every) == 0:
            inserted = _prewarm_with_predictor_and_insert(
                predictor, makers_all, lookahead_sec, prob_th, max_compile, cache,
                now_t=t, key2label=key2label,
                est_compile_ema=est_compile_ema, last_used_t=last_used_t,
                default_compile_est=default_compile_est
            )
            _evict_if_needed(
                predictor, lookahead_sec, cache, cache_capacity, t,
                key2label, est_compile_ema, last_used_t,
                protect=set(inserted), default_compile_est=default_compile_est
            )
            cache_size_series.append((t, len(cache)))

        # ---- 到达并执行 ----
        # ★ 关键修复：把模拟时间 t 作为 ts 传递给 record_arrival
        meta = run_once_with_cache(it["maker_run"], cache, shots=shots, ts=t)
        run_dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind": "run", "label": lab, "start": t, "dur": run_dur})

        # 使用/编译后的 key 与统计
        k = meta["key"]
        key2label.setdefault(k, lab)
        if meta["cache_hit"]:
            hit_by_label[lab] += 1
            total_hits += 1
        else:
            # miss 在线编译：更新 compile 估计
            prev = est_compile_ema.get(lab, default_compile_est)
            est_compile_ema[lab] = 0.7 * prev + 0.3 * float(meta["compile_sec"])

        # 推进模拟时间
        t += run_dur
        # 更新新近性为“完成时间”更合理
        last_used_t[k] = t

        # 运行后如超限，保护“刚刚用过”的 k 再淘汰
        _evict_if_needed(
            predictor, lookahead_sec, cache, cache_capacity, t,
            key2label, est_compile_ema, last_used_t,
            protect={k}, default_compile_est=default_compile_est
        )
        cache_size_series.append((t, len(cache)))

    metrics = {
        "hit_by_label": dict(hit_by_label),
        "total_hits": int(total_hits),
        "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series],
        "cache_capacity": int(cache_capacity)
    }
    return {"events": events, "metrics": metrics}
