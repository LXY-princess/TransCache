# v15_strategy_tcache_series_idle_capped.py
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import time

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
    与 S4 相同：计算候选并编译插入 cache；但这里会更新统计信息，返回刚刚插入的 key 列表（用于保护不被立刻淘汰）。
    """
    t0 = time.perf_counter()
    decided = predictor.score_candidates(makers_all, lookahead_sec, prob_th)
    _ = time.perf_counter() - t0  # S4 语义：不把 predictor 时间体现在时间线上

    inserted_keys: List[str] = []
    compiled = 0
    for item in decided:
        if compiled >= max_compile: break
        key = item["key"]
        if key in cache:
            # 已存在，更新新近性
            lbl = key2label.get(key, None)
            if lbl is not None:
                last_used_t[key] = now_t
            continue

        qc_raw = item["qc_raw"]
        t1 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        cost = time.perf_counter() - t1

        cache[key] = qc_exec
        lbl = label_of(item["info"]["circ"], item["info"]["q"], item["info"]["d"])
        key2label[key] = lbl
        last_used_t[key] = now_t  # 预热视作“最近活动”
        # 更新 compile 估计
        prev = est_compile_ema.get(lbl, default_compile_est)
        est_compile_ema[lbl] = 0.7*prev + 0.3*float(cost)
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
    score = p_next(lookahead) * est_compile * freshness(age)
    返回淘汰的条数。
    """
    if capacity <= 0:  # 保护：不做淘汰
        return 0
    evicted = 0
    protect = protect or set()
    while len(cache) > capacity:
        worst_key, worst_score = None, float("inf")
        for k in list(cache.keys()):
            if k in protect:
                continue
            lam = predictor.est_lambda(k, now_t)
            p = predictor.prob_within(lam, lookahead_sec)
            lbl = key2label.get(k, "")
            est_c = est_compile_ema.get(lbl, default_compile_est)
            age = max(0.0, now_t - last_used_t.get(k, now_t))
            # 新近性：越久未用越小
            freshness = 1.0 / (1.0 + age / max(1e-9, predictor.sliding_window_sec))
            score = p * est_c * freshness
            if score < worst_score:
                worst_score = score
                worst_key = k
        if worst_key is None:
            break
        # 执行淘汰
        print(f"Evicted {worst_key} from {worst_score}")
        cache.pop(worst_key, None)
        key2label.pop(worst_key, None)
        last_used_t.pop(worst_key, None)
        evicted += 1
    return evicted


def run_strategy(
    workload, makers_all, predictor_cfg,
    prewarm_every: int = 5,           # 与 runner 默认一致
    lookahead_sec: float = 8.0, prob_th: float = 0.45,
    max_compile: int = 3, shots: int = 256,
    # 新增：cache 容量与评分参数
    cache_capacity: int = 20,
    default_compile_est: float = 0.08
):
    """
    Tcache-Idle(Capped): 预热时间不计入时间线（与 S4 相同），但对 cache 大小做上限控制。
    超限时按“概率×节省×新近性”得分最低的优先淘汰。
    """
    predictor = PoissonPredictor(**predictor_cfg)
    seed_recent_calls_for_predictor(predictor_cfg["sliding_window_sec"], makers_all, workload)

    cache: Dict[str, Any] = {}  # idle cache
    events = []
    t = 0.0

    # 命中与状态统计
    hit_by_label: Dict[str,int] = Counter()
    total_hits = 0
    # 评分数据
    key2label: Dict[str, str] = {}
    est_compile_ema: Dict[str, float] = defaultdict(lambda: default_compile_est)
    last_used_t: Dict[str, float] = {}

    # cache size 轨迹（用 t 轴）
    cache_size_series: List[Tuple[float, int]] = []

    for idx, it in enumerate(workload):
        # —— 与 S4 相同：每 prewarm_every 次，预热但不把耗时计入 timeline —— #
        if (idx % prewarm_every) == 0:
            inserted = _prewarm_with_predictor_and_insert(
                predictor, makers_all, lookahead_sec, prob_th, max_compile, cache,
                now_t=t, key2label=key2label,
                est_compile_ema=est_compile_ema, last_used_t=last_used_t,
                default_compile_est=default_compile_est
            )
            # 预热后如超限，先保护本批刚插入的 key，再淘汰
            _evict_if_needed(
                predictor, lookahead_sec, cache, cache_capacity, t,
                key2label, est_compile_ema, last_used_t,
                protect=set(inserted), default_compile_est=default_compile_est
            )
            # 记录一次 cache size（时间不前进，与 S4 语义一致）
            cache_size_series.append((t, len(cache)))

        # —— 到达并执行 —— #
        meta = run_once_with_cache(it["maker_run"], cache, shots=shots)
        run_dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind":"run","label":lab,"start":t,"dur":run_dur})
        # 刚刚使用/编译的 key
        k = meta["key"]
        last_used_t[k] = t
        key2label.setdefault(k, lab)
        if meta["compile_sec"] > 0.0:
            # miss 在线编译：更新 compile 估计
            prev = est_compile_ema.get(lab, default_compile_est)
            est_compile_ema[lab] = 0.7*prev + 0.3*float(meta["compile_sec"])

        # 命中统计
        if meta["cache_hit"]:
            hit_by_label[lab] += 1; total_hits += 1

        # 运行推进时间
        t += run_dur

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
