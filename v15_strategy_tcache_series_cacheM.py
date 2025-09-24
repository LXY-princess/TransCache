# strategy_tcache_series.py
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import time

from v15_core import (PoissonPredictor, run_once_with_cache, label_of,
                      seed_recent_calls_for_predictor)

def prewarm_with_measured_predictor(predictor, candidates, lookahead_sec, prob_th, max_compile, cache):
    # 计时 + 编译（与现有实现一致）
    import time
    t0 = time.perf_counter()
    decided = predictor.score_candidates(candidates, lookahead_sec, prob_th)
    pred_sec = time.perf_counter() - t0

    done = []
    inserted_keys: List[str] = []
    compiled = 0
    from qiskit import transpile
    from v15_core import _prepare_kwargs
    for item in decided:
        if compiled >= max_compile: break
        key = item["key"]
        if key in cache:
            done.append({"key": key, "action":"skip_in_idle", **item}); continue
        qc_raw = item["qc_raw"]
        t1 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        cost = time.perf_counter() - t1
        cache[key] = qc_exec; compiled += 1
        done.append({"key": key, "action":"compile",
                     "prob": item["prob"], "lambda": item["lambda"],
                     "compile_sec": cost,
                     "n_qubits": qc_raw.num_qubits, "depth": qc_raw.depth(),
                     "circ": item["info"]["circ"], "q": item["info"]["q"], "d": item["info"]["d"]})
        inserted_keys.append(key)
    return done, pred_sec, inserted_keys

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

def run_strategy(workload, makers_all, predictor_cfg, prewarm_every=3,
                 lookahead_sec=8.0, prob_th=0.45, cache_capacity: int = 20,
                 max_compile=3, default_compile_est: float = 0.08, shots=256):
    """
    Series-Tcache: 每 prewarm_every 次 run 前，同步地预测 + 预编译（插入主时间轴）。
    """
    # predictor & seed
    predictor = PoissonPredictor(**predictor_cfg)
    seed_recent_calls_for_predictor(predictor_cfg["sliding_window_sec"], makers_all, workload)

    cache: Dict[str, Any] = {}  # idle cache
    events = []
    t = 0.0

    # 命中统计
    hit_by_label: Dict[str,int] = Counter()
    total_hits = 0

    # 评分数据
    key2label: Dict[str, str] = {}
    est_compile_ema: Dict[str, float] = defaultdict(lambda: default_compile_est)
    last_used_t: Dict[str, float] = {}

    for idx, it in enumerate(workload):
        if (idx % prewarm_every) == 0:
            decisions, pred_sec, inserted = prewarm_with_measured_predictor(predictor, makers_all,
                                                                  lookahead_sec, prob_th, max_compile, cache)
            if pred_sec > 0:
                events.append({"kind":"predict","label":"__predictor__","start":t,"dur":pred_sec}); t += pred_sec
            for d in decisions:
                if d.get("action")=="compile" and d.get("compile_sec",0)>0:
                    events.append({"kind":"prewarm","label":label_of(d["circ"], d["q"], d["d"]),
                                   "start":t,"dur":float(d["compile_sec"])})
                    t += float(d["compile_sec"])

            # 预热后如超限，先保护本批刚插入的 key，再淘汰
            _evict_if_needed(
                predictor, lookahead_sec, cache, cache_capacity, t,
                key2label, est_compile_ema, last_used_t,
                protect=set(inserted), default_compile_est=default_compile_est
            )

        meta = run_once_with_cache(it["maker_run"], cache, shots=shots)
        dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind":"run","label":lab,"start":t,"dur":dur}); t += dur
        if meta["cache_hit"]:
            hit_by_label[lab] += 1; total_hits += 1

    metrics = {"hit_by_label": dict(hit_by_label), "total_hits": int(total_hits)}
    return {"events": events, "metrics": metrics}
