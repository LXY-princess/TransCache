# strat_FS_Pre.py
# ccache + predictor prewarm + long unseen removal (ttl: time-to-live) (add cache memory management)
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict
import time

from v18_core import (PoissonPredictor, run_once_with_cache, label_of,
                      seed_recent_calls_for_predictor)
from v18_prewarm import (prewarm_simple_topk)


def run_strategy(workload, makers_all,
                 # predictor
                 predictor_cfg,

                 # prewarm
                 prewarm_every: int = 5,
                 lookahead_sec: float = 8.0,
                 prob_th: float = 0.6,
                 max_compile: int = 3,

                 # TTL
                 ttl_factor: float = 0.1, # 8.0

                 shots=256, include_exec: bool = True):
    # create cache
    cache: Dict[str, Any] = {}  # idle cache
    t = 0.0

    """Initialize records"""
    # record events
    events = []
    # record hit number
    hit_by_label: Dict[str,int] = Counter()
    total_hits = 0
    # record cache size change
    cache_size_series: List[Tuple[float, int]] = []

    """Initialize predictor"""
    predictor = PoissonPredictor(**predictor_cfg)
    seed_recent_calls_for_predictor(
        predictor_cfg["sliding_window_sec"], makers_all, workload,
        seed_keys=4, per_key_samples=2, spacing_sec=3.0,
        use_sim_time=True, base_now=0.0
    )

    # prepare last-used times for TTL
    last_used_t: Dict[str, float] = {}
    ttl_sec = float(ttl_factor) * float(predictor_cfg["sliding_window_sec"])
    # print(f"ttl_sec: {ttl_sec}")

    """ run over workloads"""
    for idx, it in enumerate(workload):
        # Predictor prewarm
        if (idx % prewarm_every) == 0:
            inserted = prewarm_simple_topk(
                predictor=predictor, makers_all=makers_all, now_t=t,
                lookahead_sec=lookahead_sec, prob_th=prob_th,
                cache=cache, max_compile=max_compile,
            )
            # last_used_t is used for freshness of cache, not wl history
            # wl history can be traced in RECENT_CALLS, no need to record another copy
            # every time an item is called, last_used_t will also update t
            for k in inserted:
                last_used_t[k] = t

        meta = run_once_with_cache(it["maker_run"], cache, shots=shots, ts=t, include_exec=include_exec)
        cache_size_series.append((t, len(cache)))
        dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind":"run","label":lab,"start":t,"dur":dur})
        t += dur
        if meta["cache_hit"]:
            hit_by_label[lab] += 1
            total_hits += 1

        # ttl
        # 只要用到了某个 key（命中或 miss 后加入），就刷新其 last_used
        k = meta.get("key", None)
        if k and k in cache:
            last_used_t[k] = t + dur
        # TTL 去陈 —— 最小侵入、只这一小块
        if ttl_sec > 0:
            stale = [kk for kk, lu in list(last_used_t.items()) if (t - lu) > ttl_sec]
            # if len(stale) > 0:
            #     print(f"ttl remove {len(stale)} items")
            for kk in stale:
                cache.pop(kk, None)
                last_used_t.pop(kk, None)

    metrics = {"hit_by_label": dict(hit_by_label), "total_hits": int(total_hits),
               "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series]}
    return {"events": events, "metrics": metrics}
