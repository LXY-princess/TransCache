# strategy_tcache_series.py
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict
import time

from v15_core import (PoissonPredictor, run_once_with_cache, label_of,
                      seed_recent_calls_for_predictor)


def run_strategy(workload, makers_all, predictor_cfg, prewarm_every=3,
                 lookahead_sec=8.0, prob_th=0.45, max_compile=3, shots=256):

    cache: Dict[str, Any] = {}  # idle cache
    events = []
    t = 0.0

    # 命中统计
    hit_by_label: Dict[str,int] = Counter()
    total_hits = 0

    cache_size_series: List[Tuple[float, int]] = []

    for idx, it in enumerate(workload):
        meta = run_once_with_cache(it["maker_run"], cache, shots=shots)
        cache_size_series.append((t, len(cache)))
        dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind":"run","label":lab,"start":t,"dur":dur}); t += dur
        if meta["cache_hit"]:
            hit_by_label[lab] += 1; total_hits += 1

    metrics = {"hit_by_label": dict(hit_by_label), "total_hits": int(total_hits),
               "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series]}
    return {"events": events, "metrics": metrics}
