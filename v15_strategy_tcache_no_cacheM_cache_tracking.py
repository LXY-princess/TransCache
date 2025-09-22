# v15_strategy_tcache_series_idle_tracked.py
from typing import Any, Dict, List, Tuple
from collections import Counter
import time
from qiskit import transpile

from v15_core import (
    PoissonPredictor, run_once_with_cache, label_of,
    seed_recent_calls_for_predictor, _prepare_kwargs
)

def _prewarm_with_predictor_and_insert(predictor, makers_all, lookahead_sec, prob_th, max_compile, cache):
    t0 = time.perf_counter()
    decided = predictor.score_candidates(makers_all, lookahead_sec, prob_th)
    _ = time.perf_counter() - t0  # 不计时到 timeline
    inserted = 0
    for item in decided:
        if inserted >= max_compile: break
        key = item["key"]
        if key in cache: continue
        qc_exec = transpile(item["qc_raw"], **_prepare_kwargs())
        cache[key] = qc_exec
        inserted += 1

def run_strategy(workload, makers_all, predictor_cfg,
                 prewarm_every=5, lookahead_sec=8.0, prob_th=0.45, max_compile=3, shots=256):
    predictor = PoissonPredictor(**predictor_cfg)
    seed_recent_calls_for_predictor(predictor_cfg["sliding_window_sec"], makers_all, workload)

    cache: Dict[str, Any] = {}
    events = []
    t = 0.0
    hit_by_label: Dict[str,int] = Counter()
    total_hits = 0

    cache_size_series: List[Tuple[float,int]] = []

    for idx, it in enumerate(workload):
        if (idx % prewarm_every) == 0:
            _prewarm_with_predictor_and_insert(predictor, makers_all, lookahead_sec, prob_th, max_compile, cache)
            cache_size_series.append((t, len(cache)))

        meta = run_once_with_cache(it["maker_run"], cache, shots=shots)
        dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind":"run","label":lab,"start":t,"dur":dur})
        if meta["cache_hit"]:
            hit_by_label[lab] += 1; total_hits += 1
        t += dur
        cache_size_series.append((t, len(cache)))

    metrics = {
        "hit_by_label": dict(hit_by_label),
        "total_hits": int(total_hits),
        "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series]
    }
    return {"events": events, "metrics": metrics}
