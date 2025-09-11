# strategy_tcache_series.py
from typing import Any, Dict, List, Tuple
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
    return done, pred_sec

def run_strategy(workload, makers_all, predictor_cfg, prewarm_every=3,
                 lookahead_sec=8.0, prob_th=0.45, max_compile=3, shots=256):
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

    for idx, it in enumerate(workload):
        if (idx % prewarm_every) == 0:
            decisions, pred_sec = prewarm_with_measured_predictor(predictor, makers_all,
                                                                  lookahead_sec, prob_th, max_compile, cache)
            if pred_sec > 0:
                events.append({"kind":"predict","label":"__predictor__","start":t,"dur":pred_sec}); t += pred_sec
            for d in decisions:
                if d.get("action")=="compile" and d.get("compile_sec",0)>0:
                    events.append({"kind":"prewarm","label":label_of(d["circ"], d["q"], d["d"]),
                                   "start":t,"dur":float(d["compile_sec"])})
                    t += float(d["compile_sec"])

        meta = run_once_with_cache(it["maker_run"], cache, shots=shots)
        dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind":"run","label":lab,"start":t,"dur":dur}); t += dur
        if meta["cache_hit"]:
            hit_by_label[lab] += 1; total_hits += 1

    metrics = {"hit_by_label": dict(hit_by_label), "total_hits": int(total_hits)}
    return {"events": events, "metrics": metrics}
