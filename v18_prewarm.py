from typing import Any, Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict, deque
import time
import math
import numpy as np

from qiskit import transpile

from v18_core import (
    PoissonPredictor, run_once_with_cache, label_of,
    seed_recent_calls_for_predictor, _prepare_kwargs
)

def prewarm_simple_topk(
    predictor, makers_all, now_t: float,
    lookahead_sec: float, prob_th: float,
    cache: Dict[str, Any], max_compile: int = 3
) -> List[str]:
    """
    决策 = 只按概率排序；执行 = 编译前 max_compile 个未命中 key 并放入 cache。
    不涉及分段/容量/EMA/可调度性，接口稳定，适合最小策略（FS_Pre）。
    """
    decided = predictor.score_candidates(makers_all, lookahead_sec, prob_th, now=now_t)
    # 按出现概率排序
    decided.sort(key=lambda x: x["prob"], reverse=True)

    inserted: List[str] = []
    for item in decided:
        if len(inserted) >= max_compile:
            break
        k = item["key"]
        if k in cache:
            continue
        qc_exec = transpile(item["qc_raw"], **_prepare_kwargs())
        cache[k] = qc_exec
        inserted.append(k)
    return inserted

def _median_or(x: List[float], fallback: float) -> float:
    arr = [float(v) for v in x if v is not None]
    return float(np.median(arr)) if arr else float(fallback)

def _p_within_compile(predictor: PoissonPredictor, lam: float, compile_est: float) -> float:
    return predictor.prob_within(lam, max(0.0, float(compile_est)))

def prewarm_with_predictor_and_insert(
    predictor: PoissonPredictor, makers_all, now_t: float,
    lookahead_sec: float, prob_th: float,
    cache: Dict[str, Any], segQ: Set[str], segP: Set[str],
    capQ: int, cap_total: int,
    key2label: Dict[str, str], est_compile_ema: Dict[str, float], last_used_t: Dict[str, float],

    # usued for the search in est_compile_ema, if a key does not exist, create one with value default_compile_est
    default_compile_est: float,
    max_compile: int,
    pre_kappa: float = 0.25, p_within_compile_th: float = 0.35
) -> List[str]:
    """只用 Probation 的空余预算做预热；候选按 EV 排序并过可调度性门槛。"""
    # 1) 候选（基于 sim-time now_t）
    decided = predictor.score_candidates(makers_all, lookahead_sec, prob_th, now=now_t)

    # 2) 动态阈值 θ_pre
    med_est = _median_or(list(est_compile_ema.values()), default_compile_est)
    occ = len(segQ) + len(segP)
    theta_pre = float(pre_kappa * med_est * (1.0 + max(0.0, math.log((occ + 1) / max(1, cap_total)))))

    # 3) 计算 EV 与可调度性；仅保留性价比高的候选
    enriched = []
    for item in decided:
        lbl = label_of(item["info"]["circ"], item["info"]["q"], item["info"]["d"])
        est_c = est_compile_ema.get(lbl, default_compile_est)
        lam = item["lambda"]
        p = item["prob"]
        ev = p * est_c
        sched_ok = (_p_within_compile(predictor, lam, est_c) >= p_within_compile_th)
        if (ev >= theta_pre) and sched_ok:
            enriched.append((ev, item, est_c))
    enriched.sort(key=lambda x: x[0], reverse=True)

    # 4) 仅用 Probation 的“空余预算”
    budget = max(0, min(max_compile, capQ - len(segQ)))
    inserted: List[str] = []
    for ev, item, est_c in enriched[:budget]:
        key = item["key"]
        if key in cache:  # 已存在（在 P 或 Q）
            last_used_t[key] = now_t
            continue
        qc_raw = item["qc_raw"]
        t1 = time.perf_counter()
        qc_exec = transpile(qc_raw, **_prepare_kwargs())
        cost = time.perf_counter() - t1

        cache[key] = qc_exec
        lbl = label_of(item["info"]["circ"], item["info"]["q"], item["info"]["d"])
        key2label[key] = lbl
        last_used_t[key] = now_t
        segQ.add(key)  # 预热统一进入 Probation

        # 回灌编译时长估计
        prev = est_compile_ema.get(lbl, default_compile_est)
        est_compile_ema[lbl] = 0.7 * prev + 0.3 * float(cost)
        inserted.append(key)

    return inserted
