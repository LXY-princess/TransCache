# strategy_tcache_idle_gap.py
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

from v15_core import (PoissonPredictor, run_once_with_cache, label_of,
                      seed_recent_calls_for_predictor)

def run_strategy(workload, makers_all, predictor_cfg,
                 lookahead_sec=8.0, prob_th=0.5, max_compile=2,
                 gap_usage_ratio=0.7,  # 只占用间隙的 70%
                 default_compile_est=0.08,  # 初始估计
                 shots=256):
    """
    Idle‑Tcache（间隙型）：使用 workload 的 t_arr，利用 [now, t_arr) 的空隙做预测+预热，
    仅在“预测时间 + 编译估计”不超过 gap_usage_ratio*gap 时才执行。
    """
    predictor = PoissonPredictor(**predictor_cfg)
    seed_recent_calls_for_predictor(predictor_cfg["sliding_window_sec"], makers_all, workload)

    cache: Dict[str, Any] = {}
    events = []
    now = 0.0

    # 编译时长经验估计
    est: Dict[str, float] = defaultdict(lambda: default_compile_est)

    hit_by_label: Dict[str,int] = Counter()
    total_hits = 0

    for it in workload:
        lab = label_of(it["name"], it["q"], it["d"])
        t_arr = float(it.get("t_arr", now))  # 没有时间戳则认为无间隙

        # —— 利用到达前间隙做预测+预热 —— #
        if now < t_arr:
            gap = t_arr - now
            # 1) predictor 时间
            import time as _t
            t0 = _t.perf_counter()
            decided = predictor.score_candidates(makers_all, lookahead_sec, prob_th)
            pred_sec = _t.perf_counter() - t0

            budget = max(0.0, gap_usage_ratio*gap - pred_sec)
            if pred_sec > 0:
                events.append({"kind":"predict","label":"__predictor__","start":now,"dur":pred_sec})
                now += pred_sec

            # 2) 选择最多 max_compile 条，按 p 从高到低，且累计“估计编译时长”不超过 budget
            compiled = 0
            from qiskit import transpile
            from v15_core import _prepare_kwargs
            for dec in decided:
                if compiled >= max_compile or budget <= 0: break
                plab = label_of(dec["info"]["circ"], dec["info"]["q"], dec["info"]["d"])
                need = est[plab]
                if need <= budget:
                    # 真正编译（消耗真实时间），并记录事件
                    t1 = _t.perf_counter()
                    qc_exec = transpile(dec["qc_raw"], **_prepare_kwargs())
                    comp_sec = _t.perf_counter() - t1
                    cache[dec["key"]] = qc_exec
                    events.append({"kind":"prewarm","label":plab,"start":now,"dur":comp_sec})
                    now += comp_sec
                    budget -= need
                    est[plab] = 0.7*est[plab] + 0.3*comp_sec  # 更新估计
                    compiled += 1
                else:
                    continue
            # 如果超了 t_arr，我们不再推进 now（这里保证使用估计控制，不应越界）
            if now < t_arr:
                events.append({"kind":"queue_wait","label":"__gap__","start":now,"dur":t_arr-now})
                now = t_arr

        # —— 到达，执行 run —— #
        meta = run_once_with_cache(it["maker_run"], cache, shots=shots)
        run_dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        events.append({"kind":"run","label":lab,"start":now,"dur":run_dur}); now += run_dur
        if meta["cache_hit"]:
            hit_by_label[lab] += 1; total_hits += 1
        # 更新估计（若本次发生了 miss，得到真实 compile_sec）
        if meta["compile_sec"] > 0:
            est[lab] = 0.7*est[lab] + 0.3*float(meta["compile_sec"])

    metrics = {"hit_by_label": dict(hit_by_label), "total_hits": int(total_hits)}
    return {"events": events, "metrics": metrics}
