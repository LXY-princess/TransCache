# strat_FS_Pre.py
# ccache + predictor prewarm + long unseen removal (ttl: time-to-live) (add cache memory management)
# + score eviction (add another cache memory management)
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict
import time, math

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

                 shots=256, include_exec: bool = True,

                 # Score Eviction
                 # === NEW: 周期性清理（sweep） ===
                 sweep_every: int = 20,  # 每处理多少个请求做一次“清扫”
                 sweep_evict_k: int = 2,  # 每次清扫最多移除多少条
                 sweep_fresh_half_life: float = None,  # 新鲜度半衰期；默认= sliding_window_sec
                 sweep_min_score: float = None,  # 可选阈值，小于阈值的才考虑淘汰；默认 None 表示不设下限

                 # 其他
                 default_compile_est: float = 0.08,  # 编译时长估计的冷启动
                 ):
    """
    FirstSeen + 预测预热 + TTL + 周期性清理（基于 p、编译时长估计与新鲜度的轻量打分）。
    - 不做 miss 准入拦截（FirstSeen 保底仍然成立）；
    - 清扫是温和的后台动作：周期性执行，逐步淘汰“价值低且久未用”的条目。
    """
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
    sliding = float(predictor_cfg["sliding_window_sec"])
    ttl_sec = float(ttl_factor) * sliding
    # print(f"ttl_sec: {ttl_sec}")

    # SE
    # 编译时长 EMA（按 label 聚合，更稳）
    est_compile_ema: Dict[str, float] = defaultdict(lambda: float(default_compile_est))
    # key -> label（用于从 key 反查其 label）
    key2label: Dict[str, str] = {}
    tau_fresh = float(sweep_fresh_half_life) if sweep_fresh_half_life is not None else sliding

    # ---- 评分函数（用于 sweep）----
    def _score_for_eviction(key: str, now_t: float) -> float:
        """分数越低越先被清理。"""
        lam = predictor.est_lambda(key, now=now_t)
        p = predictor.prob_within(lam, lookahead_sec)
        lab = key2label.get(key, None)
        estC = est_compile_ema[lab] if lab is not None else float(default_compile_est)
        age = now_t - last_used_t.get(key, 0.0)  # 未记录过视为“很久未用”
        fresh = math.exp(- max(0.0, age) / max(1e-9, tau_fresh))
        return math.log1p(max(0.0, p * estC)) * fresh

    def _sweep(now_t: float) -> List[str]:
        """周期性清理：返回被移除的 key 列表。"""
        if not cache:
            return []
        scored = []
        for k in list(cache.keys()):
            s = _score_for_eviction(k, now_t)
            if (sweep_min_score is None) or (s <= float(sweep_min_score)):
                scored.append((s, k))
        if not scored:
            return []
        scored.sort(key=lambda x: x[0])  # 分数低 → 先移
        evicted = []
        for _, k in scored[:max(0, int(sweep_evict_k))]:
            cache.pop(k, None)
            last_used_t.pop(k, None)
            evicted.append(k)
        return evicted

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

        # 运行一次；**用 ts=t 统一时间基**（极重要）
        meta = run_once_with_cache(it["maker_run"], cache, shots=shots, ts=t, include_exec=include_exec)

        # 时间推进与记录
        dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind":"run","label":lab,"start":t,"dur":dur})

        t += dur
        cache_size_series.append((t, len(cache)))
        if meta["cache_hit"]:
            hit_by_label[lab] += 1
            total_hits += 1

        # 编译时长 EMA（仅在 miss 编译发生时回灌）
        if meta["compile_sec"] > 0:
            prev = est_compile_ema[lab]
            est_compile_ema[lab] = 0.7 * prev + 0.3 * float(meta["compile_sec"])

        # ttl
        # 只要用到了某个 key（命中或 miss 后加入），就刷新其 last_used
        k = meta.get("key", None)
        if k:
            # 绑定 key->label（无论命中与否，只要有 key）
            key2label[k] = lab
            if k in cache:
                last_used_t[k] = t

        # TTL 去陈 —— 最小侵入、只这一小块
        if ttl_sec > 0:
            stale = [kk for kk, lu in list(last_used_t.items()) if (t - lu) > ttl_sec]
            # if len(stale) > 0:
            #     print(f"ttl remove {len(stale)} items")
            for kk in stale:
                cache.pop(kk, None)
                last_used_t.pop(kk, None)

        # 周期性清理（sweep）
        if (idx > 0) and (idx % max(1, int(sweep_every)) == 0):
            evicted = _sweep(now_t=t)
            # if evicted:
            #     print(f"evicted {len(evicted)} items")
        cache_size_series.append((t, len(cache)))

    metrics = {"hit_by_label": dict(hit_by_label), "total_hits": int(total_hits),
               "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series]}
    return {"events": events, "metrics": metrics}
