# v15_strategy_tcache_adaptive.py (SIM-TIME CONSISTENT VERSION)
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict, deque
import time
import math
import numpy as np

from qiskit import transpile

from v15_core import (
    PoissonPredictor, run_once_with_cache, label_of,
    seed_recent_calls_for_predictor, _prepare_kwargs
)

# ------------------ helper: prewarm ------------------ #
def _prewarm_with_predictor_and_insert(
    predictor: PoissonPredictor, makers_all, lookahead_sec: float, prob_th: float,
    max_compile: int, cache: Dict[str, Any],
    # for eviction scoring / stats
    now_t: float, key2label: Dict[str, str],
    est_compile_ema: Dict[str, float], last_used_t: Dict[str, float],
    default_compile_est: float,
) -> List[str]:
    # ★ 使用模拟时间 now_t
    t0 = time.perf_counter()
    decided = predictor.score_candidates(makers_all, lookahead_sec, prob_th, now=now_t)
    _ = time.perf_counter() - t0  # predictor时间不计入时间线

    inserted_keys: List[str] = []
    compiled = 0
    for item in decided:
        if compiled >= max_compile: break
        key = item["key"]
        if key in cache:
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
        last_used_t[key] = now_t
        prev = est_compile_ema.get(lbl, default_compile_est)
        est_compile_ema[lbl] = 0.7*prev + 0.3*float(cost)
        compiled += 1
        inserted_keys.append(key)
    return inserted_keys

# ------------------ helper: scored eviction --------------- #
def _evict_if_needed(
    predictor: PoissonPredictor, lookahead_sec: float,
    cache: Dict[str, Any], capacity: int, now_t: float,
    key2label: Dict[str, str],
    est_compile_ema: Dict[str, float], last_used_t: Dict[str, float],
    protect: Optional[set] = None, default_compile_est: float = 0.08,
    score_mode: str = "minmax",  # "minmax" or "log"
    alpha: float = 1.0, beta: float = 1.6, gamma: float = 1.0
) -> int:
    if capacity <= 0:
        return 0

    evicted = 0
    protect = protect or set()
    eps = 1e-12

    def _freshness(age, window):
        return 1.0 / (1.0 + age / max(1e-9, window))

    while len(cache) > capacity:
        candidates = []
        for k in list(cache.keys()):
            if k in protect:
                continue
            lam = predictor.est_lambda(k, now=now_t)  # ★ 使用模拟时间
            p = predictor.prob_within(lam, lookahead_sec)
            lbl = key2label.get(k, "")
            est_c = est_compile_ema.get(lbl, default_compile_est)
            age = max(0.0, now_t - last_used_t.get(k, now_t))
            freshness = _freshness(age, predictor.sliding_window_sec)
            candidates.append((k, p, est_c, freshness))

        if not candidates:
            break

        if score_mode == "minmax":
            p_vals = [c[1] for c in candidates]
            c_vals = [c[2] for c in candidates]
            f_vals = [c[3] for c in candidates]
            p_min, p_max = min(p_vals), max(p_vals)
            c_min, c_max = min(c_vals), max(c_vals)
            f_min, f_max = min(f_vals), max(f_vals)
            def _norm(x, lo, hi): return 0.0 if hi <= lo else (x - lo) / (hi - lo)

        worst_key, worst_score = None, float("inf")
        for (k, p, est_c, freshness) in candidates:
            if score_mode == "minmax":
                sp = _norm(p, p_min, p_max)
                sc = _norm(est_c, c_min, c_max)
                sf = _norm(freshness, f_min, f_max)
                score_keep = alpha*sp + beta*sc + gamma*sf
            else:
                score_keep = (
                    alpha * math.log(max(p, eps))
                    + beta  * math.log(max(est_c, eps))
                    + gamma * math.log(max(freshness, eps))
                )
            if score_keep < worst_score:
                worst_score = score_keep; worst_key = k

        if worst_key is None:
            break
        cache.pop(worst_key, None)
        key2label.pop(worst_key, None)
        last_used_t.pop(worst_key, None)
        evicted += 1
    return evicted

# ------------------ Adaptive capacity controller ------------------ #
class AdaptiveCapCtrl:
    def __init__(
        self, cap_init=25, cap_min=5, cap_max=200,
        target_miss=0.15, target_compile_ratio=0.25,
        adjust_every=25, step_frac=0.20, hysteresis=0.02
    ):
        self.cap = int(cap_init)
        self.cap_min = int(cap_min)
        self.cap_max = int(cap_max)
        self.target_miss = float(target_miss)
        self.target_compile_ratio = float(target_compile_ratio)
        self.adjust_every = int(adjust_every)
        self.step_frac = float(step_frac)
        self.hysteresis = float(hysteresis)

        self.win_req = deque(maxlen=self.adjust_every)
        self.win_miss = deque(maxlen=self.adjust_every)
        self.win_compile = deque(maxlen=self.adjust_every)
        self.win_wall = deque(maxlen=self.adjust_every)

    def update_kpi(self, miss: bool, compile_sec: float, wall_sec: float):
        self.win_req.append(1)
        self.win_miss.append(1 if miss else 0)
        self.win_compile.append(max(0.0, float(compile_sec)))
        self.win_wall.append(max(1e-9, float(wall_sec)))

    def should_adjust(self, idx: int) -> bool:
        return (idx > 0) and (idx % self.adjust_every == 0)

    def current_cap(self) -> int:
        return self.cap

    def adjust(self) -> int:
        req = sum(self.win_req) or 1
        miss_ratio = (sum(self.win_miss) / req)
        comp_ratio = (sum(self.win_compile) / max(1e-9, sum(self.win_wall)))

        up_trigger = (miss_ratio > self.target_miss * (1 + self.hysteresis)) or \
                     (comp_ratio > self.target_compile_ratio * (1 + self.hysteresis))
        down_trigger = (miss_ratio < self.target_miss * (1 - self.hysteresis)) and \
                       (comp_ratio < self.target_compile_ratio * (1 - self.hysteresis))

        delta = 0
        step = max(1, int(self.cap * self.step_frac))
        if up_trigger and not down_trigger:
            delta = +step
        elif down_trigger and not up_trigger:
            delta = -step

        self.cap = int(np.clip(self.cap + delta, self.cap_min, self.cap_max))
        return self.cap, dict(miss_ratio=miss_ratio, compile_ratio=comp_ratio, delta=delta)

# ------------------ admission filter ------------------ #
def _should_admit_on_miss(
    key: str, lab: str, predictor: PoissonPredictor, now_t: float, lookahead_sec: float,
    est_compile_ema: Dict[str, float], default_compile_est: float, admit_threshold: float
) -> bool:
    lam = predictor.est_lambda(key, now=now_t)  # ★ 使用模拟时间
    p = predictor.prob_within(lam, lookahead_sec)
    est_c = est_compile_ema.get(lab, default_compile_est)
    expected_saving = p * est_c
    return expected_saving >= admit_threshold

# ------------------ main strategy ------------------ #
def run_strategy(
    workload, makers_all, predictor_cfg,
    prewarm_every: int = 5,
    lookahead_sec: float = 8.0, prob_th: float = 0.45,
    max_compile: int = 3, shots: int = 256,

    # adaptive capacity params
    cap_init: int = 25, cap_min: int = 5, cap_max: int = 200,
    target_miss: float = 0.15, target_compile_ratio: float = 0.25,
    adjust_every: int = 25, step_frac: float = 0.20, hysteresis: float = 0.02,

    # eviction score params
    score_mode: str = "minmax", alpha: float = 1.0, beta: float = 1.6, gamma: float = 1.0,

    # admission threshold (seconds of expected compile saving within lookahead)
    admit_threshold: float = 0.01,

    # stale vacuum (optional)
    ttl_factor: float = 8.0,

    default_compile_est: float = 0.08
):
    """
    自适应 TransCache（统一“模拟时间”）：
    - 自适应容量控制；
    - 在线 miss 按收益准入（p×est_compile）；
    - 预热/淘汰/准入/记录到达都用同一时间轴 t。
    """
    predictor = PoissonPredictor(**predictor_cfg)
    # ★ 用模拟时间做 seeding
    seed_recent_calls_for_predictor(
        predictor_cfg["sliding_window_sec"], makers_all, workload,
        seed_keys=4, per_key_samples=2, spacing_sec=3.0,
        use_sim_time=True, base_now=0.0
    )

    cache: Dict[str, Any] = {}
    events = []
    t = 0.0

    hit_by_label: Dict[str, int] = Counter()
    total_hits = 0

    key2label: Dict[str, str] = {}
    est_compile_ema: Dict[str, float] = defaultdict(lambda: default_compile_est)
    last_used_t: Dict[str, float] = {}

    cache_size_series: List[Tuple[float, int]] = []
    cap_series: List[Tuple[float, int]] = []

    ctrl = AdaptiveCapCtrl(
        cap_init=cap_init, cap_min=cap_min, cap_max=cap_max,
        target_miss=target_miss, target_compile_ratio=target_compile_ratio,
        adjust_every=adjust_every, step_frac=step_frac, hysteresis=hysteresis
    )
    cur_cap = ctrl.current_cap()
    cap_series.append((t, int(cur_cap)))

    ttl_sec = float(ttl_factor) * float(predictor_cfg["sliding_window_sec"])

    for idx, it in enumerate(workload):
        # ----- 预热（不计入时间线） -----
        if (idx % prewarm_every) == 0:
            inserted = _prewarm_with_predictor_and_insert(
                predictor, makers_all, lookahead_sec, prob_th, max_compile, cache,
                now_t=t, key2label=key2label,
                est_compile_ema=est_compile_ema, last_used_t=last_used_t,
                default_compile_est=default_compile_est
            )
            _evict_if_needed(
                predictor, lookahead_sec, cache, cur_cap, t,
                key2label, est_compile_ema, last_used_t,
                protect=set(inserted), default_compile_est=default_compile_est,
                score_mode=score_mode, alpha=alpha, beta=beta, gamma=gamma
            )
            cache_size_series.append((t, len(cache)))

        # ----- 到达并执行（记录到达用模拟时间） -----
        meta = run_once_with_cache(it["maker_run"], cache, shots=shots, ts=t)
        run_dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind": "run", "label": lab, "start": t, "dur": run_dur})

        k = meta["key"]
        key2label.setdefault(k, lab)

        was_miss = (meta["compile_sec"] > 0.0)
        if was_miss:
            prev = est_compile_ema.get(lab, default_compile_est)
            est_compile_ema[lab] = 0.7*prev + 0.3*float(meta["compile_sec"])
            # 按收益准入（用模拟时间）
            if not _should_admit_on_miss(
                key=k, lab=lab, predictor=predictor, now_t=t, lookahead_sec=lookahead_sec,
                est_compile_ema=est_compile_ema, default_compile_est=default_compile_est,
                admit_threshold=admit_threshold
            ):
                cache.pop(k, None)
                key2label.pop(k, None)
                last_used_t.pop(k, None)

        if meta["cache_hit"]:
            hit_by_label[lab] += 1; total_hits += 1

        # 推进时间，并把新近性更新为“完成时间”
        t += run_dur
        last_used_t[k] = t

        # 自适应调 cap
        ctrl.update_kpi(miss=was_miss, compile_sec=float(meta["compile_sec"]), wall_sec=run_dur)
        if ctrl.should_adjust(idx+1):
            new_cap, info = ctrl.adjust()
            if new_cap != cur_cap:
                cur_cap = new_cap
                cap_series.append((t, int(cur_cap)))
                _evict_if_needed(
                    predictor, lookahead_sec, cache, cur_cap, t,
                    key2label, est_compile_ema, last_used_t,
                    protect={k}, default_compile_est=default_compile_est,
                    score_mode=score_mode, alpha=alpha, beta=beta, gamma=gamma
                )

        # TTL 真空
        if ttl_sec > 0:
            stale_keys = [kk for kk, lu in list(last_used_t.items()) if (t - lu) > ttl_sec]
            for kk in stale_keys:
                cache.pop(kk, None)
                key2label.pop(kk, None)
                last_used_t.pop(kk, None)

        cache_size_series.append((t, len(cache)))

    metrics = {
        "hit_by_label": dict(hit_by_label),
        "total_hits": int(total_hits),
        "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series],
        "cap_series": [{"t": float(tt), "cap": int(cc)} for (tt, cc) in cap_series],
        "note": "Adaptive capacity + benefit-based admission; SIM-TIME unified.",
        "params": {
            "cap_init": cap_init, "cap_min": cap_min, "cap_max": cap_max,
            "target_miss": target_miss, "target_compile_ratio": target_compile_ratio,
            "adjust_every": adjust_every, "step_frac": step_frac, "hysteresis": hysteresis,
            "admit_threshold": admit_threshold, "ttl_factor": ttl_factor,
            "score_mode": score_mode, "alpha": alpha, "beta": beta, "gamma": gamma
        }
    }
    return {"events": events, "metrics": metrics}
