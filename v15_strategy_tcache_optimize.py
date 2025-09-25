# v15_strategy_tcache.py
# -----------------------------------------------------------------------------
# Tcache (enhanced):
#  1) Prewarm admission control: only pre-compile & insert when EV >= threshold,
#     and (if cache full) only if it improves upon the current worst entry.
#  2) Eviction scoring switched from product to log-weighted sum:
#       log_value = w_p*log(eps+p) + w_est*log(eps+est) + w_fr*log(eps+freshness)
#     We evict the entry with the MINIMUM log_value (i.e., the least valuable).
#  3) Short-term freezing (pin/TTL): recently inserted or recently hit keys
#     are frozen for a short time window and won't be eviction candidates.
#  4) Conditional write-back on miss: after an online compile (miss),
#     decide to keep or remove the compiled key using the same admission policy.
#
# Notes:
#  - Prewarm time is NOT added to the timeline (S4-like semantics).
#  - We reuse PoissonPredictor / run_once_with_cache / seed_recent_calls
#    from v15_core.py as in your v15 codebase.
# -----------------------------------------------------------------------------

from typing import Any, Dict, List, Tuple, Optional, Iterable
from collections import Counter, defaultdict
import time, math

from v15_core import (
    PoissonPredictor, run_once_with_cache, label_of,
    seed_recent_calls_for_predictor, _prepare_kwargs
)

from qiskit import transpile


# ----------------------------- Helpers (scoring) -----------------------------

def _freshness(now_t: float, last_used_t: Dict[str, float], key: str, win_sec: float) -> float:
    """Freshness in (0,1], larger is better; older entries decay towards 0."""
    age = max(0.0, now_t - last_used_t.get(key, now_t))
    return 1.0 / (1.0 + age / max(1e-9, win_sec))


def _log_value(
    p: float, est_compile: float, fresh: float,
    w_p: float, w_est: float, w_fr: float, eps: float = 1e-9
) -> float:
    """
    Log-weighted value used for eviction order (bigger is better).
    We will evict the MINIMUM log_value.
    """
    return (
        w_p   * math.log(eps + max(0.0, p)) +
        w_est * math.log(eps + max(0.0, est_compile)) +
        w_fr  * math.log(eps + max(0.0, fresh))
    )


def _expected_value(p: float, est_compile: float) -> float:
    """Expected latency saving (seconds) if this entry hits next: EV = p * est."""
    return float(p) * float(est_compile)


def _pick_worst_key_by_log_value(
    keys: Iterable[str],
    now_t: float,
    predictor: PoissonPredictor, lookahead_sec: float,
    key2label: Dict[str, str], est_compile_ema: Dict[str, float],
    last_used_t: Dict[str, float], freeze_until: Dict[str, float],
    w_p: float, w_est: float, w_fr: float,
    default_compile_est: float, eps: float
) -> Tuple[Optional[str], float]:
    """
    Among candidate keys (already filtered by 'protect' etc.), pick the one with
    the MINIMUM log_value => the least valuable entry to keep.
    Return (key, value) where value is the log_value.
    """
    worst_key, worst_val = None, float("inf")
    for k in keys:
        # Frozen entries shouldn't be evicted.
        if freeze_until.get(k, 0.0) > now_t:
            continue
        lam = predictor.est_lambda(k, now_t)
        p = predictor.prob_within(lam, lookahead_sec)
        lbl = key2label.get(k, "")
        est_c = est_compile_ema.get(lbl, default_compile_est)
        fr = _freshness(now_t, last_used_t, k, predictor.sliding_window_sec)
        val = _log_value(p, est_c, fr, w_p, w_est, w_fr, eps=eps)
        if val < worst_val:
            worst_val, worst_key = val, k
    return worst_key, worst_val


# ----------------------- Prewarm admission (no timing) -----------------------

def _prewarm_with_admission(
    predictor: PoissonPredictor,
    makers_all,
    lookahead_sec: float, prob_th: float,
    cache: Dict[str, Any],
    key2label: Dict[str, str],
    est_compile_ema: Dict[str, float],
    last_used_t: Dict[str, float],
    freeze_until: Dict[str, float],
    now_t: float,
    # policy params
    max_compile: int,
    cache_capacity: int,
    admit_ev_min: float,
    admit_allow_replace: bool,
    w_p: float, w_est: float, w_fr: float,
    default_compile_est: float,
    eps: float,
    freeze_sec_on_insert: float,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Score candidates (Poisson), and compile+insert only if they pass admission.
    - If cache not full: EV >= admit_ev_min.
    - If cache full: new_value_log > current_worst_value_log (replacement).
    Return:
      inserted_keys (to be protected in the following eviction),
      counters ({"admit":..., "skip":..., "replace":...}).
    NOTE: predictor time and prewarm compile time are INTENTIONALLY not recorded
          into timeline (S4 semantics).
    """
    stats = {"admit": 0, "skip": 0, "replace": 0}
    inserted: List[str] = []
    compiled = 0

    # Candidate list (already filtered by p>=prob_th, sorted by prob desc)
    decided = predictor.score_candidates(makers_all, lookahead_sec, prob_th)

    for item in decided:
        if compiled >= max_compile:
            break

        key = item["key"]
        p = float(item["prob"])
        lbl = label_of(item["info"]["circ"], item["info"]["q"], item["info"]["d"])
        est_c = est_compile_ema.get(lbl, default_compile_est)
        EV = _expected_value(p, est_c)

        # 1) Quick admission gate by EV threshold
        if EV < admit_ev_min:
            stats["skip"] += 1
            continue

        # 2) If cache has space, admit directly (no replacement needed)
        if len(cache) < cache_capacity:
            # compile and insert
            t1 = time.perf_counter()
            qc_exec = transpile(item["qc_raw"], **_prepare_kwargs())
            _ = time.perf_counter() - t1
            cache[key] = qc_exec
            key2label[key] = lbl
            last_used_t[key] = now_t
            freeze_until[key] = now_t + float(freeze_sec_on_insert)
            # refine compile estimate with prewarm compile time:
            # (We don't measure cost here to keep S4 semantics, so leave EMA as-is.)
            stats["admit"] += 1
            inserted.append(key)
            compiled += 1
            continue

        # 3) Cache full: consider replacement by log_value
        if not admit_allow_replace:
            stats["skip"] += 1
            continue

        # Current worst (among all entries; frozen entries are ignored)
        worst_key, worst_val = _pick_worst_key_by_log_value(
            (k for k in cache.keys() if k != key),
            now_t, predictor, lookahead_sec,
            key2label, est_compile_ema, last_used_t, freeze_until,
            w_p, w_est, w_fr, default_compile_est, eps
        )

        # New value_log (freshness≈1.0 for a new insert)
        new_val = _log_value(p, est_c, 1.0, w_p, w_est, w_fr, eps=eps)

        if worst_key is not None and new_val > worst_val:
            # Replace: evict worst, then insert
            cache.pop(worst_key, None)
            key2label.pop(worst_key, None)
            last_used_t.pop(worst_key, None)
            freeze_until.pop(worst_key, None)

            t1 = time.perf_counter()
            qc_exec = transpile(item["qc_raw"], **_prepare_kwargs())
            _ = time.perf_counter() - t1
            cache[key] = qc_exec
            key2label[key] = lbl
            last_used_t[key] = now_t
            freeze_until[key] = now_t + float(freeze_sec_on_insert)

            stats["replace"] += 1
            inserted.append(key)
            compiled += 1
        else:
            stats["skip"] += 1

    return inserted, stats


# ------------------------------ Eviction policy ------------------------------

def _evict_if_needed_log(
    predictor: PoissonPredictor, lookahead_sec: float,
    cache: Dict[str, Any], now_t: float,
    key2label: Dict[str, str],
    est_compile_ema: Dict[str, float], last_used_t: Dict[str, float],
    freeze_until: Dict[str, float],
    protect: Optional[Iterable[str]],
    capacity: int,
    w_p: float, w_est: float, w_fr: float,
    default_compile_est: float, eps: float
) -> int:
    """
    Evict until len(cache) <= capacity using the MINIMUM log_value rule.
    Frozen keys and 'protect' set won't be considered as victims.
    """
    if capacity <= 0:
        return 0
    evicted = 0
    protect = set(protect or [])
    while len(cache) > capacity:
        candidates = [k for k in cache.keys()
                      if k not in protect and freeze_until.get(k, 0.0) <= now_t]
        if not candidates:
            break
        worst_key, _ = _pick_worst_key_by_log_value(
            candidates, now_t, predictor, lookahead_sec,
            key2label, est_compile_ema, last_used_t, freeze_until,
            w_p, w_est, w_fr, default_compile_est, eps
        )
        if worst_key is None:
            break
        cache.pop(worst_key, None)
        key2label.pop(worst_key, None)
        last_used_t.pop(worst_key, None)
        freeze_until.pop(worst_key, None)
        evicted += 1
    return evicted


# ----------------------------- Main entry (strategy) -------------------------

def run_strategy(
    workload, makers_all, predictor_cfg,
    # prewarm cadence
    prewarm_every: int = 5,
    lookahead_sec: float = 8.0, prob_th: float = 0.45,
    max_compile: int = 3, shots: int = 256,
    # cache & policy params
    cache_capacity: int = 128,
    default_compile_est: float = 0.08,
    # (1) Admission control
    admit_ev_min: float = 0.01,       # seconds of expected saving
    admit_allow_replace: bool = True,
    # (2) Log-weight params
    log_w_p: float = 1.0, log_w_est: float = 1.0, log_w_fresh: float = 0.5,
    log_eps: float = 1e-9,
    # (3) Short-term freezing
    freeze_sec_on_insert: float = 8.0,
    freeze_sec_on_hit: float = 6.0,
    # (4) Conditional write-back on miss will reuse admission+replacement policy
):
    """
    Enhanced Tcache with:
      - prewarm admission,
      - log-weighted eviction,
      - short-term freezing,
      - conditional write-back on miss.

    Prewarm time is not shown on the timeline (S4 semantics). Only 'run' bars are drawn.
    """
    # Predictor & initial seeding so that first prewarm can work (same as your v15).
    predictor = PoissonPredictor(**predictor_cfg)
    seed_recent_calls_for_predictor(
        predictor_cfg["sliding_window_sec"], makers_all, workload
    )

    cache: Dict[str, Any] = {}
    key2label: Dict[str, str] = {}
    est_compile_ema: Dict[str, float] = defaultdict(lambda: default_compile_est)
    last_used_t: Dict[str, float] = {}
    freeze_until: Dict[str, float] = {}

    events: List[Dict[str, Any]] = []
    t = 0.0

    # Metrics
    hit_by_label: Dict[str, int] = Counter()
    total_hits = 0
    cache_size_series: List[Tuple[float, int]] = []
    stats_prewarm = {"admit": 0, "skip": 0, "replace": 0}
    stats_miss_keep = 0
    stats_miss_drop = 0
    stats_evicted = 0

    for idx, it in enumerate(workload):

        # ---------- Prewarm (cadence), with admission & no timeline ----------
        if (idx % prewarm_every) == 0:
            inserted, s = _prewarm_with_admission(
                predictor, makers_all, lookahead_sec, prob_th,
                cache, key2label, est_compile_ema, last_used_t, freeze_until, t,
                max_compile=max_compile, cache_capacity=cache_capacity,
                admit_ev_min=admit_ev_min, admit_allow_replace=admit_allow_replace,
                w_p=log_w_p, w_est=log_w_est, w_fr=log_w_fresh,
                default_compile_est=default_compile_est, eps=log_eps,
                freeze_sec_on_insert=freeze_sec_on_insert,
            )
            stats_prewarm["admit"]   += s["admit"]
            stats_prewarm["skip"]    += s["skip"]
            stats_prewarm["replace"] += s["replace"]

            # capacity check after prewarm (protect inserted)
            stats_evicted += _evict_if_needed_log(
                predictor, lookahead_sec, cache, t,
                key2label, est_compile_ema, last_used_t, freeze_until,
                protect=inserted, capacity=cache_capacity,
                w_p=log_w_p, w_est=log_w_est, w_fr=log_w_fresh,
                default_compile_est=default_compile_est, eps=log_eps
            )
            cache_size_series.append((t, len(cache)))

        # ------------------------------ RUN one job --------------------------
        meta = run_once_with_cache(it["maker_run"], cache, shots=shots)
        lab = label_of(it["name"], it["q"], it["d"])
        run_dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        events.append({"kind": "run", "label": lab, "start": t, "dur": run_dur})

        k = meta["key"]
        key2label.setdefault(k, lab)
        last_used_t[k] = t
        # If it was a miss (compile_sec>0), refine compile estimate
        if meta["compile_sec"] > 0.0:
            prev = est_compile_ema.get(lab, default_compile_est)
            est_compile_ema[lab] = 0.7 * prev + 0.3 * float(meta["compile_sec"])

        # hit stats
        if meta["cache_hit"]:
            hit_by_label[lab] += 1
            total_hits += 1
            # short-term freeze on hit
            freeze_until[k] = max(freeze_until.get(k, 0.0), t + float(freeze_sec_on_hit))

        # Advance timeline
        t += run_dur

        # -------------------------- Conditional write-back -------------------
        # If it was a miss (the compiled key has been inserted by v15_core),
        # decide whether to keep it or drop it immediately based on admission+replacement.
        if meta["compile_sec"] > 0.0:
            # Now we have an arrival recorded by run_once_with_cache(); compute p from predictor.
            lam = predictor.est_lambda(k, t)  # use current time base
            p = predictor.prob_within(lam, lookahead_sec)
            est_c = est_compile_ema.get(lab, default_compile_est)
            EV_new = _expected_value(p, est_c)

            keep = False
            replaced_victim = None

            if EV_new >= admit_ev_min:
                if len(cache) < cache_capacity:
                    keep = True
                elif admit_allow_replace:
                    # Compare new vs current worst by log_value
                    worst_key, worst_val = _pick_worst_key_by_log_value(
                        (x for x in cache.keys() if x != k),
                        t, predictor, lookahead_sec,
                        key2label, est_compile_ema, last_used_t, freeze_until,
                        log_w_p, log_w_est, log_w_fresh, default_compile_est, log_eps
                    )
                    new_val = _log_value(p, est_c, 1.0, log_w_p, log_w_est, log_w_fresh, eps=log_eps)
                    if worst_key is not None and new_val > worst_val:
                        # Replace worst; keep 'k'
                        cache.pop(worst_key, None)
                        key2label.pop(worst_key, None)
                        last_used_t.pop(worst_key, None)
                        freeze_until.pop(worst_key, None)
                        keep = True
                        replaced_victim = worst_key

            if keep:
                # Freeze the newly inserted key briefly
                freeze_until[k] = max(freeze_until.get(k, 0.0), t + float(freeze_sec_on_insert))
                stats_miss_keep += 1
            else:
                # Drop it: simulate "no write-back"
                cache.pop(k, None)
                key2label.pop(k, None)
                last_used_t.pop(k, None)
                freeze_until.pop(k, None)
                stats_miss_drop += 1

        # Capacity check after run (protect the just-used key)
        stats_evicted += _evict_if_needed_log(
            predictor, lookahead_sec, cache, t,
            key2label, est_compile_ema, last_used_t, freeze_until,
            protect={k}, capacity=cache_capacity,
            w_p=log_w_p, w_est=log_w_est, w_fr=log_w_fresh,
            default_compile_est=default_compile_est, eps=log_eps
        )
        cache_size_series.append((t, len(cache)))

    metrics = {
        "hit_by_label": dict(hit_by_label),
        "total_hits": int(total_hits),
        "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series],
        "prewarm_stats": stats_prewarm,
        "miss_keep": int(stats_miss_keep),
        "miss_drop": int(stats_miss_drop),
        "evicted": int(stats_evicted),
        "params": {
            "cache_capacity": int(cache_capacity),
            "admit_ev_min": float(admit_ev_min),
            "admit_allow_replace": bool(admit_allow_replace),
            "log_w": [float(log_w_p), float(log_w_est), float(log_w_fresh)],
            "freeze_insert": float(freeze_sec_on_insert),
            "freeze_hit": float(freeze_sec_on_hit),
        }
    }
    return {"events": events, "metrics": metrics}
