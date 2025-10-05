# v15_strategy_tcache_adaptive.py  -- SLRU + EV-gated prewarm + working-set clamp (SIM-TIME)
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict, deque
import time
import math
import numpy as np

from qiskit import transpile

from v15_core import (
    PoissonPredictor, run_once_with_cache, label_of,
    seed_recent_calls_for_predictor, _prepare_kwargs
)

# ---------- small utils ----------
def _median_or(x: List[float], fallback: float) -> float:
    arr = [float(v) for v in x if v is not None]
    return float(np.median(arr)) if arr else float(fallback)

def _p_within_compile(predictor: PoissonPredictor, lam: float, compile_est: float) -> float:
    return predictor.prob_within(lam, max(0.0, float(compile_est)))

def _freshness(age: float, window: float) -> float:
    return 1.0 / (1.0 + age / max(1e-9, float(window)))

# ---------- eviction (value-based) helpers ----------
def _build_candidates(keys: List[str],
                      predictor: PoissonPredictor, now_t: float, lookahead_sec: float,
                      key2label: Dict[str, str], est_compile_ema: Dict[str, float],
                      last_used_t: Dict[str, float], default_compile_est: float):
    cand = []
    for k in keys:
        lam = predictor.est_lambda(k, now=now_t)
        p = predictor.prob_within(lam, lookahead_sec)
        lbl = key2label.get(k, "")
        est_c = est_compile_ema.get(lbl, default_compile_est)
        age = max(0.0, now_t - last_used_t.get(k, now_t))
        fr = _freshness(age, predictor.sliding_window_sec)
        cand.append((k, p, est_c, fr))
    return cand

def _pick_worst(candidates, score_mode: str, alpha: float, beta: float, gamma: float):
    """Return (worst_key, worst_score)."""
    if not candidates:
        return None, float("inf")
    eps = 1e-12
    if score_mode == "minmax":
        p_vals = [c[1] for c in candidates]
        c_vals = [c[2] for c in candidates]
        f_vals = [c[3] for c in candidates]
        p_min, p_max = min(p_vals), max(p_vals)
        c_min, c_max = min(c_vals), max(c_vals)
        f_min, f_max = min(f_vals), max(f_vals)
        def _norm(x, lo, hi): return 0.0 if hi <= lo else (x - lo) / (hi - lo)

    worst_key, worst_score = None, float("inf")
    for (k, p, est_c, fr) in candidates:
        if score_mode == "minmax":
            sp = _norm(p, p_min, p_max)
            sc = _norm(est_c, c_min, c_max)
            sf = _norm(fr, f_min, f_max)
            keep = alpha*sp + beta*sc + gamma*sf
        else:
            keep = (alpha*math.log(max(p, eps))
                    + beta*math.log(max(est_c, eps))
                    + gamma*math.log(max(fr, eps)))
        if keep < worst_score:
            worst_score, worst_key = keep, k
    return worst_key, worst_score

def _evict_segmented(
    predictor: PoissonPredictor, lookahead_sec: float, now_t: float,
    cache: Dict[str, Any], segP: Set[str], segQ: Set[str],
    capP: int, capQ: int,
    key2label: Dict[str, str], est_compile_ema: Dict[str, float], last_used_t: Dict[str, float],
    score_mode: str = "minmax", alpha: float = 1.0, beta: float = 1.6, gamma: float = 1.0,
    protect: Optional[Set[str]] = None, default_compile_est: float = 0.08
) -> int:
    """SLRU：优先从 Probation(evict)，Protected 溢出先 demote 再 evict。"""
    prot = protect or set()
    evicted = 0

    def _erase(k: str):
        nonlocal evicted
        cache.pop(k, None)
        segP.discard(k); segQ.discard(k)
        key2label.pop(k, None); last_used_t.pop(k, None)
        evicted += 1

    # 先把 Probation 压到 capQ
    while len(segQ) > capQ:
        cand = [k for k in segQ if k not in prot]
        if not cand: break
        worst, _ = _pick_worst(
            _build_candidates(cand, predictor, now_t, lookahead_sec, key2label, est_compile_ema, last_used_t, default_compile_est),
            score_mode, alpha, beta, gamma
        )
        if worst is None: break
        _erase(worst)

    # Protected 超额：先找“价值最差”的 P，demote 到 Q，再按需从 Q evict
    while len(segP) > capP:
        candP = [k for k in segP if k not in prot]
        if not candP:
            break
        demote_key, _ = _pick_worst(
            _build_candidates(candP, predictor, now_t, lookahead_sec, key2label, est_compile_ema, last_used_t, default_compile_est),
            score_mode, alpha, beta, gamma
        )
        if demote_key is None:
            break
        segP.discard(demote_key); segQ.add(demote_key)
        # 如超额，再从 Q 中淘汰一个
        if len(segQ) > capQ:
            candQ = [k for k in segQ if k not in prot]
            worst, _ = _pick_worst(
                _build_candidates(candQ, predictor, now_t, lookahead_sec, key2label, est_compile_ema, last_used_t, default_compile_est),
                score_mode, alpha, beta, gamma
            )
            if worst is None: break
            _erase(worst)

    return evicted

# ---------- prewarm with EV gating ----------
def _prewarm_with_predictor_and_insert(
    predictor: PoissonPredictor, makers_all, now_t: float,
    lookahead_sec: float, prob_th: float,
    cache: Dict[str, Any], segQ: Set[str], segP: Set[str],
    capQ: int, cap_total: int,
    key2label: Dict[str, str], est_compile_ema: Dict[str, float], last_used_t: Dict[str, float],
    default_compile_est: float, max_compile: int,
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

# ---------- adaptive capacity controller with working-set clamp ----------
class AdaptiveCapCtrl:
    def __init__(
        self, cap_init=25, cap_min=5, cap_max=200,
        target_miss=0.15, target_compile_ratio=0.25,
        adjust_every=25, step_frac_up=0.15, step_frac_down=0.25, hysteresis=0.02,
        overprov: float = 1.25
    ):
        self.cap = int(cap_init)
        self.cap_min = int(cap_min); self.cap_max = int(cap_max)
        self.target_miss = float(target_miss)
        self.target_compile_ratio = float(target_compile_ratio)
        self.adjust_every = int(adjust_every)
        self.step_frac_up = float(step_frac_up)
        self.step_frac_down = float(step_frac_down)
        self.hysteresis = float(hysteresis)
        self.overprov = float(overprov)

        self.win_req = deque(maxlen=self.adjust_every)
        self.win_miss = deque(maxlen=self.adjust_every)
        self.win_compile = deque(maxlen=self.adjust_every)
        self.win_wall = deque(maxlen=self.adjust_every)
        self.win_keys = deque(maxlen=self.adjust_every)

    def update_kpi(self, key: str, miss: bool, compile_sec: float, wall_sec: float):
        self.win_req.append(1)
        self.win_miss.append(1 if miss else 0)
        self.win_compile.append(max(0.0, float(compile_sec)))
        self.win_wall.append(max(1e-9, float(wall_sec)))
        self.win_keys.append(key)

    def should_adjust(self, idx: int) -> bool:
        return (idx > 0) and (idx % self.adjust_every == 0)

    def current_cap(self) -> int:
        return self.cap

    def adjust(self) -> Tuple[int, Dict[str, float]]:
        req = sum(self.win_req) or 1
        miss_ratio = (sum(self.win_miss) / req)
        comp_ratio = (sum(self.win_compile) / max(1e-9, sum(self.win_wall)))

        up_trigger = (miss_ratio > self.target_miss * (1 + self.hysteresis)) or \
                     (comp_ratio > self.target_compile_ratio * (1 + self.hysteresis))
        down_trigger = (miss_ratio < self.target_miss * (1 - self.hysteresis)) and \
                       (comp_ratio < self.target_compile_ratio * (1 - self.hysteresis))

        # 工作集上限：最近窗口的“唯一键数”
        uniq_recent = len(set(self.win_keys))
        cap_ws_max = max(self.cap_min, int(self.overprov * max(1, uniq_recent)))

        # 非对称步长：涨得慢、降得快
        delta = 0
        step_up = max(1, int(self.cap * self.step_frac_up))
        step_dn = max(1, int(self.cap * self.step_frac_down))
        if up_trigger and not down_trigger:
            delta = +step_up
        elif down_trigger and not up_trigger:
            delta = -step_dn

        new_cap = int(np.clip(self.cap + delta, self.cap_min, min(self.cap_max, cap_ws_max)))
        self.cap = new_cap
        return self.cap, dict(miss_ratio=miss_ratio, compile_ratio=comp_ratio,
                              uniq_recent=float(uniq_recent), cap_ws_max=float(cap_ws_max), delta=float(delta))

# ---------- admission on miss (benefit-based) ----------
def _should_admit_on_miss(
    key: str, lab: str, predictor: PoissonPredictor, now_t: float, lookahead_sec: float,
    est_compile_ema: Dict[str, float], default_compile_est: float,
    admit_threshold_base: float, p_admit_min: float,
    occ: int, cap_total: int
) -> bool:
    lam = predictor.est_lambda(key, now=now_t)
    p = predictor.prob_within(lam, lookahead_sec)
    if p < float(p_admit_min):
        return False
    est_c = est_compile_ema.get(lab, default_compile_est)
    # 动态阈值：占用越满越严格
    theta = float(admit_threshold_base * (1.0 + max(0.0, math.log((occ + 1) / max(1, cap_total)))))
    return (p * est_c) >= theta

# ---------- main strategy ----------
def run_strategy(
    workload, makers_all, predictor_cfg,
    prewarm_every: int = 5,
    lookahead_sec: float = 8.0, prob_th: float = 0.6,  # 稍提阈值
    max_compile: int = 3, shots: int = 256,
    include_exec: bool = True,

    # adaptive capacity
    cap_init: int = 25, cap_min: int = 5, cap_max: int = 200,
    target_miss: float = 0.15, target_compile_ratio: float = 0.25,
    adjust_every: int = 25, step_frac_up: float = 0.15, step_frac_down: float = 0.25,
    hysteresis: float = 0.02, overprov: float = 1.25,

    # SLRU segment split
    protected_frac: float = 0.7,

    # eviction score params
    score_mode: str = "minmax", alpha: float = 1.0, beta: float = 1.6, gamma: float = 1.0,

    # admission thresholds
    admit_threshold: float = 0.01, p_admit_min: float = 0.20,

    # prewarm EV gates
    pre_kappa: float = 0.25, p_within_compile_th: float = 0.35,

    # stale vacuum
    ttl_factor: float = 8.0,

    default_compile_est: float = 0.08
):
    """
    Adaptive TransCache (enhanced):
      - SLRU 两段缓存（Probation/Protected），先 Probation 命中再晋升，淘汰优先 Probation；
      - 预热按 EV=p×est_compile 排序 + 可调度性门槛，仅用 Probation 空余预算；
      - 自适应 cap 引入“工作集上限钳制（uniq_recent×overprov）”，抑制规模增大时的膨胀；
      - Miss 准入阈值随占用率动态增长。
    """
    predictor = PoissonPredictor(**predictor_cfg)

    # —— 用模拟时间做 seeding（与随后 now_t 对齐） —— #
    try:
        seed_recent_calls_for_predictor(
            predictor_cfg["sliding_window_sec"], makers_all, workload,
            seed_keys=4, per_key_samples=2, spacing_sec=3.0,
            use_sim_time=True, base_now=0.0
        )
    except TypeError:
        # 兼容旧版 v15_core（无 use_sim_time）
        seed_recent_calls_for_predictor(predictor_cfg["sliding_window_sec"], makers_all, workload)

    cache: Dict[str, Any] = {}    # 实体缓存（两段共享）
    segP: Set[str] = set()        # Protected 集
    segQ: Set[str] = set()        # Probation 集

    events = []
    t = 0.0

    hit_by_label: Dict[str, int] = Counter()
    total_hits = 0

    key2label: Dict[str, str] = {}
    est_compile_ema: Dict[str, float] = defaultdict(lambda: default_compile_est)
    last_used_t: Dict[str, float] = {}

    cache_size_series: List[Tuple[float, int]] = []
    cap_series: List[Tuple[float, int]] = []

    # 自适应控制器（工作集钳制）
    ctrl = AdaptiveCapCtrl(
        cap_init=cap_init, cap_min=cap_min, cap_max=cap_max,
        target_miss=target_miss, target_compile_ratio=target_compile_ratio,
        adjust_every=adjust_every, step_frac_up=step_frac_up, step_frac_down=step_frac_down,
        hysteresis=hysteresis, overprov=overprov
    )
    cur_cap = ctrl.current_cap()
    cap_series.append((t, int(cur_cap)))
    ttl_sec = float(ttl_factor) * float(predictor_cfg["sliding_window_sec"])

    # 兼容 run_once 接口（新：支持 ts；旧：不支持）
    def _run_once_sim(qc_func, cache_dict, shots_, ts_, include_exec: bool = True):
        try:
            return run_once_with_cache(qc_func, cache_dict, shots=shots_, ts=ts_, include_exec = include_exec)
        except TypeError:
            return run_once_with_cache(qc_func, cache_dict, shots=shots_, include_exec = include_exec)

    for idx, it in enumerate(workload):
        # ---- 计算本轮 segment cap ----
        capP = int(round(cur_cap * float(protected_frac)))
        capQ = int(max(0, cur_cap - capP))

        # ---- 预热（不计入时间线；仅用 Probation 空余预算） ----
        if (idx % prewarm_every) == 0:
            inserted = _prewarm_with_predictor_and_insert(
                predictor=predictor, makers_all=makers_all, now_t=t,
                lookahead_sec=lookahead_sec, prob_th=prob_th,
                cache=cache, segQ=segQ, segP=segP,
                capQ=capQ, cap_total=cur_cap,
                key2label=key2label, est_compile_ema=est_compile_ema, last_used_t=last_used_t,
                default_compile_est=default_compile_est, max_compile=max_compile,
                pre_kappa=pre_kappa, p_within_compile_th=p_within_compile_th
            )
            _ = _evict_segmented(
                predictor, lookahead_sec, t, cache, segP, segQ, capP, capQ,
                key2label, est_compile_ema, last_used_t,
                score_mode=score_mode, alpha=alpha, beta=beta, gamma=gamma,
                protect=set(inserted), default_compile_est=default_compile_est
            )
            cache_size_series.append((t, len(cache)))

        # ---- 到达并执行（记录到达用模拟时间） ----
        meta = _run_once_sim(it["maker_run"], cache, shots, t, include_exec = include_exec,)
        run_dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind": "run", "label": lab, "start": t, "dur": run_dur})
        k = meta["key"]
        key2label.setdefault(k, lab)

        was_miss = (meta["compile_sec"] > 0.0)
        if was_miss:
            # 在线编译：先更新编译时长估计
            prev = est_compile_ema.get(lab, default_compile_est)
            est_compile_ema[lab] = 0.7 * prev + 0.3 * float(meta["compile_sec"])
            # 准入（放入 Probation 或丢弃）
            occ_now = len(segP) + len(segQ)
            if _should_admit_on_miss(
                key=k, lab=lab, predictor=predictor, now_t=t, lookahead_sec=lookahead_sec,
                est_compile_ema=est_compile_ema, default_compile_est=default_compile_est,
                admit_threshold_base=admit_threshold, p_admit_min=p_admit_min,
                occ=occ_now, cap_total=cur_cap
            ):
                segQ.add(k)
                last_used_t[k] = t + run_dur
            else:
                # 用完即扔
                cache.pop(k, None); key2label.pop(k, None); last_used_t.pop(k, None)
        else:
            # 命中：Probation 命中则晋升 Protected；Protected 命中仅更新新近性
            if k in segQ:
                segQ.discard(k); segP.add(k)
            last_used_t[k] = t + run_dur
            hit_by_label[lab] += 1; total_hits += 1

        # 推进时间
        t += run_dur

        # ---- 运行后：按 segment cap 做清理（先 Q 后 P demote→Q→evict） ----
        _ = _evict_segmented(
            predictor, lookahead_sec, t, cache, segP, segQ, capP, capQ,
            key2label, est_compile_ema, last_used_t,
            score_mode=score_mode, alpha=alpha, beta=beta, gamma=gamma,
            protect={k}, default_compile_est=default_compile_est
        )

        # ---- 自适应调 cap（含工作集钳制） ----
        ctrl.update_kpi(key=k, miss=was_miss, compile_sec=float(meta["compile_sec"]), wall_sec=run_dur)
        if ctrl.should_adjust(idx + 1):
            new_cap, info = ctrl.adjust()
            if new_cap != cur_cap:
                cur_cap = new_cap
                cap_series.append((t, int(cur_cap)))
                # 新 cap 下再清理一次（保护最近用过的 k）
                capP = int(round(cur_cap * float(protected_frac)))
                capQ = int(max(0, cur_cap - capP))
                _ = _evict_segmented(
                    predictor, lookahead_sec, t, cache, segP, segQ, capP, capQ,
                    key2label, est_compile_ema, last_used_t,
                    score_mode=score_mode, alpha=alpha, beta=beta, gamma=gamma,
                    protect={k}, default_compile_est=default_compile_est
                )

        # ---- TTL 真空（去陈） ----
        if ttl_sec > 0:
            stale = [kk for kk, lu in list(last_used_t.items()) if (t - lu) > ttl_sec]
            for kk in stale:
                cache.pop(kk, None); key2label.pop(kk, None); last_used_t.pop(kk, None)
                segP.discard(kk); segQ.discard(kk)

        cache_size_series.append((t, len(cache)))

    metrics = {
        "hit_by_label": dict(hit_by_label),
        "total_hits": int(total_hits),
        "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series],
        "cap_series": [{"t": float(tt), "cap": int(cc)} for (tt, cc) in cap_series],
        "note": "Adaptive+SLRU+EV-prewarm+schedulability+WS-clamp (sim-time unified).",
        "params": {
            "cap_init": cap_init, "cap_min": cap_min, "cap_max": cap_max,
            "target_miss": target_miss, "target_compile_ratio": target_compile_ratio,
            "adjust_every": adjust_every, "step_frac_up": step_frac_up, "step_frac_down": step_frac_down,
            "hysteresis": hysteresis, "overprov": overprov, "protected_frac": protected_frac,
            "admit_threshold": admit_threshold, "p_admit_min": p_admit_min,
            "pre_kappa": pre_kappa, "p_within_compile_th": p_within_compile_th,
            "score_mode": score_mode, "alpha": alpha, "beta": beta, "gamma": gamma,
            "ttl_factor": ttl_factor
        }
    }
    return {"events": events, "metrics": metrics}
