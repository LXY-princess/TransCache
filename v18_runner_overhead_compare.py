# -*- coding: utf-8 -*-
"""
v18_runner_peakrss.py

Compare peak RSS (MB) across strategies (FS / PR / TransCache) with per-process isolation.

- For each workload size and each method, spawn a child process and:
  * rebuild makers_all/meta from v18_core.build_catalog(...)
  * inflate a slim workload (name/q/d/t_arr) to attach maker_run
  * run the strategy once
  * measure peak RSS via both psutil-sampling and ru_maxrss, take the max
  * also record peak tracemalloc heap as reference (not the main metric)

Outputs:
  - CSV summary: figs/v18_peakrss/summaries/peakrss_summary.csv
  - JSON summary: .../peakrss_summary.json
  - Plot: figs/v18/plots/scaling_peak_rss.png

Usage:
  python v18_runner_peakrss.py --sizes 50,100 --include_exec false
"""
from __future__ import annotations
import argparse, json, sys, os, time, math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import multiprocessing as mp
import threading

# plotting / table
import pandas as pd
import matplotlib.pyplot as plt

# ---- core helpers & paths (reuse your existing codebase style) ----
from v18_core import (
    ROOT, PLOT_DIR,
    build_catalog,
    clear_recent,
    build_workload_poisson_superposition_exact,
)

# ---------- optional psutil & resource ----------
try:
    import psutil
except Exception:
    psutil = None

try:
    import resource  # not on Windows
except Exception:
    resource = None

# ---------- strategies (import names only used in parent for existence check) ----------
import importlib
def _ensure_importable(modname: str):
    try:
        importlib.import_module(modname)
        return True
    except Exception as e:
        print(f"[warn] cannot import {modname}: {e}")
        return False

STRAT_SPECS = [
    ("TransCache", "v18_strat_FS_Pre_ttl_SE_ema", {"family": "pre"}),
    ("FS",         "v18_strat_FS",                     {"family": "fs"}),
    ("PR",         "v18_strat_PR",                     {"family": "pr"}),
    # ("BL",         "v18_strat_PR",                     {"family": "pr"}),
]
STRAT_SPECS = [(n,m,k) for (n,m,k) in STRAT_SPECS if _ensure_importable(m)]

# ---------- slim workload utils ----------
def slim_workload_for_dump(workload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slim=[]
    for it in workload:
        slim.append({
            "name": it.get("name"),
            "q": int(it.get("q")) if it.get("q") is not None else None,
            "d": int(it.get("d")) if it.get("d") is not None else None,
            # 尽量同时保留两种字段名，兼容不同策略
            "t_arr": float(it.get("t_arr", it.get("ts", 0.0))),
            "ts": float(it.get("t_arr", it.get("ts", 0.0))),
        })
    return slim

def inflate_workload_from_slim(slim: List[Dict[str, Any]], meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 建立 (name,q,d) -> maker_run 的映射
    key2maker = {}
    for m in meta:
        key2maker[(m["name"], int(m["q"]), int(m["d"]))] = m["maker_run"]
    wl = []
    for it in slim:
        tup=(it["name"], int(it["q"]), int(it["d"]))
        mk = key2maker.get(tup)
        if mk is None:
            raise KeyError(f"maker_run not found for {tup}")
        rec = dict(name=it["name"], q=int(it["q"]), d=int(it["d"]),
                   maker_run=mk, t_arr=float(it["t_arr"]), ts=float(it["ts"]))
        wl.append(rec)
    return wl

# ---------- memory measurement helpers ----------
class _Sampler:
    def __init__(self, interval_sec: float = 0.02):
        self.interval = float(interval_sec)
        self._stop = threading.Event()
        self._peak = 0
        self._thr = None
        self._proc = psutil.Process(os.getpid()) if psutil else None
    def start(self):
        if not self._proc: return
        def loop():
            while not self._stop.is_set():
                try:
                    rss = self._proc.memory_info().rss
                    if rss > self._peak: self._peak = rss
                except Exception:
                    pass
                self._stop.wait(self.interval)
        self._thr = threading.Thread(target=loop, daemon=True); self._thr.start()
    def stop(self):
        if not self._proc: return
        self._stop.set()
        if self._thr: self._thr.join(timeout=1.0)
    @property
    def peak_mb(self) -> Optional[float]:
        if not self._proc: return None
        return self._peak / (1024.0*1024.0)

def _ru_maxrss_mb() -> Optional[float]:
    if not resource:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        v = float(usage.ru_maxrss)
        if sys.platform.startswith("darwin"):  # bytes
            return v / (1024.0*1024.0)
        else:  # Linux: kilobytes
            return v / 1024.0
    except Exception:
        return None

# ---------- child process entry ----------
def _child_measure_peakrss(args):
    """
    Args: dict with fields:
      - modname: str, e.g., 'v18_strat_FS'
      - method: 'FS' | 'PR' | 'TransCache'
      - q_list, d_list: lists
      - shots, include_exec: runtime flags
      - predictor: dict for TransCache (sliding_window_sec, min_samples, ...)
      - prewarm kwargs: lookahead_sec, prob_th, max_compile, prewarm_every, ...
      - slim_workload: list of dicts (name,q,d,t_arr,ts)
    Returns: dict {peak_rss_mb, peak_tracemalloc_mb, ru_maxrss_mb, ok, err}
    """
    import importlib, tracemalloc, gc
    from v18_core import build_catalog, clear_recent  # reuse in child
    out = dict(ok=False, err=None, peak_rss_mb=None, peak_tracemalloc_mb=None, ru_maxrss_mb=None)
    try:
        clear_recent()
        mod = importlib.import_module(args["modname"])

        # rebuild catalog & workload
        makers_all, meta = build_catalog(args["q_list"], args["d_list"])
        workload = inflate_workload_from_slim(args["slim_workload"], meta)

        # prepare run kwargs
        run_kwargs = dict(shots=args["shots"], include_exec=args["include_exec"])
        if args["method"] == "TransCache":
            run_kwargs.update(
                dict(makers_all=makers_all, predictor_cfg=args["predictor_cfg"])
            )
            # optional prewarm/ttl/sweep knobs
            for k in ["prewarm_every","lookahead_sec","prob_th","max_compile",
                      "ttl_factor","sweep_every","sweep_evict_k","sweep_fresh_half_life","sweep_min_score",
                      "default_compile_est"]:
                if k in args and args[k] is not None:
                    run_kwargs[k] = args[k]

        # start memory instruments
        sampler = _Sampler(interval_sec=0.02); sampler.start()
        tracemalloc.start()
        # run the strategy
        _ = getattr(mod, "run_strategy")(workload, **run_kwargs)

        # stop instruments
        sampler.stop()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # collect
        out["peak_tracemalloc_mb"] = peak / (1024.0*1024.0)
        out["ru_maxrss_mb"] = _ru_maxrss_mb()
        s = sampler.peak_mb
        # prefer the max of (ru_maxrss, sampler), then fall back to either
        if out["ru_maxrss_mb"] is not None and s is not None:
            out["peak_rss_mb"] = max(out["ru_maxrss_mb"], s)
        else:
            out["peak_rss_mb"] = out["ru_maxrss_mb"] if out["ru_maxrss_mb"] is not None else s

        gc.collect()
        out["ok"] = True
        return out
    except Exception as e:
        out["err"] = f"{type(e).__name__}: {e}"
        return out

# ---------- plotting ----------
def _set_plot_style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})

def plot_peakrss(methods: List[str], sizes: List[int], method2rss: Dict[str, List[float]], out_png: Path):
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    markers=["o","s","^","D","v","P"]
    for i, name in enumerate(methods):
        ys = method2rss.get(name, [])
        ax.plot(sizes, ys, marker=markers[i%len(markers)], label=name,
                markerfacecolor='none', markersize=8, markeredgewidth=2.0)
    ax.set_xlabel("Workload size (requests)")
    ax.set_ylabel("Peak RSS (MB)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600)
    print(f"[save] {out_png}")

# ---------- runner ----------
@dataclass
class Row:
    size: int
    method: str
    peak_rss_mb: Optional[float]
    ru_maxrss_mb: Optional[float]
    peak_tracemalloc_mb: Optional[float]
    ok: bool
    err: Optional[str]

def run(args):
    sizes = [int(x) for x in args.sizes.split(",") if x]
    q_list = [int(x) for x in args.q_list.split(",") if x]
    d_list = [int(x) for x in args.d_list.split(",") if x]

    # --- build once: makers_all/meta (only to get meta shape); workload per N with fixed seeds
    save_root = (ROOT / "peakrss").resolve()
    (save_root/"workloads").mkdir(parents=True, exist_ok=True)
    (save_root/"summaries").mkdir(parents=True, exist_ok=True)

    predictor_cfg = {"sliding_window_sec": args.sliding_window_sec,
                     "min_samples": args.min_samples}

    # storage
    method_names = [name for (name, _, _) in STRAT_SPECS]
    method2rss: Dict[str, List[float]] = {name: [] for name in method_names}
    rows: List[Row] = []

    ctx = mp.get_context("spawn") if sys.platform.startswith("win") else mp.get_context("fork")

    for idx, N in enumerate(sizes):
        rng_seed = args.rng_seed_base + idx * 1009
        # 构造一次 workload（父进程），转“瘦身版”跨进程传递
        makers_all_, meta_ = build_catalog(q_list, d_list)
        workload, info = build_workload_poisson_superposition_exact(
            meta_, N, args.hot_fraction, args.hot_boost, args.rps, rng_seed, return_timestamps=True
        )
        slim = slim_workload_for_dump(workload)
        (save_root/"workloads"/f"wl_N{N}_seed{rng_seed}.json").write_text(
            json.dumps(slim, ensure_ascii=False, indent=2)
        )

        print(f"\n[run] workload size = {N}")
        print("-"*80)

        # 每个方法独立子进程测量
        for (name, modname, meta) in STRAT_SPECS:
            clear_recent()  # fairness
            payload = dict(
                modname=modname, method=name, q_list=q_list, d_list=d_list,
                shots=args.shots, include_exec=bool(args.include_exec),
                predictor_cfg=predictor_cfg,
                slim_workload=slim,
                prewarm_every=args.prewarm_every,
                lookahead_sec=args.lookahead,
                prob_th=args.prob_th,
                max_compile=args.max_compile,
                ttl_factor=args.ttl_factor,
                sweep_every=args.sweep_every,
                sweep_evict_k=args.sweep_evict_k,
                sweep_fresh_half_life=args.sweep_fresh_half_life,
                sweep_min_score=args.sweep_min_score,
                default_compile_est=args.default_compile_est,
            )
            with ctx.Pool(processes=1) as pool:
                res = pool.apply(_child_measure_peakrss, (payload,))

            rss = res.get("peak_rss_mb", None)
            method2rss[name].append(float(rss) if rss is not None else float("nan"))
            rows.append(Row(
                size=N, method=name,
                peak_rss_mb=res.get("peak_rss_mb"),
                ru_maxrss_mb=res.get("ru_maxrss_mb"),
                peak_tracemalloc_mb=res.get("peak_tracemalloc_mb"),
                ok=bool(res.get("ok", False)),
                err=res.get("err")
            ))
            print(f"{name:>12s} | peak_rss_mb = {str(rss):>8s} | tracemalloc = {res.get('peak_tracemalloc_mb')} | ok={res.get('ok')} err={res.get('err')}")

    # --- summarize & plot
    df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = (save_root/"summaries"/"peakrss_summary.csv")
    df.to_csv(csv_path, index=False); print(f"[save] {csv_path}")

    summary = {
        "sizes": sizes,
        "methods": method_names,
        "peak_rss_mb": {m: method2rss[m] for m in method_names},
        "config": {
            "q_list": q_list, "d_list": d_list, "shots": args.shots,
            "hot_fraction": args.hot_fraction, "hot_boost": args.hot_boost,
            "rps": args.rps, "rng_seed_base": args.rng_seed_base,
            "include_exec": bool(args.include_exec),
            "predictor_cfg": predictor_cfg,
            "prewarm_every": args.prewarm_every, "lookahead": args.lookahead,
            "prob_th": args.prob_th, "max_compile": args.max_compile,
        }
    }
    json_path = (save_root/"summaries"/"peakrss_summary.json")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] {json_path}")

    out_png = PLOT_DIR / "scaling_peak_rss.png"
    plot_peakrss(method_names, sizes, method2rss, out_png)

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="50,100,150,200,300,350,400,450,500", help="Comma-separated workload sizes.")
    ap.add_argument("--q_list", type=str, default="5,7,11,13")
    ap.add_argument("--d_list", type=str, default="2,4,6")
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--include_exec", type=lambda s: s.lower() in ("1","true","t","yes","y"), default=True)
    ap.add_argument("--hot_fraction", type=float, default=0.25)
    ap.add_argument("--hot_boost", type=float, default=8.0)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--rng_seed_base", type=int, default=123)

    # predictor / prewarm knobs for TransCache family
    ap.add_argument("--lookahead", type=float, default=8.0)
    ap.add_argument("--prob_th", type=float, default=0.45)
    ap.add_argument("--max_compile", type=int, default=3)
    ap.add_argument("--sliding_window_sec", type=float, default=60.0)
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--prewarm_every", type=int, default=10)
    # extra (optional) knobs to forward if your TransCache supports them
    ap.add_argument("--ttl_factor", type=float, default=None)
    ap.add_argument("--sweep_every", type=int, default=None)
    ap.add_argument("--sweep_evict_k", type=int, default=None)
    ap.add_argument("--sweep_fresh_half_life", type=float, default=None)
    ap.add_argument("--sweep_min_score", type=float, default=None)
    ap.add_argument("--default_compile_est", type=float, default=None)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
