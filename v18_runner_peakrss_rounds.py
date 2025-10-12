# -*- coding: utf-8 -*-
"""
v18_runner_peakrss_repeats.py

Compare peak RSS (MB) across strategies (FS / PR / TransCache) with per-process isolation,
run multiple repeats per (size, method), keep all raw data, and finally plot
ONLY peak_rss_mb with error bars (mean ± std).

Outputs:
  - peakrss/repeats/peakrss_runs.csv                (every run row)
  - peakrss/repeats/peakrss_agg_per_size.csv        (per size & method stats)
  - figs/v18/plots/scaling_peak_rss_errbars.png     (mean ± std)
"""

from __future__ import annotations
import argparse, json, sys, os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import multiprocessing as mp
import threading

import pandas as pd
import numpy as np
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

# ---------- strategies (import check) ----------
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
    ("FS",         "v18_strat_FS",                {"family": "fs"}),
    ("PR",         "v18_strat_PR",                {"family": "pr"}),
]
STRAT_SPECS = [(n,m,k) for (n,m,k) in STRAT_SPECS if _ensure_importable(m)]

# ---------- workload slimming ----------
def slim_workload_for_dump(workload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slim=[]
    for it in workload:
        slim.append({
            "name": it.get("name"),
            "q": int(it.get("q")) if it.get("q") is not None else None,
            "d": int(it.get("d")) if it.get("d") is not None else None,
            "t_arr": float(it.get("t_arr", it.get("ts", 0.0))),
            "ts": float(it.get("t_arr", it.get("ts", 0.0))),
        })
    return slim

def inflate_workload_from_slim(slim: List[Dict[str, Any]], meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    key2maker = {(m["name"], int(m["q"]), int(m["d"])): m["maker_run"] for m in meta}
    wl=[]
    for it in slim:
        tup=(it["name"], int(it["q"]), int(it["d"]))
        mk = key2maker.get(tup)
        if mk is None:
            raise KeyError(f"maker_run not found for {tup}")
        wl.append(dict(name=it["name"], q=int(it["q"]), d=int(it["d"]),
                       maker_run=mk, t_arr=float(it["t_arr"]), ts=float(it["ts"])))
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
    Returns: dict {peak_rss_mb, peak_tracemalloc_mb, ru_maxrss_mb, ok, err}
    """
    import importlib, tracemalloc, gc
    from v18_core import build_catalog, clear_recent  # reuse in child
    out = dict(ok=False, err=None, peak_rss_mb=None, peak_tracemalloc_mb=None, ru_maxrss_mb=None)
    try:
        clear_recent()
        mod = importlib.import_module(args["modname"])
        makers_all, meta = build_catalog(args["q_list"], args["d_list"])
        workload = inflate_workload_from_slim(args["slim_workload"], meta)

        run_kwargs = dict(shots=args["shots"], include_exec=args["include_exec"])
        if args["method"] == "TransCache":
            run_kwargs.update(dict(makers_all=makers_all, predictor_cfg=args["predictor_cfg"]))
            for k in ["prewarm_every","lookahead_sec","prob_th","max_compile",
                      "ttl_factor","sweep_every","sweep_evict_k","sweep_fresh_half_life","sweep_min_score",
                      "default_compile_est"]:
                if k in args and args[k] is not None:
                    run_kwargs[k] = args[k]

        sampler = _Sampler(interval_sec=0.02); sampler.start()
        tracemalloc.start()
        _ = getattr(mod, "run_strategy")(workload, **run_kwargs)
        sampler.stop()
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        out["peak_tracemalloc_mb"] = peak / (1024.0*1024.0)
        out["ru_maxrss_mb"] = _ru_maxrss_mb()
        s = sampler.peak_mb
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

def plot_peakrss_errbars(methods: List[str], sizes: List[int],
                         mean_map: Dict[str, List[float]],
                         std_map: Dict[str, List[float]],
                         out_png: Path):
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o","s","^","D","v","P"]

    import numpy as np
    x = np.asarray(sizes, dtype=float)

    for i, name in enumerate(methods):
        y  = np.asarray(mean_map.get(name, []), dtype=float)
        ye = np.asarray(std_map.get(name,  []), dtype=float)

        # 误差阴影（mean ± std）
        ax.fill_between(x, y - ye, y + ye, alpha=0.18, linewidth=0)

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
    repeat: int
    method: str
    peak_rss_mb: Optional[float]
    ru_maxrss_mb: Optional[float]
    peak_tracemalloc_mb: Optional[float]
    ok: bool
    err: Optional[str]

def run(args):
    sizes  = [int(x) for x in args.sizes.split(",") if x]
    q_list = [int(x) for x in args.q_list.split(",") if x]
    d_list = [int(x) for x in args.d_list.split(",") if x]
    repeats = int(args.repeats)

    save_root = (ROOT / "peakrss").resolve()
    (save_root/"workloads").mkdir(parents=True, exist_ok=True)
    (save_root/"repeats").mkdir(parents=True, exist_ok=True)

    predictor_cfg = {"sliding_window_sec": args.sliding_window_sec,
                     "min_samples": args.min_samples}

    method_names = [name for (name, _, _) in STRAT_SPECS]
    rows: List[Row] = []

    ctx = mp.get_context("spawn") if sys.platform.startswith("win") else mp.get_context("fork")

    for idx, N in enumerate(sizes):
        rng_seed = args.rng_seed_base + idx * 1009
        makers_all_, meta_ = build_catalog(q_list, d_list)
        workload, info = build_workload_poisson_superposition_exact(
            meta_, N, args.hot_fraction, args.hot_boost, args.rps, rng_seed, return_timestamps=True
        )
        slim = slim_workload_for_dump(workload)
        (save_root/"workloads"/f"wl_N{N}_seed{rng_seed}.json").write_text(
            json.dumps(slim, ensure_ascii=False, indent=2)
        )

        print(f"\n[run] workload size = {N}, repeats = {repeats}")
        print("-"*80)

        for rep in range(repeats):
            for (name, modname, meta) in STRAT_SPECS:
                clear_recent()
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

                rows.append(Row(
                    size=N, repeat=rep, method=name,
                    peak_rss_mb=res.get("peak_rss_mb"),
                    ru_maxrss_mb=res.get("ru_maxrss_mb"),
                    peak_tracemalloc_mb=res.get("peak_tracemalloc_mb"),
                    ok=bool(res.get("ok", False)),
                    err=res.get("err")
                ))
                print(f"[N={N:>4}, rep={rep:>2}] {name:>10s} | rss={res.get('peak_rss_mb')} | ru={res.get('ru_maxrss_mb')} | "
                      f"pyheap={res.get('peak_tracemalloc_mb')} | ok={res.get('ok')} err={res.get('err')}")

    # ---- save all runs
    df_runs = pd.DataFrame([asdict(r) for r in rows])
    runs_csv = (save_root/"repeats"/"peakrss_runs.csv")
    df_runs.to_csv(runs_csv, index=False); print(f"[save] {runs_csv}")

    # keep only ok rows for stats/plot
    df_ok = df_runs[df_runs["ok"] == True].copy()

    # ---- aggregate per (size, method)
    def _agg(series):
        arr = series.astype(float).values
        return pd.Series({
            "n":   len(arr),
            "mean": float(np.nanmean(arr)) if len(arr) else np.nan,
            "std":  float(np.nanstd(arr, ddof=0)) if len(arr) else np.nan,
            "median": float(np.nanmedian(arr)) if len(arr) else np.nan,
            "p25": float(np.nanpercentile(arr, 25)) if len(arr) else np.nan,
            "p75": float(np.nanpercentile(arr, 75)) if len(arr) else np.nan,
        })

    agg = (df_ok.groupby(["size","method"])["peak_rss_mb"].apply(_agg).unstack().reset_index())

    # add n_fail
    fails = (df_runs.groupby(["size","method"])["ok"].apply(lambda x: int((~x).sum()))
             .reset_index(name="n_fail"))
    agg = agg.merge(fails, on=["size","method"], how="left")

    agg_csv = (save_root/"repeats"/"peakrss_agg_per_size.csv")
    agg.to_csv(agg_csv, index=False); print(f"[save] {agg_csv}")

    # ---- build mean/std maps for errorbar plot
    sizes_sorted = sorted(df_runs["size"].unique().tolist())
    mean_map: Dict[str, List[float]] = {m: [np.nan]*len(sizes_sorted) for m in method_names}
    std_map:  Dict[str, List[float]] = {m: [np.nan]*len(sizes_sorted) for m in method_names}

    for i, N in enumerate(sizes_sorted):
        for m in method_names:
            sub = df_ok[(df_ok["size"] == N) & (df_ok["method"] == m)]
            if len(sub) > 0:
                vals = sub["peak_rss_mb"].astype(float).values
                mean_map[m][i] = float(np.nanmean(vals))
                std_map[m][i]  = float(np.nanstd(vals, ddof=0))

    # ---- plot mean ± std error bars
    out_png = PLOT_DIR / "scaling_peak_rss_errbars.png"
    plot_peakrss_errbars(method_names, sizes_sorted, mean_map, std_map, out_png)

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="250") #50,100,150,200,250,300,350,400,450,500
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
    ap.add_argument("--prewarm_every", type=int, default=5)
    # extra (optional) knobs to forward if your TransCache supports them
    ap.add_argument("--ttl_factor", type=float, default=None)
    ap.add_argument("--sweep_every", type=int, default=None)
    ap.add_argument("--sweep_evict_k", type=int, default=None)
    ap.add_argument("--sweep_fresh_half_life", type=float, default=None)
    ap.add_argument("--sweep_min_score", type=float, default=None)
    ap.add_argument("--default_compile_est", type=float, default=None)

    # repeats
    ap.add_argument("--repeats", type=int, default=5,
                    help="Number of independent repeats per (size, method).")
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
