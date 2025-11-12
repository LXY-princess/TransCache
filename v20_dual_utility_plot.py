#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
not used in paper results graphs
Compute utility-domain harmonic mean (F1-like) from E2E latency and final cache size,
then plot utility-vs-size curves with mean ± std shading across rounds.

This follows the same calling pattern as the cost-domain script, but uses desirability
mapping (smaller-is-better) for each metric and combines them via a harmonic mean.

Usage examples:
python v20_dual_utility_plot.py \
  --csv /mnt/data/scaling_multi_rounds_long.csv \
  --methods "FS+Pre+ttl+SE+ema,FS,PR" \
  --alpha 0.65 \
  --TL 1.0 --UL 2.0 --sL 2.0 \
  --TC 1.0 --UC 2.0 --sC 2.0 \
  --outdir ./figs/v20

If your column names differ, use --latency-col/--cache-col/--size-col/--method-col/--round-col.
"""

import argparse
import os
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Desirability mapping (smaller-is-better) ----------

def desirability_smaller_is_better(x: float, T: float, U: float, s: float = 2.0) -> float:
    """
    Map a 'smaller is better' metric x to [0,1] using lower target T (good) and upper bound U (bad).
    - x <= T -> 1.0
    - x >= U -> 0.0
    - otherwise ((U - x) / (U - T)) ** s
    """
    if T <= 0 or U <= 0 or U <= T:
        # invalid thresholds; fall back to a safe mapping
        return 0.0 if x <= 0 else 1.0 / (1.0 + math.log1p(x))
    if x <= T:
        return 1.0
    if x >= U:
        return 0.0
    return ((U - x) / (U - T)) ** max(s, 1e-9)


def utility_harmonic(uL: float, uC: float, alpha: float = 0.5, eps: float = 1e-12) -> float:
    """
    Harmonic mean with weights (F1-like in utility space).
    Higher is better. Strongly penalizes imbalance.
    HM = 1 / ( alpha/uL + (1-alpha)/uC )
    """
    uL = max(min(uL, 1.0), eps)
    uC = max(min(uC, 1.0), eps)
    denom = alpha / uL + (1.0 - alpha) / uC
    return 1.0 / denom


# ---------- Column detection ----------

def _first_match(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for pat in candidates:
        for k, orig in cols_lower.items():
            if pat in k:
                return orig
    return None


def autodetect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    latency_col = _first_match(cols, ["e2e_latency", "end_to_end", "e2e", "latency", "runtime", "time"])
    cache_col   = _first_match(cols, ["final_cache_size", "cache_size", "cache", "mem", "memory", "rss"])
    size_col    = _first_match(cols, ["workload_size", "size", "n_circuits", "num", "count", "scale"])
    method_col  = _first_match(cols, ["method", "algo", "algorithm", "name", "technique", "approach"])
    round_col   = _first_match(cols, ["round", "repeat", "trial", "run_id", "rep"])
    return {
        "latency_col": latency_col,
        "cache_col": cache_col,
        "size_col": size_col,
        "method_col": method_col,
        "round_col": round_col,
    }


# ---------- Plotting ----------

def plot_utility_curves(summary: pd.DataFrame, out_png: str, title: str = None):
    """
    summary columns: method, size, util_mean, util_std
    """
    methods = summary["method"].unique().tolist()
    plt.figure(figsize=(8, 5), dpi=140)

    for m in methods:
        dfm = summary[summary["method"] == m].sort_values("size")
        x = dfm["size"].values
        y = dfm["util_mean"].values
        s = dfm["util_std"].values
        plt.plot(x, y, label=str(m))
        plt.fill_between(x, y - s, y + s, alpha=0.2)

    plt.xlabel("Size")
    plt.ylabel("Utility (harmonic mean, higher is better)")
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="figs/v18_score_eviction_rounds5/scaling/summaries/scaling_multi_rounds_long.csv",
                        help="Path to input CSV")
    parser.add_argument("--outdir", type=str, default="./figs/v20_utility", help="Directory to save outputs")
    parser.add_argument("--methods", type=str, default="FS+Pre+ttl+SE+ema, FS, PR",
                        help="Comma-separated method names to include; default=all")

    # Optional explicit column names (auto-detect if omitted)
    parser.add_argument("--latency-col", type=str, default=None, help="Column for E2E latency (lower=better)")
    parser.add_argument("--cache-col", type=str, default=None, help="Column for final cache size (lower=better)")
    parser.add_argument("--size-col", type=str, default=None, help="Column for Size axis")
    parser.add_argument("--method-col", type=str, default=None, help="Column for method name")
    parser.add_argument("--round-col", type=str, default=None, help="Column for round/repeat id")

    # Utility (desirability) hyperparameters
    parser.add_argument("--alpha", type=float, default=0.65, help="Weight on latency utility; 1-alpha on cache utility")
    parser.add_argument("--TL", type=float, default=1.0, help="Latency: target (best) threshold")
    parser.add_argument("--UL", type=float, default=2.0, help="Latency: unacceptable (worst) threshold")
    parser.add_argument("--sL", type=float, default=2.0, help="Latency: curve steepness (>1 harsher)")
    parser.add_argument("--TC", type=float, default=1.0, help="Cache: target (best) threshold")
    parser.add_argument("--UC", type=float, default=2.0, help="Cache: unacceptable (worst) threshold")
    parser.add_argument("--sC", type=float, default=2.0, help="Cache: curve steepness (>1 harsher)")
    parser.add_argument("--eps", type=float, default=1e-12, help="Numerical epsilon to clamp utilities")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Auto-detect columns
    auto = autodetect_columns(df)
    latency_col = args.latency_col or auto["latency_col"]
    cache_col   = args.cache_col   or auto["cache_col"]
    size_col    = args.size_col    or auto["size_col"]
    method_col  = args.method_col  or auto["method_col"]
    round_col   = args.round_col   or auto["round_col"]

    # Validate
    required = {
        "latency": latency_col,
        "cache": cache_col,
        "size": size_col,
        "method": method_col,
        "round": round_col,
    }
    missing = [k for k, v in required.items() if v is None or v not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Auto-detected={auto}. "
            f"Pass explicit --latency-col/--cache-col/--size-col/--method-col/--round-col as needed."
        )

    # Filter by methods if provided
    if args.methods.strip():
        wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
        df = df[df[method_col].astype(str).isin(wanted)].copy()
        if df.empty:
            raise ValueError("After filtering by --methods, no rows remain. Check method names.")

    # Compute utility per row
    def _row_util(r):
        L = float(r[latency_col])
        C = float(r[cache_col])
        uL = desirability_smaller_is_better(L, args.TL, args.UL, args.sL)
        uC = desirability_smaller_is_better(C, args.TC, args.UC, args.sC)
        return utility_harmonic(uL, uC, alpha=args.alpha, eps=args.eps)

    df["_util"] = df.apply(_row_util, axis=1)

    # Save per-row utilities
    perrow_csv = os.path.join(args.outdir, "utility_per_row.csv")
    df_out = df[[method_col, size_col, round_col, latency_col, cache_col, "_util"]].copy()
    df_out.columns = ["method", "size", "round", "e2e_latency", "final_cache_size", "utility"]
    df_out.to_csv(perrow_csv, index=False)

    # Aggregate mean/std across rounds
    summary = (
        df_out.groupby(["method", "size"], as_index=False)
              .agg(util_mean=("utility", "mean"), util_std=("utility", "std"))
              .sort_values(["method", "size"])
    )
    summary_csv = os.path.join(args.outdir, "utility_summary_by_method_size.csv")
    summary.to_csv(summary_csv, index=False)

    # Plot curves
    png_path = os.path.join(args.outdir, "utility_vs_size_mean_std.png")
    title = f"Utility (harmonic mean) vs Size (alpha={args.alpha})"
    plot_utility_curves(summary, png_path, title=title)

    print(f"[OK] Wrote per-row utilities: {perrow_csv}")
    print(f"[OK] Wrote summary:          {summary_csv}")
    print(f"[OK] Wrote plot:             {png_path}")


if __name__ == "__main__":
    main()
