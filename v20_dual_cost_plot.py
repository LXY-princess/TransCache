#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute softmax-style cost from E2E latency and final cache size,
then plot cost-vs-size curves with mean ± std shading across rounds.

Usage example:
python compute_softmax_cost_plot.py /mnt/data/scaling_multi_rounds_long.csv \
  --methods "Baseline,TransCache v19" \
  --alpha 0.5 --tau 0.1 --kL 1.0 --kC 1.0 \
  --outdir ./outputs

If your column names differ, use --latency-col/--cache-col/--size-col/--method-col/--round-col
to specify them explicitly.
"""
import argparse
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import bold_unicode


# ---------- Cost definition (softmax / log-sum-exp in cost domain) ----------

def cost_softmax(latency: float, cache: float,
                 kL: float = 1.0, kC: float = 1.0,
                 alpha: float = 0.5, tau: float = 0.1) -> float:
    """
    Softmax-style (log-sum-exp) aggregation in cost domain.
    Lower is better. Strictness increases as tau -> 0.
    """
    if latency <= 0 or cache <= 0:
        # Guard to avoid invalid logs; treat non-positive as a large penalty
        return float('inf')
    cL = math.log(latency / kL)
    cC = math.log(cache / kC)
    # log-sum-exp with weights; multiply inside-log arguments by 1/tau in exponent
    x = alpha * cL / tau
    y = (1.0 - alpha) * cC / tau
    m = max(x, y)
    return tau * (m + math.log(math.exp(x - m) + math.exp(y - m)))


# ---------- Column name helpers (robust auto-detection) ----------

def _first_match(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for pat in candidates:
        for k, orig in cols_lower.items():
            if pat in k:
                return orig
    return None


def autodetect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    # Try to find typical names (case-insensitive, contains match)
    latency_col = _first_match(cols, ["e2e_latency"])
    cache_col   = _first_match(cols, ["final_cache_size"])
    size_col    = _first_match(cols, ["size"])
    method_col  = _first_match(cols, ["method"])
    round_col   = _first_match(cols, ["round"])
    return {
        "latency_col": latency_col,
        "cache_col": cache_col,
        "size_col": size_col,
        "method_col": method_col,
        "round_col": round_col,
    }


# ---------- Plotting ----------

def plot_cost_curves(summary: pd.DataFrame, out_png: str, title: str = None):
    """
    summary columns: method, size, cost_mean, cost_std
    作用：为每种方法分配不同的“空心”marker，并画 mean±std 阴影
    """
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 22, "axes.linewidth": 2})
    # methods = summary["method"].unique().tolist()
    methods = ["PR", "FS", "FS+Pre+ttl+SE+ema",]

    plt.figure(figsize=(8, 5), dpi=140)

    # 常见可区分度高的 marker；数量不够会循环复用
    marker_cycle = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    lables = {"FS+Pre+ttl+SE+ema": "TransCache",
              "FS":"CCache",
              "PR": "Braket"
    }

    colors = {"FS+Pre+ttl+SE+ema": "#213d69",
              "FS":"#6b80d6", # #6b84d6
              "PR": "#64a6d1"
    }

    for i, m in enumerate(methods):
        dfm = summary[summary["method"] == m].sort_values("size")
        x = dfm["size"].values
        y = dfm["cost_mean"].values
        s = dfm["cost_std"].values

        mk = marker_cycle[i % len(marker_cycle)]

        # 画曲线 + 空心 marker（markerfacecolor='none'）
        (line,) = plt.plot(
            x, y,
            label=lables.get(str(m)),
            marker=mk,
            markersize=10,
            markerfacecolor='none',   # 空心
            markeredgewidth=3,
            linewidth=3,
            color=colors.get(str(m)),
        )

        # 用与曲线相同的颜色填充阴影
        color = line.get_color()
        plt.fill_between(x, y - s, y + s, alpha=0.20, color=color)

    plt.xlabel("Workload Size", fontsize=26)
    plt.ylabel("BOSC", fontsize=26)
    if title:
        plt.title(title)
    plt.legend(frameon=False, fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.4, linewidth=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close()


# ---------- Main pipeline ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="figs/v18_score_eviction_rounds5/scaling/summaries/scaling_multi_rounds_long.csv" ,help="Path to input CSV (e.g., /mnt/data/scaling_multi_rounds_long.csv)")
    parser.add_argument("--outdir", type=str, default="figs/v20_cost", help="Directory to save outputs")
    parser.add_argument("--methods", type=str, default="FS+Pre+ttl+SE+ema, FS, PR", help="Comma-separated method names to include; default=all")
    # Explicit column names (optional). If omitted, we auto-detect.
    parser.add_argument("--latency-col", type=str, default=None, help="Column for E2E latency (lower=better)")
    parser.add_argument("--cache-col", type=str, default=None, help="Column for final cache size (lower=better)")
    parser.add_argument("--size-col", type=str, default=None, help="Column for Size axis")
    parser.add_argument("--method-col", type=str, default=None, help="Column for method name")
    parser.add_argument("--round-col", type=str, default=None, help="Column for round/repeat id")
    # Cost hyperparams
    parser.add_argument("--alpha", type=float, default=0.65, help="Weight on latency (0..1); 1-alpha on cache")
    parser.add_argument("--tau", type=float, default=0.1, help="Softmax temperature; smaller = more like max")
    parser.add_argument("--kL", type=float, default=1.0, help="Latency neutral unit (for log scaling)")
    parser.add_argument("--kC", type=float, default=1.0, help="Cache neutral unit (for log scaling)")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    # Auto-detect columns if not provided
    auto = autodetect_columns(df)
    latency_col = args.latency_col or auto["latency_col"]
    cache_col   = args.cache_col   or auto["cache_col"]
    size_col    = args.size_col    or auto["size_col"]
    method_col  = args.method_col  or auto["method_col"]
    round_col   = args.round_col   or auto["round_col"]

    # Validate column existence
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

    # Filter methods if requested
    if args.methods.strip():
        wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
        df = df[df[method_col].astype(str).isin(wanted)].copy()
        if df.empty:
            raise ValueError("After filtering by --methods, no rows remain. Check method names.")

    # Compute cost per row
    df["_cost"] = df.apply(
        lambda r: cost_softmax(
            latency=float(r[latency_col]),
            cache=float(r[cache_col]),
            kL=args.kL, kC=args.kC,
            alpha=args.alpha, tau=args.tau
        ),
        axis=1
    )

    # Save per-row with cost
    perrow_csv = os.path.join(args.outdir, "cost_per_row.csv")
    df_out = df[[method_col, size_col, round_col, latency_col, cache_col, "_cost"]].copy()
    df_out.columns = ["method", "size", "round", "e2e_latency", "final_cache_size", "cost"]
    df_out.to_csv(perrow_csv, index=False)

    # Aggregate mean/std across rounds for each (method, size)
    summary = (
        df_out.groupby(["method", "size"], as_index=False)
              .agg(cost_mean=("cost", "mean"), cost_std=("cost", "std"))
              .sort_values(["method", "size"])
    )
    summary_csv = os.path.join(args.outdir, "cost_summary_by_method_size.csv")
    summary.to_csv(summary_csv, index=False)

    # Plot curves
    png_path = os.path.join(args.outdir, "cost_vs_size_mean_std.png")
    plot_title = f"Cost vs Size (alpha={args.alpha}, tau={args.tau})"
    plot_cost_curves(summary, png_path)

    print(f"[OK] Wrote per-row costs: {perrow_csv}")
    print(f"[OK] Wrote summary:      {summary_csv}")
    print(f"[OK] Wrote plot:         {png_path}")


if __name__ == "__main__":
    main()
