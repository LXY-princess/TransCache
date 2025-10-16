#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot final_hitrate (mean ± std across rounds) vs Size for each method.

Keeps the calling/structure style consistent with v20_dual_cost_plot.py:
- Auto-detects common column names, but can be overridden via CLI flags
- Optional --methods filter (comma-separated)
- Saves per-row and summary CSVs, plus a PNG plot with mean±std shading
- Uses distinct hollow markers per method

Usage:
python v20_hitrate_plot.py \
  --csv /mnt/data/scaling_multi_rounds_long.csv \
  --methods "FS+Pre+ttl+SE+ema, FS, PR" \
  --outdir ./figs/v20

If your column names differ, specify:
  --hitrate-col final_hitrate --size-col size --method-col method --round-col round
"""
import argparse
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    hitrate_col = _first_match(cols, ["final_hitrate"])
    size_col    = _first_match(cols, ["size"])
    method_col  = _first_match(cols, ["method"])
    round_col   = _first_match(cols, ["round"])
    return {
        "hitrate_col": hitrate_col,
        "size_col": size_col,
        "method_col": method_col,
        "round_col": round_col,
    }


# ---------- Plotting ----------

def plot_hitrate_curves(summary: pd.DataFrame, out_png: str, title: str = None, ylabel: str = "Final hit rate"):
    """
    summary columns: method, size, hitrate_mean, hitrate_std
    Draws per-method curve with hollow markers and mean±std ribbon.
    """
    # methods = summary["method"].unique().tolist()
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 22, "axes.linewidth": 2,  })
    methods = ["PR", "FS", "FS+Pre+ttl+SE+ema", ]
    plt.figure(figsize=(8, 5), dpi=140)

    marker_cycle = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

    lables = {"FS+Pre+ttl+SE+ema": "TransCache",
              "FS": "CCache",
              "PR": "Braket"
              }

    colors = {"FS+Pre+ttl+SE+ema": "#213d69",
              "FS": "#6b80d6",  # #6b84d6
              "PR": "#64a6d1"
              }

    for i, m in enumerate(methods):
        dfm = summary[summary["method"] == m].sort_values("size")
        x = dfm["size"].values
        y = dfm["hitrate_mean"].values
        s = dfm["hitrate_std"].values

        mk = marker_cycle[i % len(marker_cycle)]
        # 画曲线 + 空心 marker（markerfacecolor='none'）
        (line,) = plt.plot(
            x, y,
            label=lables.get(str(m)),
            marker=mk,
            markersize=10,
            markerfacecolor='none',  # 空心
            markeredgewidth=3,
            linewidth=3,
            color=colors.get(str(m)),
        )
        color = line.get_color()
        plt.fill_between(x, y - s, y + s, alpha=0.20, color=color)

    plt.xlabel("Workload Size", fontsize=26)
    plt.ylabel("Workload Hitrate", fontsize=26)
    if title:
        plt.title(title)
    plt.legend(frameon=False, fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.4, linewidth=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="figs/v18_score_eviction_rounds5/scaling/summaries/scaling_multi_rounds_long.csv",
                        help="Path to input CSV")
    parser.add_argument("--outdir", type=str, default="./figs/v20_hitrate", help="Directory to save outputs")
    parser.add_argument("--methods", type=str, default="FS+Pre+ttl+SE+ema, FS, PR",
                        help="Comma-separated method names to include; default=all")

    # Optional explicit column names (auto-detect if omitted)
    parser.add_argument("--hitrate-col", type=str, default=None, help="Column for final hit rate")
    parser.add_argument("--size-col", type=str, default=None, help="Column for Size axis")
    parser.add_argument("--method-col", type=str, default=None, help="Column for method name")
    parser.add_argument("--round-col", type=str, default=None, help="Column for round/repeat id")

    # Formatting options
    parser.add_argument("--percent", action="store_true",
                        help="If set, multiply hit rate by 100 and label y-axis as '%'")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Auto-detect columns
    auto = autodetect_columns(df)
    hitrate_col = args.hitrate_col or auto["hitrate_col"]
    size_col    = args.size_col    or auto["size_col"]
    method_col  = args.method_col  or auto["method_col"]
    round_col   = args.round_col   or auto["round_col"]

    # Validate
    required = {
        "final_hitrate": hitrate_col,
        "size": size_col,
        "method": method_col,
        "round": round_col,
    }
    missing = [k for k, v in required.items() if v is None or v not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Auto-detected={auto}. "
            f"Pass explicit --hitrate-col/--size-col/--method-col/--round-col as needed."
        )

    # Filter by methods if provided
    if args.methods.strip():
        wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
        df = df[df[method_col].astype(str).isin(wanted)].copy()
        if df.empty:
            raise ValueError("After filtering by --methods, no rows remain. Check method names.")

    # Convert to percent if requested
    yfactor = 100.0 if args.percent else 1.0

    # Save per-row hitrate (optionally percent)
    perrow_csv = os.path.join(args.outdir, "hitrate_per_row.csv")
    df_out = df[[method_col, size_col, round_col, hitrate_col]].copy()
    df_out.columns = ["method", "size", "round", "final_hitrate"]
    df_out["final_hitrate"] = df_out["final_hitrate"] * yfactor
    df_out.to_csv(perrow_csv, index=False)

    # Aggregate mean/std across rounds for each (method, size)
    summary = (
        df_out.groupby(["method", "size"], as_index=False)
              .agg(hitrate_mean=("final_hitrate", "mean"),
                   hitrate_std=("final_hitrate", "std"))
              .sort_values(["method", "size"])
    )
    summary_csv = os.path.join(args.outdir, "hitrate_summary_by_method_size.csv")
    summary.to_csv(summary_csv, index=False)

    # Plot curves
    png_path = os.path.join(args.outdir, "hitrate_vs_size_mean_std.png")
    ylabel = "Final hit rate (%)" if args.percent else "Final hit rate"
    title = "Final hit rate vs Size (mean ± std across rounds)"
    plot_hitrate_curves(summary, png_path, ylabel=ylabel)

    print(f"[OK] Wrote per-row hitrate: {perrow_csv}")
    print(f"[OK] Wrote summary:         {summary_csv}")
    print(f"[OK] Wrote plot:            {png_path}")


if __name__ == "__main__":
    main()
