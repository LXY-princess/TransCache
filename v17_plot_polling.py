#!/usr/bin/env python3
# plot_backend_timeseries.py
# 读取每后端 CSV（每行为一次 snapshot），画选定指标的时间序列图。
#
# 依赖: pandas, matplotlib
# pip install pandas matplotlib

import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pathlib
from matplotlib.lines import Line2D
import itertools

DEFAULT_DIR = pathlib.Path("./figs/v17_polling")
BACKEBND_SERIES_DIR = DEFAULT_DIR/"backends"
DEFAULT_PLOT_DIR = DEFAULT_DIR/"plots"

def parse_window(s):
    if s is None: return None
    s = str(s).strip().lower()
    if s == "all": return None
    try:
        if s.endswith("h"):
            return timedelta(hours=float(s[:-1]))
        if s.endswith("d"):
            return timedelta(days=float(s[:-1]))
        # number without suffix -> hours
        return timedelta(hours=float(s))
    except Exception:
        return None

def load_csv(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None
    if "ts_utc" not in df.columns:
        # try common fallback column names
        if "ts" in df.columns:
            df["ts_utc"] = df["ts"]
        else:
            print(f"No ts_utc column in {path.name}")
            return None
    # parse timestamps (coerce errors)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    # sort
    df = df.sort_values("ts_utc").reset_index(drop=True)
    return df

def filter_window(df, window_td):
    if window_td is None or df is None: return df
    cutoff = pd.Timestamp.now(tz='UTC') - window_td
    return df[df["ts_utc"] >= cutoff]

# pd.Timestamp.now(tz='UTC')

def plot_backend(df, backend_name, metrics, window_td, save_dir=None, figsize=(10,4.5),
                 styles=None, vline_style=None):
    """
    Plot time series for a backend with customizable styles.

    Args:
      df: full dataframe (must have ts_utc column already parsed as datetimes)
      backend_name: name (title)
      metrics: list of metric column names to plot (may include "pending_jobs")
      window_td: window filter argument used by filter_window
      save_dir: optional output directory
      figsize: figure size
      styles: optional dict mapping metric -> dict of style props:
              e.g. styles = {
                "readout_error": {"color":"C0","marker":"o","linestyle":"-","label":"Readout err"},
                "pending_jobs": {"color":"C3","marker":"s","linestyle":"--","label":"Pending"}
              }
              Allowed keys: color, marker, linestyle, linewidth, alpha, label
      vline_style: dict for props-update vline, same keys as above; label recommended
    """
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    if df is None or df.empty:
        print(f"No data for {backend_name}")
        return
    dfw = filter_window(df, window_td)
    if dfw is None or dfw.empty:
        print(f"No data in window for {backend_name}")
        return

    # ensure ts_utc is datetime tz-aware
    dfw = dfw.copy()
    dfw["ts_utc"] = pd.to_datetime(dfw["ts_utc"], errors="coerce", utc=True)

    # default style cycle from rcParams
    color_cycle = itertools.cycle([d['color'] for d in plt.rcParams['axes.prop_cycle']])
    marker_cycle = itertools.cycle(["o", "s", "D", "^", "v", "P", "X", "*"])  # markers fallback

    styles = styles or {}
    vline_style = vline_style or {"color": "tab:gray", "linestyle": ":", "alpha": 0.5, "label": "props update"}

    left_metrics = [m for m in metrics if m != "pending_jobs"]
    right_metric = "pending_jobs" if "pending_jobs" in metrics and "pending_jobs" in dfw.columns else None

    fig, ax = plt.subplots(figsize=figsize)
    plotted_any = False

    # plot left-axis metrics, with customizable styles
    for m in left_metrics:
        if m not in dfw.columns:
            print(f"Warning: column '{m}' not in CSV for {backend_name}")
            continue
        y = pd.to_numeric(dfw[m], errors="coerce")
        if y.notna().sum() == 0:
            continue

        # style selection: user-provided or defaults from cycles
        sty = styles.get(m, {})
        color = sty.get("color", next(color_cycle))
        marker = sty.get("marker", next(marker_cycle))
        linestyle = sty.get("linestyle", sty.get("ls", "-"))
        linewidth = sty.get("linewidth", 1)
        alpha = sty.get("alpha", 1.0)
        label = sty.get("label", m)

        ax.plot(dfw["ts_utc"], y, color=color, marker=marker, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha, label=label,
                markerfacecolor='none',  # 空心
                markeredgecolor=color,  # 用线条颜色做描边
                markersize=2,  # 缩小
                markeredgewidth=1.0,  # 描边更清晰

                )
        plotted_any = True

    ax.set_xlabel("UTC time")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M', tz=None))

    ax2 = None
    if right_metric:
        ax2 = ax.twinx()
        yq = pd.to_numeric(dfw[right_metric], errors="coerce")
        sty = styles.get(right_metric, {})
        color = sty.get("color", next(color_cycle))
        marker = sty.get("marker", next(marker_cycle))
        linestyle = sty.get("linestyle", sty.get("ls", "-"))
        linewidth = sty.get("linewidth", 1)
        alpha = sty.get("alpha", 0.8)
        label = sty.get("label", right_metric)

        ax2.plot(dfw["ts_utc"], yq, color=color, marker=marker, linestyle=linestyle,
                 linewidth=linewidth, alpha=alpha, label=label,
                 markerfacecolor='none',  # 空心
                 markeredgecolor=color,  # 用线条颜色做描边
                 markersize=2,  # 缩小
                 markeredgewidth=1.0,  # 描边更清晰
                 )
        ax2.set_ylabel(right_metric)
        plotted_any = True

    # ==== vlines for last_update_date_iso (optionally use reported time or snapshot time) ====
    vline_present = False
    vline_label = vline_style.get("label", "props update")
    vline_handle = None

    if "last_update_date_iso" in dfw.columns:
        luv_dt = pd.to_datetime(dfw["last_update_date_iso"], errors="coerce", utc=True)
        changed = luv_dt.ne(luv_dt.shift()) & luv_dt.notna()

        xs_snapshot = dfw.loc[changed, "ts_utc"]
        xs_reported = luv_dt[changed]
        use_reported = True  # keep this behavior; change to False if you prefer snapshot detection time
        xs = xs_reported if use_reported else xs_snapshot

        xs = pd.to_datetime(xs, errors="coerce", utc=True).dropna()
        if not xs.empty:
            ymin, ymax = ax.get_ylim()
            # vline style parameters
            vcolor = vline_style.get("color", "tab:gray")
            vlinestyle = vline_style.get("linestyle", ":")
            valpha = vline_style.get("alpha", 0.5)
            vzorder = vline_style.get("zorder", 1)
            ax.vlines(xs.values, ymin, ymax, colors=vcolor, linestyles=vlinestyle, alpha=valpha, zorder=vzorder)
            vline_present = True
            # make a legend handle for vline (so it appears in legend)
            vline_handle = Line2D([0], [0], color=vcolor, linestyle=vlinestyle, alpha=valpha, label=vline_label)

    # ==== create unified legend AFTER all plotting ====
    handles = []
    labels = []
    for line in ax.get_lines():
        handles.append(line)
        labels.append(line.get_label())
    if ax2 is not None:
        for line in ax2.get_lines():
            handles.append(line)
            labels.append(line.get_label())

    if vline_present and vline_handle is not None:
        handles.append(vline_handle)
        labels.append(vline_label)

    # place legend at top of figure (outside plot)
    if handles:
        # reasonable ncol: try to keep legend not too tall
        ncol = min(max(1, len(labels)), 6)
        fig.legend(handles, labels,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.05),
                   ncol=ncol,
                   frameon=True)
        leg = fig.legends[-1]
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("none")
        frame.set_alpha(1.0)
        leg.set_zorder(100)
        plt.subplots_adjust(top=0.82)

    # finalize
    ax.set_title(backend_name)
    fig.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"{backend_name}.png"
        fig.savefig(fname, dpi=600, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close(fig)
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser(description="Plot backend time series from monitor CSVs")
    ap.add_argument("--dir", "-d", default=BACKEBND_SERIES_DIR, help="Directory containing per-backend CSVs")
    ap.add_argument("--backend", "-b", nargs="*", default=None, help="Backend names (file stems) to plot; default=all")
    ap.add_argument("--metrics", "-m", nargs="*", default=None, help="Columns to plot (default common set)")
    ap.add_argument("--window", "-w", default="48h", help="Time window like 48h, 7d, all")
    ap.add_argument("--save", "-s", default=DEFAULT_PLOT_DIR, help="If provided, save PNGs to this folder instead of showing")
    ap.add_argument("--size", type=float, nargs=2, metavar=("W","H"), default=(10,4.5), help="Figure size in inches")
    args = ap.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        raise SystemExit(f"Directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("*.csv"))
    if not all_files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    backends = args.backend if args.backend else [p.stem for p in all_files]

    default_metrics = ["twoq_gate_err_median","sx_gate_err_median","readout_err_mean","pending_jobs"]
    metrics = args.metrics if args.metrics else default_metrics

    window_td = parse_window(args.window)

    for b in backends:
        path = data_dir / f"{b}.csv"
        df = load_csv(path)
        plot_backend(df, b, metrics, window_td, save_dir=args.save, figsize=tuple(args.size))

if __name__ == "__main__":
    main()
