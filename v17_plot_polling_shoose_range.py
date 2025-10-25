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

# ------------------ 工具函数 ------------------

def parse_window(s: str):
    """
    解析相对窗口字符串: 'all', '48h', '7d', '2w' 等（小时/天/周）
    返回 timedelta 或 None（表示 all）
    """
    if not s or s.lower() == "all":
        return None
    s = s.strip().lower()
    try:
        if s.endswith("h"):
            return timedelta(hours=float(s[:-1]))
        if s.endswith("d"):
            return timedelta(days=float(s[:-1]))
        if s.endswith("w"):
            return timedelta(weeks=float(s[:-1]))
        # 无后缀：按小时
        return timedelta(hours=float(s))
    except Exception:
        return None

def load_csv(path: Path):
    """
    读取 CSV，要求至少有一列时间戳：
    - 优先使用 'ts_utc'
    - 否则退化使用 'ts'
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

    # 统一时间戳列为 ts_utc（tz-aware）
    if "ts_utc" not in df.columns:
        if "ts" in df.columns:
            df["ts_utc"] = df["ts"]
        else:
            print(f"No ts_utc column in {path.name}")
            return None

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    return df

def filter_window(df, window_td):
    """按相对窗口过滤（若 window_td 为 None，返回原 df）"""
    if window_td is None or df is None:
        return df
    cutoff = pd.Timestamp.now(tz="UTC") - window_td
    return df[df["ts_utc"] >= cutoff]

def parse_abs_range(start_str: str, end_str: str):
    """
    解析 --start/--end，要求 ISO8601 格式（如 2025-09-30T15:56:33Z）。
    返回 (start_ts, end_ts)；若未提供则返回 None。
    """
    if not start_str and not end_str:
        return None
    if not start_str or not end_str:
        raise ValueError("Both --start and --end must be provided together.")
    s = pd.to_datetime(start_str, utc=True, errors="raise")
    e = pd.to_datetime(end_str,   utc=True, errors="raise")
    if e <= s:
        raise ValueError(f"Invalid range: end ({e}) must be after start ({s}).")
    return (s, e)

def filter_abs_range(df, abs_range):
    """按绝对起止 (start, end) 过滤"""
    if not abs_range or df is None:
        return df
    s, e = abs_range
    return df[(df["ts_utc"] >= s) & (df["ts_utc"] <= e)]

def parse_segments(seg_list):
    """
    解析多段绝对时间区间。
    输入示例：
      -S "2025-10-21T00:00:00Z..2025-10-21T06:00:00Z" \
      -S "2025-10-22T09:00:00Z..2025-10-22T12:00:00Z"
    返回 [(start,end), ...]（tz-aware, UTC）
    """
    if not seg_list:
        return None
    out = []
    for s in seg_list:
        s = s.strip()
        if not s:
            continue
        # 允许用户用逗号当分隔
        s = s.replace(",", "..")
        parts = [p.strip() for p in s.split("..") if p.strip()]
        if len(parts) != 2:
            print(f"[warn] Bad segment format: {s}")
            continue
        start = pd.to_datetime(parts[0], utc=True, errors="coerce")
        end   = pd.to_datetime(parts[1], utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end) or end <= start:
            print(f"[warn] Invalid segment: {s}")
            continue
        out.append((start, end))
    return out or None

def filter_by_segments(df, segs):
    """仅保留落入任意一段 (start<=ts<=end) 的数据"""
    if not segs or df is None:
        return df
    mask = pd.Series(False, index=df.index)
    for s, e in segs:
        mask |= ((df["ts_utc"] >= s) & (df["ts_utc"] <= e))
    return df[mask]

def add_stitched_time(df, segs, col_name="t_stitch_h"):
    """
    把多段区间首尾拼接，新增一列“拼接时间（小时）”。
    规则：每一段从 0 小时开始递增；下一段接在上一段末尾。
    """
    df = df.copy()
    df[col_name] = np.nan
    cum = pd.Timedelta(0)
    for s, e in segs:
        mask = (df["ts_utc"] >= s) & (df["ts_utc"] <= e)
        df.loc[mask, col_name] = (
            (df.loc[mask, "ts_utc"] - s).dt.total_seconds() / 3600.0
            + cum.total_seconds() / 3600.0
        )
        cum += (e - s)
    return df

# ------------------ 画图主函数 ------------------

def plot_backend(df, backend_name, metrics, window_td, save_dir=None, figsize=(10, 4.5),
                 styles=None, vline_style=None, segments=None, stitch=False, abs_range=None):
    """
    根据优先级筛选数据并画图：
      优先级：segments > abs_range > window
    segments + stitch=True -> 使用拼接时间轴（小时）
    其它情况 -> 使用 UTC 时间轴
    """
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 28,
                         "axes.linewidth": 4,
                         })

    if df is None or df.empty:
        print(f"No data for {backend_name}")
        return

    # ----- 选择过滤方式 -----
    if segments:
        dfw = filter_by_segments(df, segments)
    elif abs_range:
        dfw = filter_abs_range(df, abs_range)
    else:
        dfw = filter_window(df, window_td)

    if dfw is None or dfw.empty:
        print(f"No data in selected range for {backend_name}")
        return

    dfw = dfw.copy()
    dfw["ts_utc"] = pd.to_datetime(dfw["ts_utc"], errors="coerce", utc=True)

    # ----- 选择 X 轴列 -----
    x_is_stitched = False
    if segments and stitch:
        dfw = add_stitched_time(dfw, segments, col_name="t_stitch_h")
        xcol = "t_stitch_h"
        x_is_stitched = True
    else:
        xcol = "ts_utc"

    # ----- 画图样式 -----
    color_cycle = itertools.cycle([d["color"] for d in plt.rcParams["axes.prop_cycle"]])
    marker_cycle = itertools.cycle(["o", "s", "D", "^", "v", "P", "X", "*"])
    styles = styles or {}
    vline_style = vline_style or {"color": "tab:gray", "linestyle": ":", "alpha": 0.5, "label": "props update"}

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, which="both", zorder=0, alpha=0.2)

    # 若存在右轴（例如 pending_jobs 与 error 同图），按需使用
    right_metric = None
    if "pending_jobs" in metrics and len(metrics) > 1:
        right_metric = "pending_jobs"

    plotted_any = False

    metric_label = {
        "twoq_gate_err_median": "TwoQ Err",
        "sx_gate_err_median": "SX Err",
        "readout_err_mean": "Readout Err",
    }

    for m in metrics:
        if m not in dfw.columns:
            print(f"[warn] column not found: {m} (backend={backend_name})")
            continue

        y = dfw[m].values
        # 如果是右轴指标，后面单独处理
        if right_metric and m == right_metric:
            continue

        sty = styles.get(m, {})
        color = sty.get("color", next(color_cycle))
        marker = sty.get("marker", next(marker_cycle))
        linestyle = sty.get("linestyle", "-")
        linewidth = sty.get("linewidth", 2)
        alpha = sty.get("alpha", 0.8)
        label = metric_label.get(m)

        ax.plot(
            dfw[xcol],
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=2,
            markeredgewidth=1,
            zorder=3,
        )
        ax.set_ylabel("Error Rate")
        plotted_any = True

    if right_metric and right_metric in dfw.columns:
        ax2 = ax.twinx()
        yq = dfw[right_metric].values

        sty = styles.get(right_metric, {})
        color = sty.get("color", next(color_cycle))
        marker = sty.get("marker", next(marker_cycle))
        linestyle = sty.get("linestyle", "--")
        linewidth = sty.get("linewidth", 2)
        alpha = sty.get("alpha", 0.8)
        label = "Pending Jobs"
        ax2.plot(
            dfw[xcol],
            yq,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=2,
            markeredgewidth=1.0,
            zorder=3,
        )
        ax2.set_ylabel("Pending Jobs")
        plotted_any = True
    else:
        ax2 = None

    # 在非拼接模式下，x 轴按时间格式化；拼接模式下用数值轴
    if x_is_stitched:
        ax.set_xlabel("Stitched time (hours across selected segments)")
    else:
        ax.set_xlabel("UTC time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=None))

    # （可选）非拼接模式下，用浅色标出各段区间
    if (segments is not None) and (not x_is_stitched):
        for s, e in segments:
            ax.axvspan(s, e, color="k", alpha=0.05, zorder=1)

    # last_update_date_iso 的竖线标记（若有）
    vline_present = False
    vline_label = vline_style.get("label", "update")
    vline_handle = None
    if "last_update_date_iso" in dfw.columns:
        luv = pd.to_datetime(dfw["last_update_date_iso"], errors="coerce", utc=True)
        changed = luv.ne(luv.shift()) & luv.notna()
        xs = pd.to_datetime(luv[changed], errors="coerce", utc=True).dropna()
        if not xs.empty and not x_is_stitched:
            for x in xs:
                vline_handle = ax.axvline(x, **{k:v for k,v in vline_style.items() if k != "label"})
            vline_present = True

    # 图例
    handles, labels = ax.get_legend_handles_labels()
    if ax2:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels  += l2
    if vline_present and vline_handle is not None:
        handles.append(vline_handle)
        labels.append(vline_label)

    if handles:
        # reasonable ncol: try to keep legend not too tall
        ncol = min(max(1, len(labels)), 3)
        fig.legend(handles, labels,
                   loc='upper center',
                   bbox_to_anchor=(0.5, 1.25),
                   ncol=ncol,
                   frameon=True)
        leg = fig.legends[-1]
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("none")
        frame.set_alpha(1.0)
        leg.set_zorder(100)
        plt.subplots_adjust(top=0.82)

    # ax.set_title(backend_name)
    fig.tight_layout()

    # 保存或显示
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"{backend_name}.png"
        fig.savefig(fname, dpi=600, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close(fig)
    else:
        plt.show()

# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description="Plot backend time series from monitor CSVs")
    ap.add_argument("--dir", "-d", default=BACKEBND_SERIES_DIR,
                    help="Directory containing per-backend CSVs")
    ap.add_argument("--backend", "-b", nargs="*", default=None,
                    help="Backend names (CSV file stems). Default=all CSVs in dir.")
    ap.add_argument("--metrics", "-m", nargs="*", default=None,
                    help="Columns to plot. Default: twoq_gate_err_median sx_gate_err_median readout_err_mean pending_jobs")
    ap.add_argument("--window", "-w", default="all",
                    help="Relative time window like 48h, 7d, 2w, all")

    # 1) 任意绝对起止时间（ISO8601，如 2025-09-30T15:56:33Z）
    ap.add_argument("--start",help="Absolute start time (ISO8601, e.g. 2025-09-30T15:56:33Z)") #, default="2025-09-30T15:47:00Z"
    ap.add_argument("--end",  help="Absolute end time   (ISO8601, e.g. 2025-10-01T03:00:00Z)") # , default="2025-10-11T01:05:58Z"

    # 2) 多段时间并可拼接
    ap.add_argument("--segments", "-S", nargs="*", default=None,
                    help='Multiple absolute UTC ranges, each as "START..END" in ISO8601, e.g. 2025-09-30T15:47:00Z..2025-10-03T23:56:10Z, 2025-10-21T00:02:31Z..2025-10-23T21:52:41Z')
    ap.add_argument("--stitch", action="store_true",
                    help="If set with --segments, stitch segments into a continuous time axis (hours)")

    ap.add_argument("--save", "-s", default=DEFAULT_PLOT_DIR,
                    help="Save PNGs to this folder (omit to show)")
    ap.add_argument("--size", type=float, nargs=2, metavar=("W","H"), default=(10,4.5),
                    help="Figure size in inches")

    args = ap.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        raise SystemExit(f"Directory not found: {data_dir}")

    all_files = sorted(data_dir.glob("*.csv"))
    if not all_files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    backends = args.backend if args.backend else [p.stem for p in all_files]

    default_metrics = ["twoq_gate_err_median", "sx_gate_err_median", "readout_err_mean", "pending_jobs"]
    metrics = args.metrics if args.metrics else default_metrics

    window_td = parse_window(args.window)
    abs_range = parse_abs_range(args.start, args.end) if (args.start or args.end) else None
    segments  = parse_segments(args.segments) if args.segments else None

    for b in backends:
        path = data_dir / f"{b}.csv"
        df = load_csv(path)
        plot_backend(
            df, b, metrics, window_td,
            save_dir=args.save, figsize=tuple(args.size),
            segments=segments, stitch=args.stitch, abs_range=abs_range
        )

if __name__ == "__main__":
    main()