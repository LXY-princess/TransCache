#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制：x=qubits, y=∑_{所有 depth} total latency。
- 每条线 = 一个 circuit
- baseline 实线 / tcache 虚线
- 不同 circuit 用不同颜色和 marker 区分
- 与 v11_caculate_table_load_save_csv.py 的文件命名和默认目录保持一致
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from oqpy import frame
from sympy import false

# ===== 与原代码保持一致的默认配置（可用命令行覆盖） =====
CODE_TAG = "v16"
DEFAULT_FIGDIR = Path("figs/v16_run_per_circuit")  # 读取 JSON 的目录
DEFAULT_OUTDIR = Path("figs/v16_run_per_circuit/v16_lineplots");DEFAULT_OUTDIR.mkdir(exist_ok=True)

PHASE_02 = "02_cache_search"
PHASE_03 = "03_transpile"
PHASE_07 = "07_cache_write"

DEFAULT_CIRCS = [
    "GHZ-Chain", "LinearEnt", "QFT-Like",
    "QSIM-XXZ", "QAOA-3reg", "VQE-Full",
]

# DEFAULT_CIRCS = [
#     "GHZ-Chain", "LinearEnt",
#     "QSIM-XXZ", "QAOA-3reg", "VQE-Full",
# ]

DEFAULT_DEPTHS = [1, 4]
DEFAULT_QUBITS = [8, 16, 32, 64, 96, 112, 127] #[8, 16, 32, 64, 96, 112, 127]

MODES = ["baseline", "tcache"]  # 两种方法
BASELINE_COLOR = "#BA55D3"   # 默认 baseline 颜色（蓝）
TCACHE_COLOR   = "#FFA500"   # 默认 tcache 颜色（橙）

LABEL_MAP = {
    "LinearEnt": "Linear",
    "QFT-Like": "QFT",
    "GHZ-Chain": "GHZ",
    "QSIM-XXZ": "QSIM",
    "QAOA-3reg": "QAOA",
    "VQE-Full": "VQE",
}


def load_sum(figdir: Path, code_tag: str, q: int, d: int, circ: str, mode: str) -> Dict[str, float]:
    """
    读取单个 (q,d,circ,mode) 的 *_sum.json，返回 {phase: float}
    若文件不存在或异常，返回 {}
    """
    p = figdir / f"{code_tag}_q{q}_d{d}_sim_{circ}_{mode}_sum.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
        # 强制转换为 float
        return {k: float(v) for k, v in obj.items()}
    except Exception:
        return {}


def aggregate_sum_total_over_depths(
    figdir: Path,
    code_tag: str,
    circs: List[str],
    qubits: List[int],
    depths: List[int],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    聚合：对每个 circuit、每个 qubit，分别对 baseline / tcache 的
    该 qubit 下所有 depth 的 "total" 累加。
    返回结构：
    agg[circ][mode][q] = sum_total
    若某(qubit, mode)在所有 depth 下都缺失，则值为 np.nan（绘图时会自动断线）
    """
    agg: Dict[str, Dict[str, Dict[int, float]]] = {}
    for circ in circs:
        agg[circ] = {mode: {} for mode in MODES}
        for q in qubits:
            for mode in MODES:
                total = 0.0
                found = False
                for d in depths:
                    data = load_sum(figdir, code_tag, q, d, circ, mode)
                    if data:
                        total += float(data.get("total", 0.0))
                        found = True
                agg[circ][mode][q] = total if found else float("nan")
    return agg

def aggregate_sum_total_over_depths_tTime(
    figdir: Path,
    code_tag: str,
    circs: List[str],
    qubits: List[int],
    depths: List[int],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    返回 agg[circ][mode][q] = compile_latency_sumDepth
      baseline:  Σ depth  phase03
      tcache  :  Σ depth (phase02 + 03 + 07)
    """
    agg: Dict[str, Dict[str, Dict[int, float]]] = {}
    for circ in circs:
        agg[circ] = {m: {} for m in MODES}
        for q in qubits:
            # baseline
            base_sum = 0.0
            # tcache
            tc_sum   = 0.0
            found_b = found_t = False
            for d in depths:
                bdat = load_sum(figdir, code_tag, q, d, circ, "baseline")
                tdat = load_sum(figdir, code_tag, q, d, circ, "tcache")
                if bdat:
                    base_sum += bdat.get(PHASE_03, 0.0)
                    found_b = True
                if tdat:
                    tc_sum += (tdat.get(PHASE_02, 0.0) +
                               tdat.get(PHASE_03, 0.0) +
                               tdat.get(PHASE_07, 0.0))
                    found_t = True
            agg[circ]["baseline"][q] = base_sum if found_b else float("nan")
            agg[circ]["tcache"][q]   = tc_sum   if found_t else float("nan")
    return agg

def aggregate_sum_total_over_qubits(
    figdir: Path,
    code_tag: str,
    circs: List[str],
    qubits: List[int],
    depths: List[int],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    聚合：对每个 circuit、每个 qubit，分别对 baseline / tcache 的
    该 qubit 下所有 depth 的 "total" 累加。
    返回结构：
    agg[circ][mode][q] = sum_total
    若某(qubit, mode)在所有 depth 下都缺失，则值为 np.nan（绘图时会自动断线）
    """
    agg: Dict[str, Dict[str, Dict[int, float]]] = {}
    for circ in circs:
        agg[circ] = {mode: {} for mode in MODES}
        for d in depths:
            for mode in MODES:
                total = 0.0
                found = False
                for q in qubits:
                    data = load_sum(figdir, code_tag, q, d, circ, mode)
                    if data:
                        total += float(data.get("total", 0.0))
                        found = True
                agg[circ][mode][d] = total if found else float("nan")
    return agg

def build_style_maps(circs: List[str]) -> Tuple[Dict[str, str], Dict[str, Tuple[float, float, float, float]]]:
    """
    为每个 circuit 生成 marker 和颜色：
    - 颜色来自 tab10 调色板
    - marker 从固定列表循环
    """
    markers_cycle = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "h", "*"]
    cmap = plt.get_cmap("tab10")

    marker_map: Dict[str, str] = {}
    color_map: Dict[str, Tuple[float, float, float, float]] = {}

    for i, circ in enumerate(circs):
        marker_map[circ] = markers_cycle[i % len(markers_cycle)]
        color_map[circ] = cmap(i % 10)
    return marker_map, color_map

def plot_lines_sumD(
    agg: Dict[str, Dict[str, Dict[int, float]]],
    qubits: List[int],
    ylog: bool,
    outdir: Path,
):
    """
    绘图：基于 agg 数据，画 baseline（实线）与 tcache（虚线），
    不同 circuit 用不同颜色与 marker。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})

    fig, ax = plt.subplots(figsize=(8,4))

    circs = list(agg.keys())
    marker_map, color_map = build_style_maps(circs)

    # 保证 x 轴按给定顺序
    xs = list(qubits)

    # 逐 circuit 画两条线（baseline / tcache）
    for circ in circs:
        color = color_map[circ]
        marker = marker_map[circ]

        y_base = [agg[circ]["baseline"].get(q, float("nan")) for q in xs]
        y_tc   = [agg[circ]["tcache"].get(q, float("nan")) for q in xs]

        # baseline = 实线
        ax.plot(
            xs, y_base,
            linestyle="-", linewidth=2.0,
            marker=marker, markersize=6, markerfacecolor="none", markeredgewidth=1.5,
            color=color,
            label=f"{circ} (baseline)",
        )
        # tcache = 虚线
        ax.plot(
            xs, y_tc,
            linestyle="--", linewidth=2.0,
            marker=marker, markersize=6, markerfacecolor="none", markeredgewidth=1.5,
            color=color,
            label=f"{circ} (tcache)",
        )

    ax.set_xlabel("Qubits")
    ax.set_ylabel("Latency (sum over depths) (s)")
    ax.set_xticks(xs)
    if ylog:
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", alpha=0.3)

    # --- 构建双图例 ---
    # 方法图例（线型）：baseline 实线；tcache 虚线
    method_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="Baseline (solid)"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="Tcache (dashed)"),
    ]
    # leg1 = ax.legend(
    #     handles=method_handles,
    #     title="Method",
    #     loc="upper center",
    #     frameon=False,
    # )
    leg1 = ax.legend(handles=method_handles, title="Method",
                     loc="upper left",
                     bbox_to_anchor=(1.02, 1.00),  # 右侧顶端
                     borderaxespad=0, frameon=False)
    ax.add_artist(leg1)
    ax.add_artist(leg1)

    # 电路图例（颜色+marker，不带线型）
    circuit_handles = []
    for circ in circs:
        circuit_handles.append(
            Line2D(
                [0], [0],
                color=color_map[circ], marker=marker_map[circ],
                linestyle="none", markersize=8, markerfacecolor="none", markeredgewidth=1.5,
                label=circ,
            )
        )
    # ax.legend(
    #     handles=circuit_handles,
    #     title="Circuit",
    #     loc="upper left",
    #     frameon=False,
    #     ncol=1,
    #
    # )
    ax.legend(handles=circuit_handles, title="Circuit",
              loc="upper left",
              bbox_to_anchor=(1.02, 0.7),  # 右侧靠下，避免与方法图例重叠
              borderaxespad=0, frameon=False)

    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # 保存图片
    png_path = outdir / "qubits_vs_latency_sumDepths.png"
    # pdf_path = outdir / "qubits_vs_latency_sumDepths.pdf"
    fig.savefig(png_path, dpi=600)
    # fig.savefig(pdf_path)
    print(f"✓ Figure saved: {png_path}")
    # print(f"✓ Figure saved: {pdf_path}")

def plot_lines_sunQ(
    agg: Dict[str, Dict[str, Dict[int, float]]],
    depths: List[int],
    ylog: bool,
    outdir: Path,
):
    """
    绘图：基于 agg 数据，画 baseline（实线）与 tcache（虚线），
    不同 circuit 用不同颜色与 marker。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})

    fig, ax = plt.subplots(figsize=(8, 3))

    circs = list(agg.keys())
    marker_map, color_map = build_style_maps(circs)

    # 保证 x 轴按给定顺序
    xs = list(depths)

    # 逐 circuit 画两条线（baseline / tcache）
    for circ in circs:
        color = color_map[circ]
        marker = marker_map[circ]

        y_base = [agg[circ]["baseline"].get(q, float("nan")) for q in xs]
        y_tc   = [agg[circ]["tcache"].get(q, float("nan")) for q in xs]

        # baseline = 实线
        ax.plot(
            xs, y_base,
            linestyle="-", linewidth=2.0,
            marker=marker, markersize=6, markerfacecolor="none", markeredgewidth=1.5,
            color=color,
            label=f"{circ} (baseline)",
        )
        # tcache = 虚线
        ax.plot(
            xs, y_tc,
            linestyle="--", linewidth=2.0,
            marker=marker, markersize=6, markerfacecolor="none", markeredgewidth=1.5,
            color=color,
            label=f"{circ} (tcache)",
        )

    ax.set_xlabel("Depths")
    ax.set_ylabel("Latency (sum over qubits) (s)")
    ax.set_xticks(xs)
    if ylog:
        ax.set_yscale("log")

    ax.grid(True, linestyle="--", alpha=0.3)

    # --- 构建双图例 ---
    # 方法图例（线型）：baseline 实线；tcache 虚线
    method_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="Baseline (solid)"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="Tcache (dashed)"),
    ]
    # leg1 = ax.legend(
    #     handles=method_handles,
    #     title="Method",
    #     loc="upper center",
    #     frameon=False,
    # )
    leg1 = ax.legend(handles=method_handles, title="Method",
                     loc="upper left",
                     bbox_to_anchor=(1.02, 1.00),  # 右侧顶端
                     borderaxespad=0, frameon=False)
    ax.add_artist(leg1)

    # 电路图例（颜色+marker，不带线型）
    circuit_handles = []
    for circ in circs:
        circuit_handles.append(
            Line2D(
                [0], [0],
                color=color_map[circ], marker=marker_map[circ],
                linestyle="none", markersize=8, markerfacecolor="none", markeredgewidth=1.5,
                label=circ,
            )
        )
    # ax.legend(
    #     handles=circuit_handles,
    #     title="Circuit",
    #     loc="upper left",
    #     frameon=False,
    #     ncol=1,
    #
    # )
    ax.legend(handles=circuit_handles, title="Circuit",
              loc="upper left",
              bbox_to_anchor=(1.02, 0.7),  # 右侧靠下，避免与方法图例重叠
              borderaxespad=0, frameon=False)

    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    # 保存图片
    png_path = outdir / "qubits_vs_latency_sumQubits.png"
    pdf_path = outdir / "qubits_vs_latency_sumQubits.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    print(f"✓ Figure saved: {png_path}")
    print(f"✓ Figure saved: {pdf_path}")

def plot_single_circuit(
    agg_one: Dict[str, Dict[int, float]],
    circ: str,
    qubits: List[int],
    ylog: bool,
    outdir: Path,
):
    """
    agg_one   = {"baseline": {q:val, ...}, "tcache": {q:val, ...}}
    circ      = 电路名，用于标题/文件名
    """
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 20})

    fig, ax = plt.subplots(figsize=(6, 4))

    xs = list(qubits)
    y_base = [agg_one["baseline"].get(q, float("nan")) for q in xs]
    y_tc   = [agg_one["tcache"  ].get(q, float("nan")) for q in xs]

    ax.plot(xs, y_base, "-",  lw=5, marker="o", ms=12,
            mfc="none", mew=2, label="Baseline")
    ax.plot(xs, y_tc,   "--", lw=5, marker="v", ms=12,
            mfc="none", mew=2, label="Tcache")

    ax.set_xlabel("Qubits", fontweight="bold")
    ax.set_ylabel("Latency (s)", fontweight="bold")
    ax.set_xticks(xs)
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, ls="--", alpha=.5)

    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # 只需要方法图例即可
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()

    fname = f"qubits_vs_latency_sumDepths_{circ.replace('/', '_')}"
    fig.savefig(outdir / f"{fname}.png", dpi=600)
    # fig.savefig(outdir / f"{fname}.pdf")
    print("✓ Figure saved:", outdir / f"{fname}.png")

# ========== 绘图：横向面板（每个子图独立坐标轴） ==========
def _finite_xy(xs: List[int], ys: List[float]):
    """从 (xs, ys) 中筛出 y 非 NaN 的点"""
    x2, y2 = [], []
    for x, y in zip(xs, ys):
        if not (y is None or (isinstance(y, float) and np.isnan(y))):
            x2.append(x); y2.append(y)
    return x2, y2

def plot_all_circuits_row(
    agg_all: Dict[str, Dict[str, Dict[int, float]]],
    circs: List[str],
    qubits: List[int],
    ylog: bool,
    outdir: Path,
    fig_name: str = "qubits_vs_latency_row_ind_axes.png",
):
    """
    每个电路一列子图，横向排布在同一张图上；
    每个子图拥有独立的 x/y 轴范围（不共享尺度）。
    """
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})

    n = len(circs)
    # 依据面板数自适应画布宽度
    fig_w = max(4.3 * n, 6.0)
    fig_h = 3.6

    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    if n == 1:
        axes = [axes]

    legend_lines, legend_labels = None, None

    for i, circ in enumerate(circs):
        ax = axes[i]

        # 仅用该电路有数据的 x 值，以保证每个子图的 x 轴范围“按需”设置
        xs_all = list(qubits)
        y_base_all = [agg_all[circ]["baseline"].get(q, float("nan")) for q in xs_all]
        y_tc_all   = [agg_all[circ]["tcache"  ].get(q, float("nan")) for q in xs_all]

        xs_b, y_b = _finite_xy(xs_all, y_base_all)
        xs_t, y_t = _finite_xy(xs_all, y_tc_all)
        xs = sorted(set(xs_b) | set(xs_t), key=lambda x: xs_all.index(x))
        # 若两条线都没点，跳过绘制
        if len(xs) == 0:
            ax.text(0.5, 0.5, f"No data: {circ}", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        line_base, = ax.plot(
            xs_b, y_b,
            linestyle="-", linewidth=2.0,
            marker="o", markersize=6, markerfacecolor="none", markeredgewidth=1.5,
            color=BASELINE_COLOR,
            label="Baseline"
        )
        line_tc, = ax.plot(
            xs_t, y_t,
            linestyle="--", linewidth=2.0,
            marker="v", markersize=6, markerfacecolor="none", markeredgewidth=1.5,
            color=TCACHE_COLOR,
            label="TransCache"
        )

        # 轴标签：只在最左侧画 ylabel，避免拥挤
        ax.set_xlabel("Qubits")
        if i == 0:
            ax.set_ylabel("Compilation Latency (s)")

        # 该子图的独立刻度与范围
        ax.set_xticks(xs)
        if ylog:
            ax.set_yscale("log")

        # --- 独立自适应 x/y 范围 ---
        # xlim
        x_min, x_max = min(xs), max(xs)
        if x_min == x_max:
            pad_x = 0.5 if isinstance(x_min, (int, float)) else 0.5
        else:
            pad_x = 0.05 * (x_max - x_min)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)

        # ylim
        y_vals = np.array((y_b or []) + (y_t or []), dtype=float)
        y_vals = y_vals[~np.isnan(y_vals)]
        if y_vals.size > 0:
            y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
            if ylog:
                # 避免 log(<=0)
                y_min = max(y_min, 1e-12)
                ax.set_ylim(y_min / 1.5, y_max * 1.5)
            else:
                if y_min == y_max:
                    pad_y = max(0.1 * (y_max if y_max > 0 else 1.0), 1e-9)
                else:
                    pad_y = 0.10 * (y_max - y_min)
                ax.set_ylim(max(0.0, y_min - pad_y), y_max + pad_y)

        ax.grid(True, linestyle="--", alpha=0.45)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 子图下方标签（（a） linear）
        letter = chr(ord('a') + i)
        label_text = f"({letter}) {LABEL_MAP.get(circ, circ)}"
        ax.text(0.5, -0.35, label_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=18, fontweight="bold")

        # 保留用于全局共享图例的线对象
        if legend_lines is None:
            legend_lines = [line_base, line_tc]
            legend_labels = ["Baseline", "TransCache"]

    # 全局共享图例（顶部居中）
    if legend_lines is not None:
        fig.legend(
            legend_lines, legend_labels,
            loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02)
        )

    # 留出上下边距给顶部图例与底部标签
    plt.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.30, wspace=0.28)

    out_png = outdir / fig_name
    fig.savefig(out_png, dpi=600)
    print(f"✓ Figure saved: {out_png}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot latency vs qubits (sum over depths) for each circuit; baseline solid, tcache dashed."
    )
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR, help="Directory containing *_sum.json files.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory to save figures.")
    p.add_argument("--code_tag", type=str, default=CODE_TAG, help="Prefix used in filenames, e.g., 'v11'.")
    p.add_argument("--circs", nargs="*", default=DEFAULT_CIRCS, help="Subset of circuits to include.")
    p.add_argument("--qubits", nargs="*", type=int, default=DEFAULT_QUBITS, help="List of qubits to include.")
    p.add_argument("--depths", nargs="*", type=int, default=DEFAULT_DEPTHS, help="List of depths to sum over.")
    p.add_argument("--ylog", action="store_true", help="Use log scale for Y axis.")
    p.add_argument("--name", type=str, default="qubits_vs_latency_rowpanels.png",
                   help="Output file name.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.figdir.exists():
        raise SystemExit(f"[Error] 数据目录不存在：{args.figdir}")

    # 去重并排序，确保绘图 x 轴稳定
    circs = list(dict.fromkeys(args.circs))
    qubits = sorted(set(args.qubits), key=lambda x: args.qubits.index(x))  # 保持用户给定顺序
    depths = list(dict.fromkeys(args.depths))

    agg_depth_tTime = aggregate_sum_total_over_depths_tTime(
        figdir=args.figdir,
        code_tag=args.code_tag,
        circs=circs,
        qubits=qubits,
        depths=depths,
    )

    plot_all_circuits_row(
        agg_all=agg_depth_tTime,
        circs=circs,
        qubits=qubits,
        ylog=args.ylog,
        outdir=args.outdir,
        fig_name=args.name,
    )


if __name__ == "__main__":
    main()
