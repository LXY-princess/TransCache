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
DEFAULT_FIGDIR = Path("./figs/v16")  # 读取 JSON 的目录
DEFAULT_OUTDIR = Path("./figs/v16_lineplots");DEFAULT_OUTDIR.mkdir(exist_ok=True)

PHASE_02 = "02_cache_search"
PHASE_03 = "03_transpile"
PHASE_07 = "07_cache_write"

# DEFAULT_CIRCS = [
#     "GHZ-Chain", "LinearEnt", "QFT-Like",
#     "RCA", "QSIM-XXZ", "QAOA-3reg", "VQE-Full",
# ]
DEFAULT_CIRCS = [
    "LinearEnt",
]
# DEFAULT_CIRCS = [
#     "GHZ-Chain", "LinearEnt",
#     "RCA", "QSIM-XXZ", "QAOA-3reg", "VQE-Full",
# ]
DEFAULT_DEPTHS = [4]
DEFAULT_QUBITS = [8, 16, 32, 64, 96, 112, 127] #[3, 7, 11, 15, 19, 21, 23]

MODES = ["baseline", "tcache"]  # 两种方法


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
    return p.parse_args()


def main():
    args = parse_args()

    if not args.figdir.exists():
        raise SystemExit(f"[Error] 数据目录不存在：{args.figdir}")

    # 去重并排序，确保绘图 x 轴稳定
    circs = list(dict.fromkeys(args.circs))
    qubits = sorted(set(args.qubits), key=lambda x: args.qubits.index(x))  # 保持用户给定顺序
    depths = list(dict.fromkeys(args.depths))

    agg_depth = aggregate_sum_total_over_depths(
        figdir=args.figdir,
        code_tag=args.code_tag,
        circs=circs,
        qubits=qubits,
        depths=depths,
    )
    agg_qubits = aggregate_sum_total_over_qubits(
        figdir=args.figdir,
        code_tag=args.code_tag,
        circs=circs,
        qubits=qubits,
        depths=depths,
    )

    agg_depth_tTime = aggregate_sum_total_over_depths_tTime(
        figdir=args.figdir,
        code_tag=args.code_tag,
        circs=circs,
        qubits=qubits,
        depths=depths,
    )
    print(agg_depth_tTime)

    # plot_lines_sumD(agg_depth, qubits=qubits, ylog=True, outdir=args.outdir)
    # plot_lines_sunQ(agg_qubits, depths=depths, ylog=True, outdir=args.outdir)
    # plot_lines_sumD(agg_depth_tTime, qubits=qubits, ylog=False, outdir=args.outdir)

    # === 每电路各自作图 ===
    for circ in circs:
        plot_single_circuit(
            agg_one=agg_depth_tTime[circ],
            #
            circ=circ,
            qubits=qubits,
            ylog=False,
            outdir=args.outdir,
        )


if __name__ == "__main__":
    main()
