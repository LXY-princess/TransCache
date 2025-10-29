# -*- coding: utf-8 -*-
"""
v18_wl_batch_builder_visualizer.py

基于:
  - v18_wl_kitchen.py 里的各类 workload 生成器
  - v18_runner_workload_scaling_multi.py 里的 slim_workload_for_dump 与部分可视化思路

功能（满足你的三点要求）：
1) 一次性调用“所有”workload 变体，以**相同长度 N**生成请求序列；
   同时把每个 workload 的 `workload.json`（瘦身版）与 `info.json` 落盘，
   以及一个可复绘的 `bundle.json`（包含直方图与热力图所需数据）。
2) 把原先合在一个函数里的多种图拆成**独立函数**，每种图各一个 function；
   并为每个 workload 分别绘制：
      - inter_arrival_hist.png
      - counting_process.png
      - fano_curve.png
      - acf_counts.png
      - class_zipf.png
      - heat_topk_classes.png
   （如需 “全部类热力图”，可把 `draw_heatmap(..., topk_classes=None)` 设为 None）
3) 每种 workload 的数据与图分别放入**不同子文件夹**：
      <out_root>/<workload_name>/data/
      <out_root>/<workload_name>/plots/

运行示例：
  python v18_wl_batch_builder_visualizer.py \
      --N 300 --q_list 5,7,11,13,15,17 --d_list 2,4,6 \
      --rps 1.0 --out_root wl_out

注意：脚本依赖你现有工程中的 v18_core 和 v18_wl_kitchen 等模块。
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# ==== 依赖你工程内的工具 ====
from v18_core import build_catalog  # 用于生成 meta（包含 maker_run）
from v18_wl_kitchen import (
    build_workload_poisson_superposition_exact,
    build_workload_renewal_exact,
    build_workload_nhpp_exact,
    build_workload_mmpp2_exact,
    build_workload_hawkes_exact,
    build_workload_compound_exact,
    build_workload_sessions_dirichlet_exact,
    build_workload_change_point_exact,
)

# ---------------- JSON utils ----------------
def _json_default(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    return str(o)

def json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)

def slim_workload_for_dump(workload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """仅保存 name/q/d/ts（把 t_arr 重命名为 ts），避免不可序列化字段。"""
    slim = []
    for it in workload:
        slim.append({
            "name": it.get("name"),
            "q": int(it.get("q")) if it.get("q") is not None else None,
            "d": int(it.get("d")) if it.get("d") is not None else None,
            "ts": float(it.get("t_arr", it.get("ts", 0.0))),
        })
    return slim

# ---------------- 基础绘图与统计工具 ----------------
def _set_plot_style_inter_arrival():
    plt.rcParams.update({
        # 字体与字号
        "font.family": "Times New Roman",
        "font.size": 28,
        # 坐标轴 & 标题
        "axes.linewidth": 4,
        "axes.labelsize": 30,  # xlabel/ylabel 字号
        "axes.labelweight": "bold",  # xlabel/ylabel 加粗
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        # 网格
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 4,
        "grid.alpha": 0.4,
        # 线条/marker 的默认（仍可在 plot 里覆盖）
        "lines.linewidth": 6,
        "lines.markersize": 18,
        "lines.markerfacecolor": "none",
        "lines.markeredgewidth": 4,
        # 刻度线（粗细 & 长度）
        "xtick.major.width": 3.0,
        "ytick.major.width": 3.0,
        "xtick.minor.width": 2.0,
        "ytick.minor.width": 2.0,
        "xtick.major.size": 8.0,
        "ytick.major.size": 8.0,
        "xtick.minor.size": 5.0,
        "ytick.minor.size": 5.0,
        # 刻度标签字号（注意：粗体不能用 rcParams 设置，见下方循环）
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        # 图例（字号可配，但粗体仍需手动 prop）
        "legend.frameon": False,
        "legend.fontsize": 28,
        # 保存
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })

def _set_plot_style():
    plt.rcParams.update({
        # 字体与字号
        "font.family": "Times New Roman",
        "font.size": 12,
        # # 坐标轴 & 标题
        # "axes.linewidth": 4,
        # "axes.labelsize": 40,  # xlabel/ylabel 字号
        # "axes.labelweight": "bold",  # xlabel/ylabel 加粗
        # "axes.titlesize": 24,
        # "axes.titleweight": "bold",
        # # 网格
        # "axes.grid": True,
        # "grid.linestyle": "--",
        # "grid.linewidth": 4,
        # "grid.alpha": 0.4,
        # # 线条/marker 的默认（仍可在 plot 里覆盖）
        # "lines.linewidth": 6,
        # "lines.markersize": 18,
        # "lines.markerfacecolor": "none",
        # "lines.markeredgewidth": 4,
        # # 刻度线（粗细 & 长度）
        # "xtick.major.width": 3.0,
        # "ytick.major.width": 3.0,
        # "xtick.minor.width": 2.0,
        # "ytick.minor.width": 2.0,
        # "xtick.major.size": 8.0,
        # "ytick.major.size": 8.0,
        # "xtick.minor.size": 5.0,
        # "ytick.minor.size": 5.0,
        # # 刻度标签字号（注意：粗体不能用 rcParams 设置，见下方循环）
        # "xtick.labelsize": 26,
        # "ytick.labelsize": 26,
        # # 图例（字号可配，但粗体仍需手动 prop）
        # "legend.frameon": False,
        # "legend.fontsize": 22,
        # # 保存
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })

def _sorted_ts(workload: List[Dict[str, Any]]) -> np.ndarray:
    ts = np.array([float(it.get("t_arr", it.get("ts", 0.0))) for it in workload], dtype=float)
    ts.sort()
    return ts

# ========== 单图一个函数（独立可复用） ==========
def draw_interarrival_hist(workload: List[Dict[str, Any]], info: Dict[str, Any], out_path: Path, bins: int = 40):
    ts = _sorted_ts(workload)
    if ts.size < 2:
        return
    inter = np.diff(ts)
    # 速率估计/注记：优先 info["Lambda"] 或 info["rps"]
    Lam = info.get("Lambda", info.get("rps", None))
    if Lam is None:
        Lam = 1.0 / max(float(inter.mean()), 1e-12)

    _set_plot_style_inter_arrival()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(inter, bins=bins, density=True, alpha=1, label="Inter-arrival", color="#68aed9")
    x = np.linspace(0, max(inter.max(), 1e-6), 200)
    ax.plot(x, float(Lam) * np.exp(-float(Lam) * x), "--", lw=6, label=f"Ideal Poisson", color="#2b3c8f")
    # ax.set_title("Inter-arrival histogram")
    ax.set_xlabel("Δt (s)"); ax.set_ylabel("Prob Density (1/s)")
    # ax.legend(frameon= False)
    # —— 必须逐轴处理的部分（rcParams 无法直接控制粗体） ——
    # 1) 刻度标签加粗（rcParams 没有 tick label weight）
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    # 2) 图例加粗（rcParams 仅能设字号；粗体需用 prop 或循环设置）
    # plt.legend(prop={"weight": "bold"})
    # fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, dpi=600); plt.close(fig)

def _nhpp_expectation_curve_from_params(tline: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    根据保存的 NHPP 参数，计算 E[N(t)] 的**累计积分曲线**。
    目前支持：'sinusoid' 形式： λ(t) = base * (1 + amp * sin(2π t / T))
    """
    _set_plot_style
    form = (params or {}).get("form", "sinusoid").lower()
    if form != "sinusoid":
        # 回退：直接用均值速率 base 近似
        base = float((params or {}).get("lam_base", 1.0))
        return base * tline
    base = float(params.get("lam_base", 1.0))
    amp  = float(params.get("amp", 0.0))
    T    = float(params.get("period", 10.0))
    # E[N(t)] = ∫ λ(u) du = base * ( t - (amp*T/(2π)) * (cos(2π t/T) - 1) )
    # 为了绘“曲线”，我们输出累积期望值
    two_pi_over_T = 2.0 * math.pi / max(T, 1e-12)
    return base * (tline - (amp * (1.0 / two_pi_over_T)) * (np.cos(two_pi_over_T * tline) - 1.0))

def draw_counting_process(workload: List[Dict[str, Any]], info: Dict[str, Any], out_path: Path):
    ts = _sorted_ts(workload)
    if ts.size == 0:
        return
    t0, t1 = 0.0, float(ts[-1])
    tline = np.linspace(t0, t1, 300)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(ts, np.arange(1, ts.size + 1), where="post", label="N(t)")

    if info.get("kind") == "nhpp" and "lambda_params" in info:
        En = _nhpp_expectation_curve_from_params(tline, info["lambda_params"])
        ax.plot(tline, En, "--", lw=2, label="E[N(t)] via λ(t)")
    else:
        Lam = info.get("Lambda", info.get("rps", None))
        if Lam is None:
            Lam = float(ts.size / max(t1, 1e-12))
        ax.plot(tline, float(Lam) * tline, "--", lw=2, label=f"E[N(t)]≈Λ t, Λ≈{float(Lam):.2f}")

    ax.set_title("Counting process vs expectation"); ax.set_xlabel("t"); ax.set_ylabel("N(t)")
    ax.legend(frameon=False)
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, dpi=600); plt.close(fig)

def draw_fano_curve(workload: List[Dict[str, Any]], out_path: Path, grid: Optional[List[float]] = None):
    ts = _sorted_ts(workload)
    if ts.size == 0:
        return
    T = float(ts[-1])
    if grid is None:
        grid = np.geomspace(max(T / 200.0, 1e-3), max(T / 6.0, 1e-3), 8)
    fanos = []
    for w in grid:
        edges = np.arange(0.0, T + w, w)
        cnts, _ = np.histogram(ts, bins=edges)
        mu = float(np.mean(cnts))
        va = float(np.var(cnts, ddof=1)) if cnts.size > 1 else 0.0
        fanos.append(va / max(mu, 1e-12))

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, fanos, marker="o")
    ax.axhline(1.0, linestyle="--", lw=1)
    ax.set_xscale("log"); ax.set_title("Fano factor vs bin width")
    ax.set_xlabel("bin width (seconds)"); ax.set_ylabel("Fano = Var/Mean")
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, dpi=600); plt.close(fig)

def draw_acf_counts(workload: List[Dict[str, Any]], out_path: Path, binsize: float | None = None):
    ts = _sorted_ts(workload)
    if ts.size < 3:
        return
    T = float(ts[-1])
    if binsize is None:
        binsize = max(T / 100.0, 1e-3)
    edges = np.arange(0.0, T + binsize, binsize)
    cnts, _ = np.histogram(ts, bins=edges)
    cnts = cnts - np.mean(cnts)
    acf_lags = min(20, cnts.size - 2) if cnts.size >= 3 else 0
    if acf_lags <= 0:
        return
    acf = [1.0]
    den = float(np.dot(cnts, cnts))
    for k in range(1, acf_lags + 1):
        num = float(np.dot(cnts[:-k], cnts[k:]))
        acf.append(num / max(den, 1e-12))

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stem(np.arange(0, acf_lags + 1), acf)
    ax.set_title(f"ACF of binned counts (binsize={binsize:.3g}s)")
    ax.set_xlabel("lag"); ax.set_ylabel("acf")
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, dpi=600); plt.close(fig)

def draw_class_zipf(workload: List[Dict[str, Any]], meta: List[Dict[str, Any]], out_path: Path):
    # 统计每个 meta 类的出现次数（用 name|q|d 匹配）
    label2idx = {f'{m["name"]}|q{m["q"]}|d{m["d"]}': i for i, m in enumerate(meta)}
    counts = np.zeros(len(meta), dtype=int)
    for it in workload:
        k = f'{it["name"]}|q{it["q"]}|d{it["d"]}'
        if k in label2idx:
            counts[label2idx[k]] += 1
    freq = np.sort(counts)[::-1]
    ranks = np.arange(1, len(freq) + 1)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, np.maximum(freq, 1), marker=".", linestyle="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Class frequency Zipf plot")
    ax.set_xlabel("rank (log)"); ax.set_ylabel("count (log)")
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, dpi=600); plt.close(fig)

def draw_heatmap(workload: List[Dict[str, Any]], meta: List[Dict[str, Any]], out_path: Path,
                 topk_classes: Optional[int] = 12):
    _set_plot_style()
    """
    若 topk_classes 为 None，则绘制出现过的“全部类”的热力图；否则绘制 top-k。
    """
    # 先按类聚到达时刻
    from collections import defaultdict
    label2idx = {f'{m["name"]}|q{m["q"]}|d{m["d"]}': i for i, m in enumerate(meta)}
    groups = defaultdict(list)
    ts_all = []
    for it in workload:
        key = f'{it["name"]}|q{it["q"]}|d{it["d"]}'
        if key in label2idx:
            groups[label2idx[key]].append(float(it.get("t_arr", it.get("ts", 0.0))))
            ts_all.append(float(it.get("t_arr", it.get("ts", 0.0))))
    if not ts_all:
        return
    ts_all = np.array(ts_all, dtype=float); ts_all.sort()
    T = float(ts_all[-1])

    # 统计各类总次数，并按需要选出 top-k 或全部出现过的类
    total_counts = {i: len(v) for i, v in groups.items()}
    if topk_classes is None:
        idx_use = [i for i, c in sorted(total_counts.items(), key=lambda x: -x[1]) if c > 0]
        title = f"All classes (nonzero) counts over time (K={len(idx_use)})"
    else:
        idx_sorted = [i for i, _ in sorted(total_counts.items(), key=lambda x: -x[1])]
        idx_use = idx_sorted[:min(topk_classes, len(idx_sorted))]
        title = "Top-K class counts over time"

    # 时间分箱：自适应
    B = min(120, max(20, int(ts_all.size / 50)))
    edges = np.linspace(0.0, T, B + 1)
    mat = np.zeros((len(idx_use), B), dtype=int)
    for row, i in enumerate(idx_use):
        arr = np.array(groups.get(i, []), dtype=float)
        if arr.size > 0:
            c, _ = np.histogram(arr, bins=edges)
            mat[row, :] = c

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * B), max(4, 0.35 * len(idx_use))))
    im = ax.imshow(mat, aspect="auto", origin="lower", extent=[0.0, T, 0, len(idx_use)])
    ax.set_yticks(np.arange(len(idx_use)) + 0.5)
    ax.set_yticklabels([f'{meta[i]["name"]}|q{meta[i]["q"]}|d{meta[i]["d"]}' for i in idx_use])
    ax.set_xlabel("Time"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_path, dpi=600); plt.close(fig)

# ----------------- 可复绘 bundle（便于保存“可视化所需的信息”） -----------------
def make_bundle(workload: List[Dict[str, Any]], meta: List[Dict[str, Any]],
                bins_inter: int = 40, B: int | None = None,
                include_zero_classes: bool = False) -> Dict[str, Any]:
    """
    与 v18_wl_kitchen.dump_interarrival_heat_bundle 同理念：
      - ts / inter: 画间隔直方图
      - class_labels / edges_time / counts_time: 画热力图
    这里直接返回 dict，外部负责写 JSON。
    """
    ts = sorted(float(it.get("t_arr", it.get("ts", 0.0))) for it in workload)
    inter = (np.diff(np.array(ts, dtype=float)).tolist() if len(ts) > 1 else [])
    T = ts[-1] if ts else 0.0

    # 类标签集
    ev_labels = [f'{w["name"]}|q{w["q"]}|d{w["d"]}' for w in workload]
    uniq_labels, counts = np.unique(ev_labels, return_counts=True)
    if include_zero_classes:
        class_labels = [f'{m["name"]}|q{m["q"]}|d{m["d"]}' for m in meta]
    else:
        class_labels = list(uniq_labels)

    # 时间分箱
    if B is None:
        B = min(120, max(20, int(len(ts) / 50))) if ts else 20
    edges = np.linspace(0.0, T, B + 1)

    # 各类按时间分箱统计
    from collections import defaultdict
    group_ts = defaultdict(list)
    for w in workload:
        key = f'{w["name"]}|q{w["q"]}|d{w["d"]}'
        group_ts[key].append(float(w.get("t_arr", w.get("ts", 0.0))))
    counts_time = []
    for lab in class_labels:
        arr = np.array(group_ts.get(lab, []), dtype=float)
        c, _ = np.histogram(arr, bins=edges)
        counts_time.append(c.astype(int).tolist())

    return {
        "N": len(ts),
        "T_end": float(T),
        "bins_inter": int(bins_inter),
        "ts": ts,
        "inter": inter,
        "class_labels": class_labels,
        "edges_time": edges.tolist(),
        "counts_time": counts_time,
    }

# ----------------- 构建所有 workload（统一长度 N） -----------------
def build_all_workloads(meta: List[Dict[str, Any]], N: int, rps: float, rng_seed_base: int = 123
                        ) -> Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    out: Dict[str, Tuple[List[Dict[str, Any]], Dict[str, Any]]] = {}
    seed = int(rng_seed_base)

    # 1) Poisson superposition
    wl, info = build_workload_poisson_superposition_exact(meta, N, hot_fraction=0.25, hot_boost=8.0,
                                                          rps=rps, rng_seed=seed, return_timestamps=True)
    out["poisson_superposition"] = (wl, info)

    # 2) Renewal family
    wl, info = build_workload_renewal_exact(meta, N, rps=rps, renewal_kind="gamma", shape=3.0,
                                            class_mode="weighted", rng_seed=seed)
    out["renewal_gamma_k3"] = (wl, info)

    # wl, info = build_workload_renewal_exact(meta, N, rps=rps, renewal_kind="weibull", shape=1.5,
    #                                         class_mode="weighted", rng_seed=seed + 3)
    # out["renewal_weibull_k1p5"] = (wl, info)
    #
    # wl, info = build_workload_renewal_exact(meta, N, rps=rps, renewal_kind="lognormal", sigma=0.75,
    #                                         class_mode="weighted", rng_seed=seed + 4)
    # out["renewal_lognormal_s0p75"] = (wl, info)
    #
    # wl, info = build_workload_renewal_exact(meta, N, rps=rps, renewal_kind="lomax", shape=2.5,
    #                                         class_mode="weighted", rng_seed=seed + 5)
    # out["renewal_lomax_a2p5"] = (wl, info)

    # 3) NHPP（正弦波速率）
    # lam_base = rps
    # amp = 0.6       # 0~<1
    # period = 10.0   # 秒
    # def _lam_fn(t: float) -> float:
    #     return lam_base * (1.0 + amp * math.sin(2.0 * math.pi * float(t) / period))
    # lambda_max = lam_base * (1.0 + amp)  # 上界
    # wl, info = build_workload_nhpp_exact(meta, N, lambda_of_t=_lam_fn, lambda_max=lambda_max,
    #                                      class_mode="weighted", rng_seed=seed + 6, return_timestamps=True)
    # # 额外保存参数以便可视化重建期望曲线
    # info["lambda_params"] = {"form": "sinusoid", "lam_base": lam_base, "amp": amp, "period": period, "lambda_max": lambda_max}
    # out["nhpp_sinusoid"] = (wl, info)

    # 4) 两态 MMPP
    # wl, info = build_workload_mmpp2_exact(meta, N, rps=rps, contrast=8.0, pi1=0.20, a=0.05, b=0.25,
    #                                       class_mode="weighted", rng_seed=seed + 7, return_timestamps=True)
    # out["mmpp2_contrast8"] = (wl, info)

    # 5) Hawkes
    wl, info = build_workload_hawkes_exact(meta, N, rps=rps, eta=0.5, beta=3.0,
                                           class_mode="weighted", rng_seed=seed, return_timestamps=True)
    out["hawkes_eta0p5_b3"] = (wl, info)

    # 6) 复合突发（簇）
    # wl, info = build_workload_compound_exact(meta, N, rps=rps, mean_burst_size=5.0, in_burst_jitter=0.01,
    #                                          class_mode="weighted", rng_seed=seed + 9, return_timestamps=True)
    # out["compound_bursty"] = (wl, info)

    # 7) 会话 + Dirichlet 类别
    # wl, info = build_workload_sessions_dirichlet_exact(meta, N, rps=rps, mean_session_events=20,
    #                                                    mean_idle_gap=5.0, in_session_rps=None, alpha=0.1,
    #                                                    rng_seed=seed + 10, return_timestamps=True)
    # out["sessions_dirichlet"] = (wl, info)

    # 8) 简单突变（速率+类别）
    # wl, info = build_workload_change_point_exact(meta, N, rps_left=0.8 * rps, rps_right=1.6 * rps, frac_left=0.5,
    #                                              class_mode_left="weighted", class_mode_right="zipf",
    #                                              rng_seed=seed + 11, return_timestamps=True)
    # out["change_point"] = (wl, info)

    return out

def visualize_wl(
    name: Optional[str],
    out_root: Path,
    topk_classes: int,
    meta: List[Dict[str, Any]],
    workload_pair: Tuple[List[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """
    将单个 workload (wl, info) 的数据与图形输出到:
      <out_root>/<name>/{data,plots}/
    并返回一个包含文件路径的索引字典，便于外部汇总。

    Args:
      name: 子目录名；若为 None，则用 info.get("kind", "workload")
      out_root: 输出根目录 Path
      topk_classes: 热力图展示的 Top-K 类目
      meta: catalog 元信息（用于类目匹配/标签）
      workload_pair: 形如 (wl, info)
    Returns:
      index: {"data": {...}, "plots": {...}, "meta": {...}}
    """
    # ---- 解包并兜底 name ----
    workload, info = workload_pair
    if name is None:
        name = str(info.get("kind", "workload"))

    # ---- 目录 ----
    root = out_root / name
    data_dir = root / "wl_data"
    plot_dir = root / "wl_plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- 保存数据 ----
    # 1) workload（瘦身版）
    (data_dir / "workload.json").write_text(json_dump(slim_workload_for_dump(workload)), encoding="utf-8")
    # 2) info（包含生成参数；NHPP 额外存了 lambda_params）
    (data_dir / "info.json").write_text(json_dump(info), encoding="utf-8")
    # 3) bundle（直方图/热力图可复绘信息）
    bundle = make_bundle(workload, meta, bins_inter=40, B=None, include_zero_classes=False)
    (data_dir / "bundle.json").write_text(json_dump(bundle), encoding="utf-8")

    # ---- 单图函数：分别绘制 ----
    # draw_interarrival_hist(workload, info, plot_dir / "inter_arrival_hist.png", bins=40)
    # draw_counting_process(workload, info, plot_dir / "counting_process.png")
    # draw_fano_curve(workload, plot_dir / "fano_curve.png")
    # draw_acf_counts(workload, plot_dir / "acf_counts.png")
    # draw_class_zipf(workload, meta, plot_dir / "class_zipf.png")
    draw_heatmap(workload, meta, plot_dir / "heat_topk_classes.png", topk_classes=topk_classes)
    # 如果还需要“全部类”热力图，请取消下一行注释：
    # draw_heatmap(workload, meta, plot_dir / "heat_all_classes.png", topk_classes=None)
    draw_interarrival_hist(workload, info, plot_dir / "inter_arrival_hist.png", bins=40)


# ----------------- 主流程：构建、保存、绘图 -----------------
def run_all(N: int, q_list: List[int], d_list: List[int], rps: float, out_root: Path, rng_seed_base: int = 123,
            topk_classes: int = 12):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    # catalog（meta 与 maker_run 来自你的工程）
    makers_all, meta = build_catalog(q_list, d_list)

    # 统一长度构建全部 workloads
    all_wls = build_all_workloads(meta, N=N, rps=rps, rng_seed_base=rng_seed_base)

    index = {}
    for name, pair in all_wls.items():  # pair 就是 (workload, info)
        visualize_wl(
            name=name,
            out_root=out_root,  # Path
            topk_classes=topk_classes,  # int
            meta=meta,  # build_catalog(...) 得到的 meta
            workload_pair=pair
        )
    # for name, (workload, info) in all_wls.items():
        # root = out_root / name
        # data_dir  = root / "data"
        # plot_dir  = root / "plots"
        # data_dir.mkdir(parents=True, exist_ok=True)
        # plot_dir.mkdir(parents=True, exist_ok=True)
        #
        # # ---- 保存数据 ----
        # # 1) workload（瘦身版）
        # (data_dir / "workload.json").write_text(json_dump(slim_workload_for_dump(workload)), encoding="utf-8")
        # # 2) info（包含生成参数；NHPP 额外存了 lambda_params）
        # (data_dir / "info.json").write_text(json_dump(info), encoding="utf-8")
        # # 3) bundle（直方图/热力图可复绘信息）
        # bundle = make_bundle(workload, meta, bins_inter=40, B=None, include_zero_classes=False)
        # (data_dir / "bundle.json").write_text(json_dump(bundle), encoding="utf-8")
        #
        # # ---- 单图函数：分别绘制 ----
        # # draw_interarrival_hist(workload, info, plot_dir / "inter_arrival_hist.png", bins=40)
        # # draw_counting_process(workload, info, plot_dir / "counting_process.png")
        # # draw_fano_curve(workload, plot_dir / "fano_curve.png")
        # # draw_acf_counts(workload, plot_dir / "acf_counts.png")
        # # draw_class_zipf(workload, meta, plot_dir / "class_zipf.png")
        # draw_heatmap(workload, meta, plot_dir / "heat_topk_classes.png", topk_classes=topk_classes)
        # # 如果还需要“全部类”热力图，请取消下一行注释：
        # # draw_heatmap(workload, meta, plot_dir / "heat_all_classes.png", topk_classes=None)
        # draw_interarrival_hist(workload, info, plot_dir / "inter_arrival_hist.png", bins=40)
        #
        # index[name] = {
        #     "data": {
        #         "workload_json": str((data_dir / "workload.json").resolve()),
        #         "info_json":     str((data_dir / "info.json").resolve()),
        #         "bundle_json":   str((data_dir / "bundle.json").resolve()),
        #     },
        #     "plots": {
        #         "inter_arrival": str((plot_dir / "inter_arrival_hist.png").resolve()),
        #         "counting":      str((plot_dir / "counting_process.png").resolve()),
        #         "fano":          str((plot_dir / "fano_curve.png").resolve()),
        #         "acf":           str((plot_dir / "acf_counts.png").resolve()),
        #         "zipf":          str((plot_dir / "class_zipf.png").resolve()),
        #         "heat_topk":     str((plot_dir / "heat_topk_classes.png").resolve()),
        #         # "heat_all":    str((plot_dir / "heat_all_classes.png").resolve()),
        #     },
        #     "meta": {"N": N, "rps": rps}
        # }

    # 顶层索引，便于巡查
    (out_root / "index.json").write_text(json_dump(index), encoding="utf-8")
    print(f"[DONE] Saved all workloads & plots to: {out_root}")

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=150, help="每个 workload 的统一长度（请求条数）")
    ap.add_argument("--q_list", type=str, default="5,7,11,13,15,17")
    ap.add_argument("--d_list", type=str, default="2,4,6")
    ap.add_argument("--rps", type=float, default=1.0, help="目标平均到达速率（部分模型为估计值）")
    ap.add_argument("--out_root", type=str, default="figs/v18_wl_gen", help="输出根目录")
    ap.add_argument("--rng_seed_base", type=int, default=123)
    ap.add_argument("--topk_classes", type=int, default=12)
    args = ap.parse_args()
    q_list = [int(x) for x in args.q_list.split(",") if x]
    d_list = [int(x) for x in args.d_list.split(",") if x]
    return args.N, q_list, d_list, args.rps, Path(args.out_root), args.rng_seed_base, args.topk_classes

def main():
    N, q_list, d_list, rps, out_root, seed, topk = parse_args()
    run_all(N=N, q_list=q_list, d_list=d_list, rps=rps, out_root=out_root,
            rng_seed_base=seed, topk_classes=topk)

if __name__ == "__main__":
    main()
