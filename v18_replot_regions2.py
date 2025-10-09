# v18_replot_regions.py
import json, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 依赖你的工程路径变量
from v18_core import PLOT_DIR, LOAD_ROOT  # 与原 runner 保持一致
# 直接读取你原 summary 的结构（由 v18_runner_wl_rounds.py 写出）

# ---------- 基础小工具 ----------
def _set_plot_style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})

def _flatten_points(raw_lat: Dict[str, List[List[float]]],
                    raw_csz: Dict[str, List[List[float]]],
                    methods: List[str]) -> Dict[str, np.ndarray]:
    """把某方法在所有 workload size & round 的点拉平成 Nx2 的数组：[[cache, latency], ...]"""
    points = {}
    for m in methods:
        Xs, Ys = [], []
        per_size_lat = raw_lat[m]
        per_size_csz = raw_csz[m]
        for si in range(min(len(per_size_lat), len(per_size_csz))):
            lats = per_size_lat[si]
            cszs = per_size_csz[si]
            k = min(len(lats), len(cszs))
            if k <= 0: continue
            Ys.extend(lats[:k])
            Xs.extend(cszs[:k])
        if Xs and Ys:
            points[m] = np.column_stack([np.array(Xs, float), np.array(Ys, float)])
        else:
            points[m] = np.zeros((0, 2), float)
    return points

def _geometric_median(P: np.ndarray, max_iter: int = 512, eps: float = 1e-6) -> np.ndarray:
    """Weiszfeld 算法，求几何中位数（稳健代表点）；P 为 Nx2。"""
    if P.shape[0] == 0: return np.zeros(2, float)
    x = np.median(P, axis=0)  # 初始化：坐标中位数
    for _ in range(max_iter):
        d = np.linalg.norm(P - x, axis=1)
        # 防止除0
        if np.any(d < 1e-12):
            return P[d.argmin()]
        w = 1.0 / np.clip(d, 1e-12, None)
        x_new = (P * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(x_new - x) < eps:
            return x_new
        x = x_new
    return x

def _ellipse_from_cov(mean: np.ndarray, cov: np.ndarray, q: float = 0.90) -> Tuple[float, float, float]:
    """从均值/协方差得到椭圆参数：width,height,angle(deg)。椭圆对应 (x-μ)^T Σ^{-1} (x-μ) = chi2(df=2, q)"""
    chi2_map = {0.50: 1.386, 0.68: 2.279, 0.80: 3.219, 0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
    q_key = min(chi2_map.keys(), key=lambda v: abs(v - q))
    s = chi2_map[q_key]
    evals, evecs = np.linalg.eigh(cov)  # 升序
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    width  = 2.0 * math.sqrt(max(evals[0], 1e-12) * s)
    height = 2.0 * math.sqrt(max(evals[1], 1e-12) * s)
    angle  = math.degrees(math.atan2(evecs[1, 0], evecs[0, 0]))
    return width, height, angle

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew 单调链凸包，返回顶点顺序（闭合不重复最后点）。输入 Nx2（x=cache, y=lat）."""
    if points.shape[0] <= 2:
        return points
    pts = np.unique(points, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # 先 x 后 y
    def cross(o, a, b):  # 2D叉积
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    # 下链
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    # 上链
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

# def _draw_sizebar(ax_main, sizes: List[int], size_to_s,
#                   x: float = 1.02, y: float = 0.10, w: float = 0.035, h: float = 0.80):
#     """
#     在主轴右侧插入“连续尺寸条”（非颜色），解释 workload size → marker 大小。
#     使用主轴的归一化坐标指定位置与大小，避免被 tight_layout 裁切。
#       x,y,w,h: 在 ax_main.transAxes 下的 [left, bottom, width, height]
#     """
#     vmin, vmed, vmax = min(sizes), int(np.median(sizes)), max(sizes)
#     Ns = np.linspace(vmin, vmax, 28)
#
#     # 直接用相对主轴的坐标安放（更稳，不依赖 bbox_to_anchor）
#     ax_bar = ax_main.inset_axes([x, y, w, h], transform=ax_main.transAxes)
#     ax_bar.set_facecolor("none"); ax_bar.set_xticks([])
#
#     ygrid = np.linspace(0, 1, Ns.size)
#     ax_bar.scatter(np.zeros_like(ygrid), ygrid, s=[size_to_s(n) for n in Ns],
#                    facecolors="#cfcfcf", edgecolors="black", linewidths=0.4)
#
#     ax_bar.set_ylim(-0.05, 1.05); ax_bar.set_xlim(-0.5, 0.5)
#     for spine in ax_bar.spines.values():
#         spine.set_visible(False)
#
#     # min / median / max 三个刻度（放在右侧）
#     def _norm(n): return (n - vmin) / max(1e-12, (vmax - vmin))
#     ax_bar.set_yticks([_norm(vmin), _norm(vmed), _norm(vmax)])
#     ax_bar.set_yticklabels([f"{vmin}", f"{vmed}", f"{vmax}"])
#     ax_bar.yaxis.tick_right()
#     ax_bar.set_title("Workload\n size", fontsize=12, pad=6)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

def _draw_sizebar(
    ax_main,
    sizes: list[int],
    size_to_s,
    *,
    # 在主轴坐标中的位置与大小（可以微调）
    x: float = 1.02, y: float = 0.10, w: float = 0.055, h: float = 0.80,
    # 想要显示的刻度数量（包含首尾）
    n_ticks: int = 7,
    # 刻度是否取整到最近的10
    round_to_10: bool = True,
):
    """
    在主轴右侧插入“连续尺寸条”（非颜色），解释 workload size → marker 大小。
    - sizes: 全部 N 值（决定范围）
    - size_to_s: N→scatter 面积(s) 的映射（需与主图一致）
    """
    vmin, vmax = int(min(sizes)), int(max(sizes))
    Ns = np.linspace(vmin, vmax, 28)

    # 在主轴的归一化坐标系下安放尺寸条
    ax_bar = ax_main.inset_axes([x, y, w, h], transform=ax_main.transAxes)
    ax_bar.set_facecolor("none")
    ax_bar.set_xticks([])

    # 竖直方向从下到上画一列点，形成“连续”视觉
    ygrid = np.linspace(0, 1, Ns.size)
    ax_bar.scatter(
        np.zeros_like(ygrid), ygrid,
        s=[size_to_s(n) for n in Ns],
        facecolors="#cfcfcf", edgecolors="black", linewidths=0.4
    )
    ax_bar.set_ylim(-0.05, 1.05)
    ax_bar.set_xlim(-0.5, 0.5)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)

    # —— 刻度：min→max 之间放 n_ticks 个数值（默认 7 个）
    if n_ticks < 2:
        n_ticks = 2
    tick_vals = np.linspace(vmin, vmax, n_ticks)
    if round_to_10 and (vmax - vmin) >= 50:
        tick_vals = np.round(tick_vals / 10.0) * 10.0
    # 去重，避免 vmin/vmax 过近导致重复
    tick_vals = np.unique(tick_vals.astype(int)).tolist()

    def _norm(n):  # 线性归一化到 [0,1]
        return (n - vmin) / max(1e-12, (vmax - vmin))

    ax_bar.set_yticks([_norm(v) for v in tick_vals])
    ax_bar.set_yticklabels([f"{v}" for v in tick_vals])

    # —— 把刻度与标题都放到右侧，并把标题放在数字的右边
    ax_bar.yaxis.tick_right()
    ax_bar.yaxis.set_label_position("right")
    ax_bar.set_ylabel("Workload size (#requests)", rotation=90, va="center", ha="center", labelpad=12, fontsize=20)

    # ax_bar.set_yticks([_norm(v) for v in tick_vals])
    # ax_bar.set_yticklabels([f"{v}" for v in tick_vals])
    # ax_bar.yaxis.tick_right()
    #
    # # —— 竖排标题（避免与主图重叠可调 labelpad）
    # ax_bar.set_ylabel("Workload size (requests)", rotation=90, labelpad=22, fontsize=12)


# NEW: 把 workload size 线性映射到 marker 面积（s）
def _size_scaler(values: List[int], s_min: float = 40, s_max: float = 160):
    v = np.asarray(values, float)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin + 1e-12:
        return lambda x: (s_min + s_max) / 2.0
    def f(x):
        x = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
        return s_min + x * (s_max - s_min)
    return f

# ---------- 主绘图函数 ----------
def load_and_replot_regions(load_dir: str,
                            out_png: str = "scaling_latency_vs_cache_scatter_regions.png",
                            show_scatter: bool = True,
                            region_kind: str = "ellipse",   # "ellipse" | "hull" | "both"
                            ellipse_q: float = 0.90,
                            emphasize_method: str = None,   # 例如 "FS+Pre+ttl+SE+ema"
                            point_kind: str = "geom_median", # "geom_median" | "median" | "mean"
                            methods_keep: List[str] | None = None
                            ):
    """从 summary JSON 载入所有点，叠加“每方法的区域 + 代表点”，并保存图片。"""
    LOAD_DIR = Path(load_dir)
    json_path = LOAD_DIR / "summaries" / "scaling_multi_rounds_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Summary not found: {json_path}")

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    sizes: List[int] = summary["sizes"]
    methods_all: List[str] = summary["methods"]
    raw_lat_full = summary["raw"]["e2e_latency"]
    raw_csz_full = summary["raw"]["final_cache_size"]

    # 只保留需要的方法（按给定顺序）
    methods = [m for m in (methods_keep or methods_all) if m in methods_all]
    raw_lat = {m: raw_lat_full[m] for m in methods}
    raw_csz = {m: raw_csz_full[m] for m in methods}

    # 收集每方法所有点
    points = _flatten_points(raw_lat, raw_csz, methods)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    fig.subplots_adjust(right=0.86)  # ← 给右侧尺寸条留出空间

    # 方法色盘（固定颜色，区域/代表点也用这套颜色）
    palette = ['#fcdd28', '#55A868', '#8172B3', '#C44E52',
               '#4C72B0', '#CCB974', '#64B5CD']

    method_color = {m: palette[i % len(palette)] for i, m in enumerate(methods)}
    markers = ["*", "o", "^", "H", "D", "v", "P", "s", "X", "v"]

    # --- 背景散点：颜色=方法；大小=workload size（替代 colorbar） ---
    if show_scatter:
        size_to_s = _size_scaler(sizes, s_min=40, s_max=160)
        for mi, name in enumerate(methods):
            col = method_color[name]
            mkr = markers[mi % len(markers)]
            per_size_lat = raw_lat[name]
            per_size_csz = raw_csz[name]
            for si, N in enumerate(sizes):
                lats = per_size_lat[si] if si < len(per_size_lat) else []
                cszs = per_size_csz[si] if si < len(per_size_csz) else []
                k = min(len(lats), len(cszs))
                if k <= 0: continue
                xs = np.array(cszs[:k], float); ys = np.array(lats[:k], float)
                ax.scatter(xs, ys, s=size_to_s(N),
                           marker=mkr,
                           facecolors=col,
                           edgecolors="black", linewidths=0.6, alpha=0.80)

        # 方法图例（颜色+形状）
        labels = {"FS+Pre+ttl+SE+ema": "TransCache (Proposed)",
                  "FS+Pre": "CCache + Predictor",
                  "FS": "CCache",
                  "PR": "Braket"}
        method_handles = [
            Line2D([0],[0], marker=markers[i % len(markers)], linestyle="None",
                   markerfacecolor=palette[i % len(palette)], markeredgecolor="black",
                   markersize=10, label=labels.get(methods[i]))
            for i in range(len(methods))
        ]

        # 仅追加一个“白底蓝边 = median”的示意项（不区分方法）
        median_handle = Line2D([0], [0], marker="o", linestyle="None",
                               markerfacecolor="white", markeredgecolor="blue", markeredgewidth=1.6,
                               markersize=10, label="Median (blue edge)")
        legend_handles = method_handles + [median_handle]
        leg1 = ax.legend(handles=legend_handles, title="Method", frameon=False, loc="upper right")

        # 尺寸“colorbar”式连续条（替代 size legend）
        size_to_s = _size_scaler(sizes, s_min=40, s_max=160)
        _draw_sizebar(ax, sizes, size_to_s)  # ← 新增：右侧连续尺寸条
        ax.add_artist(leg1)  # 保留方法图例


    # 叠加“区域 + 代表点”
    for mi, name in enumerate(methods):
        P = points[name]
        if P.shape[0] == 0:
            continue

        # 代表点
        if point_kind == "geom_median":
            center = _geometric_median(P)
        elif point_kind == "median":
            center = np.median(P, axis=0)
        else:
            center = P.mean(axis=0)

        # 区域
        color = method_color[name]
        lw = 2.4 if (emphasize_method and name == emphasize_method) else 1.6
        alpha_fill = 0.18 if (emphasize_method and name == emphasize_method) else 0.12

        if region_kind in ("ellipse", "both"):
            if P.shape[0] >= 3:
                cov = np.cov(P.T)  # 2x2
                width, height, angle = _ellipse_from_cov(center, cov, q=ellipse_q)
                e = Ellipse(xy=(center[0], center[1]),
                            width=width, height=height, angle=angle,
                            facecolor=color, edgecolor=color, alpha=alpha_fill, lw=lw)
                ax.add_patch(e)

        if region_kind in ("hull", "both"):
            if P.shape[0] >= 3:
                hull = _convex_hull(P)
                poly = Polygon(hull, closed=True, facecolor=color, edgecolor=color,
                               alpha=alpha_fill*0.9, lw=lw)
                ax.add_patch(poly)

        # 代表点：用同方法色的大号 marker 标出（几何中位数）
        ax.scatter([center[0]], [center[1]], s=400 if name==emphasize_method else 200,
                   marker=markers[mi % len(markers)], color=color,
                   edgecolors="blue", linewidths=1.6, zorder=5)

    ax.set_xlabel("Final Cache Size (#Circuits)", fontsize=20)
    ax.set_ylabel("Compilation Latency (s)", fontsize=20)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_path = PLOT_DIR / out_png
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"[save] {out_path}")
    print("[load] Redraw with regions from:", json_path)


if __name__ == "__main__":
    # 示例：从默认目录加载并重绘
    load_and_replot_regions(
        load_dir=str((LOAD_ROOT/"scaling").resolve()),
        out_png="scaling_latency_vs_cache_scatter_regions.png",
        show_scatter=True,
        region_kind="ellipse",          # "ellipse" | "hull" | "both"
        ellipse_q=0.90,                 # 椭圆覆盖概率（常用 0.90/0.95）
        emphasize_method="FS+Pre+ttl+SE+ema",  # 高亮你的方法
        point_kind="geom_median",        # 代表点：几何中位数
        methods_keep = ["FS+Pre+ttl+SE+ema", "FS+Pre", "FS", "PR"]
    )
