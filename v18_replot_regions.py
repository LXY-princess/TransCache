# v18_replot_regions.py
import json, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.patches import Ellipse, Polygon

# 依赖你的工程路径变量
from v18_core import PLOT_DIR, LOAD_ROOT  # 与原 runner 保持一致
# 直接读取你原 summary 的结构（由 v18_runner_wl_rounds.py 写出）
# 参考：load_and_redraw / plot_latency_vs_cache_scatter_multi 的用法与数据结构。:contentReference[oaicite:1]{index=1}


# ---------- 基础小工具 ----------
def _set_plot_style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})

def _flatten_points(raw_lat: Dict[str, List[List[float]]],
                    raw_csz: Dict[str, List[List[float]]],
                    methods: List[str]) -> Dict[str, np.ndarray]:
    """
    把某方法在所有 workload size & round 的点拉平成 Nx2 的数组：[[cache, latency], ...]
    """
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
    """
    Weiszfeld 算法，求几何中位数（稳健代表点）；P 为 Nx2。
    """
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
    """
    从均值/协方差得到椭圆参数：width,height,angle(deg)
    椭圆对应 (x-μ)^T Σ^{-1} (x-μ) = chi2(df=2, q)
    """
    # 常见分位数（df=2）的近似值，避免 SciPy 依赖
    chi2_map = {0.50: 1.386, 0.68: 2.279, 0.80: 3.219, 0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
    # 找最接近的 key
    q_key = min(chi2_map.keys(), key=lambda v: abs(v - q))
    s = chi2_map[q_key]

    # 特征分解：Σ = V diag(λ) V^T
    evals, evecs = np.linalg.eigh(cov)  # 升序
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]

    # 主轴长度：sqrt(s * λ_i)；宽高为 2*轴长
    width, height = 2.0 * math.sqrt(max(evals[0], 1e-12) * s), 2.0 * math.sqrt(max(evals[1], 1e-12) * s)
    # 椭圆旋转角：主特征向量与 x 轴的夹角
    angle = math.degrees(math.atan2(evecs[1, 0], evecs[0, 0]))
    return width, height, angle

def _convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Andrew 单调链凸包，返回顶点顺序（闭合不重复最后点）。
    输入 Nx2（x=cache, y=lat）.
    """
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
    """
    从 summary JSON 载入所有点，叠加“每方法的区域 + 代表点”，并保存图片。
    """
    LOAD_DIR = Path(load_dir)
    json_path = LOAD_DIR / "summaries" / "scaling_multi_rounds_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Summary not found: {json_path}")

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    sizes: List[int] = summary["sizes"]
    # methods: List[str] = summary["methods"]
    # raw_lat = summary["raw"]["e2e_latency"]
    # raw_csz = summary["raw"]["final_cache_size"]
    methods_all: List[str] = summary["methods"]
    interval_kind = summary.get("interval_kind", "std")
    agg_lat_full = summary["aggregate"]["e2e_latency"]
    agg_csz_full = summary["aggregate"]["final_cache_size"]
    agg_hit_full = summary["aggregate"]["final_hitrate"]
    raw_lat_full = summary["raw"]["e2e_latency"]
    raw_csz_full = summary["raw"]["final_cache_size"]

    # --- 选择要画的 methods（保持你给定的顺序；大小写精确匹配） ---
    if methods_keep:
        # 只保留 summary 里确实存在的方法；其余静默忽略
        methods = [m for m in methods_keep if m in methods_all]
    else:
        methods = methods_all

    # --- 同步裁剪 aggregate/raw 字典 ---
    agg_lat = {m: agg_lat_full[m] for m in methods}
    agg_csz = {m: agg_csz_full[m] for m in methods}
    agg_hit = {m: agg_hit_full[m] for m in methods}
    raw_lat = {m: raw_lat_full[m] for m in methods}
    raw_csz = {m: raw_csz_full[m] for m in methods}

    # 收集每方法所有点
    points = _flatten_points(raw_lat, raw_csz, methods)

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9.0, 6.4))

    markers = ["*", "o", "^", "H", "D", "v", "P", "s", "X", "v"]
    # 背景散点（可选）：颜色编码 size
    if show_scatter:
        cmap = plt.colormaps["plasma"].reversed()
        norm = Normalize(vmin=min(sizes), vmax=max(sizes))
        for mi, name in enumerate(methods):
            per_size_lat = raw_lat[name]
            per_size_csz = raw_csz[name]
            for si, N in enumerate(sizes):
                lats = per_size_lat[si] if si < len(per_size_lat) else []
                cszs = per_size_csz[si] if si < len(per_size_csz) else []
                k = min(len(lats), len(cszs))
                if k <= 0: continue
                xs = np.array(cszs[:k], float); ys = np.array(lats[:k], float)
                ax.scatter(xs, ys, s=60,
                           marker=markers[mi % len(markers)],
                           facecolors=cmap(norm(N)),
                           edgecolors="black", linewidths=0.6, alpha=0.75,
                           label=name if si == 0 else None)

        # 图例（只按方法区分 marker）
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0],[0], marker=markers[i % len(markers)], linestyle="None",
                   markerfacecolor="white", markeredgecolor="black", markersize=8, label=methods[i])
            for i in range(len(methods))
        ]
        ax.legend(handles=legend_handles, title="Method", frameon=False, loc="best")

        # 颜色条：编码 workload size
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.012); cbar.set_label("Workload size (requests)")

    # 方法色盘（固定颜色，便于区域/代表点着色）
    # palette = plt.rcParams['axes.prop_cycle'].by_key().get('color',
    #            ['#fcdd28', '#55A868','#4C72B0','#C44E52','#8172B3','#CCB974','#64B5CD'])
    palette = ['#fcdd28', '#55A868', '#4C72B0', '#C44E52', '#8172B3', '#CCB974', '#64B5CD']

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
        color = palette[mi % len(palette)]
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

        # 代表点：用大号 marker 标出（几何中位数）
        if mi == 0:
            ax.scatter([center[0]], [center[1]], s=300,
                       marker=markers[mi % len(markers)], color=color, edgecolors="black", linewidths=1.2, zorder=5)
        else:
            ax.scatter([center[0]], [center[1]], s=160,
                       marker=markers[mi % len(markers)], color=color, edgecolors="black", linewidths=1.2, zorder=5)

    ax.set_xlabel("Final cache size (#circuits)")
    ax.set_ylabel("E2E latency (s)")
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
