# v18_runner_workload_scaling_multi.py
# Multi-round scaling runner: run each workload size for R rounds (different seeds),
# aggregate stats, plot mean+interval lines, scatter all rounds, and save per-round timelines.
import argparse, json, csv, math, time
from typing import Dict, Any, List, Tuple, Optional, DefaultDict
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path
from dataclasses import asdict, dataclass

# ---- core helpers & paths (keep aligned with your v18_core) ----
from v18_core import (
    ROOT, PLOT_DIR, build_catalog,
    clear_recent, LOAD_ROOT,
    build_workload_poisson_superposition_exact,  # 总过程：Exp(Λ) + 类别 Multinomial
    draw_timeline_multi,                         # 时间线绘图（我们每轮都落盘）
    plot_cache_size_change,                      # cache 随时间变化（我们每轮都落盘）
    compute_freq_and_hits                        # 计算命中率
)

# ---- strategies (与现有 runner 保持一致) ----
import v18_strat_FS as S_FS                     # FirstSeen
import v18_strat_FS_Pre as S_FS_Pre             # FirstSeen + predictor prewarm
import v18_strat_PR as S_PR
import v18_strat_FS_Pre_ttl as S_FS_Pre_ttl

# ---------------- JSON utilities (safe for numpy) ----------------
def _json_default(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)):    return bool(o)
    return str(o)

def json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)

def slim_workload_for_dump(workload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """仅保存 name/q/d/t_arr（重命名为 ts），去掉 maker_run 等不可序列化字段。"""
    slim = []
    for it in workload:
        slim.append({
            "name": it.get("name"),
            "q": int(it.get("q")) if it.get("q") is not None else None,
            "d": int(it.get("d")) if it.get("d") is not None else None,
            "ts": float(it.get("t_arr", it.get("ts", 0.0))),
        })
    return slim

# ---------------- basic metrics helpers ----------------
def e2e_latency_from_events(events: List[Dict[str, Any]]) -> float:
    """End-to-end latency = max(start+dur)"""
    if not events: return 0.0
    return max(float(e.get("start", 0.0)) + float(e.get("dur", 0.0)) for e in events)

def final_cache_size_from_metrics(metrics: Dict[str, Any]) -> int:
    """last cache size from metrics['cache_size_series']."""
    series = metrics.get("cache_size_series") or []
    if not series: return 0
    last = series[-1]
    if isinstance(last, dict): return int(last.get("size", 0))
    if isinstance(last, (list, tuple)) and len(last) >= 2: return int(last[1])
    return 0

def final_hitrate_from_metrics(workload, metrics: Dict[str, Any]) -> float:
    hit_by_label = metrics.get("hit_by_label", {})
    _, _, overall = compute_freq_and_hits(workload, hit_by_label)
    return overall

@dataclass
class Row:
    size: int
    round: int
    method: str
    e2e_latency: float
    final_cache_size: int
    final_hitrate: float

# ---------------- plotting ----------------
def _set_plot_style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})

def _mean_std_minmax(xs: List[float]) -> Tuple[float, float, float, float]:
    if not xs: return 0.0, 0.0, 0.0, 0.0
    arr = np.array(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), float(arr.min()), float(arr.max())

def plot_with_interval(methods: List[str],
                       sizes: List[int],
                       agg: Dict[str, Dict[str, List[float]]],
                       ylabel: str,
                       out_path: Path,
                       interval_kind: str = "std"):
    """
    画“均值曲线 + 区间阴影”：
      - agg[name] = {"mean":[...], "std":[...], "min":[...], "max":[...]}
      - interval_kind: "std" 用 mean±std；"minmax" 用 [min,max]
    """
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for idx, name in enumerate(methods):
        stats = agg[name]
        mean = np.array(stats["mean"], dtype=float)
        std  = np.array(stats["std"], dtype=float)
        lo   = mean - std if interval_kind == "std" else np.array(stats["min"], dtype=float)
        hi   = mean + std if interval_kind == "std" else np.array(stats["max"], dtype=float)

        ax.plot(sizes, mean, marker=markers[idx % len(markers)], label=name,
                markerfacecolor='none', markersize=8, markeredgewidth=2.0,
                color=colors[idx % len(colors)])
        ax.fill_between(sizes, lo, hi, color=colors[idx % len(colors)], alpha=0.18, linewidth=0)

    ax.set_xlabel("Workload size (requests)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"[save] {out_path}")

def plot_latency_vs_cache_scatter_multi(methods: List[str],
                                        sizes: List[int],
                                        lat_all: Dict[str, List[List[float]]],
                                        csz_all: Dict[str, List[List[float]]],
                                        out_path: Path,
                                        cmap_name: str = "plasma",
                                        jitter: float = 0.0):
    """
    多轮散点：
      - 对每个方法、每个 size，把“多轮”的 (cache, latency) 全画出来。
      - 颜色编码 workload size；同一 size 的点颜色相同。
      - 可选 jitter (0~0.02) 轻微扰动避免重合。
    """
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 6))
    markers = ["o", "s", "^", "P", "*", "X", "D", "v"]
    cmap = plt.colormaps[cmap_name].reversed()
    norm = Normalize(vmin=min(sizes), vmax=max(sizes))

    for mi, name in enumerate(methods):
        per_size_lat = lat_all[name]  # List over sizes: [ [lat_r1, lat_r2, ...], ... ]
        per_size_csz = csz_all[name]
        for si, N in enumerate(sizes):
            lats = per_size_lat[si] if si < len(per_size_lat) else []
            cszs = per_size_csz[si] if si < len(per_size_csz) else []
            if not lats or not cszs: continue
            k = min(len(lats), len(cszs))
            xs = np.array(cszs[:k], dtype=float)
            ys = np.array(lats[:k], dtype=float)
            # jitter
            if jitter > 0:
                jx = (np.random.rand(k) - 0.5) * jitter * max(1.0, xs.max() - xs.min() + 1.0)
                jy = (np.random.rand(k) - 0.5) * jitter * max(1.0, ys.max() - ys.min() + 1.0)
                xs = xs + jx; ys = ys + jy
            ax.scatter(xs, ys, s=70,
                       marker=markers[mi % len(markers)],
                       facecolors=cmap(norm(N)),
                       edgecolors="black", linewidths=0.6, alpha=0.95,
                       label=name if si == 0 else None)  # 每个方法只加一次图例

    # 形状图例：仅编码方法
    legend_handles = [
        Line2D([0],[0], marker=markers[i % len(markers)], linestyle="None",
               markerfacecolor="white", markeredgecolor="black", markersize=8, label=methods[i])
        for i in range(len(methods))
    ]
    ax.legend(handles=legend_handles, title="Method", frameon=False, loc="best")

    # 颜色条：编码 workload size
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.012)
    cbar.set_label("Workload size (requests)")

    ax.set_xlabel("Final cache size (#circuits)")
    ax.set_ylabel("E2E latency (s)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"[save] {out_path}")

# ----------- optional: workload distribution quick visualization & save -----------
def save_workload_distribution(workload: List[Dict[str, Any]],
                               info: Dict[str, Any],
                               meta: List[Dict[str, Any]],
                               out_path: Path,
                               bins: int = 40, topk_classes: int = 10):
    """
    组合图（3合1）：间隔直方图、计数过程、按类计数（top-k）。
    注：简要版，仅依赖 workload t_arr，便于每轮落盘；不调用交互 show()。
    """
    ts = np.array([it.get("t_arr", 0.0) for it in workload], dtype=float)
    ts.sort()
    if ts.size == 0:
        return
    inter = np.diff(ts)
    H = float(ts[-1]) if ts.size > 0 else 1.0
    N = len(ts)

    # 类别计数（以 name|q|d 聚合）
    labels = [f'{it["name"]}|q{it["q"]}|d{it["d"]}' for it in workload]
    uniq, counts = np.unique(labels, return_counts=True)
    idx_top = np.argsort(-counts)[:min(topk_classes, len(counts))]

    _set_plot_style()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) inter-arrival histogram
    ax = axs[0]
    if inter.size > 0:
        ax.hist(inter, bins=bins, density=True, alpha=0.7)
        Λ = float(info.get("Lambda", 1.0))
        x = np.linspace(0, max(inter.max(), 1e-6), 200)
        ax.plot(x, Λ * np.exp(-Λ * x), "r--", lw=2)
    ax.set_title("Δt histogram")
    ax.set_xlabel("Δt"); ax.set_ylabel("density")

    # (2) counting process
    ax = axs[1]
    ax.step(ts, np.arange(1, N+1), where="post", label="N(t)")
    tline = np.linspace(0, ts[-1], 200)
    Λ = float(info.get("Lambda", 1.0))
    ax.plot(tline, Λ * tline, "r--", lw=2, label="Λ t")
    ax.set_title("Counting process"); ax.legend(frameon=False)
    ax.set_xlabel("t"); ax.set_ylabel("N(t)")

    # (3) per-class counts (top-k)
    ax = axs[2]
    x = np.arange(len(idx_top))
    ax.bar(x, counts[idx_top])
    xt = [uniq[i] for i in idx_top]
    ax.set_xticks(x); ax.set_xticklabels(xt, rotation=45, ha="right")
    ax.set_title("Per-class counts (top-k)")
    ax.set_ylabel("#events")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    print(f"[save] {out_path}")

# ---------------- run-mode core ----------------
def main_run(args):
    sizes = [int(x) for x in args.sizes.split(",") if x]
    q_list = [int(x) for x in args.q_list.split(",") if x]
    d_list = [int(x) for x in args.d_list.split(",") if x]
    R = int(args.rounds)
    interval_kind = args.interval_kind.lower().strip()

    # ensure save directory structure
    SAVE_DIR = Path(args.save_dir)
    (SAVE_DIR/"workloads").mkdir(parents=True, exist_ok=True)
    (SAVE_DIR/"summaries").mkdir(parents=True, exist_ok=True)

    # catalog independent of N
    makers_all, meta = build_catalog(q_list, d_list)

    predictor_cfg = {"sliding_window_sec": args.sliding_window_sec,
                     "min_samples": args.min_samples}

    # strategies to compare
    def _common_kwargs(workload):
        return dict(
            workload=workload, makers_all=makers_all,
            predictor_cfg=predictor_cfg, prewarm_every=args.prewarm_every,
            lookahead_sec=args.lookahead, prob_th=args.prob_th,
            max_compile=args.max_compile, shots=args.shots,
            include_exec=False,   # compile-only 为主；如需 E2E 改成 True
        )
    def _baseline_kwargs(workload):
        return dict(workload=workload, shots=args.shots, include_exec=False)

    STRATS = [
        ("FS+Pre+ttl", S_FS_Pre_ttl.run_strategy, _common_kwargs),
        ("FS+Pre", S_FS_Pre.run_strategy, _common_kwargs),
        ("FS",     S_FS.run_strategy,     _baseline_kwargs),
        ("PR",     S_PR.run_strategy,     _baseline_kwargs),
    ]

    methods = [name for (name, _, _) in STRATS]

    # storage for aggregation
    # 每个方法、每个 size，存多轮的原始值列表
    lat_all: Dict[str, List[List[float]]] = {m: [[] for _ in sizes] for m in methods}
    csz_all: Dict[str, List[List[float]]] = {m: [[] for _ in sizes] for m in methods}
    hit_all: Dict[str, List[List[float]]] = {m: [[] for _ in sizes] for m in methods}

    rows: List[Row] = []  # long-form rows

    # iterate over workload sizes and rounds
    for si, N in enumerate(sizes):
        print(f"\n========== Workload size N={N} ==========")
        for r in range(1, R+1):
            rng_seed = args.rng_seed_base + si * 1009 + r * 17  # 不同 size/round 有不同 seed（可复现）
            workload, info = build_workload_poisson_superposition_exact(
                meta, N, args.hot_fraction, args.hot_boost, args.rps, rng_seed, return_timestamps=True
            )

            # persist workload (slim)
            wl_path = SAVE_DIR/"workloads"/f"workload_N{N}_seed{rng_seed}_r{r}.json"
            wl_path.write_text(json_dump(slim_workload_for_dump(workload)))

            print(f"[run] N={N}, round={r}, seed={rng_seed}  (saved {wl_path.name})")
            perN_dir = SAVE_DIR/f"N{N}"/f"round_{r}"
            (perN_dir/"events").mkdir(parents=True, exist_ok=True)
            (perN_dir/"metrics").mkdir(parents=True, exist_ok=True)

            # ---- 运行各策略 ----
            events_series: Dict[str, List[Dict[str, Any]]] = {}
            metrics_series: Dict[str, Dict[str, Any]] = {}

            for name, fn, kw_builder in STRATS:
                clear_recent()  # reset predictor stats per run for fairness
                kwargs = kw_builder(workload)
                out = fn(**kwargs)
                events = out["events"]
                metrics = out.get("metrics", {})

                # save events & metrics
                (perN_dir/"events"/f"{name}.json").write_text(json_dump(events))
                (perN_dir/"metrics"/f"{name}.json").write_text(json_dump(metrics))
                events_series[name]  = events
                metrics_series[name] = metrics

                # collect metrics
                e2e = e2e_latency_from_events(events)
                csz = final_cache_size_from_metrics(metrics)
                hits = final_hitrate_from_metrics(workload, metrics)

                lat_all[name][si].append(e2e)
                csz_all[name][si].append(csz)
                hit_all[name][si].append(hits)

                rows.append(Row(size=N, round=r, method=name,
                                e2e_latency=e2e, final_cache_size=csz, final_hitrate=hits))
                print(f"{name:>16s} | e2e={e2e:8.3f}s | cache={csz:4d} | hit={hits:6.2f}")

            # ---- 每轮 timeline & cache-size 曲线 ----
            out_tl = PLOT_DIR / f"timeline_wl_{N}_r{r}.png"
            draw_timeline_multi(events_series, out_tl)  # 该函数内部会保存

            cache_size_changes = {name: metrics_series[name].get("cache_size_series", [])
                                  for (name, _, _) in STRATS}
            out_sz = PLOT_DIR / f"cache_change_wl_{N}_r{r}.png"
            plot_cache_size_change(cache_size_changes, out_sz)

            # ---- 每轮 workload 分布图 ----
            out_wl = PLOT_DIR / f"wl_dist_N{N}_r{r}.png"
            save_workload_distribution(workload, info, meta, out_wl, bins=40, topk_classes=12)

    # ---------------- 统计聚合 & 绘图 ----------------
    # 把每个方法/每个 size 的多轮值聚成 mean/std/min/max
    agg_lat: Dict[str, Dict[str, List[float]]] = {m: {"mean": [], "std": [], "min": [], "max": []} for m in methods}
    agg_csz: Dict[str, Dict[str, List[float]]] = {m: {"mean": [], "std": [], "min": [], "max": []} for m in methods}
    agg_hit: Dict[str, Dict[str, List[float]]] = {m: {"mean": [], "std": [], "min": [], "max": []} for m in methods}

    for m in methods:
        for si, N in enumerate(sizes):
            mu, sd, mn, mx = _mean_std_minmax(lat_all[m][si])
            agg_lat[m]["mean"].append(mu); agg_lat[m]["std"].append(sd); agg_lat[m]["min"].append(mn); agg_lat[m]["max"].append(mx)
            mu, sd, mn, mx = _mean_std_minmax(csz_all[m][si])
            agg_csz[m]["mean"].append(mu); agg_csz[m]["std"].append(sd); agg_csz[m]["min"].append(mn); agg_csz[m]["max"].append(mx)
            mu, sd, mn, mx = _mean_std_minmax(hit_all[m][si])
            agg_hit[m]["mean"].append(mu); agg_hit[m]["std"].append(sd); agg_hit[m]["min"].append(mn); agg_hit[m]["max"].append(mx)

    # 路径
    out_latency = PLOT_DIR / args.out_latency          # e2e (mean+interval)
    out_cache   = PLOT_DIR / args.out_cache            # final cache size (mean+interval)
    out_hitrate = PLOT_DIR / args.out_hitrate          # final hitrate (mean+interval)
    out_scatter = PLOT_DIR / args.out_scatter          # scatter: all rounds

    # 线图（带区间）
    plot_with_interval(methods, sizes, agg_lat, "E2E latency (s)", out_latency, interval_kind=interval_kind)
    plot_with_interval(methods, sizes, agg_csz, "Final cache size (#circuits)", out_cache, interval_kind=interval_kind)
    plot_with_interval(methods, sizes, agg_hit, "Final hitrate (%)", out_hitrate, interval_kind=interval_kind)

    # 散点：按回合把所有点都画上
    plot_latency_vs_cache_scatter_multi(methods, sizes, lat_all, csz_all, out_scatter,
                                        cmap_name="plasma", jitter=args.jitter)

    # ---------------- persist summary ----------------
    SAVE_DIR = Path(args.save_dir)
    (SAVE_DIR/"summaries").mkdir(parents=True, exist_ok=True)

    # 1) long-form CSV（每行=一个(size, round, method)）
    csv_path = SAVE_DIR/"summaries"/"scaling_multi_rounds_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["size", "round", "method", "e2e_latency", "final_cache_size", "final_hitrate"])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"[save] {csv_path}")

    # 2) JSON summary（聚合 + 原始）
    summary = {
        "sizes": sizes,
        "rounds": R,
        "methods": methods,
        "interval_kind": interval_kind,
        "aggregate": {
            "e2e_latency": agg_lat,
            "final_cache_size": agg_csz,
            "final_hitrate": agg_hit,
        },
        "raw": {
            "e2e_latency": {m: lat_all[m] for m in methods},        # shape: [size_idx][round_vals...]
            "final_cache_size": {m: csz_all[m] for m in methods},
            "final_hitrate": {m: hit_all[m] for m in methods},
        },
        "config": {
            "q_list": q_list, "d_list": d_list, "shots": args.shots,
            "hot_fraction": args.hot_fraction, "hot_boost": args.hot_boost,
            "rps": args.rps, "rng_seed_base": args.rng_seed_base,
            "lookahead": args.lookahead, "prob_th": args.prob_th,
            "max_compile": args.max_compile, "sliding_window_sec": args.sliding_window_sec,
            "min_samples": args.min_samples, "prewarm_every": args.prewarm_every,
        },
        "plots": {
            "latency_png": str(out_latency),
            "cache_png":   str(out_cache),
            "hitrate_png": str(out_hitrate),
            "scatter_png": str(out_scatter),
        },
    }
    json_path = SAVE_DIR/"summaries"/"scaling_multi_rounds_summary.json"
    json_path.write_text(json_dump(summary))
    print(f"[save] {json_path}")

    # console pretty
    print("-"*92)
    print("E2E latency (mean ± std) by size:")
    hdr = ["N"] + methods
    print(" | ".join(f"{h:>20s}" for h in hdr))
    for si, N in enumerate(sizes):
        cells = []
        for m in methods:
            mu, sd = agg_lat[m]["mean"][si], agg_lat[m]["std"][si]
            cells.append(f"{mu:7.3f} ± {sd:6.3f}")
        print(" | ".join([f"{N:>20d}"] + [f"{c:>20s}" for c in cells]))

# ---------------- load-mode: read summary json and redraw ----------------
def load_and_redraw(load_dir: str,
                    out_latency: str = "scaling_e2e_latency.png",
                    out_cache: str   = "scaling_final_cache_size.png",
                    out_hitrate: str = "scaling_final_hitrate.png",
                    out_scatter: str = "scaling_latency_vs_cache_scatter.png"):
    """
    从 SAVE_DIR/'summaries'/'scaling_multi_rounds_summary.json' 载入并重绘。
    """
    LOAD_DIR = Path(load_dir)
    json_path = LOAD_DIR / "summaries" / "scaling_multi_rounds_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Summary not found: {json_path}")

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    sizes: List[int] = summary["sizes"]
    methods: List[str] = summary["methods"]
    interval_kind = summary.get("interval_kind", "std")
    agg_lat = summary["aggregate"]["e2e_latency"]
    agg_csz = summary["aggregate"]["final_cache_size"]
    agg_hit = summary["aggregate"]["final_hitrate"]
    raw_lat = summary["raw"]["e2e_latency"]
    raw_csz = summary["raw"]["final_cache_size"]

    out_latency_path = PLOT_DIR / out_latency
    out_cache_path   = PLOT_DIR / out_cache
    out_hitrate_path = PLOT_DIR / out_hitrate
    out_scatter_path = PLOT_DIR / out_scatter

    plot_with_interval(methods, sizes, agg_lat, "E2E latency (s)", out_latency_path, interval_kind=interval_kind)
    plot_with_interval(methods, sizes, agg_csz, "Final cache size (#circuits)", out_cache_path, interval_kind=interval_kind)
    plot_with_interval(methods, sizes, agg_hit, "Final hitrate (%)", out_hitrate_path, interval_kind=interval_kind)
    plot_latency_vs_cache_scatter_multi(methods, sizes, raw_lat, raw_csz, out_scatter_path, cmap_name="plasma", jitter=0.0)

    print("-"*80)
    print("[load] Redraw finished from:", json_path)

# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["run", "load"], default="run",
                    help="run: 运行仿真并绘图；load: 从 summary 重绘")

    # workload shape
    ap.add_argument("--sizes", type=str, default="50,100,150,200,300,350,400,450,500",
                    help="Comma-separated workload lengths to test.")
    ap.add_argument("--q_list", type=str, default="5,7,11,13,15,17")
    ap.add_argument("--d_list", type=str, default="2,4,6")
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--hot_fraction", type=float, default=0.25)
    ap.add_argument("--hot_boost", type=float, default=8.0)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--rng_seed_base", type=int, default=123)

    # predictor / prewarm (for FS family)
    ap.add_argument("--lookahead", type=float, default=8.0)
    ap.add_argument("--prob_th", type=float, default=0.45)
    ap.add_argument("--max_compile", type=int, default=3)
    ap.add_argument("--sliding_window_sec", type=float, default=60.0)
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--prewarm_every", type=int, default=5)

    # multi-round
    ap.add_argument("--rounds", type=int, default=3, help="每个 workload 大小的重复轮数")
    ap.add_argument("--interval_kind", type=str, choices=["std", "minmax"], default="std",
                    help="线图区间：std=均值±1σ；minmax=[最小, 最大]")
    ap.add_argument("--jitter", type=float, default=0.01,
                    help="散点轻微抖动系数（0~0.05），避免多轮点完全重合")

    # IO
    ap.add_argument("--out_latency", type=str, default="scaling_e2e_latency_mean_interval.png")
    ap.add_argument("--out_cache",   type=str, default="scaling_final_cache_size_mean_interval.png")
    ap.add_argument("--out_hitrate", type=str, default="scaling_final_hitrate_mean_interval.png")
    ap.add_argument("--out_scatter", type=str, default="scaling_latency_vs_cache_scatter_all_rounds.png")
    ap.add_argument("--load_dir", type=str, default=str((LOAD_ROOT/"scaling").resolve()))
    ap.add_argument("--save_dir", type=str, default=str((ROOT / "scaling").resolve()))
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()
    if args.mode == "run":
        main_run(args)
    else:
        load_and_redraw(load_dir=args.load_dir,
                        out_latency=args.out_latency,
                        out_cache=args.out_cache,
                        out_hitrate=args.out_hitrate,
                        out_scatter=args.out_scatter)

if __name__ == "__main__":
    main()
