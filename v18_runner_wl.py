# v15_runner_workload_scaling.py
# Compare E2E latency & final cache size across workload sizes for 4 strategies.
import argparse, json, csv
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import asdict, dataclass
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# core helpers & paths
from v18_core import (
    ROOT, PLOT_DIR, build_catalog,
    clear_recent, LOAD_ROOT,
    build_workload_poisson_superposition_exact,
    visualize_superposed_poisson_exact,
    draw_timeline_multi,
    plot_cache_size_change,
    compute_freq_and_hits
)

# strategies
# import v18_strategy_baseline as S0          # FullCompilation (Baseline)
import v18_strat_FS as S_FS  # FirstSeen
import v18_strat_FS_Pre as S_FS_Pre  # FirstSeen + predictor prewarm
import v18_strat_PR as S_PR
import v18_strat_FS_Pre_ttl as S_FS_Pre_ttl
# import v18_strategy_tcache_optimize_score_log as S6S       # TransCache(Proposed)
# import v18_strategy_param_reuse as SPR      # ParamReuse (Braket-like)
import v18_strat_FS_Pre as SA

# ---------------- JSON utilities (safe for numpy & callables) ----------------
def _json_default(o):
    # 安全处理 numpy 标量 / bool
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    # 其他未知类型（包括不可序列化对象）退化为字符串
    return str(o)

def json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)

def slim_workload_for_dump(workload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    把 workload 瘦身为可 JSON 序列化的列表：
    仅保存 name/q/d/ts；去掉 maker_run（callable）等不可序列化字段。
    """
    slim = []
    for it in workload:
        slim.append({
            "name": it.get("name"),
            "q": int(it.get("q")) if it.get("q") is not None else None,
            "d": int(it.get("d")) if it.get("d") is not None else None,
            "ts": float(it.get("ts", 0.0)),
        })
    return slim

# ---------------- helpers ----------------
def e2e_latency_from_events(events: List[Dict[str, Any]]) -> float:
    """Compute end-to-end latency (seconds) from events timeline."""
    if not events:
        return 0.0
    return max(float(e.get("start", 0.0)) + float(e.get("dur", 0.0)) for e in events)

def final_cache_size_from_metrics(metrics: Dict[str, Any]) -> int:
    """
    From metrics['cache_size_series'] pick the last size.
    Supports both [{'t':..., 'size':...}, ...] and [(t, size), ...] shapes.
    Baseline (no cache) will return 0.
    """
    series = metrics.get("cache_size_series") or []
    if not series:
        return 0
    last = series[-1]
    if isinstance(last, dict):
        return int(last.get("size", 0))
    if isinstance(last, (list, tuple)) and len(last) >= 2:
        return int(last[1])
    return 0

def final_hitrate_from_metrics(workload, metrics: Dict[str, Any]):
    hit_by_label = metrics.get("hit_by_label", {})
    _, _, overall = compute_freq_and_hits(workload, hit_by_label)
    return overall

@dataclass
class Row:
    size: int
    method: str
    e2e_latency: float
    final_cache_size: int

# ---------------- plotting (shared by run & load) ----------------
def _set_plot_style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})

def plot_e2e_latency(methods: List[str],
                     sizes: List[int],
                     method2lat: Dict[str, List[float]],
                     out_path: Path) -> None:
    """
    绘制 E2E latency vs workload size 并保存到 out_path。
    methods: 方法名称顺序
    sizes:   x 轴工作负载大小（所有方法共用）
    method2lat: 每个方法对应的 y 序列
    """
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, name in enumerate(methods):
        ys = method2lat.get(name, [])
        ax.plot(sizes, ys, marker=markers[idx % len(markers)], label=name,
                markerfacecolor='none',  # 空心
                # markeredgecolor=color,  # 用线条颜色做描边
                markersize=8,  # 缩小
                markeredgewidth=2.0,)
    ax.set_xlabel("Workload size (requests)")
    ax.set_ylabel("E2E latency (s)")
    # ax.set_title("E2E latency vs. workload size")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"[save] {out_path}")

def plot_final_cache_size(methods: List[str],
                          sizes: List[int],
                          method2csz: Dict[str, List[int]],
                          out_path: Path) -> None:
    """
    绘制 Final cache size vs workload size 并保存到 out_path。
    """
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, name in enumerate(methods):
        ys = method2csz.get(name, [])
        ax.plot(sizes, ys, marker=markers[idx % len(markers)], label=name,
                markerfacecolor='none',  # 空心
                # markeredgecolor=color,  # 用线条颜色做描边
                markersize=8,  # 缩小
                markeredgewidth=2.0,)
    ax.set_xlabel("Workload size (requests)")
    ax.set_ylabel("Final cache size (#circuits)")
    # ax.set_title("Final cache size vs. workload size")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"[save] {out_path}")

def plot_final_hitrate(methods: List[str],
                          sizes: List[int],
                          method2hits: Dict[str, List[int]],
                          out_path: Path) -> None:
    """
    绘制 Final cache size vs workload size 并保存到 out_path。
    """
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, name in enumerate(methods):
        ys = method2hits.get(name, [])
        ax.plot(sizes, ys, marker=markers[idx % len(markers)], label=name,
                markerfacecolor='none',  # 空心
                # markeredgecolor=color,  # 用线条颜色做描边
                markersize=8,  # 缩小
                markeredgewidth=2.0,)
    ax.set_xlabel("Workload size (requests)")
    ax.set_ylabel("Final hitrate (%)")
    # ax.set_title("Final cache size vs. workload size")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    print(f"[save] {out_path}")

# ---------- NEW: Scatter plot of latency vs cache size (color=workload, marker=method) ----------
def plot_latency_vs_cache_scatter(methods: List[str],
                                  sizes: List[int],
                                  method2lat: Dict[str, List[float]],
                                  method2csz: Dict[str, List[int]],
                                  out_path: Path,
                                  cmap_name: str = "plasma") -> None:
    """
    从 summary 中的 {sizes, methods, e2e_latency, final_cache_size} 生成一张散点图：
      - x: final cache size
      - y: E2E latency
      - marker 形状：方法(method)
      - 颜色深浅：workload size（越深越大）
      - 侧边 colorbar 表示 workload 数值
    """
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # 不同方法使用不同 marker；保证方法数>markers数时循环使用
    markers = ["o", "s", "^", "P", "*", "X", "D", "v",]
    # cmap = plt.cm.get_cmap(cmap_name)
    cmap = plt.colormaps[cmap_name].reversed()
    if len(sizes) == 0:
        raise ValueError("sizes is empty; nothing to plot.")
    norm = Normalize(vmin=min(sizes), vmax=max(sizes))

    # 逐方法绘制散点；每个方法的一组点在不同 workload 下有不同颜色
    for idx, name in enumerate(methods):
        ys = method2lat.get(name, [])
        xs = method2csz.get(name, [])
        k = min(len(xs), len(ys), len(sizes))
        if k == 0:
            continue
        nvals = np.array(sizes[:k])
        colors = cmap(norm(nvals))
        ax.scatter(
            np.array(xs[:k]),
            np.array(ys[:k]),
            s=70,
            marker=markers[idx % len(markers)],
            facecolors=colors,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            label=name,  # 用于构造 legend（我们会自定义以只体现形状）
        )

    # 形状图例：仅编码“方法”，不受颜色影响，便于阅读
    legend_handles = [
        Line2D([0], [0],
               marker=markers[i % len(markers)],
               linestyle="None",
               markerfacecolor="white",
               markeredgecolor="black",
               markersize=8,
               label=methods[i])
        for i in range(len(methods))
    ]
    ax.legend(handles=legend_handles, title="Method", frameon=False, loc="best")

    # 颜色条：编码 workload size，颜色越深数值越大（Greys: 高值更黑）
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

# ---------------- run-mode core (was original main body) ----------------
def main_run(args):
    sizes = [int(x) for x in args.sizes.split(",") if x]
    q_list = [int(x) for x in args.q_list.split(",") if x]
    d_list = [int(x) for x in args.d_list.split(",") if x]

    # ensure save directory structure
    SAVE_DIR = Path(args.save_dir)
    (SAVE_DIR/"workloads").mkdir(parents=True, exist_ok=True)
    (SAVE_DIR/"summaries").mkdir(parents=True, exist_ok=True)

    # catalog doesn't depend on workload length
    makers_all, meta = build_catalog(q_list, d_list)  # returns (makers_all, meta)

    predictor_cfg = {"sliding_window_sec": args.sliding_window_sec,
                     "min_samples": args.min_samples}

    # strategies to compare (name -> (fn, kwargs_builder))
    # Names kept consistent with your existing runner for label parity.
    def _common_kwargs(workload):
        return dict(
            workload=workload, makers_all=makers_all,
            predictor_cfg=predictor_cfg, prewarm_every=args.prewarm_every,
            lookahead_sec=args.lookahead, prob_th=args.prob_th,
            max_compile=args.max_compile, shots=args.shots,
            include_exec = False,
        )
    def _baseline_kwargs(workload):
        return dict(workload=workload, shots=args.shots, include_exec = False,)

    STRATS = [
        # ("TransCache(Adaptive)", SA.run_strategy, _common_kwargs),
        ("FS+Pre+ttl", S_FS_Pre_ttl.run_strategy, _common_kwargs),
        ("FS+Pre", S_FS_Pre.run_strategy, _common_kwargs),
        ("FS",S_FS.run_strategy,  _baseline_kwargs),
        ("PR",S_PR.run_strategy,  _baseline_kwargs),
        # ("ParamReuse",           SPR.run_strategy, _baseline_kwargs),
        # ("FullCompilation",      S0.run_strategy,  _baseline_kwargs),
    ]

    # storage for plotting
    method2xs: Dict[str, List[int]] = {name: [] for (name, _, _) in STRATS}
    method2lat: Dict[str, List[float]] = {name: [] for (name, _, _) in STRATS}
    method2csz: Dict[str, List[int]]   = {name: [] for (name, _, _) in STRATS}
    method2hits: Dict[str, List[int]]   = {name: [] for (name, _, _) in STRATS}

    # Also collect a long-form table for CSV and a JSON-friendly summary
    rows: List[Row] = []

    # iterate over workload sizes
    for i, N in enumerate(sizes):
        rng_seed = args.rng_seed_base + i * 1009  # deterministic but varies with size
        # workload = build_workload_poisson_superposition(
        #     meta, N, args.hot_fraction, args.hot_boost,
        #     args.rps, rng_seed, return_timestamps=True
        # # )
        # 2) 生成 workload（保留 expovariate 语义）
        workload, info = build_workload_poisson_superposition_exact(
            meta, N, args.hot_fraction, args.hot_boost,
            args.rps, rng_seed, return_timestamps=True
        )
        # 3) 可视化自检
        visualize_superposed_poisson_exact(workload, info, meta, bins=40, topk_classes=12)

        # persist a SLIM version of workload (no callables)
        SAVE_DIR = Path(args.save_dir)
        wl_path = SAVE_DIR/"workloads"/f"workload_N{N}_seed{rng_seed}.json"
        wl_slim = slim_workload_for_dump(workload)
        wl_path.write_text(json_dump(wl_slim))

        print(f"\n[run] workload size = {N} (saved {wl_path.name})")
        print("-"*80)

        # create a per-size folder for events/metrics
        perN_dir = SAVE_DIR/f"N{N}"
        (perN_dir/"events").mkdir(parents=True, exist_ok=True)
        (perN_dir/"metrics").mkdir(parents=True, exist_ok=True)

        events_series = {}
        metrics_series = {}
        for name, fn, kw_builder in STRATS:
            clear_recent()  # reset predictor stats for fairness
            kwargs = kw_builder(workload)
            out = fn(**kwargs)
            events = out["events"]
            metrics = out.get("metrics", {})

            # save events and metrics for this (N, method) using safe dumper
            (perN_dir/"events"/f"{name}.json").write_text(json_dump(events))
            (perN_dir/"metrics"/f"{name}.json").write_text(json_dump(metrics))

            events_series[name] = events
            metrics_series[name] = metrics

            e2e = e2e_latency_from_events(events)
            csz = final_cache_size_from_metrics(metrics)
            hits = final_hitrate_from_metrics(workload, metrics)

            method2xs[name].append(N)
            method2lat[name].append(e2e)
            method2csz[name].append(csz)
            method2hits[name].append(hits)

            rows.append(Row(size=N, method=name, e2e_latency=e2e, final_cache_size=csz))
            print(f"{name:>20s} | E2E latency = {e2e:8.3f} s | final cache size = {csz:4d}")

        # draw timeline & cache size change for each wl
        # timeline
        out_tl = PLOT_DIR / f"timeline_wl_{N}.png"
        draw_timeline_multi(events_series, out_tl)

        # cache changes compare
        cache_size_cahnges = {}
        for name, _, _ in STRATS:
            cache_size_cahnges[name] = metrics_series[name].get("cache_size_series", [])
        out_png = PLOT_DIR / f"cache_change_wl_{N}.png"
        plot_cache_size_change(cache_size_cahnges, out_png)

    # ---------------- plotting via shared functions ----------------
    methods = [name for (name, _, _) in STRATS]
    out_latency = PLOT_DIR / args.out_latency
    out_cache = PLOT_DIR / args.out_cache
    out_scatter_path = PLOT_DIR / args.out_scatter  # NEW
    out_hitrate_path = PLOT_DIR / args.out_hitrate  # NEW

    # 注意：x 轴 sizes 共用，method2xs 仅用于一致性校验
    for m in methods:
        if method2xs[m] != sizes:
            print(f"[warn] sizes mismatch for method {m}: {method2xs[m]} vs {sizes}")

    plot_e2e_latency(methods, sizes, method2lat, out_latency)
    plot_final_cache_size(methods, sizes, method2csz, out_cache)
    plot_final_hitrate(methods, sizes, method2hits, out_hitrate_path)
    plot_latency_vs_cache_scatter(methods, sizes, method2lat, method2csz, out_scatter_path)

    # ---------------- persist summary ----------------
    SAVE_DIR = Path(args.save_dir)
    (SAVE_DIR/"summaries").mkdir(parents=True, exist_ok=True)

    # 1) long-form CSV
    csv_path = SAVE_DIR/"summaries"/"scaling_summary_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["size", "method", "e2e_latency", "final_cache_size"])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"[save] {csv_path}")

    # 2) JSON summary (wide)
    summary = {
        "sizes": sizes,
        "methods": methods,
        "e2e_latency": {name: method2lat[name] for name in methods},
        "final_cache_size": {name: method2csz[name] for name in methods},
        "config": {
            "q_list": [int(x) for x in args.q_list.split(",") if x],
            "d_list": [int(x) for x in args.d_list.split(",") if x],
            "shots": args.shots, "hot_fraction": args.hot_fraction,
            "hot_boost": args.hot_boost, "rps": args.rps,
            "rng_seed_base": args.rng_seed_base,
            "lookahead": args.lookahead, "prob_th": args.prob_th,
            "max_compile": args.max_compile, "sliding_window_sec": args.sliding_window_sec,
            "min_samples": args.min_samples, "prewarm_every": args.prewarm_every,
        },
        "plots": {
            "latency_png": str(out_latency),
            "cache_png": str(out_cache),
        },
    }
    json_path = SAVE_DIR/"summaries"/"scaling_summary.json"
    json_path.write_text(json_dump(summary))
    print(f"[save] {json_path}")

    # pretty console tables
    print("-"*80)
    print("Summary (rows = workload size)")
    header = ["N"] + methods
    print(" | ".join(f"{h:>20s}" for h in header))
    for row_i, N in enumerate(sizes):
        cells_lat = [f"{summary['e2e_latency'][name][row_i]:8.3f}s" for name in methods]
        print(" | ".join([f"{N:>20d}"] + [f"{c:>20s}" for c in cells_lat]))

    print("-"*80)
    print("Final cache size")
    print(" | ".join(f"{h:>20s}" for h in header))
    for row_i, N in enumerate(sizes):
        cells_sz = [f"{summary['final_cache_size'][name][row_i]:>8d}" for name in methods]
        print(" | ".join([f"{N:>20d}"] + [f"{c:>20s}" for c in cells_sz]))


# ---------------- load-mode: read summary json and redraw ----------------
def load_and_redraw(load_dir: str,
                    out_latency: str = "scaling_e2e_latency.png",
                    out_cache: str = "scaling_final_cache_size.png",
                    out_scatter: str = "scaling_latency_vs_cache_scatter.png") -> None:
    """
    从 SAVE_DIR/'summaries'/'scaling_summary.json' 载入结果，并用与 run 相同的逻辑重绘两张图。
    仅依赖 summary，不重新运行任何策略。
    """
    LOAD_DIR = Path(load_dir)
    json_path = LOAD_DIR / "summaries" / "scaling_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Summary not found: {json_path}")

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    sizes: List[int] = summary["sizes"]
    methods: List[str] = summary["methods"]
    method2lat: Dict[str, List[float]] = summary["e2e_latency"]
    method2csz: Dict[str, List[int]] = summary["final_cache_size"]

    # 输出路径（保持与 run 模式一致，默认写到 PLOT_DIR 下）
    out_latency_path = PLOT_DIR / out_latency
    out_cache_path = PLOT_DIR / out_cache
    out_scatter_path = PLOT_DIR / out_scatter  # NEW

    # 调用与 run 相同的绘图函数
    plot_e2e_latency(methods, sizes, method2lat, out_latency_path)
    plot_final_cache_size(methods, sizes, method2csz, out_cache_path)
    # NEW: 使用 summary 数据绘制散点图
    plot_latency_vs_cache_scatter(methods, sizes, method2lat, method2csz, out_scatter_path)

    # 控制台提示
    print("-"*80)
    print("[load] Redraw finished from summary:")
    print(f" sizes   = {sizes}")
    print(f" methods = {methods}")
    print(f" source  = {json_path}")
    print(f" saved   = {out_latency_path}, {out_cache_path}")

# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["run", "load"], default="run",
                    help="run: 运行仿真并绘图；load: 仅从summary加载并重绘图")

    # workload shape
    ap.add_argument("--sizes", type=str, default="500", #50,100,150,200,300,350,400,450,500
                    help="Comma-separated workload lengths to test.")
    # ap.add_argument("--q_list", type=str, default="5,7,11,13")
    # ap.add_argument("--d_list", type=str, default="4,8")
    ap.add_argument("--q_list", type=str, default="5, 7, 11, 13, 15, 17") #5, 7, 11, 13, 15, 17, 19, 21
    ap.add_argument("--d_list", type=str, default="2,4,6") # 2,4,6,8, 12
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--hot_fraction", type=float, default=0.25)
    ap.add_argument("--hot_boost", type=float, default=8.0)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--rng_seed_base", type=int, default=123)

    # predictor / prewarm (for TransCache/FirstSeen family)
    ap.add_argument("--lookahead", type=float, default=8.0) # FS_Pre
    ap.add_argument("--prob_th", type=float, default=0.45) # FS_Pre
    ap.add_argument("--max_compile", type=int, default=3) # FS_Pre
    ap.add_argument("--sliding_window_sec", type=float, default=60.0)
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--prewarm_every", type=int, default=5) # FS_Pre

    # IO
    ap.add_argument("--out_latency", type=str, default="scaling_e2e_latency.png")
    ap.add_argument("--out_cache", type=str, default="scaling_final_cache_size.png")
    ap.add_argument("--out_scatter", type=str, default="scaling_latency_vs_cache_scatter.png")
    ap.add_argument("--out_hitrate", type=str, default="scaling_hirtrate.png")
    ap.add_argument("--load_dir", type=str, default=str((LOAD_ROOT/"scaling").resolve()),
                    help="Directory to load summaries.")
    ap.add_argument("--save_dir", type=str, default=str((ROOT / "scaling").resolve()),
                    help="Directory to store workloads, per-run events/metrics, and summaries.")
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
                        out_scatter=args.out_scatter)

if __name__ == "__main__":
    main()
