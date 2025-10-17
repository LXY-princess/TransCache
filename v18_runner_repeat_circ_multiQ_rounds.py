# -*- coding: utf-8 -*-
"""
Runner (multi-q): 固定深度 d=2，在 16 个 qubits 上跑多次（重复=5），
保存每次结果与 mean/std 的聚合结果；保持原有输出结构，并新增 *_agg.{csv,json}。
新增：为每个电路绘制 "Latency vs Qubits" 折线图（mean 线 + std 阴影）。

基于 v18_runner_repeat_circ_multiQ.py 修改：
1) 固定 d=2
2) qs = list(range(3,128,4))[::2] => 共 16 个点
3) 每种 (circuit,q,d) 的 workload 重复 5 次
4) 新增每电路折线图（方法区分颜色与 marker，阴影为 ±std）
"""

import argparse, json, csv, re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from v18_core import LOAD_ROOT
# ==== core & viz helpers ====
from v18_core import (
    ROOT, PLOT_DIR, build_catalog, clear_recent,
    draw_timeline_multi, plot_cache_size_change, compute_freq_and_hits
)

# ==== 策略 ====
import v18_strat_FS as S_FS
import v18_strat_FS_Pre as S_FS_Pre
import v18_strat_PR as S_PR
import v18_strat_FS_Pre_ttl_SE_ema as S_FS_Pre_ttl_SE_ema
import v18_strat_fullComp as S_FullComp

# ==== v11 电路 ====
from v11_quasa_bench_circuits import CIRCUITS_QUASA


# ---------------- utils ----------------
def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)

def json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)

def slim_workload_for_dump(workload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"name": it.get("name"),
         "q": int(it.get("q")) if it.get("q") is not None else None,
         "d": int(it.get("d")) if it.get("d") is not None else None,
         "ts": float(it.get("ts", 0.0))}
        for it in workload
    ]

def e2e_latency_from_events(events: List[Dict[str, Any]]) -> float:
    if not events: return 0.0
    return max(float(e.get("start", 0.0)) + float(e.get("dur", 0.0)) for e in events)

def final_cache_size_from_metrics(metrics: Dict[str, Any]) -> int:
    series = metrics.get("cache_size_series") or []
    if not series: return 0
    last = series[-1]
    if isinstance(last, dict): return int(last.get("size", 0))
    if isinstance(last, (list, tuple)) and len(last) >= 2: return int(last[1])
    return 0

def final_hitrate_from_metrics(workload, metrics: Dict[str, Any]):
    hit_by_label = metrics.get("hit_by_label", {})
    _, _, overall = compute_freq_and_hits(workload, hit_by_label)
    return overall

@dataclass
class Row:
    circuit: str
    q: int
    d: int
    N: int
    method: str
    e2e_latency: float
    final_cache_size: int
    final_hitrate: float


# ---------------- workload ----------------
def build_workload_repeated(label: str, q: int, d: int, N: int = 10, rps: float = 1.0
                            ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if N <= 0:
        raise ValueError("N 必须为正整数")
    if label not in CIRCUITS_QUASA:
        raise KeyError(f"电路 {label} 不在 v11 清单中。可选：{list(CIRCUITS_QUASA.keys())}")
    step = 0.0 if rps <= 0 else (1.0 / float(rps))
    wl = [{"name": label, "q": int(q), "d": int(d), "ts": i * step} for i in range(N)]
    info = {"label": label, "q": int(q), "d": int(d), "N": int(N), "rps": float(rps)}
    return wl, info

def _safe_label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


# ---------------- 绘图：Latency vs Q ----------------
# def _set_plot_style():
#     plt.rcParams.update({
#         "font.family": "Times New Roman",
#         "font.size": 22,
#         "axes.labelsize": 26,
#         # "axes.titlesize": 26,
#         # "legend.fontsize": 16,
#         # "xtick.labelsize": 18,
#         "ytick.labelsize": 18,
#         "figure.dpi": 600,
#         "axes.linewidth": 4,
#     })

def _set_plot_style():
    plt.rcParams.update({
        # 字体与字号
        "font.family": "Times New Roman",
        "font.size": 22,
        # 坐标轴 & 标题
        "axes.linewidth": 4,
        "axes.labelsize": 40,  # xlabel/ylabel 字号
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
        "lines.markeredgewidth": 5,
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
        "legend.fontsize": 22,
        # 保存
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def plot_latency_vs_q_per_circuit(circuit: str, qs: List[int], agg_rows: List[Dict[str, Any]], outdir: Path):
    """对给定电路，画每种方法的 E2E latency(mean) 随 qubit 增大的折线 + std 阴影。"""
    _set_plot_style()
    # 排序并去重
    qs_sorted = sorted(set(qs))
    # 方法图例与样式
    methods_order = ["TransCache", "CCache", "Braket", "FullComp"]
    label_map = {
        "TransCache": "TransCache",
        "CCache": "CCache",
        "Braket": "Braket",
        "FullComp": "FullComp"
    }

    colors = {"TransCache": "#213d69",
              "CCache": "#6b80d6",  # #6b84d6
              "Braket": "#64a6d1",
              "FullComp": "#d6d66b"
              }

    titles = {
        "QAOA-3reg": "QAOA",
        "QFT-Like": "QFT",
        "QSIM-XXZ": "QSIM",
        "RCA": "RCA",
        "VQE-Full": "VQE",
        "GHZ-Chain": "GHZ",
    }
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    # 颜色：使用 Matplotlib tab10 前几色
    # colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8,5))
    for i, m in enumerate(methods_order):
        # 从聚合结果收集该电路该方法在各 q 的 mean/std
        mean_y, std_y, xs = [], [], []
        for q in qs_sorted:
            # 找到唯一一行 (circuit, q, method)
            rows_mq = [r for r in agg_rows if r["circuit"] == circuit and r["q"] == q and r["method"] == m]
            if not rows_mq:
                continue
            r = rows_mq[0]
            xs.append(q)
            mean_y.append(r["e2e_latency_mean"])
            std_y.append(r["e2e_latency_std"])
        if not xs:
            continue
        xs = np.array(xs, dtype=int)
        mean_y = np.array(mean_y, dtype=float)
        std_y = np.array(std_y, dtype=float)

        ax.plot(xs, mean_y, marker=markers[i % len(markers)],label=label_map.get(m, m),
                color=colors.get(str(m)))
        ax.fill_between(xs, mean_y - std_y, mean_y + std_y,
                        alpha=0.20, linewidth=0, color=colors.get(str(m)))

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel("Qubits")
    ax.set_ylabel("Latency (s)")
    # ax.set_yscale("log")
    # ax.grid(True, linestyle="--", linewidth=2, alpha=0.5)
    # ax.legend(frameon=False, ncol=2, loc="upper left")
    plt.title(titles.get(circuit), fontsize=40, fontweight="bold")

    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"latency_vs_q_{_safe_label(circuit)}.png"
    # pdf = outdir / f"latency_vs_q_{_safe_label(circuit)}.pdf"
    fig.tight_layout()
    fig.savefig(png)
    # fig.savefig(pdf)
    plt.close(fig)
    print(f"[plot] {png}")

def abstract_legend(circuit: str, qs: List[int], agg_rows: List[Dict[str, Any]], outdir: Path):
    """对给定电路，画每种方法的 E2E latency(mean) 随 qubit 增大的折线 + std 阴影。"""
    _set_plot_style()
    # 排序并去重
    qs_sorted = sorted(set(qs))
    # 方法图例与样式
    methods_order = ["TransCache", "CCache", "Braket", "FullComp"]
    label_map = {
        "TransCache": "TransCache",
        "CCache": "CCache",
        "Braket": "Braket",
        "FullComp": "FullComp"
    }

    colors = {"TransCache": "#213d69",
              "CCache": "#6b80d6",  # #6b84d6
              "Braket": "#64a6d1",
              "FullComp": "#d6d66b"
              }

    titles = {
        "QAOA-3reg": "QAOA",
        "QFT-Like": "QFT",
        "QSIM-XXZ": "QSIM",
        "RCA": "RCA",
        "VQE-Full": "VQE",
        "GHZ-Chain": "GHZ",
    }
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    # 颜色：使用 Matplotlib tab10 前几色
    # colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8,5))
    for i, m in enumerate(methods_order):
        # 从聚合结果收集该电路该方法在各 q 的 mean/std
        mean_y, std_y, xs = [], [], []
        for q in qs_sorted:
            # 找到唯一一行 (circuit, q, method)
            rows_mq = [r for r in agg_rows if r["circuit"] == circuit and r["q"] == q and r["method"] == m]
            if not rows_mq:
                continue
            r = rows_mq[0]
            xs.append(q)
            mean_y.append(r["e2e_latency_mean"])
            std_y.append(r["e2e_latency_std"])
        if not xs:
            continue
        xs = np.array(xs, dtype=int)
        mean_y = np.array(mean_y, dtype=float)
        std_y = np.array(std_y, dtype=float)

        ax.plot(xs, mean_y, marker=markers[i % len(markers)],
                markerfacecolor='none',markeredgewidth=4,
                markersize=12, linewidth=4, label=label_map.get(m, m),
                color=colors.get(str(m)))
        ax.fill_between(xs, mean_y - std_y, mean_y + std_y,
                        alpha=0.20, linewidth=0, color=colors.get(str(m)))

    ax.set_xlabel("Qubits")
    ax.set_ylabel("Transpilation Latency (s)")
    # ax.set_yscale("log")
    # ax.grid(True, linestyle="--", linewidth=2, alpha=0.5)
    ax.legend(frameon=True, ncol=2, loc="upper left")
    plt.title(titles.get(circuit))

    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"latency_vs_q_{_safe_label(circuit)}.png"
    # pdf = outdir / f"latency_vs_q_{_safe_label(circuit)}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=600)
    # fig.savefig(pdf)
    plt.close(fig)
    print(f"[plot] {png}")

def reload_and_replot(save_dir: str):
    """从保存的 *_agg.json 或 *_agg.csv 重新绘制每电路的 latency vs qubits 折线图。"""
    save_dir = Path(save_dir)
    sum_dir = save_dir / "summaries"
    json_agg = sum_dir / "repeated_v11_multiq_summary_agg.json"
    csv_agg  = sum_dir / "repeated_v11_multiq_summary_agg.csv"

    agg_rows = []
    circuits, qs = [], []
    if json_agg.exists():
        data = json.loads(json_agg.read_text())
        circuits = data.get("circuits", [])
        qs       = data.get("qs", [])
        summary  = data.get("summary", {})
        # 还原成绘图函数使用的扁平列表
        for c, q_map in summary.items():
            for q, m_map in q_map.items():
                for method, v in m_map.items():
                    agg_rows.append(dict(
                        circuit=c, q=int(q), method=method, d=int(v.get("d", 2)), N=int(v.get("N", 10)),
                        repeats=int(v.get("repeats", 5)),
                        e2e_latency_mean=float(v["e2e_latency_mean"]),
                        e2e_latency_std=float(v["e2e_latency_std"]),
                        final_cache_size_mean=float(v["final_cache_size_mean"]),
                        final_cache_size_std=float(v["final_cache_size_std"]),
                        final_hitrate_mean=float(v["final_hitrate_mean"]),
                        final_hitrate_std=float(v["final_hitrate_std"]),
                    ))
    elif csv_agg.exists():
        import csv as _csv
        with csv_agg.open("r", newline="", encoding="utf-8") as f:
            rdr = _csv.DictReader(f)
            for r in rdr:
                r = {k: (int(v) if k in {"q","d","N","repeats"} else float(v) if "mean" in k or "std" in k else v)
                     for k,v in r.items()}
                agg_rows.append(r)
        circuits = sorted({r["circuit"] for r in agg_rows})
        # 若没保存 qs，可从行聚合
        qs = sorted({int(r["q"]) for r in agg_rows})
    else:
        raise FileNotFoundError(f"Not found: {json_agg} or {csv_agg}")

    # 绘图
    outdir = PLOT_DIR / "latency_vs_q"
    for c in circuits:
        plot_latency_vs_q_per_circuit(c, qs, agg_rows, outdir)
        # abstract_legend(c, qs, agg_rows, outdir)



# ---------------- main ----------------
def main_run(args):
    # circuits
    all_labels = list(CIRCUITS_QUASA.keys())
    if args.circuits.strip().upper() == "ALL":
        labels = all_labels
    else:
        want = [x.strip() for x in args.circuits.split(",") if x.strip()]
        bad = [x for x in want if x not in all_labels]
        if bad:
            raise KeyError(f"以下电路不在 v11 清单中：{bad}；可选：{all_labels}")
        labels = want

    # ===== 实验设置（按需求固定） =====
    # qubits：先取 range(3,128,4)，再隔一个取一个，得到 16 个点：3,11,19,...,123
    qs = [int(x) for x in args.qs.split(",") if x.strip()]
    ds = [int(x) for x in args.ds.split(",") if x.strip()]
    N = int(args.N)
    REPEATS = 5  # 每种 workload 重复次数

    # catalog：一次性覆盖所有 q（d=2）
    makers_all, meta = build_catalog(qs, ds)
    # (name,q,d) -> maker_run
    meta_map = {(m["name"], m["q"], m["d"]): m["maker_run"] for m in meta}

    predictor_cfg = {"sliding_window_sec": args.sliding_window_sec,
                     "min_samples": args.min_samples}

    def _common_kwargs(workload):
        return dict(
            workload=workload, makers_all=makers_all,
            predictor_cfg=predictor_cfg, prewarm_every=args.prewarm_every,
            lookahead_sec=args.lookahead, prob_th=args.prob_th,
            max_compile=args.max_compile, shots=args.shots,
            include_exec=False,
        )
    def _baseline_kwargs(workload):
        return dict(workload=workload, shots=args.shots, include_exec=False)

    STRATS = [
        ("TransCache", S_FS_Pre_ttl_SE_ema.run_strategy, _common_kwargs),
        # ("FS+Pre",            S_FS_Pre.run_strategy,            _common_kwargs),
        ("CCache",                S_FS.run_strategy,                _baseline_kwargs),
        ("Braket",                S_PR.run_strategy,                _baseline_kwargs),
        ("FullComp",              S_FullComp.run_strategy,          _baseline_kwargs),
    ]

    SAVE_DIR = Path(args.save_dir)
    (SAVE_DIR / "workloads").mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "summaries").mkdir(parents=True, exist_ok=True)
    (PLOT_DIR / "latency_vs_q").mkdir(parents=True, exist_ok=True)

    rows: List[Row] = []                 # 逐次长表（兼容原逻辑）
    agg_rows: List[Dict[str, Any]] = []  # 聚合行（mean/std）

    # 逐 (circuit, q) 跑 —— 共 len(labels) * 16 个 workload，每个重复 5 次
    for label in labels:
        for q in qs:
            for d in ds:
                wl, _ = build_workload_repeated(label, q, d, N, args.rps)

                # 注入 maker_run
                mk = meta_map.get((label, q, d))
                if mk is None:
                    raise KeyError(f"No maker_run found for {(label, q, d)}")
                for it in wl:
                    it["maker_run"] = mk

                wl_path = SAVE_DIR / "workloads" / f"wl_{_safe_label(label)}_N{N}_q{q}_d{d}.json"
                wl_path.write_text(json_dump(slim_workload_for_dump(wl)))
                print(f"\n[workload] {label} | q={q}, d={d}, N={N}  ->  saved: {wl_path.name}")

                per_dir = SAVE_DIR / f"{_safe_label(label)}_q{q}_d{d}_N{N}"
                (per_dir / "events").mkdir(parents=True, exist_ok=True)
                (per_dir / "metrics").mkdir(parents=True, exist_ok=True)

                # 累计器（用于 mean/std）
                acc: Dict[str, Dict[str, List[float]]] = {}
                for name, _, _ in STRATS:
                    acc[name] = {"e2e": [], "cache": [], "hit": []}

                # ===== 重复 REPEATS 次 =====
                for rep in range(REPEATS):
                    print(f"  [repeat {rep+1}/{REPEATS}] {label} q={q} d={d}")
                    events_series, metrics_series = {}, {}
                    for name, fn, kw_builder in STRATS:
                        clear_recent()
                        out = fn(**kw_builder(wl))
                        events = out["events"]
                        metrics = out.get("metrics", {})

                        # 保存每次
                        (per_dir / "events" / f"{name}_r{rep}.json").write_text(json_dump(events))
                        (per_dir / "metrics" / f"{name}_r{rep}.json").write_text(json_dump(metrics))

                        events_series[name] = events
                        metrics_series[name] = metrics

                        # 逐次行（兼容原长表 CSV）
                        e2e = e2e_latency_from_events(events)
                        csz = final_cache_size_from_metrics(metrics)
                        hit = final_hitrate_from_metrics(wl, metrics)
                        rows.append(Row(circuit=label, q=q, d=d, N=N,
                                        method=name, e2e_latency=e2e,
                                        final_cache_size=csz, final_hitrate=hit))
                        print(f"{name:>22s} | E2E={e2e:8.3f}s | cache={csz:4d} | hitrate={hit:6.2f}%")

                        # 累加
                        acc[name]["e2e"].append(e2e)
                        acc[name]["cache"].append(csz)
                        acc[name]["hit"].append(hit)

                    # 可视化（每次可选保留一次）
                    draw_timeline_multi(events_series, PLOT_DIR / f"timeline_{_safe_label(label)}_q{q}_d{d}_N{N}_r{rep}.png")
                    # cache_size_changes = {m: metrics_series[m].get("cache_size_series", [])
                    #                       for (m, _, _) in STRATS}
                    # plot_cache_size_change(cache_size_changes,
                    #                        PLOT_DIR / f"cache_change_{_safe_label(label)}_q{q}_d{d}_N{N}_r{rep}.png")

                # ===== 计算 mean/std（聚合一行）=====
                for name in acc:
                    e2e_arr = np.array(acc[name]["e2e"], dtype=float)
                    csz_arr = np.array(acc[name]["cache"], dtype=float)
                    hit_arr = np.array(acc[name]["hit"], dtype=float)
                    agg_rows.append(dict(
                        circuit=label, q=q, d=d, N=N, method=name, repeats=REPEATS,
                        e2e_latency_mean=float(np.mean(e2e_arr)),
                        e2e_latency_std=float(np.std(e2e_arr, ddof=1)) if REPEATS > 1 else 0.0,
                        final_cache_size_mean=float(np.mean(csz_arr)),
                        final_cache_size_std=float(np.std(csz_arr, ddof=1)) if REPEATS > 1 else 0.0,
                        final_hitrate_mean=float(np.mean(hit_arr)),
                        final_hitrate_std=float(np.std(hit_arr, ddof=1)) if REPEATS > 1 else 0.0,
                    ))

    # ===== 保存：逐次长表（原逻辑不变） =====
    SAVE_DIR = Path(args.save_dir)
    csv_path = SAVE_DIR / "summaries" / "repeated_v11_multiq_summary_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in Row.__annotations__.keys()])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\n[save] {csv_path}")

    # ===== 保存：聚合 CSV =====
    agg_csv = SAVE_DIR / "summaries" / "repeated_v11_multiq_summary_agg.csv"
    if agg_rows:
        with agg_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(agg_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in agg_rows:
                w.writerow(r)
        print(f"[save] {agg_csv}")

    # ===== 保存：summary.json（保持原结构，但用 mean 值）+ agg.json（含 mean/std）=====
    methods = ["TransCache", "CCache", "Braket", "FullComp"]
    circuits = labels

    # 原结构（存 mean）
    summary_mean = {}
    for r in agg_rows:
        summary_mean.setdefault(r["circuit"], {}).setdefault(r["q"], {})[r["method"]] = {
            "e2e_latency": r["e2e_latency_mean"],
            "final_cache_size": r["final_cache_size_mean"],
            "final_hitrate": r["final_hitrate_mean"],
            "d": r["d"], "N": r["N"],
        }
    json_mean = SAVE_DIR / "summaries" / "repeated_v11_multiq_summary.json"
    json_mean.write_text(json_dump({
        "circuits": circuits, "methods": methods, "qs": qs, "summary": summary_mean,
        "config": {
            "d": 2, "N": N, "repeats": 5, "rps": args.rps, "shots": args.shots,
            "lookahead": args.lookahead, "prob_th": args.prob_th,
            "max_compile": args.max_compile, "sliding_window_sec": args.sliding_window_sec,
            "min_samples": args.min_samples, "prewarm_every": args.prewarm_every,
        }
    }))
    print(f"[save] {json_mean}")

    # 含 mean/std 的 agg 版
    summary_agg = {}
    for r in agg_rows:
        summary_agg.setdefault(r["circuit"], {}).setdefault(r["q"], {})[r["method"]] = {
            "e2e_latency_mean": r["e2e_latency_mean"],
            "e2e_latency_std": r["e2e_latency_std"],
            "final_cache_size_mean": r["final_cache_size_mean"],
            "final_cache_size_std": r["final_cache_size_std"],
            "final_hitrate_mean": r["final_hitrate_mean"],
            "final_hitrate_std": r["final_hitrate_std"],
            "d": r["d"], "N": r["N"], "repeats": r["repeats"],
        }
    json_agg = SAVE_DIR / "summaries" / "repeated_v11_multiq_summary_agg.json"
    json_agg.write_text(json_dump({
        "circuits": circuits, "methods": methods, "qs": qs, "summary": summary_agg,
        "config": {
            "d": 2, "N": N, "repeats": 5, "rps": args.rps, "shots": args.shots,
            "lookahead": args.lookahead, "prob_th": args.prob_th,
            "max_compile": args.max_compile, "sliding_window_sec": args.sliding_window_sec,
            "min_samples": args.min_samples, "prewarm_every": args.prewarm_every,
        }
    }))
    print(f"[save] {json_agg}")

    # ===== 绘图：每个电路一张折线图（mean±std） =====
    outdir = PLOT_DIR / "latency_vs_q"
    for c in circuits:
        plot_latency_vs_q_per_circuit(c, qs, agg_rows, outdir)



def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", type=str, default="RCA",
                    help='ALL 或 逗号分隔的 v11 电路名（如 "GHZ-Chain,QFT-Like,QAOA-3reg,VQE-Full,RCA,QSIM-XXZ"）')

    # 下面两个参数仍保留，但本次实验在 main_run 中已覆盖为 qs[::2] 与 d=2
    ap.add_argument("--qs", type=str, default="3,7,15,31,63,95,127", # 3,7,15,31,63,95,127
                    help="逗号分隔的 qubits 列表（本实验固定为 range(3,128,4)[::2] 共16点）")
    ap.add_argument("--ds", type=str, default="2",
                    help="逗号分隔的 depth 列表（本实验固定为 2）")

    ap.add_argument("--N", type=int, default=10, help="同一电路连续调用次数")
    ap.add_argument("--rps", type=float, default=1.0, help="请求到达率（requests per second）")

    # predictor / prewarm（与现有 runner 对齐）
    ap.add_argument("--lookahead", type=float, default=8.0)
    ap.add_argument("--prob_th", type=float, default=0.45)
    ap.add_argument("--max_compile", type=int, default=3)
    ap.add_argument("--sliding_window_sec", type=float, default=60.0)
    ap.add_argument("--min_samples", type=int, default=2)
    ap.add_argument("--prewarm_every", type=int, default=5)
    ap.add_argument("--shots", type=int, default=256)

    ap.add_argument("--save_dir", type=str, default=str((ROOT / "repeated_v11_multiq").resolve()),
                    help="保存 workloads / events / metrics / summaries 的根目录")
    ap.add_argument("--load_dir", type=str, default=str((LOAD_ROOT).resolve()),
                    help="load workloads / events / metrics / summaries 的根目录")
    return ap

def main():
    args = build_argparser().parse_args()
    # main_run(args)
    name = ["ghz", "qaoa", "qsim", "rca", "vqe", "qft"]
    # name = ["qft"]
    for n in name:
        # LOAD_ROOT = pathlib.Path("./figs")/f"v{VNUM}_test_127q"
        dir = args.load_dir + f"_{n}/repeated_v11_multiq"
        reload_and_replot(dir)

if __name__ == "__main__":
    main()
