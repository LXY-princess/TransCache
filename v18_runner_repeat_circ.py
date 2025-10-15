# -*- coding: utf-8 -*-
"""
Runner: 固定 (q, d)，让指定 v11 电路连续调用 N 次形成一个 workload，
并按你工程现有策略运行与记录结果。

- 兼容你现有 v18_runner_wl.py 的调用/保存/绘图习惯
- 工作负载生成：build_workload_repeated_v11()
- 电路来源：v11 CIRCUITS_QUASA（与工程保持同名 key）
"""

import argparse, json, csv, re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# ==== core & viz helpers（与现有 runner 保持一致） ====
from v18_core import (
    ROOT, PLOT_DIR, build_catalog, clear_recent,
    draw_timeline_multi, plot_cache_size_change, compute_freq_and_hits
)

# ==== 策略（与现有 runner 一致的别名与参数） ====
import v18_strat_FS as S_FS                  # FirstSeen
import v18_strat_FS_Pre as S_FS_Pre          # FirstSeen + predictor prewarm
import v18_strat_PR as S_PR                  # Param-Reuse-like（无预测器）
import v18_strat_FS_Pre_ttl_SE_ema as S_FS_Pre_ttl_SE_ema  # 强化版 FS+Pre
import v18_strat_fullComp as S_FullComp

# ==== v11 电路清单（名称需与 catalog 一致） ====
from v11_quasa_bench_circuits import CIRCUITS_QUASA  # v11 的电路集合

# ---------------- JSON utilities（与现有 runner 一致，安全序列化） ----------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)

def json_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)

def slim_workload_for_dump(workload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """仅保存 name/q/d/ts 以便 JSON 序列化。"""
    return [
        {"name": it.get("name"),
         "q": int(it.get("q")) if it.get("q") is not None else None,
         "d": int(it.get("d")) if it.get("d") is not None else None,
         "ts": float(it.get("ts", 0.0))}
        for it in workload
    ]

# ---------------- 基本指标提取（与现有 runner 一致） ----------------
def e2e_latency_from_events(events: List[Dict[str, Any]]) -> float:
    if not events:
        return 0.0
    return max(float(e.get("start", 0.0)) + float(e.get("dur", 0.0)) for e in events)

def final_cache_size_from_metrics(metrics: Dict[str, Any]) -> int:
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
    circuit: str
    q: int
    d: int
    N: int
    method: str
    e2e_latency: float
    final_cache_size: int
    final_hitrate: float

# ---------------- Workload：v11 电路连续调用 N 次 ----------------
def build_workload_repeated(label: str, q: int, d: int, N: int = 10, rps: float = 1.0
                                ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    生成 workload：同一电路 (label, q, d) 连续调用 N 次，间隔 1/rps 秒。
    返回 (workload, info)；workload 每项包含 {name,q,d,ts}。
    """
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

# ---------------- 主流程：按电路循环，每个电路一个 workload ----------------
def main_run(args):
    # 1) 选择电路集合
    all_labels = list(CIRCUITS_QUASA.keys())
    if args.circuits.strip().upper() == "ALL":
        labels = all_labels
    else:
        want = [x.strip() for x in args.circuits.split(",") if x.strip()]
        bad = [x for x in want if x not in all_labels]
        if bad:
            raise KeyError(f"以下电路不在 v11 清单中：{bad}；可选：{all_labels}")
        labels = want

    q, d, N = int(args.q), int(args.d), int(args.N)

    # 2) 构建 catalog（给策略查 maker 用，限制在单一 q/d 组合即可）
    makers_all, meta = build_catalog([q], [d])

    # NEW: (name,q,d) -> maker_run 的映射
    meta_map = {(m["name"], m["q"], m["d"]): m["maker_run"] for m in meta}

    # 3) 预测器/预热等共有参数
    predictor_cfg = {"sliding_window_sec": args.sliding_window_sec,
                     "min_samples": args.min_samples}
    def _common_kwargs(workload):
        return dict(
            workload=workload, makers_all=makers_all,
            predictor_cfg=predictor_cfg, prewarm_every=args.prewarm_every,
            lookahead_sec=args.lookahead, prob_th=args.prob_th,
            max_compile=args.max_compile, shots=args.shots,
            include_exec=True,
        )
    def _baseline_kwargs(workload):
        return dict(workload=workload, shots=args.shots, include_exec=True)

    STRATS = [
        ("FS+Pre+ttl+SE+ema", S_FS_Pre_ttl_SE_ema.run_strategy, _common_kwargs),
        # ("FS+Pre",            S_FS_Pre.run_strategy,            _common_kwargs),
        ("FS",                S_FS.run_strategy,                _baseline_kwargs),
        ("PR",                S_PR.run_strategy,                _baseline_kwargs),
        ("Full",                S_FullComp.run_strategy,        _baseline_kwargs),
    ]

    # 4) 目录结构
    SAVE_DIR = Path(args.save_dir)
    (SAVE_DIR / "workloads").mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "summaries").mkdir(parents=True, exist_ok=True)

    rows: List[Row] = []

    # 5) 每个电路跑一个 workload
    for label in labels:
        wl, info = build_workload_repeated(label, q, d, N, args.rps)

        # NEW: 注入 maker_run（策略会读取 it["maker_run"]）
        mk = meta_map.get((label, q, d))
        if mk is None:
            raise KeyError(f"No maker_run found for {(label, q, d)}")
        for it in wl:
            it["maker_run"] = mk

        # 保存 workload（瘦身版）
        wl_path = SAVE_DIR / "workloads" / f"wl_{_safe_label(label)}_N{N}_q{q}_d{d}.json"
        wl_path.write_text(json_dump(slim_workload_for_dump(wl)))
        print(f"\n[workload] {label} | q={q}, d={d}, N={N}  ->  saved: {wl_path.name}")

        # 每个电路单独一个目录装 events/metrics
        per_label_dir = SAVE_DIR / f"{_safe_label(label)}_q{q}_d{d}_N{N}"
        (per_label_dir / "events").mkdir(parents=True, exist_ok=True)
        (per_label_dir / "metrics").mkdir(parents=True, exist_ok=True)

        # 跑各策略
        events_series = {}
        metrics_series = {}
        for name, fn, kw_builder in STRATS:
            clear_recent()
            out = fn(**kw_builder(wl))
            events = out["events"]
            metrics = out.get("metrics", {})

            # 保存单策略的结果
            (per_label_dir / "events" / f"{name}.json").write_text(json_dump(events))
            (per_label_dir / "metrics" / f"{name}.json").write_text(json_dump(metrics))

            events_series[name] = events
            metrics_series[name] = metrics

            e2e = e2e_latency_from_events(events)
            csz = final_cache_size_from_metrics(metrics)
            hit = final_hitrate_from_metrics(wl, metrics)

            rows.append(Row(circuit=label, q=q, d=d, N=N,
                            method=name, e2e_latency=e2e,
                            final_cache_size=csz, final_hitrate=hit))
            print(f"{name:>22s} | E2E={e2e:8.3f}s | cache={csz:4d} | hitrate={hit:6.2f}%")

        # 可视化：时间线 & 缓存规模变化
        draw_timeline_multi(events_series, PLOT_DIR / f"timeline_{_safe_label(label)}_q{q}_d{d}_N{N}.png")
        cache_size_changes = {m: metrics_series[m].get("cache_size_series", [])
                              for (m, _, _) in STRATS}
        plot_cache_size_change(cache_size_changes,
                               PLOT_DIR / f"cache_change_{_safe_label(label)}_q{q}_d{d}_N{N}.png")

    # 6) 汇总保存
    csv_path = SAVE_DIR / "summaries" / "repeated_v11_summary_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in Row.__annotations__.keys()])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\n[save] {csv_path}")

    # 便于二次处理的 wide JSON
    methods = [name for (name, _, _) in STRATS]
    circuits = labels
    # wide 结构：按 (circuit -> method -> 指标)
    wide = {c: {m: {} for m in methods} for c in circuits}
    for r in rows:
        wide[r.circuit][r.method] = {
            "e2e_latency": r.e2e_latency,
            "final_cache_size": r.final_cache_size,
            "final_hitrate": r.final_hitrate,
            "q": r.q, "d": r.d, "N": r.N,
        }
    json_path = SAVE_DIR / "summaries" / "repeated_v11_summary.json"
    json_path.write_text(json_dump({
        "circuits": circuits, "methods": methods, "summary": wide,
        "config": {
            "q": q, "d": d, "N": N, "rps": args.rps, "shots": args.shots,
            "lookahead": args.lookahead, "prob_th": args.prob_th,
            "max_compile": args.max_compile, "sliding_window_sec": args.sliding_window_sec,
            "min_samples": args.min_samples, "prewarm_every": args.prewarm_every,
        }
    }))
    print(f"[save] {json_path}")

# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", type=str, default="ALL",
                    help='ALL 或 逗号分隔的 v11 电路名（如 "GHZ-Chain,RCA"）')
    ap.add_argument("--q", type=int, default=3, help="固定 qubits 数")
    ap.add_argument("--d", type=int, default=4, help="固定电路深度")
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

    ap.add_argument("--save_dir", type=str, default=str((ROOT / "repeated_v11").resolve()),
                    help="保存 workloads / events / metrics / summaries 的根目录")
    return ap

def main():
    args = build_argparser().parse_args()
    main_run(args)

if __name__ == "__main__":
    main()
