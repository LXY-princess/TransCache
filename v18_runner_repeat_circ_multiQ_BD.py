# -*- coding: utf-8 -*-
"""
Runner (multi-q): V18_runner_repeat_circ_multiQ.py:基础上，
额外记录transT 和execT，这两个之前只记录了sum（e2e_latency=transT+execT），
反馈到summary中row多两个内容。
固定深度 d，针对指定 v11 电路集合，在多个 qubits 值上分别
生成“同一电路连续调用 N 次”的 workload，并按既有策略运行与记录结果。

- 与 v18_runner_repeat_circ.py 的风格/输出保持一致
- 新增 --qs 参数：逗号分隔 qubits 列表（默认 3,11,17,23）
"""

import argparse, json, csv, re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

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

def sum_latency_from_events(events: List[Dict[str, Any]], tag:str) -> float:
    """统计所有事件的 execT 之和；若事件没有 execT 字段，则用 default_execT。"""
    if not events:
        print(f"error: no events {tag}")
        return 0.0
    total = 0.0
    for e in events:
        try:
            total += float(e.get(tag))
        except (TypeError, ValueError):
            # 非法值则跳过
            pass
    return total

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
    transT: float
    execT: float
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

    # qubits list（默认 3,11,17,23）
    qs = [int(x) for x in args.qs.split(",") if x.strip()]
    ds = [int(x) for x in args.ds.split(",") if x.strip()]
    N = int(args.N)

    # catalog：一次性覆盖所有 q
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
            include_exec=True,
        )
    def _baseline_kwargs(workload):
        return dict(workload=workload, shots=args.shots, include_exec=True)

    STRATS = [
        ("TransCache", S_FS_Pre_ttl_SE_ema.run_strategy, _common_kwargs),
        # ("FS+Pre",            S_FS_Pre.run_strategy,            _common_kwargs),
        # ("CCache",                S_FS.run_strategy,                _baseline_kwargs),
        # ("Braket",                S_PR.run_strategy,                _baseline_kwargs),
        ("FullComp",              S_FullComp.run_strategy,          _baseline_kwargs),
    ]

    SAVE_DIR = Path(args.save_dir)
    (SAVE_DIR / "workloads").mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "summaries").mkdir(parents=True, exist_ok=True)

    rows: List[Row] = []

    # 逐 (circuit, q) 跑 —— 共 len(labels) * len(qs) 个 workload
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

                events_series, metrics_series = {}, {}
                for name, fn, kw_builder in STRATS:
                    clear_recent()
                    out = fn(**kw_builder(wl))
                    events = out["events"]
                    metrics = out.get("metrics", {})

                    (per_dir / "events" / f"{name}.json").write_text(json_dump(events))
                    (per_dir / "metrics" / f"{name}.json").write_text(json_dump(metrics))

                    events_series[name] = events
                    metrics_series[name] = metrics

                    e2e = e2e_latency_from_events(events)
                    csz = final_cache_size_from_metrics(metrics)
                    hit = final_hitrate_from_metrics(wl, metrics)
                    tranT = sum_latency_from_events(events, "transT")
                    execT = sum_latency_from_events(events, "execT")
                    rows.append(Row(circuit=label, q=q, d=d, N=N,
                                    method=name, transT=tranT, execT=execT, e2e_latency=e2e,
                                    final_cache_size=csz, final_hitrate=hit))
                    print(f"{name:>22s} | transT={tranT:8.3f}s |execT={execT:8.3f}s |E2E={e2e:8.3f}s | cache={csz:4d} | hitrate={hit:6.2f}%")

                # 可视化
                draw_timeline_multi(events_series, PLOT_DIR / f"timeline_{_safe_label(label)}_q{q}_d{d}_N{N}.png")
                cache_size_changes = {m: metrics_series[m].get("cache_size_series", [])
                                      for (m, _, _) in STRATS}
                plot_cache_size_change(cache_size_changes,
                                       PLOT_DIR / f"cache_change_{_safe_label(label)}_q{q}_d{d}_N{N}.png")

    # 汇总
    csv_path = SAVE_DIR / "summaries" / "repeated_v11_multiq_summary_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in Row.__annotations__.keys()])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\n[save] {csv_path}")

    methods = [name for (name, _, _) in STRATS]
    circuits = labels
    summary = {}
    for r in rows:
        summary.setdefault(r.circuit, {}).setdefault(r.q, {})[r.method] = {
            "transT": r.transT,
            "execT": r.execT,
            "e2e_latency": r.e2e_latency,
            "final_cache_size": r.final_cache_size,
            "final_hitrate": r.final_hitrate,
            "d": r.d, "N": r.N,
        }
    json_path = SAVE_DIR / "summaries" / "repeated_v11_multiq_summary.json"
    json_path.write_text(json_dump({
        "circuits": circuits, "methods": methods, "qs": qs, "summary": summary,
        "config": {
            "d": d, "N": N, "rps": args.rps, "shots": args.shots,
            "lookahead": args.lookahead, "prob_th": args.prob_th,
            "max_compile": args.max_compile, "sliding_window_sec": args.sliding_window_sec,
            "min_samples": args.min_samples, "prewarm_every": args.prewarm_every,
        }
    }))
    print(f"[save] {json_path}")

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", type=str, default="ALL",
                    help='ALL 或 逗号分隔的 v11 电路名（如 "GHZ-Chain,RCA"）')
    ap.add_argument("--qs", type=str, default="3,11,17,23",
                    help="逗号分隔的 qubits 列表，例如 3,11,17,23")
    ap.add_argument("--ds", type=str, default="4",
                    help="逗号分隔的 depth 列表，例如 1,2,4,8")
    # ap.add_argument("--d", type=int, default=4, help="固定电路深度")
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
    return ap

def main():
    args = build_argparser().parse_args()
    main_run(args)

if __name__ == "__main__":
    main()
