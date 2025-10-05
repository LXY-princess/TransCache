# runner_compare.py
import argparse, pathlib
from typing import Dict, Any
from collections import Counter

from oqpy import frame

from v15_core import LOAD_EVENTS_DIR
from v15_core import (ROOT, EVENTS_DIR, PLOT_DIR, draw_timeline_multi,
                      build_catalog, build_workload_poisson_superposition,
                      compute_freq_and_hits, plot_freq_hitrate_bars,
                      save_events_json, clear_recent, load_events_json, plot_cache_size_change)


def load_and_draw_timeline(methods=None, outfile=None, title=None, legend_topk=16):
    """
    读取已保存的 events JSON，重绘 multi-track timeline。
    methods: 要显示的方法名（显示顺序），默认用你在 main 里保存过的那几种。
    """
    # 方法名 -> JSON 文件名 的映射（按你保存时的命名来）
    name2file = {
        "Baseline": "baseline.json",
        "FirstSeen": "FirstSeen.json",
        "TransCache": "TransCache_optimize_score_log.json",
        "ParamReuse": "ParamReuse.json",   # 如有
    }

    if methods is None:
        methods = ["TransCache",
                   "FirstSeen",
                   "ParamReuse",
                   "Baseline",
                   ]

    # 逐个加载
    method_events = {}
    for name in methods:
        fn = name2file.get(name)
        if not fn:
            print(f"[warn] no file mapping for method '{name}', skip.")
            continue
        p = LOAD_EVENTS_DIR / fn
        if not p.exists():
            print(f"[warn] events file not found: {p}, skip.")
            continue
        method_events[name] = load_events_json(p)

    if not method_events:
        print("[error] nothing loaded; check EVENTS_DIR and file names.")
        return

    # 标题与输出路径
    if title is None:
        title = "Timeline — redraw from saved EventList"
    if outfile is None:
        outfile = PLOT_DIR / "timeline_multi_replot.png"

    draw_timeline_multi(method_events, outfile, title, legend_topk=legend_topk)
    print(f"[ok] redraw saved to {outfile}")

import v15_strategy_baseline as S0
import v15_strategy_tcache_series as S1
import v15_strategy_tcache_series_cacheM as S1M
# import v15_strategy_tcache_idle_gap as S2
# import v15_strategy_cache_first_seen as S3
# import v15_strategy_tcache_series_idle as S4

import v15_strategy_cache_first_seen_cache_tracking as S3
import v15_strategy_tcache_no_cacheM_cache_tracking as S4   # 监测版 S4（语义同 S4）
import v15_strategy_tcache as S5     # 新策略：有上限+评分淘汰
import v15_strategy_tcache_optimize as S6     # 新策略：有上限+评分淘汰
import v15_strategy_tcache_optimize_score_log as S6S     # 新策略
import v15_strategy_param_reuse_missFirstSeen as SPR0
import v15_strategy_param_reuse as SPR
import v15_strategy_tcache_adaptive_2 as SA



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_list", type=str, default="5, 7, 11, 13")
    ap.add_argument("--d_list", type=str, default="4,8")
    # ap.add_argument("--q_list", type=str, default="5, 7, 11, 13, 15, 17, 19, 21")
    # ap.add_argument("--d_list", type=str, default="2,4,6,8, 12")
    ap.add_argument("--workload_len", type=int, default=100)
    ap.add_argument("--shots", type=int, default=256)
    # predictor / prewarm
    ap.add_argument("--lookahead", type=float, default=8.0)
    ap.add_argument("--prob_th", type=float, default=0.45)
    ap.add_argument("--max_compile", type=int, default=3)
    ap.add_argument("--sliding_window_sec", type=float, default=60.0)
    ap.add_argument("--min_samples", type=int, default=2)
    # workload shape
    ap.add_argument("--hot_fraction", type=float, default=0.25)
    ap.add_argument("--hot_boost", type=float, default=8.0)
    ap.add_argument("--rps", type=float, default=1.0)
    ap.add_argument("--rng_seed", type=int, default=123)
    # idle-gap params
    ap.add_argument("--gap_usage_ratio", type=float, default=0.7)
    ap.add_argument("--default_compile_est", type=float, default=0.08)
    # output
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    q_list = [int(x) for x in args.q_list.split(",") if x]
    d_list = [int(x) for x in args.d_list.split(",") if x]

    # 1) catalog + workload(带到达时间戳)
    makers_all, meta = build_catalog(q_list, d_list)
    workload = build_workload_poisson_superposition(
        meta, args.workload_len, args.hot_fraction, args.hot_boost,
        args.rps, args.rng_seed, return_timestamps=True
    )

    # 2) run strategies on the SAME workload
    predictor_cfg = {"sliding_window_sec": args.sliding_window_sec, "min_samples": args.min_samples}

    # 通用 kwargs（所有非 Baseline 策略共用）
    common_kwargs = dict(
        workload=workload,
        makers_all=makers_all,
        predictor_cfg=predictor_cfg,
        prewarm_every=5,
        lookahead_sec=args.lookahead,
        prob_th=args.prob_th,
        max_compile=args.max_compile,
        shots=args.shots,
        include_exec=False,
    )

    # Baseline 特殊 kwargs
    baseline_kwargs = dict(
        workload=workload,
        shots=args.shots,
        include_exec=False,
    )

    # 策略映射
    strategies = {
        "TransCache(Adaptive)": (SA.run_strategy, common_kwargs),
        "TransCache(Proposed)": (S6S.run_strategy, common_kwargs),
        "FirstSeen": (S3.run_strategy, baseline_kwargs),
        "ParamReuse": (SPR.run_strategy, baseline_kwargs),
        "FullCompilation": (S0.run_strategy, baseline_kwargs),
        # "ParamReuse_missFirstSeen": (SPR0.run_strategy, baseline_kwargs),
        # "TransCache-Series": (S1.run_strategy, common_kwargs),
        # "TransCache-Series-cacheM": (S1M.run_strategy, common_kwargs),
        # "TransCache-no-cache-management": (S4.run_strategy, common_kwargs),
        # "TransCache": (S5.run_strategy, common_kwargs),
        # "TransCache_optimize": (S6.run_strategy, common_kwargs),
    }

    method_events = {}
    metrics_all = {}

    for name, (fn, kwargs) in strategies.items():
        clear_recent()
        out = fn(**kwargs)
        save_events_json(name, out["events"])
        method_events[name] = out["events"]
        metrics_all[name] = out["metrics"]

    # 3) timeline (multi-track)
    title = (f"Timeline — q={q_list}, d={d_list}, N={args.workload_len}; lookahead={args.lookahead}s,"
             f" prob_th={args.prob_th}, max_compile={args.max_compile}, window={args.sliding_window_sec}s")
    out_tl = PLOT_DIR/"timeline_multi.png"
    draw_timeline_multi(method_events, out_tl, title)

    # 4) bars（针对 Tcache 方法）
    #   基于同一 workload 的频次，分别计算两种 Tcache 的命中率并作图
    hitrate_compare_method_tags = ["TransCache(Proposed)", "TransCache(Adaptive)", "FirstSeen", "ParamReuse", "FullCompilation"]
    for tag in hitrate_compare_method_tags:
        hit_by_label = metrics_all[tag].get("hit_by_label", {})
        freq_by_label, hitrate_by_label, overall = compute_freq_and_hits(workload, hit_by_label)
        out_bar = PLOT_DIR/f"bars_{tag}.png"
        plot_freq_hitrate_bars(freq_by_label, hitrate_by_label, overall, out_bar,
                               title=f"{tag}: Frequency & Hit Rate", top_k=(args.top_k or None))

    # 4) cache changes compare
    cache_size_cahnges = {}
    cache_compare_method_tags = ["TransCache(Proposed)", "TransCache(Adaptive)", "FirstSeen", "ParamReuse", "FullCompilation"]
    for tag in cache_compare_method_tags:
        cache_size_cahnges[tag] = metrics_all[tag].get("cache_size_series", [])
    plot_cache_size_change(cache_size_cahnges)



if __name__ == "__main__":
    main()
    # load_and_draw_timeline()
