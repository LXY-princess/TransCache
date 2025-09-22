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
        "Tcache-Series": "tcache_series.json",
        "tcache_firstSeen": "tcache_firstSeen.json",
        "tcache_preCompile_on_idle": "tcache_preCompile_on_idle.json",
        "Tcache-IdleGap": "tcache_idle.json",   # 如有
    }

    if methods is None:
        methods = ["Baseline", "Tcache-Series", "tcache_firstSeen", "tcache_preCompile_on_idle"]

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
# import v15_strategy_tcache_idle_gap as S2
# import v15_strategy_cache_first_seen as S3
# import v15_strategy_tcache_series_idle as S4

import v15_strategy_cache_first_seen_cache_tracking as S3
import v15_strategy_tcache_no_cacheM_cache_tracking as S4   # 监测版 S4（语义同 S4）
import v15_strategy_tcache as S5     # 新策略：有上限+评分淘汰



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q_list", type=str, default="5,7, 11, 13")
    ap.add_argument("--d_list", type=str, default="4,8")
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

    method_events = {}
    metrics_all = {}

    # Baseline
    clear_recent()
    out0 = S0.run_strategy(workload, shots=args.shots)
    save_events_json("baseline", out0["events"])
    method_events["Baseline"] = out0["events"]
    metrics_all["Baseline"] = out0["metrics"]

    # Tcache (cache when first seen)
    clear_recent()
    out3 = S3.run_strategy(workload, makers_all, predictor_cfg,
                           prewarm_every=5, lookahead_sec=args.lookahead, prob_th=args.prob_th,
                           max_compile=args.max_compile, shots=args.shots)
    save_events_json("FirstSeen", out3["events"])
    method_events["FirstSeen"] = out3["events"]
    metrics_all["FirstSeen"] = out3["metrics"]

    # Tcache (Series) prewarm time plotted in the workload timeline
    clear_recent()
    out1 = S1.run_strategy(workload, makers_all, predictor_cfg,
                           prewarm_every=5, lookahead_sec=args.lookahead, prob_th=args.prob_th,
                           max_compile=args.max_compile, shots=args.shots)
    save_events_json("TransCache-Series", out1["events"])
    method_events["TransCache-Series"] = out1["events"]
    metrics_all["TransCache-Series"] = out1["metrics"]

    # Tcache (remove prewarm time from results)
    clear_recent()
    out4 = S4.run_strategy(workload, makers_all, predictor_cfg,
                           prewarm_every=5, lookahead_sec=args.lookahead, prob_th=args.prob_th,
                           max_compile=args.max_compile, shots=args.shots)
    save_events_json("TransCache_no_cache_management", out4["events"])
    method_events["TransCache_no_cache_management"] = out4["events"]
    metrics_all["TransCache_no_cache_management"] = out4["metrics"]

    # # Tcache (Idle-gap) this manually add wait time between circuits in workload
    # clear_recent()
    # out2 = S2.run_strategy(workload, makers_all, predictor_cfg,
    #                        lookahead_sec=args.lookahead, prob_th=args.prob_th,
    #                        max_compile=args.max_compile, gap_usage_ratio=args.gap_usage_ratio,
    #                        default_compile_est=args.default_compile_est, shots=args.shots)
    # save_events_json("tcache_idle", out2["events"])
    # method_events["Tcache-IdleGap"] = out2["events"]
    # metrics_all["Tcache-IdleGap"] = out2["metrics"]

    # Tcache cache management
    clear_recent()
    out5 = S5.run_strategy(workload, makers_all, predictor_cfg,
                           prewarm_every=5, lookahead_sec=args.lookahead, prob_th=args.prob_th,
                           max_compile=args.max_compile, shots=args.shots)
    save_events_json("TransCache", out5["events"])
    method_events["TransCache"] = out5["events"]
    metrics_all["TransCache"] = out5["metrics"]

    # 3) timeline (multi-track)
    title = (f"Timeline — q={q_list}, d={d_list}, N={args.workload_len}; lookahead={args.lookahead}s,"
             f" prob_th={args.prob_th}, max_compile={args.max_compile}, window={args.sliding_window_sec}s")
    out_tl = PLOT_DIR/"timeline_multi.png"
    draw_timeline_multi(method_events, out_tl, title)

    # # 4) bars（针对 Tcache 方法）
    # #   基于同一 workload 的频次，分别计算两种 Tcache 的命中率并作图
    # for tag in ["Tcache-Series", "Tcache-IdleGap"]:
    #     hit_by_label = metrics_all[tag].get("hit_by_label", {})
    #     freq_by_label, hitrate_by_label, overall = compute_freq_and_hits(workload, hit_by_label)
    #     out_bar = PLOT_DIR/f"bars_{tag}.png"
    #     plot_freq_hitrate_bars(freq_by_label, hitrate_by_label, overall, out_bar,
    #                            title=f"{tag}: Frequency & Hit Rate", top_k=(args.top_k or None))


    s4_series = out4["metrics"].get("cache_size_series", [])
    s5_series = out5["metrics"].get("cache_size_series", [])
    s3_series = out3["metrics"].get("cache_size_series", [])
    cache_size_cahnges = {}
    cache_size_cahnges["TransCache_no_cache_management"] = s4_series
    cache_size_cahnges["TransCache"] = s5_series
    cache_size_cahnges["FirstSeen"] = s3_series

    plot_cache_size_change(cache_size_cahnges)



if __name__ == "__main__":
    main()
    # load_and_draw_timeline()
