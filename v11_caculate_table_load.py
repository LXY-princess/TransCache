# plot_from_sum_json_depth_stacked_select.py
# 读取 *_baseline_sum.json / *_tcache_sum.json（优先），按 (q, circuit) 分图，
# 每图包含指定 DEPTHS；每个 depth 两根相邻堆叠柱（Baseline vs TranspileCache）

import argparse, json, re, pathlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ----------------- 你可以在这里直接改筛选集合 -----------------
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})
# CIRCS  = ["GHZ-Chain", "LinearEnt", "QFT-Like", "RCA", "QSIM-XXZ"]
# CIRCS  = ["GHZ-Chain", "LinearEnt", "QFT-Like", "RCA", "QSIM-XXZ", "QAOA-3reg", "VQE-Full"]
CIRCS  = ["QAOA-3reg", "VQE-Full"]
DEPTHS = [1, 4, 8, 16, 25]  # 1, 4, 8, 16, 25
QUBITS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]  # 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25

PHASE_02 = "02_cache_search"
PHASE_03 = "03_transpile"
PHASE_07 = "07_cache_write"

# 目录与命名（需与生成脚本一致）
CODE_TAG = "v11"
DEFAULT_FIGDIR = pathlib.Path("./figs/v11sd_5runs")     # load csv from

# 阶段颜色与顺序（与生成脚本一致）
COLOR_MAP = {
    "01_setup"       : "#9ecae1",
    "02_cache_search": "#00E5EE",
    "03_transpile"   : "#4292c6",
    "04_submit"      : "#8FBC8F",
    "05_run"         : "#fedcb2",
    "06_collect"     : "#E9967A",
    "07_cache_write" : "#6e559c",
}
PHASE_ORDER = ["01_setup","02_cache_search","03_transpile","04_submit","05_run","06_collect","07_cache_write"]

# 运行级文件名（回退聚合用）：v10_q{q}_d{d}_sim_{circuit}_{mode}_{r}.json
RUN_RE = re.compile(
    r"^" + re.escape(CODE_TAG) + r"_q(?P<q>\d+)_d(?P<d>\d+)_sim_(?P<circ>[^_]+)_(?P<mode>baseline|tcache)_(?P<run>\d+)\.json$"
)
# 汇总文件名（首选读取）：v10_q{q}_d{d}_sim_{circuit}_{mode}_sum.json
SUM_RE = re.compile(
    r"^" + re.escape(CODE_TAG) + r"_q(?P<q>\d+)_d(?P<d>\d+)_sim_(?P<circ>[^_]+)_(?P<mode>baseline|tcache)_sum\.json$"
)

def sum_by_phase(laps_list: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for ph in PHASE_ORDER:
        vals = [lp.get(ph, 0.0) for lp in laps_list]
        out[ph] = float(sum(vals)) if vals else 0.0
    out["total"] = float(sum(out[ph] for ph in PHASE_ORDER))
    return out

def load_sum_or_runs(figdir: Path, q: int, d: int, circ: str, mode: str) -> Dict[str, float]:
    """
    优先读 *_sum.json；若不存在则聚合 *_r.json 的 laps 做求和。
    返回 {phase: sum_seconds, ..., 'total': ...}；若完全缺失返回 {}。
    """
    # 1) 先找 sum 文件
    sum_name = f"{CODE_TAG}_q{q}_d{d}_sim_{circ}_{mode}_sum.json"
    sum_path = figdir / sum_name
    if sum_path.exists():
        try:
            obj = json.loads(sum_path.read_text())
            # sum 文件里就是 {phase: value, ...} 的字典
            # 兜底 total
            # if "total" not in obj:
            #     obj["total"] = float(sum(obj.get(ph, 0.0) for ph in PHASE_ORDER))
            return {k: float(v) for k, v in obj.items()}
        except Exception:
            pass

    # # 2) 回退：聚合 run 级文件
    # laps_list = []
    # for p in figdir.glob(f"{CODE_TAG}_q{q}_d{d}_sim_{circ}_{mode}_*.json"):
    #     m = RUN_RE.match(p.name)
    #     if not m:  # 只聚合 *_r.json，而不是 *_sum.json
    #         continue
    #     try:
    #         obj = json.loads(p.read_text())
    #     except Exception:
    #         continue
    #     laps = obj.get("laps", obj)  # 兼容 {"laps": {...}} 或直接 laps dict
    #     if isinstance(laps, dict):
    #         laps_list.append(laps)
    # if laps_list:
    #     return sum_by_phase(laps_list)
    #
    # return {}  # 完全没有数据

def collect_stats(figdir: Path,
                  circs: List[str], depths: List[int], qubits: List[int]
                  ) -> Dict[Tuple[int,str], Dict[int, Dict[str, Dict[str,float]]]]:
    """
    返回：
      stats[(q, circ)][d]['baseline'|'tcache'] = {phase: sum_time, ..., 'total': ...}
    """
    stats = defaultdict(lambda: defaultdict(dict))
    for q in qubits:
        for circ in circs:
            for d in depths:
                base = load_sum_or_runs(figdir, q, d, circ, "baseline")
                tca  = load_sum_or_runs(figdir, q, d, circ, "tcache")
                if base: stats[(q, circ)][d]["baseline"] = base
                if tca:  stats[(q, circ)][d]["tcache"]   = tca
    return stats

def fmt_speedup(numer: float, denom: float) -> str:
    """返回 speedup= numer/denom，若 denom<=0 或 speedup<=1 则 '-'。"""
    if denom <= 0:
        return "-"
    s = numer / denom
    return f"{s:.2f}×" if s > 1.0 else "-"

def fmt(s: float) -> str:
    return f"{s:.3f}"

# ---------- Main table printer ----------
def print_table(figdir: Path, circs: List[str], qubits: List[int], depths: List[int]):
    sep = "-" * 120
    for q in qubits:
        for circ in circs:
            print(sep)
            print(f"[Circuit={circ}] [q={q}]")
            print(sep)
            # 表头
            print(
                "depth | total_base  total_tcache   speedup(TC vs Base) || "
                "base.03  tc.02  tc.03  tc.07 || tc.(02+03+07)  speedup( base.03 vs tc.sum )"
            )
            print("-" * 120)

            for d in depths:
                base = load_sum_or_runs(figdir, q, d, circ, "baseline")
                tca  = load_sum_or_runs(figdir, q, d, circ, "tcache")

                total_base   = float(base.get("total", 0.0))
                total_tcache = float(tca.get("total", 0.0))
                # 1) TCache 相对 Baseline 的 speedup（越大越好）
                speedup_total = fmt_speedup(total_base, total_tcache)

                # 2) 分阶段
                base_03 = float(base.get(PHASE_03, 0.0))
                tc_02   = float(tca.get(PHASE_02, 0.0))
                tc_03   = float(tca.get(PHASE_03, 0.0))
                tc_07   = float(tca.get(PHASE_07, 0.0))

                # 3) TCache 的 (02+03+07) 之和
                tc_sum_020307 = tc_02 + tc_03 + tc_07

                # 4) base.03 相对 tc_sum 的 speedup（>1 表示 tc_sum 更小/更快）
                speedup_compile = fmt_speedup(base_03, tc_sum_020307)

                # 输出一行
                print(
                    f"{d:>5} | {fmt(total_base):>10}  {fmt(total_tcache):>12}   {speedup_total:>16} || "
                    f"{fmt(base_03):>7}  {fmt(tc_02):>6}  {fmt(tc_03):>6}  {fmt(tc_07):>6} || "
                    f"{fmt(tc_sum_020307):>12}  {speedup_compile:>24}"
                )
            print()  # 空行分隔每个 (q,circuit)

def plot_one(q: int, circ: str, depths: List[int],
             stats_one: Dict[int, Dict[str, Dict[str, float]]],
             outdir: Path):
    """
    stats_one[d]['baseline'|'tcache'] = {phase: sum_time, ..., 'total': ...}
    每个 depth 两根相邻堆叠柱：左 baseline, 右 tcache
    """
    # 过滤只画有数据的 depth
    depths_draw = [d for d in depths if d in stats_one and
                   ("baseline" in stats_one[d] or "tcache" in stats_one[d])]
    if not depths_draw:
        print(f"⚠ 无数据：q={q}, circ={circ}")
        return

    x = np.arange(len(depths_draw), dtype=float)
    bw, gap = 0.25, 0.10
    fig, ax = plt.subplots(figsize=(11, 4.8))

    def draw_stacked(xpos, comp_key, label_prefix):
        bottoms = np.zeros_like(xpos, dtype=float)
        any_drawn = False
        for idx, ph in enumerate(PHASE_ORDER):
            heights = [stats_one[d].get(comp_key, {}).get(ph, 0.0) for d in depths_draw]
            if any(h > 0 for h in heights):
                ax.bar(
                    xpos, heights, bottom=bottoms, width=bw,
                    color=COLOR_MAP.get(ph, "#999999"), edgecolor="black",
                    # label=ph if (label_prefix == "Baseline" and not any_drawn) else None,
                    hatch=('//' if label_prefix == "Baseline" else None),
                    # linestyle="-" if label_prefix == "Baseline" else "--",
                )
                any_drawn = True
            bottoms += np.array(heights)

    # 左：Baseline；右：Cache
    x_base  = x - (bw/2 + gap/2)
    x_cache = x + (bw/2 + gap/2)
    draw_stacked(x_base,  "baseline", "Baseline")
    draw_stacked(x_cache, "tcache",   "Cache")

    # 轴 & 标题
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths_draw])
    ax.set_xlabel("Circuit depth", fontweight="bold")
    ax.set_ylabel(f"Cumulative Latency", fontweight="bold")
    # ax.set_title(f"{circ}  |  q={q}  |  Baseline vs TranspileCache (stacked)")

    # 图例（按阶段）


    phase_handles = [mpatches.Patch(color=COLOR_MAP[ph], label=ph) for ph in PHASE_ORDER]
    leg_phase = ax.legend(handles=phase_handles, bbox_to_anchor=(1.02, 1.0),
                          loc="upper left", title="Phase", frameon=False)
    mode_baseline = mpatches.Patch(facecolor="white", edgecolor="black", hatch='//', label="Baseline")
    mode_cache    = mpatches.Patch(facecolor="white", edgecolor="black", label="TranspileCache")
    leg_mode = ax.legend(handles=[mode_baseline, mode_cache],
                         loc="upper left", bbox_to_anchor=(1.02, 0.3), title="Mode", frameon=False)
    ax.add_artist(leg_phase)

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # handles = [mpatches.Patch(color=COLOR_MAP[ph], label=ph) for ph in PHASE_ORDER]
    # ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Phase")
    # ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 在柱顶标注 total（并加 B/C）
    for xi, d in zip(x, depths_draw):
        tb = stats_one[d].get("baseline", {}).get("total", 0.0)
        tc = stats_one[d].get("tcache",   {}).get("total", 0.0)
        if tb > 0:
            ax.text(xi - (bw/2 + gap/2), tb + 0.02, f"{tb:.3f}", ha="center", va="bottom", fontsize=10)
        if tc > 0:
            ax.text(xi + (bw/2 + gap/2), tc + 0.02, f"{tc:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlim(x.min() - 0.7, x.max() + 0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / f"stacked_q{q}_{circ}_depths.png"
    plt.savefig(out_png, dpi=600)
    plt.close(fig)
    print("✅ saved:", out_png)

def main():
    parser = argparse.ArgumentParser()
    # 也允许命令行覆盖默认筛选
    # parser.add_argument("--circs",  default=",".join(CIRCS), help="电路名，逗号分隔")
    # parser.add_argument("--depths", default=",".join(map(str, DEPTHS)), help="depth 列表，逗号分隔")
    # parser.add_argument("--qubits", default=",".join(map(str, QUBITS)), help="qubits 列表，逗号分隔")
    args = parser.parse_args()

    figdir = DEFAULT_FIGDIR
    # outdir = Path(args.outdir)
    if not figdir.exists():
        raise SystemExit(f"目录不存在：{figdir}")

    # ----------------- 你可以在这里直接改筛选集合 -----------------
    circs = CIRCS
    depths = DEPTHS
    qubits = QUBITS

    # 聚合：优先 *_sum.json，缺失时回退 runs 求和
    all_stats = collect_stats(figdir, circs, depths, qubits)
    # stats[(q, circ)][d]['baseline' | 'tcache'] = {phase: sum_time, ..., 'total': ...}

    print_table(figdir, circs, qubits, depths)

if __name__ == "__main__":
    main()
