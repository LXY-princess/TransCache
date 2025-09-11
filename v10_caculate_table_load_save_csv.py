# plot_from_sum_json_depth_stacked_select.py
# 读取 *_baseline_sum.json / *_tcache_sum.json（优先），按 (q, circuit) 分图，
# 每图包含指定 DEPTHS；每个 depth 两根相邻堆叠柱（Baseline vs TranspileCache）

import argparse, json, re, csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib

# ----------------- 你可以在这里直接改筛选集合 -----------------
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16})
CIRCS  = ["GHZ-Chain", "LinearEnt", "QFT-Like"]
DEPTHS = [1, 2, 4, 8, 16, 32]
QUBITS = [3, 5, 6, 7]

PHASE_02 = "02_cache_search"
PHASE_03 = "03_transpile"
PHASE_07 = "07_cache_write"

# 目录与命名（需与生成脚本一致）
CODE_TAG = "v10"
DEFAULT_FIGDIR = pathlib.Path("./figs/v10sd")     # 生成脚本写 JSON 的目录
OUTDIR_DEFAULT = pathlib.Path("./figs/v10_table");OUTDIR_DEFAULT.mkdir(exist_ok=True) # 输出图片目录

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

def load_sum(figdir: Path, q: int, d: int, circ: str, mode: str):
    p = figdir / f"{CODE_TAG}_q{q}_d{d}_sim_{circ}_{mode}_sum.json"
    if not p.exists(): return {}
    try:
        obj = json.loads(p.read_text())
        return {k: float(v) for k, v in obj.items()}
    except Exception:
        return {}

def speedup(numer: float, denom: float):
    if denom <= 0: return "-"
    s = numer / denom
    return f"{s:.2f}" if s > 1.0 else "-"

def save_to_csv_all():
    figdir = DEFAULT_FIGDIR
    outdir = OUTDIR_DEFAULT
    if not figdir.exists():
        raise SystemExit(f"目录不存在：{figdir}")

    # ----------------- 你可以在这里直接改筛选集合 -----------------
    circs = CIRCS
    depths = DEPTHS
    qubits = QUBITS

    csv_path = outdir / "save_metrics_qd_in_circuit.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        # circuit_tag 里已包含 q/d；其他列维持数值与 speedup
        w.writerow([
            "circuit_tag",
            "total_base", "total_tcache", "speedup_total",
            "base_03", "tc_02", "tc_03", "tc_07",
            "tc_sum_020307", "speedup_compile"
        ])
        for circ in circs:
            for q in qubits:
                for d in depths:
                    base = load_sum(figdir, q, d, circ, "baseline")
                    tca = load_sum(figdir, q, d, circ, "tcache")
                    if not base and not tca:
                        continue
                    tb = float(base.get("total", 0.0))
                    tc = float(tca.get("total", 0.0))
                    s_tot = speedup(tb, tc)

                    b03 = float(base.get(PHASE_03, 0.0))
                    c02 = float(tca.get(PHASE_02, 0.0))
                    c03 = float(tca.get(PHASE_03, 0.0))
                    c07 = float(tca.get(PHASE_07, 0.0))
                    csum = c02 + c03 + c07
                    s_cmp = speedup(b03, csum)

                    circuit_tag = f"{circ}\_q{q}\_d{d}"
                    w.writerow([
                        circuit_tag,
                        f"{tb:.3f}", f"{tc:.3f}", s_tot,
                        f"{b03:.3f}", f"{c02:.3f}", f"{c03:.3f}", f"{c07:.3f}",
                        f"{csum:.3f}", s_cmp
                    ])
    print("✓ CSV saved:", csv_path)
    return csv_path

def _sum_dicts(dict_list):
    """对若干 {str: float} 按键求和，缺失键按 0 处理。"""
    acc = defaultdict(float)
    for d in dict_list:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            try:
                acc[k] += float(v)
            except (TypeError, ValueError):
                pass
    return dict(acc)

def save_to_csv_sumD():
    figdir = DEFAULT_FIGDIR
    outdir = OUTDIR_DEFAULT
    if not figdir.exists():
        raise SystemExit(f"目录不存在：{figdir}")

    # ----------------- 你可以在这里直接改筛选集合 -----------------
    circs = CIRCS
    depths = DEPTHS
    qubits = QUBITS

    csv_path = outdir / "save_metrics_qd_in_circuit_sumD.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        # circuit_tag 里已包含 q/d；其他列维持数值与 speedup
        w.writerow([
            "circuit_tag",
            "total_base", "total_tcache", "speedup_total",
            "base_03", "tc_02", "tc_03", "tc_07",
            "tc_sum_020307", "speedup_compile"
        ])
        for circ in circs:
            for q in qubits:
                base_list = []
                tca_list = []
                for d in depths:
                    base = load_sum(figdir, q, d, circ, "baseline")
                    tca = load_sum(figdir, q, d, circ, "tcache")
                    if not base and not tca:
                        continue
                    base_list.append(base)
                    tca_list.append(tca)
                base_sum = _sum_dicts(base_list)
                base_tca = _sum_dicts(tca_list)
                tb = float(base_sum.get("total", 0.0))
                tc = float(base_tca.get("total", 0.0))
                s_tot = speedup(tb, tc)

                b03 = float(base_sum.get(PHASE_03, 0.0))
                c02 = float(base_tca.get(PHASE_02, 0.0))
                c03 = float(base_tca.get(PHASE_03, 0.0))
                c07 = float(base_tca.get(PHASE_07, 0.0))
                csum = c02 + c03 + c07
                s_cmp = speedup(b03, csum)

                circuit_tag = f"{circ}\_q{q}\_allD"
                w.writerow([
                    circuit_tag,
                    f"{tb:.3f}", f"{tc:.3f}", s_tot,
                    f"{b03:.3f}", f"{c02:.3f}", f"{c03:.3f}", f"{c07:.3f}",
                    f"{csum:.3f}", s_cmp
                ])
    print("✓ CSV saved:", csv_path)
    return csv_path

def save_to_csv_sumQ():
    figdir = DEFAULT_FIGDIR
    outdir = OUTDIR_DEFAULT
    if not figdir.exists():
        raise SystemExit(f"目录不存在：{figdir}")

    # ----------------- 你可以在这里直接改筛选集合 -----------------
    circs = CIRCS
    depths = DEPTHS
    qubits = QUBITS

    csv_path = outdir / "save_metrics_qd_in_circuit_sumQ.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        # circuit_tag 里已包含 q/d；其他列维持数值与 speedup
        w.writerow([
            "circuit_tag",
            "total_base", "total_tcache", "speedup_total",
            "base_03", "tc_02", "tc_03", "tc_07",
            "tc_sum_020307", "speedup_compile"
        ])
        for circ in circs:
            for d in depths:
                base_list = []
                tca_list = []
                for q in qubits:
                    base = load_sum(figdir, q, d, circ, "baseline")
                    tca = load_sum(figdir, q, d, circ, "tcache")
                    if not base and not tca:
                        continue
                    base_list.append(base)
                    tca_list.append(tca)
                base_sum = _sum_dicts(base_list)
                base_tca = _sum_dicts(tca_list)
                tb = float(base_sum.get("total", 0.0))
                tc = float(base_tca.get("total", 0.0))
                s_tot = speedup(tb, tc)

                b03 = float(base_sum.get(PHASE_03, 0.0))
                c02 = float(base_tca.get(PHASE_02, 0.0))
                c03 = float(base_tca.get(PHASE_03, 0.0))
                c07 = float(base_tca.get(PHASE_07, 0.0))
                csum = c02 + c03 + c07
                s_cmp = speedup(b03, csum)

                circuit_tag = f"{circ}\_d{d}\_allQ"
                w.writerow([
                    circuit_tag,
                    f"{tb:.3f}", f"{tc:.3f}", s_tot,
                    f"{b03:.3f}", f"{c02:.3f}", f"{c03:.3f}", f"{c07:.3f}",
                    f"{csum:.3f}", s_cmp
                ])
    print("✓ CSV saved:", csv_path)
    return csv_path

def save_to_csv_sumQD():
    figdir = DEFAULT_FIGDIR
    outdir = OUTDIR_DEFAULT
    if not figdir.exists():
        raise SystemExit(f"目录不存在：{figdir}")

    # ----------------- 你可以在这里直接改筛选集合 -----------------
    circs = CIRCS
    depths = DEPTHS
    qubits = QUBITS

    csv_path = outdir / "save_metrics_qd_in_circuit_sumQD.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        # circuit_tag 里已包含 q/d；其他列维持数值与 speedup
        w.writerow([
            "circuit_tag",
            "total_base", "total_tcache", "speedup_total",
            "base_03", "tc_02", "tc_03", "tc_07",
            "tc_sum_020307", "speedup_compile"
        ])
        for circ in circs:
            base_list = []
            tca_list = []
            for d in depths:
                for q in qubits:
                    base = load_sum(figdir, q, d, circ, "baseline")
                    tca = load_sum(figdir, q, d, circ, "tcache")
                    if not base and not tca:
                        continue
                    base_list.append(base)
                    tca_list.append(tca)
            base_sum = _sum_dicts(base_list)
            base_tca = _sum_dicts(tca_list)
            tb = float(base_sum.get("total", 0.0))
            tc = float(base_tca.get("total", 0.0))
            s_tot = speedup(tb, tc)

            b03 = float(base_sum.get(PHASE_03, 0.0))
            c02 = float(base_tca.get(PHASE_02, 0.0))
            c03 = float(base_tca.get(PHASE_03, 0.0))
            c07 = float(base_tca.get(PHASE_07, 0.0))
            csum = c02 + c03 + c07
            s_cmp = speedup(b03, csum)

            circuit_tag = f"{circ}\_all\_QD"
            w.writerow([
                circuit_tag,
                f"{tb:.3f}", f"{tc:.3f}", s_tot,
                f"{b03:.3f}", f"{c02:.3f}", f"{c03:.3f}", f"{c07:.3f}",
                f"{csum:.3f}", s_cmp
            ])
    print("✓ CSV saved:", csv_path)
    return csv_path

def save_to_csv_sumAll():
    figdir = DEFAULT_FIGDIR
    outdir = OUTDIR_DEFAULT
    if not figdir.exists():
        raise SystemExit(f"目录不存在：{figdir}")

    # ----------------- 你可以在这里直接改筛选集合 -----------------
    circs = CIRCS
    depths = DEPTHS
    qubits = QUBITS

    csv_path = outdir / "save_metrics_qd_in_circuit_sumAll.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        # circuit_tag 里已包含 q/d；其他列维持数值与 speedup
        w.writerow([
            "circuit_tag",
            "total_base", "total_tcache", "speedup_total",
            "base_03", "tc_02", "tc_03", "tc_07",
            "tc_sum_020307", "speedup_compile"
        ])
        base_list = []
        tca_list = []
        for circ in circs:
            for d in depths:
                for q in qubits:
                    base = load_sum(figdir, q, d, circ, "baseline")
                    tca = load_sum(figdir, q, d, circ, "tcache")
                    if not base and not tca:
                        continue
                    base_list.append(base)
                    tca_list.append(tca)
        base_sum = _sum_dicts(base_list)
        base_tca = _sum_dicts(tca_list)
        tb = float(base_sum.get("total", 0.0))
        tc = float(base_tca.get("total", 0.0))
        s_tot = speedup(tb, tc)

        b03 = float(base_sum.get(PHASE_03, 0.0))
        c02 = float(base_tca.get(PHASE_02, 0.0))
        c03 = float(base_tca.get(PHASE_03, 0.0))
        c07 = float(base_tca.get(PHASE_07, 0.0))
        csum = c02 + c03 + c07
        s_cmp = speedup(b03, csum)

        circuit_tag = f"sumAll"
        w.writerow([
            circuit_tag,
            f"{tb:.3f}", f"{tc:.3f}", s_tot,
            f"{b03:.3f}", f"{c02:.3f}", f"{c03:.3f}", f"{c07:.3f}",
            f"{csum:.3f}", s_cmp
        ])
    print("✓ CSV saved:", csv_path)
    return csv_path


def csv_to_latex(csv_path, out_file_name):
    outdir = OUTDIR_DEFAULT
    # from csv to latex
    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    caption = "Cumulative latency and cache breakdown across all circuit–qubit–depth configurations."
    label = "tab:cache_all"
    lex_path = outdir / out_file_name
    with lex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\resizebox{{\\textwidth}}{{!}}{{\n")
        f.write(f"\\renewcommand{{\\arraystretch}}{{1.2}}\n")
        f.write("\\begin{tabular}{| l || r r c || r r r r r c |}\n\\hline\n")
        f.write("Circuit & E2E (Base) & E2E (TC) & \\textbf{E2E Speedup} "
                "& Transpile (B) & Cache Search (TC) & Transpile (TC) & Cache Write (TC) "
                "& TC Sum $^\dagger$ & \\textbf{Compile Speedup} \\\\\n")
        f.write("\\hline\n")
        f.write("\\hline\n")
        for row in rows:
            circ = row["circuit_tag"]
            tb   = row["total_base"]
            tc   = row["total_tcache"]
            st   = row["speedup_total"].replace("x","\\times") if row["speedup_total"]!="-" else "--"
            b03  = row["base_03"]
            c02  = row["tc_02"]
            c03  = row["tc_03"]
            c07  = row["tc_07"]
            csum = row["tc_sum_020307"]
            sc   = row["speedup_compile"]
            f.write(f"{circ} & ${tb}$ & ${tc}$ & $\\textbf{{{st}}}$$\\times$ & ${b03}$ & ${c02}$ & ${c03}$ & ${c07}$ & ${csum}$ & $\\textbf{{{sc}}}$$\\times$ \\\\\n")
        f.write("\\hline\n\\end{tabular}\n}\n\\end{table*}\n")
    print("✓ LaTeX saved:", lex_path)

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    csv_path = save_to_csv_all()
    csv_to_latex(csv_path, "cache_table_all.tex")

    csv_path_sumD = save_to_csv_sumD()
    csv_to_latex(csv_path_sumD, "cache_table_sumD.tex")
    #
    csv_path_sumQ = save_to_csv_sumQ()
    csv_to_latex(csv_path_sumQ, "cache_table_sumQ.tex")
    #
    csv_path_sumQD = save_to_csv_sumQD()
    csv_to_latex(csv_path_sumQD, "cache_table_sumQD.tex")
    #
    csv_path_sumAll = save_to_csv_sumAll()
    csv_to_latex(csv_path_sumAll, "cache_table_sumAll.tex")

if __name__ == "__main__":
    main()
