# -*- coding: utf-8 -*-
"""
make_origin_tables.py
---------------------
Load a runner summary CSV and export Origin-friendly tables for grouped bar plots.

Inputs
------
--in_csv:  path to repeated_v11_multiq_summary_long.csv
--out_dir: directory to write outputs (default: same dir as input)

Outputs
-------
origin_long.csv
    Columns: Circuit, Qubits, Depth, N, Method, E2E_Latency_s, Speedup_vs_Full
origin_wide_latency.csv
    Index (first columns): Circuit, Qubits
    One column per Method: latency (seconds)
origin_wide_speedup.csv
    Index (first columns): Circuit, Qubits
    One column per Method: speedup vs Full (Full_latency / method_latency)

Notes
-----
- Baseline is method == "Full". If missing for a (Circuit, Qubits, Depth, N),
  speedup will be NaN for that group.
- Methods are kept in a stable order if detected among known names; otherwise
  alphabetical order is used.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from v18_core import (
    LOAD_ROOT, PLOT_DIR)

def load_and_process(in_csv: Path):
    df = pd.read_csv(in_csv)
    # Normalize column names
    rename_map = {
        "circuit": "Circuit",
        "q": "Qubits",
        "d": "Depth",
        "N": "N",
        "method": "Method",
        "e2e_latency": "E2E_Latency_s",
        "final_cache_size": "FinalCache",
        "final_hitrate": "FinalHitRate"
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # Keep only necessary columns for Origin
    cols_keep = ["Circuit", "Qubits", "Depth", "N", "Method", "E2E_Latency_s"]
    df = df[cols_keep].copy()
    # Baseline: Full
    base = (
        df[df["Method"] == "Full"]
        .groupby(["Circuit", "Qubits", "Depth", "N"], as_index=False)["E2E_Latency_s"]
        .min()
        .rename(columns={"E2E_Latency_s": "Baseline_Full_s"})
    )
    df = df.merge(base, on=["Circuit", "Qubits", "Depth", "N"], how="left")
    df["Speedup_vs_Full"] = df["Baseline_Full_s"] / df["E2E_Latency_s"]
    long_out = df[["Circuit", "Qubits", "Depth", "N", "Method", "E2E_Latency_s", "Speedup_vs_Full"]].copy()
    return long_out

def export_wide_tables(long_out: pd.DataFrame):
    pref_order = ["FS+Pre+ttl+SE+ema", "FS+Pre", "FS", "PR", "Full"]
    methods_present = list(long_out["Method"].unique())
    ordered_methods = [m for m in pref_order if m in methods_present] + \
                      [m for m in sorted(methods_present) if m not in pref_order]
    wide_lat = long_out.pivot_table(
        index=["Circuit", "Qubits"], columns="Method", values="E2E_Latency_s", aggfunc="mean"
    )
    wide_lat = wide_lat.reindex(columns=ordered_methods)
    wide_lat = wide_lat.reset_index()
    wide_spd = long_out.pivot_table(
        index=["Circuit", "Qubits"], columns="Method", values="Speedup_vs_Full", aggfunc="mean"
    )
    wide_spd = wide_spd.reindex(columns=ordered_methods)
    wide_spd = wide_spd.reset_index()
    return wide_lat, wide_spd, ordered_methods

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, default=LOAD_ROOT / "repeated_v11_multiq/summaries/repeated_v11_multiq_summary_long.csv",
                    help="Path to repeated_v11_multiq_summary_long.csv")
    ap.add_argument("--out_dir", type=str, default=PLOT_DIR,
                    help="Output directory; default same as input dir")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir) if args.out_dir else in_csv.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    long_out = load_and_process(in_csv)
    wide_lat, wide_spd, ordered_methods = export_wide_tables(long_out)

    long_path = out_dir / "origin_long.csv"
    wide_lat_path = out_dir / "origin_wide_latency.csv"
    wide_spd_path = out_dir / "origin_wide_speedup.csv"
    long_out.to_csv(long_path, index=False)
    wide_lat.to_csv(wide_lat_path, index=False)
    wide_spd.to_csv(wide_spd_path, index=False)

    readme = out_dir / "origin_tables_README.txt"
    readme.write_text(
        "Files:\n"
        f"- {long_path.name}: tidy table (Circuit, Qubits, Depth, N, Method, E2E_Latency_s, Speedup_vs_Full)\n"
        f"- {wide_lat_path.name}: wide latency by Method (index: Circuit, Qubits)\n"
        f"- {wide_spd_path.name}: wide speedup by Method (index: Circuit, Qubits)\n\n"
        f"Method order used: {ordered_methods}\n"
        "Speedup_vs_Full = Full_latency / method_latency; Full baseline equals 1.0.\n"
    )
    print(f"[OK] Wrote:\n  - {long_path}\n  - {wide_lat_path}\n  - {wide_spd_path}\n  - {readme}")

if __name__ == "__main__":
    main()
