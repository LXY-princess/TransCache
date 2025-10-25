#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime
from pathlib import Path
import argparse

def parse_iso(ts: str) -> datetime:
    """
    Robust ISO8601 parser for strings like '2025-10-22 20:47:01.238385+00:00'
    datetime.fromisoformat handles this format directly.
    """
    return datetime.fromisoformat(ts)

def main():
    ap = argparse.ArgumentParser(
        description="Sum over entries of (execT + (finished - running)) in seconds."
    )
    ap.add_argument(
        "json_path",
        nargs="?",
        default="figs/v22_qaoa_q11d1_s32/repeated_v11/QAOA-3reg_q11_d1_N5/events/TransCache.json",
        # default="figs/v22_qaoa_q11d1_s32/repeated_v11/QAOA-3reg_q11_d1_N5/events/FullComp.json",
        help="Path to the JSON file (default: /mnt/data/FullComp.json)",
    )
    args = ap.parse_args()

    p = Path(args.json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))

    total = 0.0
    total_trans = 0.0
    rows = []

    for i, rec in enumerate(data):
        # if i == 3:
        #     continue
        # Only process entries that have required fields
        if not all(k in rec for k in ("transT", "running", "finished")):
            continue

        transT = float(rec["transT"])
        running = parse_iso(rec["running"])
        finished = parse_iso(rec["finished"])
        wall = (finished - running).total_seconds()

        value = transT + wall
        total += value
        total_trans += transT

        rows.append({
            "idx": i,
            "label": rec.get("label", ""),
            "transT_sec": transT,
            "finished_minus_running_sec": wall,
            "sum_sec": value,
        })

    # Pretty print results
    print("Per-entry details:")
    for r in rows:
        print(
            f"- #{r['idx']:02d} {r['label']}: "
            f"transT={r['transT_sec']:.6f}s, "
            f"(finished-running)={r['finished_minus_running_sec']:.6f}s, "
            f"sum={r['sum_sec']:.6f}s"
        )
    print("\nTOTAL seconds =", f"{total:.8f}")
    print("\nTOTAL trans seconds =", f"{total_trans:.8f}")
    print("TOTAL minutes =", f"{total/60.0:.6f}")

    totale2e = total_trans + 5 * 4.058037
    print("\nTOTAL e2e seconds =", f"{totale2e:.8f}")

if __name__ == "__main__":
    main()
