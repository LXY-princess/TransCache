#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import pandas
from pathlib import Path
VNUM = 18
LOAD_ROOT = Path("./figs")/f"v{VNUM}/scaling/N150/events"
LOAD_ROOT_M = Path("./figs")/f"v{VNUM}/scaling/N150/metrics"
file_name = "TransCache" #"FullComp" "CCache" "Braket" "TransCache"
def load_events():
    in_path = LOAD_ROOT/f"{file_name}.json"
    out_path = LOAD_ROOT/f"{file_name}_flat.csv"

    # 读取 JSON（要求是列表，每个元素是包含指定键的对象）
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 扁平化为行
    rows = []
    for i, item in enumerate(data, start=1):
        rows.append({
            "index": i,
            "label": item.get("label", ""),
            "transT": item.get("transT", 0),
            "loadT": item.get("loadT", 0),
            "bindT": item.get("bindT", 0),
            "execT": item.get("execT", 0),
        })

    # 写 CSV
    fieldnames = ["index", "label", "transT", "loadT", "bindT", "execT"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")

def load_metric():
    in_path = LOAD_ROOT_M/f"{file_name}.json"
    out_path = LOAD_ROOT_M/f"{file_name}_flat.csv"

    # 读取 JSON（要求是列表，每个元素是包含指定键的对象）
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cache_series = data.get("cache_size_series")
    # 扁平化为行
    rows = []
    for i, item in enumerate(cache_series, start=1):
        if i%2 == 0:
            rows.append({
                "index": i,
                "size": item.get("size", ""),
            })

    # 写 CSV
    fieldnames = ["index", "size"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")

if __name__ == "__main__":
    # load_events()
    load_metric()
