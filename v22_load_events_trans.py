import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Iterable, Union

# === 配置区 ===
# 你的数据根目录（修正为你实际路径）
LOAD_ROOT = Path(r"figs/v22/repeated_v11/QAOA_q16_d4_N5/events")
# 需要处理的四种方法（也决定了 CSV 的行顺序）
METHOD_NAMES = ["FullComp", "Braket", "CCache", "TransCache"]
# 只导出前 N 次（例如 4）；若为 None 则导出文件里的全部次数
NUM_RUNS: Union[int, None] = None
# 输出文件名
OUTPUT_CSV = LOAD_ROOT/Path("method_transT.csv")


def _extract_runs(data: Any) -> List[Dict[str, Any]]:
    """
    允许三种结构：
      1) 顶层即为 list[dict]（每个元素是一轮）
      2) {"events": list[dict]}
      3) {"runs": list[dict]}
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("events", "runs"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    raise ValueError("Unrecognized JSON structure: expect list or dict with 'events'/'runs'.")


def read_transTs(json_path: Path) -> List[float]:
    """
    从单个 JSON 文件读取按顺序的 transT 值：
      transT_i = event.get('transT', 0.0) + event.get('bindT', 0.0)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = _extract_runs(data)

    vals: List[float] = []
    for ev in runs:
        if not isinstance(ev, dict):
            continue
        t = float(ev.get("transT", 0.0)) + float(ev.get("bindT", 0.0))
        vals.append(t)
    return vals


def main() -> None:
    # 读取每个方法的 transT 序列
    all_vals: Dict[str, List[float]] = {}
    max_len = 0

    for name in METHOD_NAMES:
        fp = LOAD_ROOT / f"{name}.json"
        if not fp.exists():
            print(f"[WARN] Missing file: {fp}")
            all_vals[name] = []
            continue

        vals = read_transTs(fp)
        if NUM_RUNS is not None:
            vals = vals[:NUM_RUNS]
        all_vals[name] = vals
        max_len = max(max_len, len(vals))

    # 如果用户指定了 NUM_RUNS，则列数按 NUM_RUNS，否则按各文件中最大次数
    num_cols = NUM_RUNS if NUM_RUNS is not None else max_len

    # 生成表头：method, transT1, transT2, ...
    header = ["method"] + [f"transT{i+1}" for i in range(num_cols)]

    # 写 CSV（4 行，对应四个方法；不足列数用空字符串填充）
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in METHOD_NAMES:
            vals = all_vals.get(name, [])
            row = [name] + [f"{v:.10f}" for v in vals] + [""] * (num_cols - len(vals))
            writer.writerow(row)

    print(f"[OK] Wrote CSV: {OUTPUT_CSV.resolve()} "
          f"({len(METHOD_NAMES)} rows × {len(header)} columns)")


if __name__ == "__main__":
    main()
