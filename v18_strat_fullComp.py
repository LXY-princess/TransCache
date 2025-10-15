# strategy_baseline.py
from typing import Any, Dict, List
from collections import defaultdict
from v18_core import run_once_nocache, label_of

def run_strategy(workload: List[Dict[str,Any]], shots: int = 256, include_exec: bool = True) -> Dict[str, List[Dict[str,Any]]]:
    """
    Baseline: no prewarm, no cache. Return {"events": [...], "metrics": {...}}
    """
    events = []
    t = 0.0
    for it in workload:
        meta = run_once_nocache(it["maker_run"], shots=shots, include_exec=include_exec)
        dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        events.append({"kind":"run","label": label_of(it["name"], it["q"], it["d"]),
                       "start": t, "dur": dur})
        t += dur
    metrics = {}  # Baseline无命中率
    return {"events": events, "metrics": metrics}
