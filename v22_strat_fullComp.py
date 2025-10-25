# strategy_baseline.py
from typing import Any, Dict, List
from collections import defaultdict
from v22_core import run_once_nocache_ibm, label_of
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


def run_strategy(workload: List[Dict[str,Any]], shots: int = 256, include_exec: bool = True) -> Dict[str, List[Dict[str,Any]]]:
    """
    Baseline: no prewarm, no cache. Return {"events": [...], "metrics": {...}}
    """
    events = []
    t = 0.0

    """Prepare ibm backend"""
    svc = QiskitRuntimeService()
    bkd_name = "ibm_torino"
    backend = svc.backend(bkd_name)
    sampler = Sampler(mode=backend)

    for it in workload:
        meta = run_once_nocache_ibm(it["maker_run"], shots=shots, include_exec=include_exec, backend=backend, sampler=sampler)
        # meta = run_once_nocache_ibm(it["maker_run"], shots=shots, include_exec=include_exec)
        dur = float(meta["compile_sec"]) + float(meta["exec_sec"])
        events.append({"kind":"run","label": label_of(it["name"], it["q"], it["d"]),
                       "start": t, "dur": dur,
                       "transT":float(meta["compile_sec"]),
                       "execT":float(meta["exec_sec"]),
                       "created": meta["created"],
                       "running": meta["running"],
                       "finished": meta["finished"],
                       "runtime": (meta["finished"]-meta["running"]).total_seconds(),
                       })
        t += dur
    metrics = {}  # Baseline无命中率
    return {"events": events, "metrics": metrics}
