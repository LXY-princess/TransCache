# v19_monitor_calibration_split_csv.py
# 对账号可见的所有免费硬件后端（非模拟器且 operational）各写一个 CSV：
# figs/v17/backends/<backend_sanitized>.csv
# 仅调用 backend.properties()（控制面），不会提交量子作业/计费。

import os
import csv
import re
import pathlib
import datetime as dt
from statistics import median
from qiskit_ibm_runtime import QiskitRuntimeService

VNUM = 17
ROOT = pathlib.Path("./figs") / f"v{VNUM}" / "backends"
ROOT.mkdir(parents=True, exist_ok=True)

service = QiskitRuntimeService()

# 可选：白/黑名单（逗号分隔）
INCLUDE = {s.strip() for s in os.environ.get("BACKEND_INCLUDE", "").split(",") if s.strip()}
EXCLUDE = {s.strip() for s in os.environ.get("BACKEND_EXCLUDE", "").split(",") if s.strip()}

# 两比特门优先顺序（可用环境变量覆盖，如 "cx,ecr"）
TWOQ_PRIORITY = [s.strip() for s in os.environ.get("TWOQ_PRIORITY", "ecr,cx,cz,iswap").split(",") if s.strip()]


# HEADER = [
#     "ts_utc","backend","last_update_date_iso","n_qubits",
#     "T1_median_us","T2_median_us",
#     "readout_err_mean","sx_gate_err_median","cx_gate_err_median",
#     "cx_gate_err_min","cx_gate_err_max","error"
# ]

HEADER = [
    "ts_utc","backend","last_update_date_iso","n_qubits",
    "T1_median_us","T2_median_us",
    "readout_err_mean","sx_gate_err_median",
    "twoq_gate_used",                   # 新增：使用到的两比特门类型
    "twoq_gate_err_median","twoq_gate_err_min","twoq_gate_err_max",
    "error"
]



def sanitize(name: str) -> str:
    """把后端名转为文件系统安全的短名（仅字母数字下划线）。"""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")

def get_value(params, name):
    for p in params:
        if getattr(p, "name", "") == name:
            return p.value
    return None

def candidate_backends():
    backs = []
    for b in service.backends():
        try:
            if b.configuration().simulator:
                continue
            if not b.status().operational:
                continue
            name = b.name
            if INCLUDE and name not in INCLUDE:
                continue
            if name in EXCLUDE:
                continue
            backs.append(b)
        except Exception:
            continue
    # 纯美观：按排队长度排序
    backs.sort(key=lambda x: getattr(x.status(), "pending_jobs", 0))
    return backs

def ensure_csv(file_path: pathlib.Path):
    if not file_path.exists():
        with open(file_path, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

def choose_twoq_gate(props):
    """返回 (gate_name, [gate_error列表])；找不到则 (None, [])."""
    # 先按优先顺序找
    gates_by_name = {}
    for g in props.gates:
        gates_by_name.setdefault(g.gate, []).append(g)
    for name in TWOQ_PRIORITY:
        if name in gates_by_name:
            errs = []
            for g in gates_by_name[name]:
                ge = get_value(g.parameters, "gate_error")
                if ge is not None:
                    errs.append(ge)
            if errs:
                return name, errs
    # 兜底：如果没有优先列表里的，但存在其他两比特门（qubits 参数长度为2）
    errs = []
    alt_name = None
    for g in props.gates:
        try:
            if hasattr(g, "qubits") and len(g.qubits) == 2:
                ge = get_value(g.parameters, "gate_error")
                if ge is not None:
                    errs.append(ge)
                    alt_name = g.gate
        except Exception:
            continue
    if errs:
        return alt_name, errs
    return None, []

def snapshot_row(b):
    try:
        props = b.properties()  # 控制面读取
        n = b.configuration().n_qubits

        # T1/T2/Readout
        T1 = []; T2 = []; readout = []
        for q in range(n):
            qs = props.qubits[q]
            t1 = get_value(qs, "T1")
            t2 = get_value(qs, "T2")
            ro = get_value(qs, "readout_error")
            if t1: T1.append(t1)
            if t2: T2.append(t2)
            if ro is not None: readout.append(ro)

        # 1Q 门（sx/x）误差
        sx_err = []
        for g in props.gates:
            if g.gate in ("sx","x"):
                ge = get_value(g.parameters, "gate_error")
                if ge is not None: sx_err.append(ge)

        # 2Q 门误差（自动识别门型）
        twoq_name, twoq_errs = choose_twoq_gate(props)

        row = [
            dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
            b.name,
            (props.last_update_date.isoformat()
             if hasattr(props.last_update_date, "isoformat")
             else str(props.last_update_date)),
            n,
            round(1e6*median(T1),2) if T1 else None,                  # s -> us
            round(1e6*median(T2),2) if T2 else None,                  # s -> us
            round(sum(readout)/len(readout),6) if readout else None,
            round(median(sx_err),6) if sx_err else None,
            twoq_name or "",
            (round(median(twoq_errs),6) if twoq_errs else None),
            (round(min(twoq_errs),6)    if twoq_errs else None),
            (round(max(twoq_errs),6)    if twoq_errs else None),
            ""
        ]
        return row
    except Exception as e:
        return [
            dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
            getattr(b, "name", "unknown"),
            None, None, None, None, None, None, "", None, None, None,
            f"{type(e).__name__}: {e}"
        ]

def main():
    backs = candidate_backends()
    if not backs:
        raise RuntimeError("没有发现可用的免费硬件后端（非模拟器且 operational）。")

    print("Monitoring", len(backs), "backends:")
    for b in backs:
        print(" -", b.name)

    for b in backs:
        fname = sanitize(b.name) + ".csv"
        fpath = ROOT / fname
        ensure_csv(fpath)
        row = snapshot_row(b)
        with open(fpath, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print("logged to", fpath.name, ":", row[:5], "...", (row[-1] or ""))

if __name__ == "__main__":
    main()
