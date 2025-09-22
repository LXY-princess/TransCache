# monitor_calibration.py
import os, time, csv, math, datetime as dt
from qiskit_ibm_runtime import QiskitRuntimeService
from statistics import median

TOKEN = os.environ["IBM_QUANTUM_TOKEN"]
INSTANCE = os.environ["IBM_QUANTUM_INSTANCE"]  # 如: 'ibm-q/open/main' 或实例 CRN
BACKEND_NAME = os.environ.get("IBM_BACKEND", "")  # 可留空，自动选一个

service = QiskitRuntimeService(token=TOKEN, instance=INSTANCE)

def pick_backend():
    if BACKEND_NAME:
        return service.backend(BACKEND_NAME)
    # 筛选：在线、可运行、物理机
    cands = [b for b in service.backends() if getattr(b, "num_qubits", 0) >= 5 and not b.configuration().simulator and b.status().operational]
    # 简单按排队长度排序
    cands.sort(key=lambda b: b.status().pending_jobs)
    if not cands:
        raise RuntimeError("No suitable backend found for your instance.")
    return cands[0]

backend = pick_backend()
print("Using backend:", backend.name)

CSV = "calib_log.csv"
header = [
    "ts_utc","backend","last_update_date_iso","n_qubits",
    "T1_median_us","T2_median_us",
    "readout_err_mean","sx_gate_err_median","cx_gate_err_median",
    "cx_gate_err_min","cx_gate_err_max"
]

if not os.path.exists(CSV):
    with open(CSV,"w",newline="") as f: csv.writer(f).writerow(header)

def get_value(params, name):
    for p in params:
        if getattr(p, "name", "") == name:
            return p.value
    return None

def snapshot():
    props = backend.properties()   # 不消耗量子时间
    n = backend.configuration().n_qubits
    # T1/T2
    T1 = []; T2 = []; readout = []
    for q in range(n):
        qs = props.qubits[q]
        t1 = get_value(qs,"T1"); t2 = get_value(qs,"T2")
        ro = get_value(qs,"readout_error")
        if t1: T1.append(t1);
        if t2: T2.append(t2);
        if ro is not None: readout.append(ro)
    # 门误差
    sx_err = []
    cx_errs = []
    for g in props.gates:
        if g.gate in ("sx","x"):
            ge = get_value(g.parameters,"gate_error")
            if ge is not None: sx_err.append(ge)
        if g.gate == "cx":
            ge = get_value(g.parameters,"gate_error")
            if ge is not None: cx_errs.append(ge)
    row = [
        dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
        backend.name,
        (props.last_update_date.isoformat() if hasattr(props.last_update_date,"isoformat") else str(props.last_update_date)),
        n,
        round(1e6*median(T1),2) if T1 else None,
        round(1e6*median(T2),2) if T2 else None,
        round(sum(readout)/len(readout),6) if readout else None,
        round(median(sx_err),6) if sx_err else None,
        round(median(cx_errs),6) if cx_errs else None,
        round(min(cx_errs),6) if cx_errs else None,
        round(max(cx_errs),6) if cx_errs else None,
    ]
    with open(CSV,"a",newline="") as f: csv.writer(f).writerow(row)
    print("logged:", row)

if __name__ == "__main__":
    snapshot()
