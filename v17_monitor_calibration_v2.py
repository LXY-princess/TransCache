import os, csv, re, json, hashlib, pathlib, datetime as dt
from statistics import median
from qiskit_ibm_runtime import QiskitRuntimeService

VNUM = 17
ROOT = pathlib.Path("./figs") / f"v{VNUM}_v2" / "backends"
ROOT.mkdir(parents=True, exist_ok=True)

INCLUDE   = {s.strip() for s in os.environ.get("BACKEND_INCLUDE","").split(",") if s.strip()}
EXCLUDE   = {s.strip() for s in os.environ.get("BACKEND_EXCLUDE","").split(",") if s.strip()}
TWOQ_PREF = [s.strip() for s in os.environ.get("TWOQ_PRIORITY","ecr,cx,cz,iswap").split(",") if s.strip()]

svc = QiskitRuntimeService()

HEADER = [
  # 时间/设备
  "ts_utc","backend","last_update_date_iso",
  # 规模/状态/队列
  "n_qubits","operational","status_msg","pending_jobs",
  # 物理时标
  "dt","sx_length_med","twoq_length_med",
  # 误差（1Q/2Q/读出/相干）
  "T1_median_us","T2_median_us","readout_err_mean","sx_gate_err_median",
  "twoq_gate_used","twoq_gate_err_median","twoq_gate_err_min","twoq_gate_err_max",
  # 重大变更兜底键
  "basis_hash","coupling_hash",
  # 采集错误
  "error"
]

def sanitize(name:str)->str: return re.sub(r"[^A-Za-z0-9_]+","_",name).strip("_")

def ensure_csv(path: pathlib.Path):
    if not path.exists():
        with open(path,"w",newline="") as f: csv.writer(f).writerow(HEADER)

def get_param(params, name):
    for p in params:
        if getattr(p,"name","")==name:
            return p.value
    return None

def basis_hash(cfg)->str:
    try:
        basis = sorted(list(getattr(cfg,"basis_gates",[]) or []))
        return hashlib.sha1(json.dumps(basis).encode()).hexdigest()[:12]
    except Exception: return ""

def coupling_hash(cfg)->str:
    try:
        cm = getattr(cfg,"coupling_map",None)
        if cm is None: return ""
        # 统一排序/格式
        edges = sorted([tuple(map(int,e)) for e in cm])
        return hashlib.sha1(json.dumps(edges).encode()).hexdigest()[:12]
    except Exception: return ""

def choose_twoq(props):
    # 优先从 TWOQ_PREF 中找
    by_name = {}
    for g in props.gates: by_name.setdefault(g.gate, []).append(g)
    for name in TWOQ_PREF:
        if name in by_name:
            errs = [get_param(g.parameters,"gate_error") for g in by_name[name]]
            errs = [e for e in errs if e is not None]
            if errs: return name, errs
    # 兜底：凡 qubits 长度为 2 的都算
    errs, alt = [], None
    for g in props.gates:
        try:
            if hasattr(g,"qubits") and len(g.qubits)==2:
                ge = get_param(g.parameters,"gate_error")
                if ge is not None:
                    errs.append(ge); alt = g.gate
        except: pass
    return (alt, errs) if errs else (None, [])

def snapshot_row(b):
    try:
        st  = b.status()
        cfg = b.configuration()
        props = b.properties()  # 控制面，不计费

        # 队列/状态
        operational = getattr(st,"operational",None)
        status_msg  = getattr(st,"status_msg","")
        pending     = getattr(st,"pending_jobs",None)

        # 规模/时标
        n = getattr(cfg,"n_qubits",None)
        dt_val = getattr(cfg,"dt",None)

        # T1/T2/读出
        T1=[]; T2=[]; read=[]
        for q in range(n or 0):
            qs = props.qubits[q]
            t1 = get_param(qs,"T1"); t2 = get_param(qs,"T2"); ro = get_param(qs,"readout_error")
            if t1: T1.append(t1)
            if t2: T2.append(t2)
            if ro is not None: read.append(ro)

        # 1Q 门误差与时长
        sx_err=[]; sx_len=[]
        for g in props.gates:
            if g.gate in ("sx","x"):
                ge = get_param(g.parameters,"gate_error")
                gl = get_param(g.parameters,"gate_length") or get_param(g.parameters,"duration")
                if ge is not None: sx_err.append(ge)
                if gl is not None: sx_len.append(gl)

        # 2Q 门误差与时长（按首选门型）
        twoq_name, twoq_errs = choose_twoq(props)
        twoq_len=[]
        if twoq_name:
            for g in props.gates:
                if g.gate==twoq_name:
                    gl = get_param(g.parameters,"gate_length") or get_param(g.parameters,"duration")
                    if gl is not None: twoq_len.append(gl)
        # # 测量时长
        # meas_len=[]
        # for g in props.gates:
        #     if g.gate in ("measure","measure_ro","readout"):
        #         gl = get_param(g.parameters,"gate_length") or get_param(g.parameters,"duration")
        #         if gl is not None: meas_len.append(gl)

        row = [
          dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
          b.name,
          (props.last_update_date.isoformat() if hasattr(props.last_update_date,"isoformat") else str(props.last_update_date)),
          n, operational, status_msg, pending,
          dt_val,
          (round(median(sx_len),9) if sx_len else None),
          (round(median(twoq_len),9) if twoq_len else None),
          # (round(median(meas_len),9) if meas_len else None),
          (round(1e6*median(T1),2) if T1 else None),
          (round(1e6*median(T2),2) if T2 else None),
          (round(sum(read)/len(read),6) if read else None),
          (round(median(sx_err),6) if sx_err else None),
          (twoq_name or ""),
          (round(median(twoq_errs),6) if twoq_errs else None),
          (round(min(twoq_errs),6) if twoq_errs else None),
          (round(max(twoq_errs),6) if twoq_errs else None),
          basis_hash(cfg), coupling_hash(cfg),
          ""
        ]
        return row
    except Exception as e:
        return [
          dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
          getattr(b,"name","unknown"), None,
          None, None, "", None,
          None, None, None,
          None, None, None, None,
          "", None, None, None,
          "", "",
          f"{type(e).__name__}: {e}"
        ]

def run_once():
    backs=[]
    for b in svc.backends():
        try:
            if b.configuration().simulator: continue
            if INCLUDE and b.name not in INCLUDE: continue
            if b.name in EXCLUDE: continue
            backs.append(b)
        except: pass
    backs.sort(key=lambda x: getattr(x.status(),"pending_jobs",0))

    for b in backs:
        f = ROOT / (sanitize(b.name)+".csv"); ensure_csv(f)
        row = snapshot_row(b)
        with open(f,"a",newline="") as fp: csv.writer(fp).writerow(row)
        print("logged", b.name, "| pending:", row[6], "| twoq:", row[15], "| err:", row[-1])

if __name__=="__main__":
    run_once()

