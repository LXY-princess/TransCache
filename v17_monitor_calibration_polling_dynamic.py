import os, csv, re, json, hashlib, pathlib, datetime as dt, time, math
from statistics import median
from qiskit_ibm_runtime import QiskitRuntimeService

# ========= 基本设置 =========
VNUM = 17
ROOT = pathlib.Path("./figs") / f"v{VNUM}_polling" / "backends"
ROOT.mkdir(parents=True, exist_ok=True)

# 轮询频率（可用环境变量覆盖）
DEFAULT_INTERVAL_SEC = int(os.environ.get("DEFAULT_INTERVAL_SEC", "1800"))  # 30 min
HIGH_INTERVAL_SEC    = int(os.environ.get("HIGH_INTERVAL_SEC", "300"))      # 5  min
HIGH_HOLD_SEC        = int(os.environ.get("HIGH_HOLD_SEC", "7200"))         # 2  h

# 触发阈值（可用环境变量覆盖）
TWOQ_REL       = float(os.environ.get("TWOQ_REL", "0.25"))
SX_REL         = float(os.environ.get("SX_REL", "0.25"))
READOUT_ABS    = float(os.environ.get("READOUT_ABS", "0.01"))
T2_REL         = float(os.environ.get("T2_REL", "0.20"))
PENDING_SPIKE  = int(os.environ.get("PENDING_SPIKE", "200"))  # 队列长度绝对阈值

# 后端筛选 & 2Q 门优先级
INCLUDE   = {s.strip() for s in os.environ.get("BACKEND_INCLUDE","").split(",") if s.strip()}
EXCLUDE   = {s.strip() for s in os.environ.get("BACKEND_EXCLUDE","").split(",") if s.strip()}
TWOQ_PREF = [s.strip() for s in os.environ.get("TWOQ_PRIORITY","ecr,cx,cz,iswap").split(",") if s.strip()]

# 连接服务（延用你原来的无参构造；若你已设置 IBM_QUANTUM_TOKEN/INSTANCE，也可自动生效）
svc = QiskitRuntimeService()

# ========= 写入表头（沿用你原版的列次序） =========
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

def sanitize(name:str)->str:
    return re.sub(r"[^A-Za-z0-9_]+","_",name).strip("_")

def ensure_csv(path: pathlib.Path):
    if not path.exists():
        with open(path,"w",newline="") as f:
            csv.writer(f).writerow(HEADER)

def get_param(params, name):
    for p in params:
        if getattr(p,"name","")==name:
            return p.value
    return None

def basis_hash(cfg)->str:
    try:
        basis = sorted(list(getattr(cfg,"basis_gates",[]) or []))
        return hashlib.sha1(json.dumps(basis).encode()).hexdigest()[:12]
    except Exception:
        return ""

def coupling_hash(cfg)->str:
    try:
        cm = getattr(cfg,"coupling_map",None)
        if cm is None: return ""
        edges = sorted([tuple(map(int,e)) for e in cm])
        return hashlib.sha1(json.dumps(edges).encode()).hexdigest()[:12]
    except Exception:
        return ""

def choose_twoq(props):
    # 优先从 TWOQ_PREF 中找
    by_name = {}
    for g in props.gates:
        by_name.setdefault(g.gate, []).append(g)
    for name in TWOQ_PREF:
        if name in by_name:
            errs = [get_param(g.parameters,"gate_error") for g in by_name[name]]
            errs = [e for e in errs if e is not None]
            if errs:
                return name, errs
    # 兜底：凡 qubits 长度为 2 的都算
    errs, alt = [], None
    for g in props.gates:
        try:
            if hasattr(g,"qubits") and len(g.qubits)==2:
                ge = get_param(g.parameters,"gate_error")
                if ge is not None:
                    errs.append(ge); alt = g.gate
        except:
            pass
    return (alt, errs) if errs else (None, [])

def snapshot_row(b):
    """抓取单个后端的一次快照 → list（与 HEADER 对齐）"""
    try:
        st  = b.status()
        cfg = b.configuration()
        props = b.properties()  # 控制面，不计费

        # 队列/状态
        operational = getattr(st,"operational",None)
        status_msg  = getattr(st,"status_msg","") or ""
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

        row = [
          dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
          b.name,
          (props.last_update_date.isoformat() if hasattr(props.last_update_date,"isoformat") else str(props.last_update_date)),
          n, operational, status_msg, pending,
          dt_val,
          (round(median(sx_len),9) if sx_len else None),
          (round(median(twoq_len),9) if twoq_len else None),
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

# ---------- 自适应触发判断 ----------
def row_list_to_dict(row_list):
    return {HEADER[i]: row_list[i] if i < len(row_list) else None for i in range(len(HEADER))}

def to_float(x):
    try:
        if x in (None,"","NaN","nan"): return math.nan
        return float(x)
    except Exception:
        return math.nan

def to_bool(x):
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in ("true","1","t","yes","y"): return True
    if s in ("false","0","f","no","n"): return False
    return False

def rel_change(a,b):
    if not (math.isfinite(a) and math.isfinite(b)) or a == 0:
        return math.inf if math.isfinite(b) and a == 0 else math.nan
    return abs(b-a)/abs(a)

def need_high_freq(prev: dict, curr: dict) -> bool:
    """是否需要提升采样频率"""
    if not prev or not curr:
        return False

    # 1) 校准/配置硬触发
    if prev.get("last_update_date_iso") != curr.get("last_update_date_iso"):
        return True
    if prev.get("twoq_gate_used","") != curr.get("twoq_gate_used",""):
        return True
    if str(prev.get("dt")) != str(curr.get("dt")):
        return True
    if prev.get("basis_hash","") != curr.get("basis_hash",""):
        return True
    if prev.get("coupling_hash","") != curr.get("coupling_hash",""):
        return True
    if to_bool(prev.get("operational")) != to_bool(curr.get("operational")):
        return True

    # 2) 队列拥堵触发（绝对阈值）
    try:
        if int(curr.get("pending_jobs") or 0) >= PENDING_SPIKE:
            return True
    except Exception:
        pass
    # 状态文案包含校准/维护关键词
    msg = (curr.get("status_msg") or "").lower()
    if any(k in msg for k in ("calib","mainten","offline","disable")):
        return True

    # 3) 误差显著变化触发（相对/绝对）
    sx_prev, sx_curr = to_float(prev.get("sx_gate_err_median")), to_float(curr.get("sx_gate_err_median"))
    tq_prev = to_float(prev.get("twoq_gate_err_median") or prev.get("cx_gate_err_median"))
    tq_curr = to_float(curr.get("twoq_gate_err_median") or curr.get("cx_gate_err_median"))
    ro_prev, ro_curr = to_float(prev.get("readout_err_mean")), to_float(curr.get("readout_err_mean"))
    t2_prev, t2_curr = to_float(prev.get("T2_median_us")), to_float(curr.get("T2_median_us"))

    if math.isfinite(tq_prev) and math.isfinite(tq_curr) and rel_change(tq_prev,tq_curr) >= TWOQ_REL:
        return True
    if math.isfinite(sx_prev) and math.isfinite(sx_curr) and rel_change(sx_prev,sx_curr) >= SX_REL:
        return True
    if math.isfinite(ro_prev) and math.isfinite(ro_curr) and abs(ro_curr-ro_prev) >= READOUT_ABS:
        return True
    if math.isfinite(t2_prev) and math.isfinite(t2_curr) and rel_change(t2_prev,t2_curr) >= T2_REL:
        return True

    return False

# ---------- 主轮询 ----------
def candidate_backends():
    backs=[]
    for b in svc.backends():
        try:
            if b.configuration().simulator:
                continue
            if INCLUDE and b.name not in INCLUDE:
                continue
            if b.name in EXCLUDE:
                continue
            backs.append(b)
        except Exception:
            pass
    # 仅美观：按队列长度排序
    backs.sort(key=lambda x: getattr(x.status(),"pending_jobs",0))
    return backs

def read_last_row_dict(csv_path: pathlib.Path):
    if not csv_path.exists():
        return None
    last = None
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            last = row
    return last

# 记录“高频维持到何时”：backend_name -> timestamp
HIGH_UNTIL = {}

def poll_once():
    """采样一次所有后端，按需决定是否进入高频"""
    backs = candidate_backends()
    any_trigger = False
    for b in backs:
        csv_path = ROOT / (sanitize(b.name)+".csv")
        ensure_csv(csv_path)

        prev = read_last_row_dict(csv_path)  # 跨进程/重启也有记忆
        row_list = snapshot_row(b)
        # 追加到 CSV
        with open(csv_path,"a",newline="") as fp:
            csv.writer(fp).writerow(row_list)
        curr = row_list_to_dict(row_list)

        # 判定是否需要升频
        if need_high_freq(prev, curr):
            HIGH_UNTIL[b.name] = time.time() + HIGH_HOLD_SEC
            any_trigger = True
            print(f"[{curr['ts_utc']}] TRIGGER high-freq: {b.name}  (hold {HIGH_HOLD_SEC//60} min)")
        else:
            # 若已在高频，继续观察是否到期
            if b.name in HIGH_UNTIL and time.time() >= HIGH_UNTIL[b.name]:
                del HIGH_UNTIL[b.name]
        # 打印关键信息
        print("logged", b.name,
              "| pending:", curr.get("pending_jobs"),
              "| twoq:", curr.get("twoq_gate_used"),
              "| twoq_med:", curr.get("twoq_gate_err_median"),
              "| sx_med:", curr.get("sx_gate_err_median"),
              "| err:", curr.get("error"))
    return any_trigger

def adaptive_loop():
    try:
        while True:
            start = time.time()
            _ = poll_once()
            # 根据是否有未到期的高频后端，决定下一次间隔
            if any(time.time() < t for t in HIGH_UNTIL.values()):
                interval = min(DEFAULT_INTERVAL_SEC, HIGH_INTERVAL_SEC)
            else:
                interval = DEFAULT_INTERVAL_SEC
            elapsed = time.time() - start
            sleep_s = max(0, interval - elapsed)
            print(f"Next poll in {int(sleep_s)}s  (active high backends: {len(HIGH_UNTIL)})")
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("Stopped by user.")

if __name__=="__main__":
    adaptive_loop()
