# v18_strategy_param_reuse.py  (Braket-like: parameterize only RX/RY/RZ/U)
from typing import Any, Dict, List, Tuple
from collections import Counter
from numbers import Real
import time

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from v22_core import md5_qasm, label_of, record_arrival
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ---- 白名单：只把这些门视为“可参数化”的单比特旋转 ----
# 默认：RX/RY/RZ/UGate（含历史 U3）
ALLOWED_PARAM_GATES = {"rz","p","rx","ry","u","u3"}
# ALLOWED_PARAM_GATES = {"rx", "rz"}

# # 若你要更贴近 Rigetti/Braket 的原生参数化能力，可启用严格模式：仅 RX/RZ
# BRAKET_RIGETTI_STRICT = True
# if BRAKET_RIGETTI_STRICT:
#     ALLOWED_PARAM_GATES = {"rx", "rz"}
#     # ALLOWED_PARAM_GATES = {"rz","p","rx","ry","u","u3"}


def _gate_name(inst) -> str:
    # Qiskit Instruction/Operator 均有 .name；统一转小写
    n = getattr(inst, "name", None) or getattr(getattr(inst, "operation", None), "name", "")
    return str(n).lower()


def _build_param_template_and_bindings(
    qc_raw: QuantumCircuit
) -> Tuple[QuantumCircuit, List[Parameter], List[float]]:
    """
    仅对白名单门 (RX/RY/RZ/U/U3) 的“数值型参数”替换为新的 Parameter p{i}；
    其它门（哪怕带数值参数）一律保留为常数（不参数化）。
    返回 (模板电路, 我们新造的参数列表, 对应数值列表)。
    """
    nq = qc_raw.num_qubits
    nc = qc_raw.num_clbits
    templ = QuantumCircuit(nq, nc)

    params: List[Parameter] = []   # 仅记录“由数值替换生成”的参数
    values: List[float] = []
    pid = 0

    for inst, qargs, cargs in qc_raw.data:
        new_inst = inst.copy()
        gname = _gate_name(inst)
        if getattr(inst, "params", None):
            new_params = []
            for p in inst.params:
                # 只有当：门在白名单 且 参数是实数（角度/相位等） 才替换为 Parameter
                if (gname in ALLOWED_PARAM_GATES) and isinstance(p, Real) and not isinstance(p, bool):
                    par = Parameter(f"p{pid}"); pid += 1
                    params.append(par); values.append(float(p))
                    new_params.append(par)
                else:
                    # 非白名单门 或 非纯数值（已有 Parameter/表达式） => 原样保留
                    new_params.append(p)
            new_inst.params = new_params
        templ.append(new_inst, qargs, cargs)

    # 透传可能存在的硬件校准
    if getattr(qc_raw, "calibrations", None):
        for gate, cals in qc_raw.calibrations.items():
            for sig, sched in cals.items():
                templ.add_calibration(gate, sig, sched)

    return templ, params, values


def run_once_param_reuse(
    qc_func, template_cache: Dict[str, QuantumCircuit], shots: int = 256, include_exec: bool = True,
        sampler: Sampler = None, backend: Any = None, bk_name: str = None
) -> Dict[str, Any]:
    """
    仅对白名单门得到的“带参模板”做缓存复用；其余电路常规编译（不缓存）。
    """
    qc_raw = qc_func()

    # 构建模板 + 本次数值绑定
    templ, replaced_params, replaced_values = _build_param_template_and_bindings(qc_raw)
    templ_params = set(templ.parameters)
    replaced_params_set = set(replaced_params)

    # A) 模板无参数 => 非带参电路 => 每次常规编译，不缓存
    if len(templ_params) == 0:
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, backend=backend, optimization_level=3)
        compile_sec = time.perf_counter() - t0

        if include_exec:
            t1 = time.perf_counter()
            # job = sampler.run([qc_exec], shots=shots)
            svc = QiskitRuntimeService()
            job_id = "d3sk546jbaoc73cne5ug"  # 1. "d3sfk6k60rgc73aa7p50" 2. "d3sjqhkv6o9s73cs0pj0" 3."d3sjr2mjbaoc73cnds90" 4."d3sjr5s60rgc73aabmv0", 5."d3sk546jbaoc73cne5ug"
            job = svc.job(job_id)

            result = job.result()
            exec_sec = time.perf_counter() - t1
        else:
            exec_sec = 0

        return {
            "key": None,
            "cache_hit": False,
            "compile_sec": float(compile_sec),
            "bind_sec": 0.0,
            "exec_sec": float(exec_sec),
            "n_qubits": qc_raw.num_qubits,
            "depth_in": qc_raw.depth(),
            "depthT": qc_exec.depth(),
            "parametric": False,
            "stratPath":"A"
        }

    # B) 模板含参，但存在“原始 Parameter”（非我们替换出的）=> 本次常规编译，不缓存
    extra_params = templ_params - replaced_params_set
    if len(extra_params) > 0:
        t0 = time.perf_counter()
        qc_exec = transpile(qc_raw, backend=backend, optimization_level=3)
        compile_sec = time.perf_counter() - t0

        if include_exec:
            t1 = time.perf_counter()
            # job = sampler.run([qc_exec], shots=shots)
            svc = QiskitRuntimeService()
            job_id = "d3sk546jbaoc73cne5ug"  # 1. "d3sfk6k60rgc73aa7p50" 2. "d3sjqhkv6o9s73cs0pj0" 3."d3sjr2mjbaoc73cnds90" 4."d3sjr5s60rgc73aabmv0", 5."d3sk546jbaoc73cne5ug"
            job = svc.job(job_id)

            result = job.result()
            exec_sec = time.perf_counter() - t1
        else:
            exec_sec = 0.0

        return {
            "key": None,
            "cache_hit": False,
            "compile_sec": float(compile_sec),
            "bind_sec": 0.0,
            "exec_sec": float(exec_sec),
            "n_qubits": qc_raw.num_qubits,
            "depth_in": qc_raw.depth(),
            "depthT": qc_exec.depth(),
            "parametric": True,
            "stratPath":"B"
        }

    # C) 仅有“我们替换出的参数” => 可缓存复用 + 按值绑定
    temp_key = f"{bk_name}:{md5_qasm(templ)}"
    qc_exec_templ = template_cache.get(temp_key)
    hit = qc_exec_templ is not None
    compile_sec = 0.0
    if not hit:
        t0 = time.perf_counter()
        qc_exec_templ = transpile(templ, backend=backend, optimization_level=3)
        template_cache[temp_key] = qc_exec_templ
        compile_sec = time.perf_counter() - t0

    # 绑定：使用 transpiled 电路中的实际 Parameter 实例（按 name 匹配）
    bind_sec = 0.0
    name2param = {p.name: p for p in qc_exec_templ.parameters}
    binding: Dict[Parameter, float] = {}
    for par_orig, val in zip(replaced_params, replaced_values):
        p_in_exec = name2param.get(par_orig.name, None)
        if p_in_exec is not None:
            binding[p_in_exec] = val

    if include_exec:
        if binding:
            t1 = time.perf_counter()
            qc_exec = qc_exec_templ.assign_parameters(binding, inplace=False)
            bind_sec = time.perf_counter() - t1
        else:
            qc_exec = qc_exec_templ  # 理论上很少发生

        # 执行
        t2 = time.perf_counter()
        # job = sampler.run([qc_exec], shots=shots)
        svc = QiskitRuntimeService()
        job_id = "d3sk546jbaoc73cne5ug" # 1. "d3sfk6k60rgc73aa7p50" 2. "d3sjqhkv6o9s73cs0pj0" 3."d3sjr2mjbaoc73cnds90" 4."d3sjr5s60rgc73aabmv0", 5."d3sk546jbaoc73cne5ug"
        job = svc.job(job_id)

        result = job.result()
        exec_sec = time.perf_counter() - t2
    else:
        exec_sec = 0.0

    record_arrival(temp_key)  # 仅对“带参模板”记录

    return {
        "key": temp_key,
        "cache_hit": hit,
        "compile_sec": float(compile_sec),
        "bind_sec": float(bind_sec),
        "exec_sec": float(exec_sec),
        "n_qubits": qc_raw.num_qubits,
        "depth_in": qc_raw.depth(),
        # "depthT": qc_exec.depth(),
        "parametric": True,
        "stratPath":"C"
    }


def run_strategy(
    workload: List[Dict[str, Any]],
    shots: int = 256,
    include_exec: bool = True,
    bkd_name: str = "ibm_torino"
):
    """
    ParamReuse（Braket-like, param-only caching）：
    - 仅对白名单门 (RX/RY/RZ/U) 参数化并缓存模板；
    - 其他带数值的门不参数化（数值变化 => 重新编译）；
    - 非带参电路：每次常规 transpilation，不缓存。
    """
    template_cache: Dict[str, QuantumCircuit] = {}
    events: List[Dict[str, Any]] = []
    t = 0.0

    hit_by_label: Dict[str, int] = Counter()
    total_hits = 0
    cache_size_series: List[Tuple[float, int]] = []

    """Prepare ibm backend"""
    svc = QiskitRuntimeService()
    backend = svc.backend(bkd_name)
    sampler = Sampler(mode=backend)

    for it in workload:
        meta = run_once_param_reuse(it["maker_run"], template_cache, shots=shots, include_exec=include_exec,
                                       sampler=sampler, backend=backend, bk_name=bkd_name)

        run_dur = float(meta.get("compile_sec", 0.0)) + float(meta.get("bind_sec", 0.0)) + float(meta.get("exec_sec", 0.0))
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind": "run", "label": lab, "start": t, "dur": run_dur,
                       "transT":float(meta["compile_sec"]),
                       "bindT":float(meta["bind_sec"]),
                       "execT":float(meta["exec_sec"]),
                       "stratPath":meta["stratPath"]})
        t += run_dur

        if meta.get("cache_hit", False):
            hit_by_label[lab] += 1
            total_hits += 1

        cache_size_series.append((t, len(template_cache)))

    metrics = {
        "hit_by_label": dict(hit_by_label),
        "total_hits": int(total_hits),
        "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series],
        "cache_capacity": None,
        "note": f"Only parameterize gates in {sorted(ALLOWED_PARAM_GATES)}; others stay constant; cache key = md5(template QASM).",
    }
    return {"events": events, "metrics": metrics}
