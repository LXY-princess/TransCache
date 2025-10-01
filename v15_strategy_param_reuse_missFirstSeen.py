# v15_strategy_param_reuse.py  (fixed)
from typing import Any, Dict, List, Tuple
from collections import Counter
from numbers import Real
import time

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer.primitives import Sampler as AerSampler

from v15_core import md5_qasm, label_of, _prepare_kwargs, record_arrival


def _build_param_template_and_bindings(qc_raw: QuantumCircuit) -> Tuple[QuantumCircuit, List[Parameter], List[float]]:
    """
    把原电路规范化为“参数化模板”：
    - 遍历每个指令，把所有数值型参数替换为新的 Parameter p{i}；
    - 返回 (模板电路, 参数列表, 对应的数值列表)，
      使得 template.assign_parameters({params[i]: values[i]}) 恢复到当前数值。
    这样即可对模板“编译一次，多次按值绑定”。
    """
    nq = qc_raw.num_qubits
    nc = qc_raw.num_clbits
    templ = QuantumCircuit(nq, nc)
    params: List[Parameter] = []
    values: List[float] = []
    pid = 0

    for inst, qargs, cargs in qc_raw.data:
        new_inst = inst.copy()
        if getattr(inst, "params", None):
            new_params = []
            for p in inst.params:
                # 连续实数（角度等）视作可绑定参数；非数值（如已有 Parameter/门对象）保持不变
                if isinstance(p, Real) and not isinstance(p, bool):
                    par = Parameter(f"p{pid}"); pid += 1
                    params.append(par); values.append(float(p))
                    new_params.append(par)
                else:
                    new_params.append(p)
            new_inst.params = new_params
        templ.append(new_inst, qargs, cargs)

    # 透传可能存在的硬件校准
    if getattr(qc_raw, "calibrations", None):
        for gate, cals in qc_raw.calibrations.items():
            for sig, sched in cals.items():
                templ.add_calibration(gate, sig, sched)

    return templ, params, values


def run_once_param_reuse(qc_func, template_cache: Dict[str, QuantumCircuit], shots: int = 256) -> Dict[str, Any]:
    qc_raw = qc_func()

    # 与现有路径一致：小于等于25比特用 statevector，其他用 MPS
    method = "statevector" if qc_raw.num_qubits <= 25 else "matrix_product_state"
    sampler = AerSampler(skip_transpilation=True, backend_options={"method": method})
    bk = "AerSV"

    # 构建参数化模板并提取本次的数值绑定
    templ, param_list, val_list = _build_param_template_and_bindings(qc_raw)
    key = f"{bk}:{md5_qasm(templ)}"

    # 以“结构模板”的 md5 作为缓存键：只对结构编译一次
    qc_exec_templ = template_cache.get(key)
    hit = qc_exec_templ is not None
    compile_sec = 0.0
    if not hit:
        t0 = time.perf_counter()
        qc_exec_templ = transpile(templ, **_prepare_kwargs())
        compile_sec = time.perf_counter() - t0
        template_cache[key] = qc_exec_templ

    # 快速参数绑定（不再走编译）
    bind_sec = 0.0
    if param_list:
        # 注意：transpile 后的 circuit 中 Parameter 实例通常不是 templ 中的同一对象
        # 所以我们需要按 name 去匹配 transpiled circuit 中实际存在的 Parameter 对象
        name2param = {p.name: p for p in qc_exec_templ.parameters}

        # 构造实际的绑定 dict：用 transpiled circuit 中的 Parameter 实例作为 key
        binding: Dict[Parameter, float] = {}
        for par_orig, val in zip(param_list, val_list):
            pname = par_orig.name
            p_in_exec = name2param.get(pname, None)
            if p_in_exec is not None:
                binding[p_in_exec] = val
            else:
                # 参数在 transpile 后不存在（可能被编译器优化掉），跳过绑定
                #（不会抛错）
                continue

        if binding:
            t1 = time.perf_counter()
            qc_exec = qc_exec_templ.assign_parameters(binding, inplace=False)
            bind_sec = time.perf_counter() - t1
        else:
            # 所有参数都在 transpile 过程中被移除/常量化 => 没有绑定开销
            qc_exec = qc_exec_templ
            bind_sec = 0.0
    else:
        qc_exec = qc_exec_templ

    # 执行
    t2 = time.perf_counter()
    _ = sampler.run([qc_exec], shots=shots).result()
    exec_sec = time.perf_counter() - t2

    # 记录到达（如后续你用预测器种子）
    record_arrival(key)

    return {
        "key": key,
        "cache_hit": hit,
        "compile_sec": float(compile_sec),
        "bind_sec": float(bind_sec),
        "exec_sec": float(exec_sec),
        "n_qubits": qc_raw.num_qubits,
        "depth_in": qc_raw.depth(),
        "depthT": qc_exec.depth(),
    }


def run_strategy(
    workload: List[Dict[str, Any]],
    shots: int = 256,
):
    """
    ParamReuse：首次编译结构模板，随后仅按值绑定。
    - 对“含连续参数的门”复用编译产物（不同角度/系数只做 assign_parameters）。
    - 对“无参数电路”自然退化为 FirstSeen。
    事件：仅产生 'run'，与现有画图兼容。
    指标：按 label 的命中计数与 cache 大小轨迹。
    """
    template_cache: Dict[str, QuantumCircuit] = {}
    events: List[Dict[str, Any]] = []
    t = 0.0

    hit_by_label: Dict[str, int] = Counter()
    total_hits = 0
    cache_size_series: List[Tuple[float, int]] = []

    for it in workload:
        meta = run_once_param_reuse(it["maker_run"], template_cache, shots=shots)

        # 时间线
        run_dur = float(meta.get("compile_sec", 0.0)) + float(meta.get("bind_sec", 0.0)) + float(meta.get("exec_sec", 0.0))
        lab = label_of(it["name"], it["q"], it["d"])
        events.append({"kind": "run", "label": lab, "start": t, "dur": run_dur})
        t += run_dur

        # 统计
        if meta["cache_hit"]:
            hit_by_label[lab] += 1
            total_hits += 1

        # cache 大小轨迹
        cache_size_series.append((t, len(template_cache)))

    metrics = {
        "hit_by_label": dict(hit_by_label),
        "total_hits": int(total_hits),
        "cache_size_series": [{"t": float(tt), "size": int(sz)} for (tt, sz) in cache_size_series],
        "cache_capacity": None,
        "note": "Cache keyed by structure-only (parameter-agnostic) template.",
    }
    return {"events": events, "metrics": metrics}
