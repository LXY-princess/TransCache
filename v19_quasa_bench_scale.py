# v19_quasa_bench_scale.py
# 环境：qiskit-terra==0.46.3, qiskit-aer==0.13.3, qiskit-ibm-runtime==0.40.1
import inspect
import pathlib
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit_aer import AerSimulator  # ← 使用 Aer 模拟后端

VNUM = 19
ROOT = pathlib.Path("./figs") / f"v{VNUM}"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 载入你的基准电路 ----
# 优先使用你提供的文件名；若老版本文件名存在也可回退
from v11_quasa_bench_circuits import CIRCUITS_QUASA  # 回退

# ---- 与你需求一致的后端准备 ----
def _prepare_kwargs():
    """
    使用 AerSimulator 的 target 做“硬件假设”：
    - target: Aer 的门集/时序等（不含物理耦合约束）
    - optimization_level: 2
    - seed_transpiler: 42
    """
    aer = AerSimulator()
    try:
        return {"target": aer.target, "optimization_level": 2, "seed_transpiler": 42}
    except Exception:
        # 兼容兜底（极少数环境下 Aer target 不可用时，退回 basis_gates）
        return {
            "basis_gates": ['rz', 'sx', 'x', 'cx', 'id'],
            "optimization_level": 2,
            "seed_transpiler": 42
        }

# --- 工具函数 ---
def build_circuit(builder, q, d):
    """把 d 映射到各生成器的参数名：d/p/steps/iters"""
    sig = inspect.signature(builder)
    kwargs = {}
    if 'q' in sig.parameters:
        kwargs['q'] = q
    elif 'num_qubits' in sig.parameters:
        kwargs['num_qubits'] = q

    if 'd' in sig.parameters:
        kwargs['d'] = d
    elif 'p' in sig.parameters:
        kwargs['p'] = d
    elif 'steps' in sig.parameters:
        kwargs['steps'] = d
    elif 'iters' in sig.parameters:
        kwargs['iters'] = d
    return builder(**kwargs)

def remove_final_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    # Terra 0.46 支持此方法
    return qc.remove_final_measurements(inplace=False)

def _dag_depth(qc: QuantumCircuit, ignore_barrier: bool) -> int:
    """用 DAG 计算深度，保证跨版本稳定；可选择忽略 barrier。"""
    dag = circuit_to_dag(qc)
    if ignore_barrier:
        for n in list(dag.op_nodes()):
            if n.name == "barrier":
                dag.remove_op_node(n)
    return dag.depth()

def _dag_twoq_depth(qc: QuantumCircuit, ignore_barrier: bool) -> int:
    """仅按两比特门（及可选 barrier）计算深度。"""
    dag = circuit_to_dag(qc)
    for n in list(dag.op_nodes()):
        if n.name == "barrier":
            if ignore_barrier:
                dag.remove_op_node(n)
        elif n.op.num_qubits < 2:
            dag.remove_op_node(n)
    return dag.depth()

def count_by_arity(qc: QuantumCircuit):
    oneq = twoq = threeq = 0
    cx = cz = swap = m = b = 0
    for inst, _, _ in qc.data:
        k = inst.num_qubits
        if inst.name == 'measure': m += 1
        if inst.name == 'barrier': b += 1
        if k == 1:
            oneq += 1
        elif k == 2:
            twoq += 1
            if inst.name == 'cx': cx += 1
            if inst.name == 'cz': cz += 1
            if inst.name == 'swap': swap += 1
        elif k == 3:
            threeq += 1
    return dict(oneq=oneq, twoq=twoq, threeq=threeq, cx=cx, cz=cz, swap=swap, measure=m, barrier=b)

def metrics_for(qc: QuantumCircuit, ignore_barrier=False):
    """统计一个电路的逻辑层指标（跨版本稳妥）。"""
    depth_all = _dag_depth(qc, ignore_barrier=ignore_barrier)
    arity = count_by_arity(qc)
    twoq_depth = _dag_twoq_depth(qc, ignore_barrier=ignore_barrier)
    return dict(
        num_qubits=qc.num_qubits,
        depth=depth_all,
        twoq_depth=twoq_depth,
        size=len(qc.data),
        **arity
    )

def sweep_metrics(circuits, q_list, d_list,
                  transpile_kwargs: dict | None = None,
                  ignore_barrier: bool = False):
    """
    对 (电路, q, d) 网格做逻辑+（可选）Aer target 映射两套统计。
    - 逻辑：直接在构造后的电路（移除末测）上统计
    - “硬件”：若给出 transpile_kwargs（默认来自 _prepare_kwargs()），则先 transpile 再统计
    """
    rows = []
    for name, builder in circuits.items():
        for q in q_list:
            for d in d_list:
                try:
                    qc_full = build_circuit(builder, q, d)
                    qc = remove_final_measurements(qc_full)
                except Exception as e:
                    rows.append(dict(circuit=name, q=q, d=d, error=str(e)))
                    continue

                # 逻辑层
                m_logic = metrics_for(qc, ignore_barrier=ignore_barrier)
                rec = dict(circuit=name, q=q, d=d, **{f'logic_{k}': v for k, v in m_logic.items()})

                # “硬件”（Aer target）映射（可选）
                if transpile_kwargs is not None:
                    qc_hw = transpile(qc, **transpile_kwargs)
                    m_hw = metrics_for(qc_hw, ignore_barrier=ignore_barrier)
                    rec.update({f'hw_{k}': v for k, v in m_hw.items()})

                rows.append(rec)
    return pd.DataFrame(rows)

# ========== 示例用法 ==========
if __name__ == "__main__":
    circuits = CIRCUITS_QUASA
    q_list = [3,11,17,23]
    d_list = [1, 2, 3,4]

    # 1) 只做逻辑层统计（不映射）
    df_logic = sweep_metrics(circuits, q_list, d_list, transpile_kwargs=None, ignore_barrier=True)
    df_logic.to_csv(PLOT_DIR / "metrics_logic.csv", index=False)

    # 2) 使用 AerSimulator target 做“硬件假设”映射
    df_hw = sweep_metrics(
        circuits, q_list, d_list,
        transpile_kwargs=_prepare_kwargs(),   # ← 关键：target=aer.target 等
        ignore_barrier=True
    )
    df_hw.to_csv(PLOT_DIR / "metrics_hw_aer_target.csv", index=False)

    # 3) 生成适合画热图的透视表（以 twoq_depth 为例）
    def pivot_for(df, circuit_name, col='hw_twoq_depth'):
        sub = df[df['circuit'] == circuit_name]
        return sub.pivot(index='d', columns='q', values=col)

    piv = pivot_for(df_hw, 'QFT-Like', col='hw_twoq_depth')
    print(piv)
