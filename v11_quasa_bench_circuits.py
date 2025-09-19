# circuits_quasa_bench.py
import numpy as np
from typing import Dict, Callable, Tuple, List
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

# ---------- 你原来的三类 ----------
def make_linear_ent(q=6, d=5):
    qc = QuantumCircuit(q, q)
    for _ in range(d):
        qc.h(range(q))
        for i in range(q-1):
            qc.cx(i, i+1)
        qc.rx(np.pi/4, range(q))
    qc.measure(range(q), range(q))
    return qc

def make_ghz_chain(q=6, d=5):
    qc = QuantumCircuit(q, q)
    qc.h(0)
    for i in range(q-1):
        qc.cx(i, i+1)
    for _ in range(d-1):
        qc.rz(np.pi/8, q-1)
        qc.barrier()
    qc.measure(range(q), range(q))
    return qc

def make_qft_like(q=6, d=5):
    qc = QuantumCircuit(q, q)
    for _ in range(d):
        qc.compose(QFT(num_qubits=q, do_swaps=False).decompose(),
                   range(q), inplace=True)
    qc.measure(range(q), range(q))
    return qc

# ---------- 论文 NISQ：QAOA / RCA / VQE ----------
# 1) QAOA on 3-regular graph (ZZ only on graph edges)
def _random_3_regular_edges(n: int, rng: np.random.Generator) -> List[Tuple[int,int]]:
    # 简洁的近似生成：循环配对三次并随机重连（不保证严格 3-regular，但足够做合成负载）
    deg = {i:0 for i in range(n)}
    edges = set()
    trials = 0
    while any(deg[i] < 3 for i in range(n)) and trials < 20*n:
        a, b = rng.integers(0, n), rng.integers(0, n)
        if a == b:
            trials += 1; continue
        u, v = min(a,b), max(a,b)
        if (u,v) in edges:
            trials += 1; continue
        if deg[u] < 3 and deg[v] < 3:
            edges.add((u,v)); deg[u]+=1; deg[v]+=1
        trials += 1
    if len(edges) == 0:  # 保底
        for i in range(n-1): edges.add((i,i+1))
    return list(edges)

def make_qaoa_3regular(q=12, p=1, seed=1):
    rng = np.random.default_rng(seed)
    edges = _random_3_regular_edges(q, rng)
    gamma, beta = np.pi/8, np.pi/6
    qc = QuantumCircuit(q, q)
    qc.h(range(q))
    for _ in range(p):
        # ZZ cost on edges
        for (u,v) in edges:
            qc.cx(u, v); qc.rz(2*gamma, v); qc.cx(u, v)
        # mixer X
        qc.rx(2*beta, range(q))
    qc.measure(range(q), range(q))
    return qc

# 2) Ripple-Carry Adder（RCA）— 简化版 n-bit 加法器
def make_rca(q: int = 9, d: int = 1):
    """
    Ripple-Carry Adder (RCA) 基准电路，统一 (q, d) 接口。
    要求 q 为奇数，q = 2n + 1：n 位 A、n 位 B、1 位进位 c。
    d 为重复的“加法层”次数，用于 depth 扫描。
    """
    if q < 3:
        raise ValueError("q 必须 ≥ 3 且为奇数 (q = 2n+1)。")
    if q % 2 == 0:
        raise ValueError(f"RCA 需要奇数条线路：收到 q={q}，应满足 q=2n+1。")

    n = (q - 1) // 2
    a = list(range(0, n))          # A 寄存器 [0..n-1]
    b = list(range(n, 2*n))        # B 寄存器 [n..2n-1]
    c = 2*n                        # 进位位

    qc = QuantumCircuit(q, q)

    # 可选：如需固定输入，可在此对 A/B 打 X 初始化特定位
    # 例如：qc.x(a[0]); qc.x(b[1])

    for layer in range(d):
        # 简化的 ripple-carry：CX 写回、CCX 产生进位、把进位带到下一位
        for i in range(n):
            qc.cx(a[i], b[i])
            qc.ccx(a[i], b[i], c)
            if i < n - 1:
                qc.cx(c, a[i+1])
        if layer != d - 1:
            qc.barrier()  # 仅用于层间分隔，便于对齐 depth 统计

    qc.measure(range(q), range(q))
    return qc

# 3) VQE full-entanglement ansatz（层数 d）
def make_vqe_fullent(q=12, d=2):
    qc = QuantumCircuit(q, q)
    for _ in range(d):
        qc.ry(np.pi/7, range(q))
        for i in range(q-1):
            qc.cx(i, i+1)
        qc.cx(q-1, 0)  # 环形纠缠
    qc.measure(range(q), range(q))
    return qc

# ---------- 论文 FT：QFT / Grover / QSIM(XXZ, Trotter) ----------
# def make_qft(q=12, d=1):
#     qc = QuantumCircuit(q, q)
#     for _ in range(d):
#         qc.compose(QFT(q, do_swaps=True).decompose(), range(q), inplace=True)
#     qc.measure(range(q), range(q))
#     return qc

def make_grover_oracle(q: int):
    # 简化 oracle：标记全零 |0...0>，用多重 Z 近似（不做多控优化）
    qc = QuantumCircuit(q)
    qc.z(0)  # 只是放个标记，真实多控会更重
    return qc.to_gate(label="Oracle")

def make_grover(q=12, iters=2):
    qc = QuantumCircuit(q, q)
    qc.h(range(q))
    oracle = make_grover_oracle(q)
    for _ in range(iters):
        qc.append(oracle, range(q))
        # diffusion
        qc.h(range(q)); qc.x(range(q))
        qc.h(q-1); qc.mcx(list(range(q-1)), q-1); qc.h(q-1)
        qc.x(range(q)); qc.h(range(q))
    qc.measure(range(q), range(q))
    return qc

def make_qsim_xxz(q=12, steps=1, theta=np.pi/12):
    # Trotter XXZ: exp(-i theta (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}))
    qc = QuantumCircuit(q, q)
    for _ in range(steps):
        for i in range(q-1):
            # 用常见分解近似（示意）
            qc.cx(i, i+1); qc.rz(2*theta, i+1); qc.cx(i, i+1)     # ZZ
            qc.cx(i+1, i); qc.rz(2*theta, i); qc.cx(i+1, i)       # XX/YY 近似
        qc.barrier()
    qc.measure(range(q), range(q))
    return qc

# --------- 暴露基准集合 ----------
CIRCUITS_QUASA: Dict[str, Callable] = {
    # 你已有
    # "GHZ-Chain": make_ghz_chain,
    # "LinearEnt": make_linear_ent,
    # "QFT-Like":  make_qft_like,
    # # # 论文 NISQ
    # "QAOA-3reg": make_qaoa_3regular,
    # # "RCA":       make_rca,
    # "VQE-Full":  make_vqe_fullent,
    # # # 论文 FT
    # # "Grover":    make_grover,
    # "QSIM-XXZ":  make_qsim_xxz,
}
