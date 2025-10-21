# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Tuple, Callable, Optional
import math, random, os
import numpy as np
import matplotlib.pyplot as plt
from v18_core import (
    ROOT, PLOT_DIR,
)

# -----------------------------
# 0) 通用：类别分布/装配/工具
# -----------------------------
def _make_class_probs(
    meta: List[Dict[str, Any]],
    mode: str = "weighted",          # 'weighted' | 'zipf' | 'uniform'
    rng: Optional[np.random.Generator] = None,
    hot_fraction: float = 0.25,
    hot_boost: float = 8.0,
    zipf_s: float = 1.1
) -> np.ndarray:
    """
    生成类别选择概率向量 p_i（与 meta 对齐）：
      - weighted: 模拟热点（与您原实现一致思路）
      - zipf:     幂律重尾（随机置乱后按 rank^-s 分配）
      - uniform:  平均
    """
    assert len(meta) > 0
    rng = rng or np.random.default_rng(123)
    m = len(meta)
    if mode == "uniform":
        probs = np.ones(m, dtype=float) / m
        return probs

    if mode == "weighted":
        weights = np.ones(m, dtype=float)
        hot_k = max(1, int(round(hot_fraction * m)))
        hot_idx = rng.choice(m, size=hot_k, replace=False)
        weights[hot_idx] *= max(1.0, float(hot_boost))
        probs = weights / weights.sum()
        return probs

    if mode == "zipf":
        # 生成 Zipf 概率，再随机打乱，避免与索引顺序相关
        ranks = np.arange(1, m + 1, dtype=float)
        weights = 1.0 / np.power(ranks, max(0.5, float(zipf_s)))
        weights /= weights.sum()
        perm = rng.permutation(m)
        probs = weights[perm]
        probs /= probs.sum()
        return probs

    raise ValueError(f"unknown class mode: {mode}")


def _assemble_workload(
    meta: List[Dict[str, Any]],
    times: np.ndarray,   # shape (N,), 非降
    cls_idx: np.ndarray, # shape (N,), 每条事件的 meta 索引
    return_timestamps: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    assert len(times) == len(cls_idx)
    workload: List[Dict[str, Any]] = []
    for t_i, i in zip(times, cls_idx):
        rec = {
            "name": meta[i]["name"],
            "q":    meta[i]["q"],
            "d":    meta[i]["d"],
            "maker_run": meta[i]["maker_run"],
        }
        if return_timestamps:
            rec["t_arr"] = float(t_i)
        workload.append(rec)
    info = {
        "T_end": float(times[-1]) if len(times) else 0.0,
        "N": int(len(times)),
    }
    return workload, info


def _draw_classes(
    N: int,
    probs: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    rng = rng or np.random.default_rng(123)
    m = len(probs)
    idx = rng.choice(m, size=N, p=probs)
    return idx


# -----------------------------
# 1) 时间维：到达过程族
# -----------------------------
def gen_times_hpp_exact(N: int, rps: float, rng: Optional[np.random.Generator]=None) -> np.ndarray:
    """HPP: Δ ~ Exp(Λ=rps)"""
    rng = rng or np.random.default_rng(123)
    inter = rng.exponential(scale=1.0/max(rps,1e-12), size=N).astype(float)
    t = np.cumsum(inter)
    return t

def gen_times_renewal_exact(
    N: int,
    rps: float,
    kind: str = "gamma",        # 'gamma' | 'weibull' | 'lognormal' | 'lomax'
    shape: float = 2.0,         # gamma.k / weibull.k / lomax.alpha；lognormal 使用 sigma
    sigma: float = 1.0,         # lognormal 的对数标准差
    rng: Optional[np.random.Generator]=None
) -> np.ndarray:
    """广义更新：调参使 E[Δ]=1/Λ，精确 N 条"""
    rng = rng or np.random.default_rng(123)
    Λ = float(rps)
    if kind == "gamma":
        k = max(shape, 1e-3)
        theta = (1.0/Λ)/k
        inter = rng.gamma(shape=k, scale=theta, size=N)
    elif kind == "weibull":
        k = max(shape, 1e-3)
        theta = (1.0/Λ) / math.gamma(1.0 + 1.0/k)
        u = rng.weibull(a=k, size=N)  # 这是“形状参数 k”的生成，尺度需外乘
        inter = theta * u
    elif kind == "lognormal":
        s = max(sigma, 1e-6)
        m = 1.0/Λ
        mu = math.log(m) - 0.5*(s**2)
        inter = rng.lognormal(mean=mu, sigma=s, size=N)
    elif kind == "lomax":
        # Lomax(shape=alpha, scale=θ): E=θ/(α-1) for α>1
        alpha = max(shape, 1.001)
        theta = (1.0/Λ)*(alpha - 1.0)
        # numpy 无 lomax，自己用 ParetoII 变换：X=θ*(U^{-1/α}-1)
        U = np.clip(rng.random(size=N), 1e-12, 1-1e-12)
        inter = theta*(np.power(1.0 - U, -1.0/alpha) - 1.0)
    else:
        raise ValueError(f"unknown renewal kind: {kind}")
    t = np.cumsum(inter)
    return t

def gen_times_nhpp_exact(
    N: int,
    lambda_of_t: Callable[[float], float],
    lambda_max: float,
    rng: Optional[np.random.Generator]=None
) -> np.ndarray:
    """
    NHPP: 用 thinning 直到收集满 N 个事件；不预设固定窗口，T_end 由过程决定。
    需要给出 λ_max ≥ sup_t λ(t)。
    """
    rng = rng or np.random.default_rng(123)
    t = 0.0
    out = []
    while len(out) < N:
        # 提议间隔 ~ Exp(λ_max)
        t += rng.exponential(scale=1.0/max(lambda_max,1e-12))
        if rng.random() < (lambda_of_t(t)/max(lambda_max,1e-12)):
            out.append(t)
    return np.array(out, dtype=float)

def gen_times_mmpp2_exact(
    N: int,
    rps: float,
    contrast: float = 8.0,   # λ1/λ0
    pi1: float = 0.15,       # 高速态稳态概率
    a: float = 0.05,         # 0->1 转移率
    b: float = 0.25,         # 1->0 转移率 (pi1 ≈ a/(a+b)；入参会覆盖)
    rng: Optional[np.random.Generator]=None
) -> np.ndarray:
    """
    两态 MMPP：状态0速率 λ0，状态1速率 λ1=contrast*λ0；
    使 E[λ]=Λ，故 λ0 = Λ / (1 + (contrast-1)*pi1)；lambda1=contrast*lambda0。
    状态持时 ~ Exp(a or b)；next event 时间取 min(下一个状态切换, 下一个到达)。
    """
    rng = rng or np.random.default_rng(123)
    Λ = float(rps)
    pi1 = float(pi1)
    lam0 = Λ / (1.0 + (max(contrast,1.0)-1.0)*pi1)
    lam1 = lam0 * max(contrast,1.0)
    # 用 a,b 保证稳态 ≈ pi1 (可直接用 a,b 入参覆盖)
    if a + b > 0:
        pi1_est = a/(a+b)
        if abs(pi1_est - pi1) > 1e-6:
            # 不强制匹配，给出注记即可
            pass

    state = 0
    lam = lam0
    t = 0.0
    out = []
    while len(out) < N:
        # 两个竞争：下次到达、下次切换
        dt_arr = rng.exponential(scale=1.0/max(lam,1e-12))
        dt_sw  = rng.exponential(scale=1.0/max((a if state==0 else b),1e-12))
        if dt_arr < dt_sw:
            t += dt_arr
            out.append(t)
        else:
            t += dt_sw
            state = 1 - state
            lam = lam1 if state==1 else lam0
    return np.array(out, dtype=float)

def gen_times_hawkes_exact(
    N: int,
    rps: float,
    eta: float = 0.5,      # branching ratio, <1
    beta: float = 3.0,     # 衰减率（越大，簇越紧凑）
    rng: Optional[np.random.Generator]=None
) -> np.ndarray:
    """
    Hawkes(μ, φ)：φ(t)=η*β*exp(-β t)，E[λ]=μ/(1-η)=Λ => μ=Λ*(1-η)。
    Ogata thinning：维护上界 λ̄，反复试采直到满 N。
    """
    rng = rng or np.random.default_rng(123)
    Λ = float(rps)
    mu = Λ*(1.0 - min(max(eta, 1e-6), 0.99))

    t = 0.0
    out = []
    # 为提升效率，维护对每个到达的残余贡献；这里只存最近若干（指数核可用递推，但简化实现）
    contrib_times = []

    # 当前强度上界（粗）：mu + sum(ηβ) = mu + ηβ * len(contrib)
    lam_bar = mu

    while len(out) < N:
        # 1) 提议时间增量
        dt = rng.exponential(scale=1.0/max(lam_bar,1e-12))
        t = t + dt

        # 2) 计算真实强度 λ(t) = mu + Σ ηβ e^{-β (t-ti)}
        if contrib_times:
            decays = np.exp(-beta*(t - np.array(contrib_times, dtype=float)))
            lam_t = mu + eta*beta*np.sum(decays)
        else:
            lam_t = mu

        # 3) 接受-拒绝
        if rng.random() <= lam_t/max(lam_bar,1e-12):
            out.append(t)
            contrib_times.append(t)
            # 更新一个保守上界（增加一个 ηβ 项）
            lam_bar = mu + eta*beta*len(contrib_times)
        else:
            # 更新上界（强度衰减导致更紧的上界，这里可以略降一点）
            lam_bar = max(mu + eta*beta*len(contrib_times) * math.exp(-beta*dt), mu + 1e-9)

        # 为避免长列表，定期清理远古贡献
        if len(contrib_times) > 0 and (t - contrib_times[0]) > 10.0/beta:
            # 贡献衰尽后移除
            while len(contrib_times)>0 and (t - contrib_times[0]) > 10.0/beta:
                contrib_times.pop(0)
    return np.array(out, dtype=float)

def gen_times_compound_bursty_exact(
    N: int,
    rps: float,
    mean_burst_size: float = 5.0,          # E[B]
    in_burst_jitter: float = 0.01,         # 突发内抖动尺度（秒）
    rng: Optional[np.random.Generator]=None
) -> np.ndarray:
    """
    复合泊松：簇到达率 Λ_c = Λ/E[B]；每簇生成 B 条（几何/对数级）近同时到达。
    用几何分布近似（mean = 1/p）。
    """
    rng = rng or np.random.default_rng(123)
    Λ = float(rps)
    meanB = max(mean_burst_size, 1.0001)
    p = 1.0/meanB                     # Geo(p) with mean 1/p
    Λ_c = Λ / meanB

    times = []
    t = 0.0
    while len(times) < N:
        # 簇到达间隔
        t += rng.exponential(scale=1.0/max(Λ_c,1e-12))
        # 簇大小
        # numpy 没有参数为 p 的 Geo(支持 {1,2,...})，手工造：
        U = max(1e-12, float(rng.random()))
        B = int(math.ceil(math.log(1-U)/math.log(1-p)))  # 期望 1/p
        B = max(1, B)
        # 生成簇内事件
        b = min(B, N - len(times))
        if in_burst_jitter <= 0:
            times.extend([t]*b)
        else:
            jitter = rng.exponential(scale=in_burst_jitter, size=b)
            times.extend((t + np.cumsum(jitter)).tolist())
    return np.array(times[:N], dtype=float)

def gen_times_sessions_exact(
    N: int,
    rps: float,
    mean_session_events: int = 20,
    mean_idle_gap: float = 5.0,          # 会话间平均间隔（秒）
    in_session_rps: float = None,        # 会话内速率（默认自动：保证全局 rps）
    rng: Optional[np.random.Generator]=None
) -> Tuple[np.ndarray, List[Tuple[int,int,float,float]]]:
    """
    会话型：把 N 条分成若干 session，每个 session 内速率高，session 之间空闲较长。
    返回 times 以及 session 划分信息（用于类别级可视化/对齐）。
    """
    rng = rng or np.random.default_rng(123)
    if in_session_rps is None:
        in_session_rps = rps*5.0   # 默认：会话内更快

    # 1) 划分簇大小
    sizes = []
    remain = N
    while remain > 0:
        s = max(1, int(rng.poisson(lam=max(1,mean_session_events))))
        s = min(s, remain)
        sizes.append(s)
        remain -= s

    times = []
    t = 0.0
    session_bounds = []
    for s in sizes:
        # 会话开始前 idle gap
        t += rng.exponential(scale=mean_idle_gap)
        # 会话内到达
        inter = rng.exponential(scale=1.0/max(in_session_rps,1e-12), size=s)
        start = t + 0.0
        t = t + np.sum(inter)
        end = t
        times.extend((start + np.cumsum(inter)).tolist())
        session_bounds.append((len(times)-s, len(times)-1, start, end))

    return np.array(times, dtype=float), session_bounds


# -----------------------------
# 2) 类别维：Polya / Session-Dirichlet
# -----------------------------
def draw_classes_polya_exact(
    N: int, m: int, alpha0: float = 1.0, reinforce: float = 1.0, rng: Optional[np.random.Generator]=None
) -> np.ndarray:
    """
    Pólya's Urn：强化采样（越抽越热），模拟短期流行度自增强。
    alpha0 为初始浓度（越大越接近均匀），reinforce 为每次抽取后对该类的加权。
    """
    rng = rng or np.random.default_rng(123)
    weights = np.ones(m, dtype=float)*max(alpha0,1e-6)
    out = np.empty(N, dtype=int)
    for k in range(N):
        p = weights/weights.sum()
        i = int(rng.choice(m, p=p))
        out[k] = i
        weights[i] += max(reinforce, 0.0)
    return out

def draw_classes_session_dirichlet_exact(
    session_bounds: List[Tuple[int,int,float,float]],
    m: int,
    alpha: float = 0.1,    # 会话内更尖锐（alpha 小）
    rng: Optional[np.random.Generator]=None
) -> np.ndarray:
    """
    每个会话抽一个 Dirichlet(α) 的类分布，随后该会话内按该分布采样类别。
    """
    rng = rng or np.random.default_rng(123)
    out = np.empty(session_bounds[-1][1]+1, dtype=int)
    for (s,e,_,_) in session_bounds:
        p = rng.dirichlet(alpha=np.ones(m)*max(alpha,1e-6))
        out[s:e+1] = rng.choice(m, size=(e-s+1), p=p)
    return out


# -----------------------------
# 3) 顶层封装：构造不同 workload（精确 N）
# -----------------------------
# HPP superposition
def build_workload_poisson_superposition_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    hot_fraction: float = 0.25,
    hot_boost: float = 8.0,
    rps: float = 1.0,
    rng_seed: int = 123,
    return_timestamps: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    叠加泊松（superposition）精确生成：
      - 设总率 Λ = rps；按热点权重分解为各类 λ_i，∑λ_i = Λ；
      - 逐个事件：Δ ~ Exp(Λ), t += Δ；类别 ~ Categorical(p_i=λ_i/Λ)；
      - 重复 N=workload_len 次，得到恰好 N 个事件；时间窗 T_end 随机，更贴近真实。

    返回：
      workload: List[dict] 与你现有策略直接兼容（含 maker_run / 可选 t_arr）
      info: 记录 Λ、λ_i、类别概率、最终 T_end 等，便于可视化与检验
    """
    assert workload_len > 0 and len(meta) > 0
    py_rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    # 1) 构造热点权重 -> 归一化 -> λ_i
    m = len(meta)
    weights = np.ones(m, dtype=float)
    hot_k = max(1, int(round(hot_fraction * m)))
    for i in py_rng.sample(range(m), hot_k):
        weights[i] *= max(1.0, float(hot_boost))
    w_sum = float(weights.sum())
    if w_sum <= 0:
        weights[:] = 1.0
        w_sum = float(weights.sum())

    Λ = float(rps)
    lambdas = (weights / w_sum) * Λ           # 每类速率 λ_i
    probs   = (lambdas / max(1e-12, Λ))       # 分类用 p_i = λ_i / Λ

    # 2) 逐事件模拟：Δ ~ Exp(Λ)，类别 ~ Categorical(probs)
    t = 0.0
    events: List[Tuple[float, int]] = []
    for _ in range(workload_len):
        u = 1.0 - py_rng.random()
        delta = -math.log(max(1e-12, u)) / max(1e-12, Λ)  # Exp(Λ)
        t += delta
        i = int(np_rng.choice(m, p=probs))                 # 类别归属
        events.append((t, i))

    # 3) 组装 workload
    out: List[Dict[str, Any]] = []
    for t_i, idx in events:
        rec = {
            "name": meta[idx]["name"],
            "q":    meta[idx]["q"],
            "d":    meta[idx]["d"],
            "maker_run": meta[idx]["maker_run"],
        }
        if return_timestamps:
            rec["t_arr"] = float(t_i)  # 模拟到达时刻
        out.append(rec)

    info = {
        "Lambda": float(Λ),
        "lambdas": lambdas.tolist(),
        "probs": probs.tolist(),
        "weights": weights.tolist(),
        "hot_fraction": float(hot_fraction),
        "hot_boost": float(hot_boost),
        "rps": float(rps),
        "T_end": float(events[-1][0]) if events else 0.0,
        "N": int(workload_len),
    }
    return out, info


def visualize_superposed_poisson_exact(
    workload: List[Dict[str, Any]],
    info: Dict[str, Any],
    meta: List[Dict[str, Any]],
    bins: int = 40,
    topk_classes: int = 10
):
    """
    针对“superposition 精确生成”的可视化/健康检查：
      1) 全局到达间隔 Δt 直方图 ≈ Exp(Λ)（检验 total 过程）；
      2) N(t) 与期望 Λ t 的对比；
      3) 类别计数与 Multinomial(N; p_i=λ_i/Λ) 期望的对比（固定 N 条的条件下）；
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import Counter

    Λ = float(info.get("Lambda", 1.0))
    probs = np.array(info.get("probs", []), dtype=float)
    N = int(info.get("N", len(workload)))

    # 取到达时刻
    ts = np.array([w["t_arr"] for w in workload], dtype=float)
    ts.sort()
    inter = np.diff(ts)  # Δt

    # 1) Δt 直方图 + 指数拟合曲线
    fig1, ax1 = plt.subplots(figsize=(6,4))
    if len(inter) > 0:
        ax1.hist(inter, bins=bins, density=True, alpha=0.7, label="empirical Δt")
        x = np.linspace(0, max(inter.max(), 1e-6), 200)
        ax1.plot(x, Λ*np.exp(-Λ*x), "r--", lw=2, label=f"Exp(Λ={Λ:.2f})")
    ax1.set_title("Inter-arrival histogram (should be ~Exp(Λ))")
    ax1.set_xlabel("Δt"); ax1.set_ylabel("density")
    ax1.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR/f"wl_{N}_inter_arrival.png", dpi=600)

    # 2) 计数过程 N(t) 与 E[N(t)]=Λ t
    fig2, ax2 = plt.subplots(figsize=(6,4))
    if len(ts) > 0:
        ax2.step(ts, np.arange(1, len(ts)+1), where="post", label="N(t)")
        tline = np.linspace(0, ts[-1], 200)
        ax2.plot(tline, Λ*tline, "r--", lw=2, label="E[N(t)] = Λ t")
    ax2.set_title("Counting process vs expectation")
    ax2.set_xlabel("t"); ax2.set_ylabel("N(t)")
    ax2.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR/f"wl_{N}_counting_process.png", dpi=600)

    # 3) 类别计数 vs Multinomial 期望
    if probs.size > 0:
        # 统计观测
        keys = [f'{w["name"]}|q{w["q"]}|d{w["d"]}' for w in workload]
        # 将 keys 映射回 meta 序号的方法：这里我们直接利用 meta 顺序的 probs 向量；
        # 统计每个 meta 索引的次数：用 name|q|d 定位到 meta 下标
        label2idx = {f'{m["name"]}|q{m["q"]}|d{m["d"]}': i for i, m in enumerate(meta)}
        counts = np.zeros(len(meta), dtype=int)
        for k in keys:
            i = label2idx.get(k, None)
            if i is not None:
                counts[i] += 1

        expect = N * probs
        # 展示 top-k
        idx_top = np.argsort(-counts)[:min(topk_classes, len(counts))]
        fig3, ax3 = plt.subplots(figsize=(max(7, 0.6*len(idx_top)+4), 4))
        x = np.arange(len(idx_top))
        ax3.bar(x - 0.2, counts[idx_top], width=0.4, label="observed")
        ax3.bar(x + 0.2, expect[idx_top], width=0.4, label="expected (N·p_i)")
        xt = [f'{meta[i]["name"]}|q{meta[i]["q"]}|d{meta[i]["d"]}' for i in idx_top]
        ax3.set_xticks(x); ax3.set_xticklabels(xt, rotation=45, ha="right")
        ax3.set_title("Per-class counts vs Multinomial expectation (fixed N)")
        ax3.set_ylabel("#events")
        ax3.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR/f"wl_{N}_counts_predict.png", dpi=600)
    # plt.show()


def build_workload_renewal_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    rps: float = 1.0,
    renewal_kind: str = "gamma",
    shape: float = 2.0,
    sigma: float = 1.0,
    class_mode: str = "weighted",
    rng_seed: int = 123,
    return_timestamps: bool = True,
    **class_kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(rng_seed)
    times = gen_times_renewal_exact(workload_len, rps, kind=renewal_kind, shape=shape, sigma=sigma, rng=rng)
    probs = _make_class_probs(meta, mode=class_mode, rng=rng, **class_kwargs)
    cls_idx = _draw_classes(workload_len, probs, rng=rng)
    workload, info = _assemble_workload(meta, times, cls_idx, return_timestamps)
    info.update({
        "kind": f"renewal/{renewal_kind}",
        "rps": float(rps),
        "params": {"shape": float(shape), "sigma": float(sigma)},
        "class_mode": class_mode,
        "class_params": class_kwargs,
    })
    return workload, info

def build_workload_nhpp_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    lambda_of_t: Callable[[float], float],
    lambda_max: float,
    class_mode: str = "weighted",
    rng_seed: int = 123,
    return_timestamps: bool = True,
    **class_kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(rng_seed)
    times = gen_times_nhpp_exact(workload_len, lambda_of_t, lambda_max, rng=rng)
    # 平均速率（用于注记）
    rps_est = float(workload_len / max(times[-1], 1e-12))
    probs = _make_class_probs(meta, mode=class_mode, rng=rng, **class_kwargs)
    cls_idx = _draw_classes(workload_len, probs, rng=rng)
    workload, info = _assemble_workload(meta, times, cls_idx, return_timestamps)
    info.update({
        "kind": "nhpp",
        "rps_mean_est": rps_est,
        "lambda_max": float(lambda_max),
        "class_mode": class_mode,
        "class_params": class_kwargs,
        "lambda_of_t": "callable_in_info",   # 供可视化叠加（见 visualize 中的注记）
    })
    info["_lambda_of_t_callable"] = lambda_of_t  # 内部使用（可视化时取出）
    return workload, info

def build_workload_mmpp2_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    rps: float = 1.0,
    contrast: float = 8.0,
    pi1: float = 0.15,
    a: float = 0.05,
    b: float = 0.25,
    class_mode: str = "weighted",
    rng_seed: int = 123,
    return_timestamps: bool = True,
    **class_kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(rng_seed)
    times = gen_times_mmpp2_exact(workload_len, rps, contrast=contrast, pi1=pi1, a=a, b=b, rng=rng)
    probs = _make_class_probs(meta, mode=class_mode, rng=rng, **class_kwargs)
    cls_idx = _draw_classes(workload_len, probs, rng=rng)
    workload, info = _assemble_workload(meta, times, cls_idx, return_timestamps)
    info.update({
        "kind": "mmpp2",
        "rps": float(rps),
        "params": {"contrast": float(contrast), "pi1": float(pi1), "a": float(a), "b": float(b)},
        "class_mode": class_mode,
        "class_params": class_kwargs,
    })
    return workload, info

def build_workload_hawkes_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    rps: float = 1.0,
    eta: float = 0.5,
    beta: float = 3.0,
    class_mode: str = "weighted",
    rng_seed: int = 123,
    return_timestamps: bool = True,
    **class_kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(rng_seed)
    times = gen_times_hawkes_exact(workload_len, rps, eta=eta, beta=beta, rng=rng)
    probs = _make_class_probs(meta, mode=class_mode, rng=rng, **class_kwargs)
    cls_idx = _draw_classes(workload_len, probs, rng=rng)
    workload, info = _assemble_workload(meta, times, cls_idx, return_timestamps)
    info.update({
        "kind": "hawkes",
        "rps": float(rps),
        "params": {"eta": float(eta), "beta": float(beta)},
        "class_mode": class_mode,
        "class_params": class_kwargs,
    })
    return workload, info

def build_workload_compound_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    rps: float = 1.0,
    mean_burst_size: float = 5.0,
    in_burst_jitter: float = 0.01,
    class_mode: str = "weighted",
    rng_seed: int = 123,
    return_timestamps: bool = True,
    **class_kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(rng_seed)
    times = gen_times_compound_bursty_exact(workload_len, rps, mean_burst_size=mean_burst_size,
                                            in_burst_jitter=in_burst_jitter, rng=rng)
    probs = _make_class_probs(meta, mode=class_mode, rng=rng, **class_kwargs)
    cls_idx = _draw_classes(workload_len, probs, rng=rng)
    workload, info = _assemble_workload(meta, times, cls_idx, return_timestamps)
    info.update({
        "kind": "compound",
        "rps": float(rps),
        "params": {"mean_burst_size": float(mean_burst_size), "in_burst_jitter": float(in_burst_jitter)},
        "class_mode": class_mode,
        "class_params": class_kwargs,
    })
    return workload, info

def build_workload_sessions_dirichlet_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    rps: float = 1.0,
    mean_session_events: int = 20,
    mean_idle_gap: float = 5.0,
    in_session_rps: float = None,
    alpha: float = 0.1,                     # 小 -> 会话内更尖锐
    rng_seed: int = 123,
    return_timestamps: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(rng_seed)
    times, session_bounds = gen_times_sessions_exact(workload_len, rps,
                                                     mean_session_events=mean_session_events,
                                                     mean_idle_gap=mean_idle_gap,
                                                     in_session_rps=in_session_rps, rng=rng)
    # 会话级 Dirichlet 类别
    m = len(meta)
    cls_idx = draw_classes_session_dirichlet_exact(session_bounds, m=m, alpha=alpha, rng=rng)
    workload, info = _assemble_workload(meta, times, cls_idx, return_timestamps)
    info.update({
        "kind": "session_dirichlet",
        "rps_est": float(workload_len/max(times[-1],1e-12)),
        "params": {"mean_session_events": int(mean_session_events),
                   "mean_idle_gap": float(mean_idle_gap),
                   "in_session_rps": float(in_session_rps) if in_session_rps else None,
                   "alpha": float(alpha)},
        "session_bounds": session_bounds
    })
    return workload, info

def build_workload_change_point_exact(
    meta: List[Dict[str, Any]],
    workload_len: int,
    rps_left: float = 1.0,
    rps_right: float = 2.0,
    frac_left: float = 0.5,
    class_mode_left: str = "weighted",
    class_mode_right: str = "zipf",
    rng_seed: int = 123,
    return_timestamps: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    简单突变：前半部分 Renewal-Gamma(k=1=泊松) 速率 rps_left，后半部分 速率 rps_right，
    且类别分布/模式也突变（如 weighted -> zipf）。
    """
    rng = np.random.default_rng(rng_seed)
    nL = max(1, int(round(frac_left*workload_len)))
    nR = workload_len - nL
    tL = gen_times_hpp_exact(nL, rps_left, rng=rng)
    tR = gen_times_hpp_exact(nR, rps_right, rng=rng)
    # 接续
    tR = tR + (tL[-1] if nL>0 else 0.0) + 1.0/max(rps_left,1e-12)  # 保持时间严格递增
    times = np.concatenate([tL, tR], axis=0)

    probsL = _make_class_probs(meta, mode=class_mode_left, rng=rng)
    probsR = _make_class_probs(meta, mode=class_mode_right, rng=rng)
    clsL = _draw_classes(nL, probsL, rng=rng)
    clsR = _draw_classes(nR, probsR, rng=rng)
    cls_idx = np.concatenate([clsL, clsR], axis=0)

    workload, info = _assemble_workload(meta, times, cls_idx, return_timestamps)
    info.update({
        "kind": "change_point",
        "params": {"rps_left": float(rps_left), "rps_right": float(rps_right), "frac_left": float(frac_left),
                   "class_mode_left": class_mode_left, "class_mode_right": class_mode_right}
    })
    return workload, info


# -----------------------------
# 4) 通用可视化仪表板
# -----------------------------
def visualize_workload_common(
    workload: List[Dict[str, Any]],
    info: Dict[str, Any],
    meta: List[Dict[str, Any]],
    out_dir: str = "plots",
    bins_inter: int = 40,
    topk_classes: int = 12,
    fano_bin_grid: Optional[List[float]] = None,   # 自定义时间粒度（秒）
    acf_binsize: Optional[float] = None,           # ACF 统计的时间粒度（秒）
):
    """
    统一输出 6 张图（保存于 out_dir）：
      A) inter_arrival_hist.png       到达间隔 Δt 直方图（若 info 含“Λ或 λ(t)”则叠加期望）
      B) counting_process.png         计数过程 N(t)（若有 Λ 或 λ(t) 积分则叠加期望）
      C) fano_curve.png               Fano 因子 vs bin 宽度（>1 过度离散，<1 欠离散）
      D) acf_counts.png               binned 计数自相关（检测突发/长记忆）
      E) class_zipf.png               类别频次 Zipf 排序（log-log）
      F) heat_topk_classes.png        top-K 类别 × 时间分箱 热力图
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = np.array([w["t_arr"] for w in workload], dtype=float)
    ts.sort()
    if len(ts) == 0:
        return
    inter = np.diff(ts)

    # A) Δt 直方图 + 可能的指数叠加
    fig, ax = plt.subplots(figsize=(6,4))
    if len(inter) > 0:
        ax.hist(inter, bins=bins_inter, density=True, alpha=0.7, label="empirical Δt")
    # 叠加：若 info 有 'Lambda' 或可估计 rps
    Λ = info.get("Lambda", None)
    if Λ is None:
        Λ = info.get("rps", None)
    if Λ is None and len(inter)>0:
        Λ = 1.0/max(np.mean(inter), 1e-12)
    if Λ is not None and len(inter) > 0:
        x = np.linspace(0, max(inter.max(), 1e-6), 200)
        ax.plot(x, float(Λ)*np.exp(-float(Λ)*x), "--", lw=2, label=f"Exp(Λ≈{float(Λ):.2f})")
    ax.set_title("Inter-arrival histogram")
    ax.set_xlabel("Δt"); ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"inter_arrival_hist.png"), dpi=300); plt.close(fig)

    # B) 计数过程 + 期望（若有 λ(t)）
    fig, ax = plt.subplots(figsize=(6,4))
    ax.step(ts, np.arange(1,len(ts)+1), where="post", label="N(t)")
    t0, t1 = 0.0, ts[-1]
    tline = np.linspace(t0, t1, 300)
    # 若 NHPP 提供了 lambda_of_t，则叠加 E[N(t)] = ∫ λ(u) du
    lam_fn = info.get("_lambda_of_t_callable", None)
    if lam_fn is not None:
        lam_vals = np.array([lam_fn(float(u)) for u in tline], dtype=float)
        En = np.trapz(lam_vals, tline)
        # 规范化为曲线（累积积分）
        cum = np.cumsum((lam_vals[1:] + lam_vals[:-1]) * (tline[1:] - tline[:-1]) / 2.0)
        ax.plot(tline[1:], cum, "--", lw=2, label="E[N(t)] via λ(t)")
    else:
        # 若没有，画直线 Λ t
        ax.plot(tline, float(Λ)*tline, "--", lw=2, label=f"E[N(t)]≈Λ t, Λ≈{float(Λ):.2f}")
    ax.set_title("Counting process vs expectation")
    ax.set_xlabel("t"); ax.set_ylabel("N(t)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"counting_process.png"), dpi=300); plt.close(fig)

    # C) Fano 因子曲线：var(count)/mean(count) 随时间粒度变化
    T = ts[-1]
    if fano_bin_grid is None:
        # 自动构建 8 个尺度：从 P1（约 50 档）到 P99（约 6 档）
        grid = np.geomspace(max(T/200, 1e-3), max(T/6, 1e-3), 8)
    else:
        grid = np.array(sorted(set([g for g in fano_bin_grid if g>0])), dtype=float)

    fanos, means = [], []
    for w in grid:
        edges = np.arange(0.0, T + w, w)
        cnts, _ = np.histogram(ts, bins=edges)
        mu = np.mean(cnts)
        va = np.var(cnts, ddof=1) if len(cnts)>1 else 0.0
        fanos.append((va / max(mu,1e-12)))
        means.append(mu)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(grid, fanos, marker="o")
    ax.axhline(1.0, linestyle="--", lw=1)
    ax.set_xscale("log")
    ax.set_title("Fano factor vs bin width")
    ax.set_xlabel("bin width (seconds)"); ax.set_ylabel("Fano = Var/Mean")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"fano_curve.png"), dpi=300); plt.close(fig)

    # D) ACF of binned counts（lag-1..20）
    if acf_binsize is None:
        acf_binsize = max(T/100.0, 1e-3)
    edges = np.arange(0.0, T + acf_binsize, acf_binsize)
    cnts, _ = np.histogram(ts, bins=edges)
    cnts = cnts - np.mean(cnts)
    acf_lags = min(20, len(cnts)-2) if len(cnts)>=3 else 0
    if acf_lags > 0:
        acf = [1.0]
        for k in range(1, acf_lags+1):
            num = np.dot(cnts[:-k], cnts[k:])
            den = np.dot(cnts, cnts)
            acf.append(float(num/max(den,1e-12)))
        lags = np.arange(0, acf_lags+1)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.stem(lags, acf)
        ax.set_title(f"ACF of binned counts (binsize={acf_binsize:.3g}s)")
        ax.set_xlabel("lag"); ax.set_ylabel("acf")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir,"acf_counts.png"), dpi=300); plt.close(fig)

    # E) 类别 Zipf
    label2idx = {f'{m["name"]}|q{m["q"]}|d{m["d"]}': i for i,m in enumerate(meta)}
    counts = np.zeros(len(meta), dtype=int)
    keys = [f'{w["name"]}|q{w["q"]}|d{w["d"]}' for w in workload]
    for k in keys:
        i = label2idx.get(k, None)
        if i is not None:
            counts[i] += 1
    freq = np.sort(counts)[::-1]
    ranks = np.arange(1, len(freq)+1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(ranks, np.maximum(freq, 1), marker=".", linestyle="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Class frequency Zipf plot")
    ax.set_xlabel("rank (log)"); ax.set_ylabel("count (log)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"class_zipf.png"), dpi=300); plt.close(fig)

    # F) Top-K × time heatmap
    idx_top = np.argsort(-counts)[:min(topk_classes, len(counts))]
    k = len(idx_top)
    if k > 0:
        # 时间分箱：取 60 档或按 N/100 自适应
        B = min(120, max(20, int(len(ts)/50)))
        edges = np.linspace(0.0, T, B+1)
        top_counts_time = np.zeros((k, B), dtype=int)
        for ci, i in enumerate(idx_top):
            # 取该类的到达时刻
            ts_i = np.array([w["t_arr"] for w in workload if label2idx[f'{w["name"]}|q{w["q"]}|d{w["d"]}']==i])
            if ts_i.size > 0:
                c, _ = np.histogram(ts_i, bins=edges)
                top_counts_time[ci,:] = c
        fig, ax = plt.subplots(figsize=(max(7, 0.35*B), max(4, 0.35*k)))
        im = ax.imshow(top_counts_time, aspect="auto", origin="lower",
                       extent=[0, T, 0, k])
        ax.set_yticks(np.arange(k)+0.5)
        ax.set_yticklabels([f'{meta[i]["name"]}|q{meta[i]["q"]}|d{meta[i]["d"]}' for i in idx_top])
        ax.set_xlabel("time"); ax.set_title("Top-K class counts over time")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir,"heat_topk_classes.png"), dpi=300); plt.close(fig)
