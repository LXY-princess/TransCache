# v10_multiC_sim_gateLevel.py — SIM backend + transpile-cache (merged v5 backend detection)

import time, json, argparse, pathlib, pickle, hashlib
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from qiskit import QuantumCircuit, transpile, qasm3
from qiskit.circuit.library import QFT
from qiskit_aer.primitives import Sampler as AerSampler

# ================= Figure & cache =================
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 15})
FIGDIR  = pathlib.Path("./figs/v10sd/"); FIGDIR.mkdir(exist_ok=True)
CACHE_P = pathlib.Path("./figs/v10sd/v10_sim_transpile_cache.pkl")   # {key: qc_transpiled}
code_tag = "v10"

COLOR_MAP = {
    "01_setup"       : "#9ecae1",
    "02_cache_search": "#00E5EE",
    "03_transpile"   : "#4292c6",
    "04_submit"      : "#8FBC8F",
    "05_run"         : "#fedcb2",
    "06_collect"     : "#E9967A",
    "07_cache_write" : "#6e559c",
}

# ================ v5-style backend detection (merged) ================
def autodetect_fake_backend():
    """
    v5 的多路径 Fake 后端探测：依次尝试不同包路径的 FakeJakarta，
    全部失败则回退 GenericBackendV2(num_qubits=7):contentReference[oaicite:2]{index=2}。
    """
    backend = None
    for mod, cls in [
        ("qiskit_ibm_runtime.fake_provider.backends.fake_jakarta", "FakeJakarta"),
        ("qiskit.providers.fake_provider.backends.fake_jakarta",   "FakeJakarta"),
        ("qiskit.providers.fake_provider",                         "FakeJakarta"),
        ("qiskit.test.mock",                                       "FakeJakarta"),
    ]:
        try:
            backend = getattr(__import__(mod, fromlist=[cls]), cls)()
            break
        except (ImportError, AttributeError):
            continue
    if backend is None:
        from qiskit.providers.fake_provider import GenericBackendV2
        backend = GenericBackendV2(num_qubits=7)
    return backend

def get_backend_by_name(name: str):
    """
    优先按名字拿指定 Fake 设备；拿不到就走 autodetect。
    等价于 v10 的 get_fake_backend，但兼容 v5 的导入路径:contentReference[oaicite:3]{index=3}。
    """
    n = (name or "").lower()
    # 尝试 ibm_runtime.fake_provider
    try:
        from qiskit_ibm_runtime.fake_provider import FakeJakarta, FakeSherbrooke  # type: ignore
        mp = {
            "fake_jakarta": FakeJakarta, "jakarta": FakeJakarta,
            "fake_sherbrooke": FakeSherbrooke, "sherbrooke": FakeSherbrooke,
        }
        if n in mp: return mp[n]()
    except Exception:
        pass
    # 回退 terra fake_provider
    try:
        from qiskit.providers.fake_provider import FakeJakarta, FakeSherbrooke  # type: ignore
        mp = {
            "fake_jakarta": FakeJakarta, "jakarta": FakeJakarta,
            "fake_sherbrooke": FakeSherbrooke, "sherbrooke": FakeSherbrooke,
        }
        if n in mp: return mp[n]()
    except Exception:
        pass
    # 最后兜底：v5 的自动探测
    return autodetect_fake_backend()

def get_backend_name(backend):
    """兼容 backend.name 是属性或方法两种实现；否则取 configuration().backend_name。"""
    if hasattr(backend, "name"):
        n = backend.name
        return n() if callable(n) else n
    try:
        return backend.configuration().backend_name
    except Exception:
        return type(backend).__name__

# ================= helpers =================
def now_perf(): return time.perf_counter()
def dsec(t0):   return round(time.perf_counter() - t0, 6)

def md5_qasm(circ: QuantumCircuit) -> str:
    """Stable MD5 on QASM (兼容 Terra 1.x/2.x)"""
    try:
        txt = circ.qasm()
    except AttributeError:
        txt = qasm3.dumps(circ)
    return hashlib.md5(txt.encode()).hexdigest()

def load_cache() -> Dict[str, QuantumCircuit]:
    return pickle.loads(CACHE_P.read_bytes()) if CACHE_P.exists() else {}

def save_cache(c: Dict[str, QuantumCircuit]) -> None:
    CACHE_P.write_bytes(pickle.dumps(c))

# ================= circuits =================
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
        qc.compose(QFT(num_qubits=q, do_swaps=False).decompose(), range(q), inplace=True)
    qc.measure(range(q), range(q))
    return qc

CIRCUITS = {
    "LinearEnt": make_linear_ent,
    "GHZ-Chain": make_ghz_chain,
    "QFT-Like":  make_qft_like,
}

# ================= single run =================
def run_once(qc_func, mode: str, backend, shots: int):
    """
    mode: 'baseline'（每次都 transpile）| 'cache'（命中则跳过 transpile）
    返回：laps(dict) 每阶段耗时；meta(dict) 结果/统计
    """
    laps, t0 = {}, None
    def tic(k): nonlocal t0; t0 = now_perf(); laps[k] = -1.0
    def toc(k): laps[k] = dsec(t0)

    # 01 setup
    tic("01_setup")
    sampler = AerSampler()
    bk_name = get_backend_name(backend)
    toc("01_setup")

    # 02 make qc
    # tic("02_make_qc")
    qc_raw = qc_func()
    # toc("02_make_qc")

    # 03 transpile  （仅缓存 transpiled QuantumCircuit）
    if mode == "cache":
        cache = load_cache()
        tic("02_cache_search")
        key = f"{bk_name}:{md5_qasm(qc_raw)}"
        if key in cache:
            qc_exec = cache[key]
            toc("02_cache_search")
        else:
            toc("02_cache_search")
            tic("03_transpile")
            qc_exec = transpile(qc_raw, backend=backend, optimization_level=3)
            toc("03_transpile")
            tic("07_cache_write")
            cache[key] = qc_exec
            save_cache(cache)
            toc("07_cache_write")
    else:
        # Mode baseline
        tic("03_transpile")
        qc_exec = transpile(qc_raw, backend=backend, optimization_level=3)
        toc("03_transpile")

    # 04 sample run
    tic("04_submit")
    job = sampler.run([qc_exec], shots=shots)
    toc("04_submit")
    tic("05_run")
    res = job.result()
    toc("05_run")

    # 05 collect
    tic("06_collect")
    quasi = res.quasi_dists[0] if hasattr(res, "quasi_dists") else None
    meta = {
        "backend": bk_name,
        "n_qubits": qc_raw.num_qubits,
        "depth_in": qc_raw.depth(),
        "depth_transpiled": qc_exec.depth(),
        "size_transpiled": qc_exec.size(),
        "quasi_dist": dict(quasi) if quasi is not None else None,
    }
    toc("06_collect")

    laps["total"] = round(sum(v for k, v in laps.items() if k != "total"), 6)
    return laps, meta

# ================= plotters =================
def plot_compare(baseline, cache, out_png: pathlib.Path):
    order = [k for k in baseline if k != "total"]
    fig, ax = plt.subplots(figsize=(11, 3))
    left = {"Baseline": 0.0, "TranspileCache": 0.0}
    for key in order:
        for label, data in [("Baseline", baseline), ("TranspileCache", cache)]:
            w = data.get(key, 0.0)
            ax.barh(label, w, left=left[label],
                    color=COLOR_MAP.get(key, "#999999"),
                    edgecolor="black", height=0.35)
            left[label] += w
    ax.set_xlabel("Latency (s)")
    ax.set_xlim(0, max(baseline.get("total",0), cache.get("total",0)) * 1.2 + 1e-9)
    ax.margins(x=0.05)
    handles = [mpatches.Patch(color=COLOR_MAP.get(k, "#999999"), label=k) for k in order]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_png, dpi=600); plt.show()
    print("figure saved ->", out_png)

def plot_all(res_dict, out_png: pathlib.Path):
    fig, ax = plt.subplots(figsize=(12, 2 + 1.4 * len(res_dict)))
    ytick, ylbl = [], []
    order = list(COLOR_MAP.keys())
    for i, (name, (base, cache)) in enumerate(res_dict.items()):
        for j, (mode, laps) in enumerate([("Baseline", base), ("TranspileCache", cache)]):
            y = i * 1.4 + 0.4 * j
            left = 0.0
            for k in order:
                w = laps.get(k, 0.0)
                ax.barh(y, w, left=left, color=COLOR_MAP.get(k, "#999999"),
                        edgecolor="black", height=0.2, zorder=2)
                left += w
            ytick.append(y); ylbl.append(f"{name} – {mode}")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_yticks(ytick); ax.set_yticklabels(ylbl, fontweight="bold")
    ax.set_xlabel("Latency (s)", fontweight="bold"); ax.margins(x=0.05)
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    handles = [mpatches.Patch(color=COLOR_MAP.get(k, "#999999"), label=k) for k in order]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(out_png, dpi=600)
    # plt.show()
    print("figure saved ->", out_png)

# ================= batch runners =================
def run_circuits(runs: int, backend, shots: int, q: int, d: int):
    results = {}
    qd_tag = f"q{q}_d{d}"
    for name, make in CIRCUITS.items():
        fn = (lambda q_=q, d_=d: (lambda: make(q_, d_)))()
        print(f"▶ {name} baseline")
        base_runs = []
        for r in range(runs):
            laps, meta = run_once(fn, "baseline", backend, shots)
            base_runs.append(laps)
            (FIGDIR / f"{code_tag}_{qd_tag}_sim_{name}_baseline_{r}.json").write_text(json.dumps({"laps": laps, "meta": meta}, indent=2))
        base_sum = {k: round(sum(rr.get(k, 0.0) for rr in base_runs), 6) for k in base_runs[0]}
        (FIGDIR / f"{code_tag}_{qd_tag}_sim_{name}_baseline_sum.json").write_text(json.dumps(base_sum, indent=2))

        print(f"▶ {name} TranspileCache")
        pc_runs = []
        for r in range(runs):
            laps, meta = run_once(fn, "cache", backend, shots)
            pc_runs.append(laps)
            (FIGDIR / f"{code_tag}_{qd_tag}_sim_{name}_tcache_{r}.json").write_text(json.dumps({"laps": laps, "meta": meta}, indent=2))
        cache_sum = {k: round(sum(rr.get(k, 0.0) for rr in pc_runs), 6) for k in pc_runs[0]}
        (FIGDIR / f"{code_tag}_{qd_tag}_sim_{name}_tcache_sum.json").write_text(json.dumps(cache_sum, indent=2))

        results[name] = (base_sum, cache_sum)

    plot_all(results, FIGDIR / f"{code_tag}_{qd_tag}_sim_compare_multi.png")

def run_compare(cir_name: str, runs: int, backend, shots: int):
    fn = CIRCUITS[cir_name]
    print(f"▶ {cir_name} baseline")
    base_runs = []
    for r in range(runs):
        laps, meta = run_once(fn, "baseline", backend, shots)
        base_runs.append(laps)
        (FIGDIR / f"sim_{cir_name}_baseline_{r}.json").write_text(json.dumps({"laps": laps, "meta": meta}, indent=2))
    base_sum = {k: round(sum(rr.get(k, 0.0) for rr in base_runs), 6) for k in base_runs[0]}
    (FIGDIR / f"sim_{cir_name}_baseline_sum.json").write_text(json.dumps(base_sum, indent=2))

    print(f"▶ {cir_name} TranspileCache")
    pc_runs = []
    for r in range(runs):
        laps, meta = run_once(fn, "cache", backend, shots)
        pc_runs.append(laps)
        (FIGDIR / f"sim_{cir_name}_tcache_{r}.json").write_text(json.dumps({"laps": laps, "meta": meta}, indent=2))
    cache_sum = {k: round(sum(rr.get(k, 0.0) for rr in pc_runs), 6) for k in pc_runs[0]}
    (FIGDIR / f"sim_{cir_name}_tcache_sum.json").write_text(json.dumps(cache_sum, indent=2))

    plot_compare(base_sum, cache_sum, FIGDIR / f"{code_tag}_sim_lat_compare.png")

# ================= CLI =================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake_jakarta", help="fake_jakarta | fake_sherbrooke | (auto if missing)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--circuit", default="", help="只跑某个电路名（留空则跑全部）")
    args = ap.parse_args()

    # 使用 v5 风格的多路径导入 + 兜底；优先按用户给定名取设备
    backend = get_backend_by_name(args.backend)   # v10 结合 v5 的取后端方式:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
    print("Using backend:", get_backend_name(backend))

    # if args.circuit:
    #     run_compare(args.circuit, args.runs, backend, args.shots)
    # else:
    DEPTHS = [1, 2, 4, 8, 16, 32]
    QUBITS = [3, 5, 7]
    # DEPTHS = [32]
    for d in DEPTHS:
        for q in QUBITS:
            run_circuits(args.runs, backend, args.shots, q, d)
