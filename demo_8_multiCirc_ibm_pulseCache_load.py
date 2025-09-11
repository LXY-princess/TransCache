# bell_real_backend_compare.py  – baseline vs PulseCache (real QPU)
# -----------------------------------------------------------------
import time, json, argparse, pathlib, pickle, hashlib, sys
from datetime import datetime, timezone
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
import matplotlib.pyplot as plt
from qiskit import qasm3
import matplotlib.patches as mpatches
from qiskit.circuit.library import QFT
import numpy as np

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size":   15,          # 视论文排版可再调
})

code_tag = "v8mload_brisbane_q6d2"
load_code_tag = "v8_brisbane/v8_qubits6_depth2/v8m"
FIGDIR   = pathlib.Path("./figs"); FIGDIR.mkdir(exist_ok=True)


# ---- build three depth-2 circuits -------------------------------------------
def make_linear_ent(q=6, d=2):
    qc = QuantumCircuit(q); import numpy as np
    for _ in range(d):
        qc.h(range(q))
        for i in range(q-1):  qc.cx(i,i+1)
        qc.rx(np.pi/4, range(q))
    qc.measure_all(); return qc

def make_ghz_chain(q=6, d=2):
    qc = QuantumCircuit(q); import numpy as np
    qc.h(0);  [qc.cx(i,i+1) for i in range(q-1)]
    for _ in range(d-1):
        qc.rz(np.pi/8, q-1); qc.barrier()
    qc.measure_all(); return qc

def make_qft_like(q=6, d=2):
    qc = QuantumCircuit(q)
    for _ in range(d):
        qc.compose(QFT(num_qubits=q, do_swaps=False).decompose(), range(q), inplace=True)
    qc.measure_all(); return qc

def make_bell():
    qc = QuantumCircuit(2,2); qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1]); return qc

# CIRCUITS = {"bell": make_bell,
#             "LinearEnt": make_linear_ent,
#             "GHZ-Chain": make_ghz_chain,
#             "QFT-Like":  make_qft_like}
CIRCUITS = {"LinearEnt": make_linear_ent,
            "GHZ-Chain": make_ghz_chain,
            "QFT-Like":  make_qft_like}
# CIRCUITS = {"bell": make_bell,}

COLOR_MAP = {
    "02a_tls_auth"       : "#9ecae1",
    "03_transpile"          : "#6baed6",
    "04_submit"          : "#4292c6",
    "05_09_prep"         : "#fedcb2",
    "10_11_bind_load"    : "#fdae6b",
    "12_qpu_exec"        : "#fd8d3c",
    "13_14_read_post"    : "#e6550d",
    # "15_return"          : "#8c6bb1",
    # "16_cache_write"     : "#6e559c",
}

# ── plotting comparison ------------------------------------------------------
def plot_compare(baseline, cache):
    order = [k for k in baseline if k not in ("total")]
    fig, ax = plt.subplots(figsize=(11, 3))
    left = {"Baseline": 0.0, "PulseCache": 0.0}

    for key in order:
        for label, data in [("Baseline", baseline), ("PulseCache", cache)]:
            width = data.get(key, 0.0)
            ax.barh(label, width, left=left[label],
                    color=COLOR_MAP[key], edgecolor="black", height=0.35)
            left[label] += width

    ax.set_xlabel("Latency (s)")
    ax.set_xlim(0, max(baseline["total"], cache["total"]) * 1.2)
    ax.margins(x=0.05)

    # -------- legend right side --------------
    handles = [mpatches.Patch(color=COLOR_MAP[k], label=k) for k in order]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = FIGDIR / F"{load_code_tag}load_lat_compare.png"
    plt.savefig(out, dpi=600)
    plt.show()
    print("figure saved ->", out)

def plot_all(res_dict):
    fig,ax=plt.subplots(figsize=(12,2+1.4*len(res_dict)))
    ytick, ylbl = [],[]
    order = list(COLOR_MAP.keys())
    for i,(name,(base,cache)) in enumerate(res_dict.items()):
        for j,(mode,laps) in enumerate([("Baseline",base),("PulseCache",cache)]):
            y = i*1.4 + 0.4*j
            left=0
            for k in order:
                w=laps.get(k,0)
                ax.barh(y,w,left=left,color=COLOR_MAP[k],edgecolor="black",height=0.2, zorder=2)
                left+=w
            # ax.text(left+0.05,y,f"{left:.2f}s",va="center",fontsize=8)
            ytick.append(y); ylbl.append(f"{name} – {mode}")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_yticks(ytick); ax.set_yticklabels(ylbl, fontweight="bold")
    ax.set_xlabel("Latency (s)", fontweight="bold"); ax.margins(x=0.05)
    handles=[mpatches.Patch(color=COLOR_MAP[k],label=k) for k in order]
    # ★ 把四条外围框线加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.legend(handles=handles,bbox_to_anchor=(1.02,1),loc="upper left")
    plt.tight_layout()
    out=FIGDIR/f"{code_tag}_compare_multi3.png"
    plt.savefig(out,dpi=600)
    plt.show()
    print("figure saved ->",out)

def load_laps(path: pathlib.Path):
    with path.open() as f:
        return json.load(f)

# ── CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_torino")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    results = {}
    for name, fn in CIRCUITS.items():
        print(f"▶ {name} baseline")
        base_runs = []
        for r in range(args.runs):
            base = load_laps(FIGDIR / f"{load_code_tag}_{name}_baseline_{r}.json")
            base_runs.append(base)
        base_sum = {k: round(sum(r[k] for r in base_runs), 4) for k in base_runs[0]}
        # base_sum = load_laps(FIGDIR / f"{load_code_tag}_{name}_baseline_sum.json")

        print(f"▶ {name} PulseCache")
        pc_runs = []
        for r in range(args.runs):
            pc = load_laps(FIGDIR / f"{load_code_tag}_{name}_pulsecache{r}.json")
            pc_runs.append(pc)
        cache_sum = {k: round(sum(r[k] for r in pc_runs), 4) for k in pc_runs[0]}
        # cache_sum = load_laps(FIGDIR / f"{load_code_tag}_{name}_pulsecache_sum.json")
        results[name] = (base_sum, cache_sum)

    plot_all(results)
