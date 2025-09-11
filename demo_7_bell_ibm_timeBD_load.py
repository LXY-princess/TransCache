import pathlib
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size":   13,          # 视论文排版可再调
})

SAVE_DIR = pathlib.Path("./figs")
SAVE_DIR.mkdir(exist_ok=True)

def plotBar(lat_json):
    manual_colors = {
        # Client-side (blue palette suggestion)
        "01_bootstrap":       "#c6dbef",
        "02a_tls_auth":       "#9ecae1",
        "02b_backend_query":  "#6baed6",
        "03_transpile":   "#3182bd",
        "04_upload_to_cloud": "#7CCD7C",
        # Cloud-side (orange palette suggestion)
        "05_to_09_queue":           "#fee6ce",
        "10_11_bindElec_loadPulse": "#f9c28b",
        "12_qpu_exec":           "#fd8d3c",
        "13_14_read_post": "#EE6A50",
        # Back to client
        "15_return_to_client": "#8B7D6B",
    }



    labels, times = zip(*sorted(lat_json.items()))
    cum = 0
    plt.figure(figsize=(10, 3))

    for l, t in zip(labels, times):
        plt.barh([0.5], t,             # y 设 0.5 可以让条居中
                 left=cum,
                 color=manual_colors[l],
                 height=0.25,
                 edgecolor="black",
                 linewidth=0.6,
                 label=l.replace("_", " "),
                 zorder=3)
        cum += t

    ax = plt.gca()
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5, zorder=-2)
    ax.margins(x=0.05)                 # ← 关键：两侧各留 5% 空白
    ax.set_ylim(0, 1)                  # 上下也留一点空白
    plt.xlabel("Time Latency (s)", fontweight="bold")
    plt.ylabel("Bell Circuit", fontweight="bold")
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    png_path = SAVE_DIR / "v7_bell_load_latencies.png"
    plt.savefig(png_path, dpi=600)
    plt.show()
    print("[✔] stacked-bar saved to", png_path)

import json, pathlib

json_path = pathlib.Path("./savedRes/bell_latencies_merge.json")
with json_path.open() as f:
    laps = json.load(f)

plotBar(laps)