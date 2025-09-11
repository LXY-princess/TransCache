import pathlib, pickle, argparse
from collections import Counter

SAVE_DIR = pathlib.Path("./savedRes")
SAVE_DIR.mkdir(exist_ok=True)

def load_primitive(tag, shots):
    path = SAVE_DIR / f"{tag}_{shots}.pkl"
    if path.exists():
        with path.open("rb") as f:
            result = pickle.load(f)
        print(f"[CACHE] Loaded PrimitiveResult from {path}")
        return result
    return None

tag = "ibm_torino"
shots = 1024
cached = load_primitive(tag, shots)
print(cached)

pub   = cached[0]                      # SamplerPubResult

# ---------- BitArray → 计数 ----------
bitarr     = pub.data.c             # BitArray(num_shots, num_bits)
raw_bool  = bitarr._array              # ← 直接拿底层 np.bool_ 矩阵
bitstrs   = ["".join("1" if b else "0" for b in row) for row in raw_bool]
counts    = Counter(bitstrs)           # {'00': 503, '11': 521}
probs     = {b: c / raw_bool.shape[0] for b, c in counts.items()}

print("Integer counts:", counts)
print("Probabilities :", probs)