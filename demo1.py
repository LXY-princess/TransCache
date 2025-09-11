#!/usr/bin/env python
# demo1.py  ——  PulseCache mock benchmark (robust version)
import warnings, time
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
from joblib import Memory
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.pulse.exceptions import PulseError

# ------------------------------------------------------------------
# 1) 选择假后端：优先 FakeJakarta，退到 GenericBackendV2
# ------------------------------------------------------------------



backend = None
try:  # Qiskit ≥0.46
    from qiskit.providers.fake_provider.backends.fake_jakarta import FakeJakarta
    backend = FakeJakarta()

except Exception:
    try:  # 0.23–0.45
        from qiskit.providers.fake_provider import FakeJakarta
        backend = FakeJakarta()
    except Exception:
        from qiskit.providers.fake_provider import GenericBackendV2
        backend = GenericBackendV2(num_qubits=7)

backend_name = type(backend).__name__
# GenericBackendV2 没有完整脉冲表 → 不尝试 schedule()
HAS_PULSE = backend_name.lower().startswith("fake") and hasattr(
    backend, "instruction_schedule_map"
)
print(f"✅ Backend in use → {backend_name}  |  HAS_PULSE={HAS_PULSE}")

# ------------------------------------------------------------------
# 2) 缓存 & 运行参数
# ------------------------------------------------------------------
CACHE_DIR = Path("./pulsecache_data")
CACHE_DIR.mkdir(exist_ok=True)
memory = Memory(location=str(CACHE_DIR / "joblib_cache"), verbose=0)

N_RUNS, SHOTS = 30, 4096

# ------------------------------------------------------------------
# 3) 构造一个简单电路
# ------------------------------------------------------------------
def build_circuit(n_qubits: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.rx(np.pi / 4, range(n_qubits))
    qc.measure_all()
    return qc


# ------------------------------------------------------------------
# 4) 冷启动路径
# ------------------------------------------------------------------
def baseline_flow(qc: QuantumCircuit) -> Tuple[float, float, float]:
    t0 = time.perf_counter()
    transpiled = transpile(qc, backend, optimization_level=3)
    t1 = time.perf_counter()

    pulse_dur = 0.0
    if HAS_PULSE:
        try:
            from qiskit import schedule

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
            schedule(transpiled, backend=backend)
        except (PulseError, QiskitError):
            # 后端缺指令或 Pulse 被移除 → 退化为 sleep
            pulse_dur = 0.3
            time.sleep(pulse_dur)
    else:
        pulse_dur = 0.3
        time.sleep(pulse_dur)

    t2 = time.perf_counter()
    time.sleep(0.5)  # 模拟校准 & 队列
    t3 = time.perf_counter()
    return (t3 - t0, t1 - t0, pulse_dur)


# ------------------------------------------------------------------
# 5) 缓存路径
# ------------------------------------------------------------------
@memory.cache
def cached_compile(qc: QuantumCircuit):
    transpiled = transpile(qc, backend, optimization_level=3)
    if HAS_PULSE:
        try:
            from qiskit import schedule

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
            _ = schedule(transpiled, backend=backend)
        except (PulseError, QiskitError):
            pass
    return transpiled, None


def cached_flow(qc: QuantumCircuit) -> Tuple[float, float, float]:
    t0 = time.perf_counter()
    if cached_compile.check_call_in_cache(qc):
        print("Cache hit ✓")
    else:
        print("Cache miss → run transpile+schedule")
    _ = cached_compile(qc)
    time.sleep(0.1)  # δ-cal 占位
    t1 = time.perf_counter()
    return (t1 - t0, 0.0, 0.0)


# ------------------------------------------------------------------
# 6) 统计工具
# ------------------------------------------------------------------
def run(flow_fn, n: int) -> List[float]:
    qc = build_circuit()
    lat_total, lat_comp, lat_pulse = [], [], []
    for _ in range(n):
        total, comp, pulse = flow_fn(qc)
        lat_total.append(total)
        lat_comp.append(comp)
        lat_pulse.append(pulse)
    print(f"{flow_fn.__name__:<12}"
          f" median={np.median(lat_total):.3f}s"
          f"  P95={np.percentile(lat_total,95):.3f}s"
          f"  P99={np.percentile(lat_total,99):.3f}s"
          f"  (compile≈{mean(lat_comp):.3f}s, pulse≈{mean(lat_pulse):.3f}s)")
    return lat_total


def throughput(latencies: List[float], shots: int = SHOTS) -> float:
    return shots / mean(latencies)


# ------------------------------------------------------------------
# 7) 主入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print("=== PulseCache Mock Benchmark ===")
    cold_lat = run(baseline_flow, N_RUNS)
    warm_lat = run(cached_flow, N_RUNS)

    print("\n--- Summary ---")
    print(f"Baseline throughput : {throughput(cold_lat):.1f} shots/s")
    print(f"Cached   throughput : {throughput(warm_lat):.1f} shots/s")
    print(
        f"P99 speed-up        : "
        f"{np.percentile(cold_lat,99)/np.percentile(warm_lat,99):.1f}×"
    )
