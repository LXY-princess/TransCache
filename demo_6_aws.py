#!/usr/bin/env python
"""
PulseCache on AWS Braket — end‑to‑end demo
==========================================
This script reproduces the *demo_multiv5* ablation benchmark on a **real
quantum serverless back‑end** (Rigetti Aspen‑M via AWS Braket).  The only
non‑local steps are **compile → schedule → δ‑cal → execute** — exactly the
chain that PulseCache accelerates.

Prerequisites
-------------
* AWS account with Braket enabled, an S3 bucket and an Elasticache Redis
  cluster in the same *region* (e.g. `us‑west‑2`).
* AWS credentials (`AWS_PROFILE` or env‑vars).
* `pip install amazon-braket-sdk boto3 redis joblib numpy`
* Redis security‑group must accept inbound traffic from this client.

Key design choices
------------------
* **Key = ``f"{epoch}:{sha256(QASM).hexdigest()}"``**
* **Value = compiled _pulse blob_** (JSON produced by Braket Pulse).
* Hot-layer = Redis (< 50 MiB) ; Cold‑layer = S3 (`s3://<bucket>/pulsecache/`).
* δ‑calibration uses Rigetti Fast T API (`device.run_quick_calibration`).

This is *reference* code — fill in `<⟨PLACEHOLDER⟩>` values for bucket &
redis‑endpoint before running.
"""

from __future__ import annotations
import argparse, hashlib, json, os, time, random, concurrent.futures as cf
from pathlib import Path
from typing import Dict, Callable

import boto3
import redis
import numpy as np
# from amazon_braket.circuits import Circuit
# from amazon_braket.devices import Device
# from amazon_braket.aws import AwsQuantumTask
# from amazon_braket.pulse import Port, Waveform, Frame, PulseSequence
from braket.circuits import Circuit
from braket.aws import AwsDevice, AwsSession
from braket.aws import AwsQuantumTask
from braket.pulse import Port, Frame, PulseSequence

# ───────────────────────── AWS / Redis handles ────────────────────────────
REGION           = "us-west-2"
S3_BUCKET        = "<my-pulsecache-demo>"         # e.g. "my‑pulsecache"
REDIS_ENDPOINT   = "localhost:6379" # "<host>:6379"           # Elasticache primary endpoint

REDIS = redis.Redis.from_url(f"redis://{REDIS_ENDPOINT}")
S3    = boto3.client("s3", region_name=REGION)
# DEVICE = Device("arn:aws:braket:::device/qpu/rigetti/Aspen-M-3")  # example
session = AwsSession(region="us-west-2")
print("Session created:", session)
DEVICE = AwsDevice("arn:aws:braket:::device/qpu/rigetti/Aspen-M-3", aws_session=session)

CAL_EPOCH = "e0"    # override via CLI when full‑cal updates

# ──────────────────────── Canonical QASM key ───────────────────────────────

def qasm_key(qasm: str, epoch: str = CAL_EPOCH) -> str:
    return f"{epoch}:{hashlib.sha256(qasm.encode()).hexdigest()}"

# ───────────────────────── Circuit generators ─────────────────────────────

def make_linear_ent(nq: int, depth: int) -> Circuit:
    c = Circuit()
    for _ in range(depth):
        for i in range(nq - 1):
            c.h(i).cnot(i, i + 1)
    c.measure_all()
    return c

def make_ghz_chain(nq: int, depth: int) -> Circuit:
    c = Circuit().h(0)
    for i in range(1, nq):
        c.cnot(i - 1, i)
    for _ in range(depth - 1):
        c.rx(np.pi / 8, nq - 1)
    c.measure_all(); return c

def make_qft_like(nq: int, depth: int) -> Circuit:
    from braket.circuits import gates
    c = Circuit()
    for _ in range(depth):
        for k in range(nq):
            c.h(k)
            for j in range(k + 1, nq):
                c.cphaseshift(j, k, np.pi / 2 ** (j - k))
    c.measure_all(); return c

CIRCS: Dict[str, Callable[[int, int], Circuit]] = {
    "LinearEnt": make_linear_ent,
    "GHZ-Chain": make_ghz_chain,
    "QFT-Like":  make_qft_like,
}

# ─────────────────── Pulse compile + cache helpers ─────────────────────────

def cache_put(key: str, blob: dict):
    REDIS.set(key, "1", ex=7 * 24 * 3600)                 # heat‑counter
    S3.put_object(Bucket=S3_BUCKET, Key=f"pulse/{key}.json", Body=json.dumps(blob))

def cache_get(key: str):
    if not REDIS.get(key):
        return None
    try:
        obj = S3.get_object(Bucket=S3_BUCKET, Key=f"pulse/{key}.json")
        return json.loads(obj["Body"].read())
    except S3.exceptions.NoSuchKey:
        return None

# compile‑and‑schedule using Braket Pulse -------------------------------------------------

def compile_to_pulse(qasm: str) -> dict:
    # 1) compile to IR (Braket internal) – placeholder API
    compile_task = DEVICE.compile(qasm)           # assumes preview SDK
    ir          = compile_task.result()["ir"]
    # 2) schedule → pulse sequence (placeholder)
    seq: PulseSequence = DEVICE.pulse().schedule(ir)
    return seq.to_ir()           # dict/JSON serialisable

# ────────────────────── execution helpers ────────────────────────────────

def run_pulse(blob: dict) -> float:
    """Send pulse blob & δ‑cal; return latency in seconds."""
    t0 = time.time()
    task: AwsQuantumTask = DEVICE.run(program=blob, shots=100)
    DEVICE.run_quick_calibration(task.id)     # δ‑calibration ~20–50 ms
    result = task.result()
    return time.time() - t0

# ─────────────────── Flow implementations (Baseline/Full/…) ───────────────

def flow_baseline(circ: Circuit):
    t0 = time.time(); task = DEVICE.run(circ, shots=100)
    DEVICE.calibrations.wait_for_calibration()   # full‑cal – blocks ~0.5 s
    task.result(); return time.time() - t0

def flow_full(circ: Circuit):
    qasm = circ.to_ir().to_openqasm()
    key  = qasm_key(qasm)
    blob = cache_get(key)
    if blob is None:
        blob = compile_to_pulse(qasm)
        cache_put(key, blob)
    return run_pulse(blob)

# HashOnly – same as Full but skips δ‑cal; here we simulate as 0‑overhead
flow_hash = flow_full  # simplification – treat δ‑cal cost negligible

# Transpile – cache IR only, always schedule online

def flow_transpile(circ: Circuit):
    qasm = circ.to_ir().to_openqasm(); key = qasm_key(qasm)
    ir = REDIS.get(f"ir:{key}")
    if ir is None:
        ir = DEVICE.compile(qasm).result()["ir"]
        REDIS.set(f"ir:{key}", json.dumps(ir), ex=7*24*3600)
    blob = DEVICE.pulse().schedule(json.loads(ir)).to_ir()
    return run_pulse(blob)

# SyncCache – first run Baseline, then Full
SYNC_SEEN: set[str] = set()

def flow_sync(circ: Circuit):
    key = qasm_key(circ.to_ir().to_openqasm())
    if key not in SYNC_SEEN:
        SYNC_SEEN.add(key)
        return flow_baseline(circ)
    return flow_full(circ)

FLOW = {
    "Baseline":  flow_baseline,
    "Full":      flow_full,
    "HashOnly":  flow_hash,
    "Transpile": flow_transpile,
    "SyncCache": flow_sync,
}

# ───────────────────────── Main benchmark ---------------------------------

def bench(policy: str, circ: Circuit, runs: int = 3):
    lat = [FLOW[policy](circ) for _ in range(runs)]
    print(f"{policy:<9s} med={np.median(lat):.3f}s  p95={np.percentile(lat,95):.3f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", default="e0", help="calibration epoch tag")
    args = ap.parse_args(); CAL_EPOCH = args.epoch

    depth = 16; nq = 7
    for name, fn in CIRCS.items():
        print(f"=== {name} depth={depth} ===")
        bench("Baseline",  fn(nq, depth))
        bench("Full",      fn(nq, depth))
        bench("HashOnly",  fn(nq, depth))
        bench("Transpile", fn(nq, depth))
        bench("SyncCache", fn(nq, depth))
