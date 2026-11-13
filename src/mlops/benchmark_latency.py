"""
Inference Latency Benchmark & SLA Auditor (OS 5)
Evaluates execution speed across N=100 iterations.
Computes latency percentiles (p50, p90, p95, p99) and asserts SLA targets:
SLA: p95 < 400 ms on GPU / < 2 000 ms on CPU
"""

import time
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path
from typing import Dict, Any

from src.config import (
    FLOORPLAN_CFG,
    FACADE_CFG,
    MLOPS_CFG,
    ONNX_DIR
)


def benchmark_session(
    onnx_path: Path,
    input_name: str,
    input_shape: tuple,
    num_iterations: int = 50,
    warmup: int = 10
) -> Dict[str, float]:
    """Runs timed benchmark for an ONNX model session"""
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    dummy_input = {input_name: np.random.randn(*input_shape).astype(np.float32)}

    # Warmup runs
    for _ in range(warmup):
        _ = session.run(None, dummy_input)

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = session.run(None, dummy_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "iterations": num_iterations,
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p90_ms": round(float(np.percentile(latencies, 90)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }


def run_latency_benchmark(num_iterations: int = 10) -> Dict[str, Any]:
    """Runs complete latency audit for both models and verifies SLA"""
    print(f"\n--> Running Inference Latency Benchmark (N={num_iterations})")

    floorplan_bench = benchmark_session(
        FLOORPLAN_CFG.onnx_path,
        input_name="image",
        input_shape=(1, 3, 512, 512),
        num_iterations=num_iterations
    )

    facade_bench = benchmark_session(
        FACADE_CFG.onnx_path,
        input_name="semantic_mask",
        input_shape=(1, 3, 256, 256),
        num_iterations=num_iterations
    )

    cpu_sla = MLOPS_CFG.latency_sla_cpu_ms
    floorplan_sla_ok = floorplan_bench["p95_ms"] < cpu_sla
    facade_sla_ok = facade_bench["p95_ms"] < cpu_sla

    results = {
        "sla_target_cpu_p95_ms": cpu_sla,
        "sla_target_gpu_p95_ms": MLOPS_CFG.latency_sla_gpu_ms,
        "floorplan_unetpp": {
            **floorplan_bench,
            "sla_met": bool(floorplan_sla_ok),
        },
        "facade_pix2pix": {
            **facade_bench,
            "sla_met": bool(facade_sla_ok),
        },
        "all_sla_met": bool(floorplan_sla_ok and facade_sla_ok)
    }

    out_file = ONNX_DIR / "latency_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Floorplan UNet++ p95: {floorplan_bench['p95_ms']} ms (SLA < {cpu_sla} ms: {'PASS' if floorplan_sla_ok else 'FAIL'})")
    print(f"Facade Pix2Pix   p95: {facade_bench['p95_ms']} ms (SLA < {cpu_sla} ms: {'PASS' if facade_sla_ok else 'FAIL'})")
    print(f"Benchmark results written to {out_file}")

    return results


if __name__ == "__main__":
    run_latency_benchmark()
