"""
Inference Latency Benchmark & SLA Auditor (OS 5)
Evaluates execution speed across N iterations (default 10, see params.yaml).
Computes latency percentiles (p50, p90, p95, p99) and checks the SLA targets:
SLA: p95 < 400 ms on GPU / < 2 000 ms on CPU

A missed SLA is reported as a GitHub Actions warning annotation (::warning::) and in the
step summary, but does NOT fail the run by default: CI runs on slow shared CPUs, where the
figure is informative only. Set ARCHIGEN_LATENCY_STRICT=1 (or pass --strict) to exit 1 on a miss.
"""

import os
import sys
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

    rows = [("Floorplan UNet++", floorplan_bench, floorplan_sla_ok), ("Facade Pix2Pix", facade_bench, facade_sla_ok)]
    for name, bench, ok in rows:
        print(f"{name:17s} p95: {bench['p95_ms']} ms (SLA < {cpu_sla} ms: {'PASS' if ok else 'FAIL'})")
        if not ok:
            # GitHub Actions turns this line into a visible warning annotation on the run summary.
            print(f"::warning title=Latency SLA missed::{name} p95 = {bench['p95_ms']} ms > {cpu_sla} ms "
                  f"(CPU, N={num_iterations}). Informative only on shared CI runners.")
    print(f"Benchmark results written to {out_file}")

    # Step summary table (only inside GitHub Actions, where GITHUB_STEP_SUMMARY is set).
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        lines = ["### Inference latency (CPU, ONNX Runtime)", "",
                 f"| Model | p50 ms | p95 ms | SLA p95 | Status |", "| :--- | ---: | ---: | ---: | :---: |"]
        for name, bench, ok in rows:
            lines.append(f"| {name} | {bench['p50_ms']} | {bench['p95_ms']} | < {cpu_sla:g} | {'PASS' if ok else 'WARN'} |")
        lines.append("")
        lines.append("A missed SLA is a warning here (shared runner); set `ARCHIGEN_LATENCY_STRICT=1` to make it fail.")
        with open(summary_path, "a", encoding="utf8") as f:
            f.write("\n".join(lines) + "\n")

    return results


if __name__ == "__main__":
    strict = "--strict" in sys.argv[1:] or os.environ.get("ARCHIGEN_LATENCY_STRICT") == "1"
    res = run_latency_benchmark()
    if strict and not res["all_sla_met"]:
        print("::error title=Latency SLA missed::Strict mode enabled: failing the run.")
        sys.exit(1)
