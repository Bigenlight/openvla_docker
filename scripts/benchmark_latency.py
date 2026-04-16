#!/usr/bin/env python3
"""Benchmark HTTP communication latency between benchmark container and VLA server.

Measures three components separately:
  1. Image encoding overhead (base64 PNG on the client side)
  2. Pure HTTP round-trip with dummy server (network + FastAPI + deserialization)
  3. Real inference round-trip (HTTP + model forward pass)

Usage (run INSIDE a container that has numpy/PIL/requests, or on host with deps):
    python scripts/benchmark_latency.py --url http://localhost:8600 --rounds 20
    python scripts/benchmark_latency.py --url http://localhost:8600 --rounds 50 --warmup 5
"""
from __future__ import annotations

import argparse
import base64
import io
import statistics
import sys
import time

import numpy as np
import requests
from PIL import Image


def encode_image(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def measure_encoding(img: np.ndarray, rounds: int) -> list[float]:
    """Measure client-side base64 PNG encoding time."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        encode_image(img)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def measure_health(url: str, rounds: int) -> list[float]:
    """Measure GET /health round-trip (minimal payload)."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        r = requests.get(f"{url}/health", timeout=10)
        r.raise_for_status()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def measure_act(url: str, payload: dict, rounds: int, warmup: int) -> list[float]:
    """Measure POST /act round-trip (includes model inference if server is real)."""
    # Warmup
    for _ in range(warmup):
        r = requests.post(f"{url}/act", json=payload, timeout=120)
        r.raise_for_status()

    times_wall = []
    times_server = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/act", json=payload, timeout=120)
        r.raise_for_status()
        wall_ms = (time.perf_counter() - t0) * 1000
        times_wall.append(wall_ms)
        result = r.json()
        if "latency_ms" in result:
            times_server.append(result["latency_ms"])

    return times_wall, times_server


def fmt_stats(times: list[float], label: str) -> str:
    if not times:
        return f"  {label}: no data"
    s = sorted(times)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[int(n * 0.95)] if n >= 20 else s[-1]
    p99 = s[int(n * 0.99)] if n >= 100 else s[-1]
    return (
        f"  {label}:\n"
        f"    n={n}  mean={statistics.mean(s):.1f}ms  std={statistics.stdev(s):.1f}ms\n"
        f"    min={s[0]:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  max={s[-1]:.1f}ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8600")
    parser.add_argument("--rounds", type=int, default=20, help="Number of measurement rounds")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup rounds for /act")
    parser.add_argument("--resolution", type=int, default=256, help="Image resolution")
    args = parser.parse_args()

    print(f"=== VLA HTTP Latency Benchmark ===")
    print(f"URL: {args.url}")
    print(f"Rounds: {args.rounds}, Warmup: {args.warmup}, Resolution: {args.resolution}x{args.resolution}")
    print()

    # Check server
    try:
        info = requests.get(f"{args.url}/health", timeout=5).json()
        print(f"Server: {info.get('model', '?')} (status={info.get('status')})")
        is_dummy = "dummy" in str(info.get("model", ""))
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.url}: {e}")
        sys.exit(1)
    print()

    # --- 1. Client-side image encoding ---
    img = (np.random.rand(args.resolution, args.resolution, 3) * 255).astype(np.uint8)
    enc_times = measure_encoding(img, args.rounds)
    print("1) Client-side base64 PNG encoding")
    print(fmt_stats(enc_times, "encoding"))
    print()

    # --- 2. GET /health round-trip ---
    health_times = measure_health(args.url, args.rounds)
    print("2) GET /health round-trip (minimal, no payload)")
    print(fmt_stats(health_times, "/health"))
    print()

    # --- 3. POST /act round-trip ---
    b64_img = encode_image(img)
    payload = {
        "task": "pick up the black bowl and place it on the plate",
        "observation.images.static": b64_img,
        "observation.state.eef_pos": [0.1, 0.2, 0.3],
        "observation.state.eef_quat": [0.0, 0.0, 0.0, 1.0],
        "observation.state.gripper_qpos": [0.04, -0.04],
    }

    print(f"3) POST /act round-trip ({'DUMMY' if is_dummy else 'REAL inference'})")
    wall_times, server_times = measure_act(args.url, payload, args.rounds, args.warmup)
    print(fmt_stats(wall_times, "wall (client→server→client)"))
    if server_times:
        print(fmt_stats(server_times, "server-side (reported by server)"))
        http_overhead = [w - s for w, s in zip(wall_times, server_times)]
        print(fmt_stats(http_overhead, "HTTP overhead (wall - server)"))
    print()

    # --- 4. Payload size ---
    import json
    payload_bytes = len(json.dumps(payload).encode())
    print(f"4) Payload stats")
    print(f"  Request payload size: {payload_bytes / 1024:.1f} KB")
    print(f"  Image resolution: {args.resolution}x{args.resolution}")
    print(f"  Base64 PNG size: {len(b64_img) / 1024:.1f} KB")
    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if server_times:
        mean_wall = statistics.mean(wall_times)
        mean_server = statistics.mean(server_times)
        mean_overhead = mean_wall - mean_server
        overhead_pct = (mean_overhead / mean_wall) * 100 if mean_wall > 0 else 0
        print(f"  Mean wall latency:    {mean_wall:.1f} ms")
        print(f"  Mean server latency:  {mean_server:.1f} ms")
        print(f"  Mean HTTP overhead:   {mean_overhead:.1f} ms ({overhead_pct:.1f}% of total)")
        print(f"  Mean encoding:        {statistics.mean(enc_times):.1f} ms")
        print(f"  Mean /health RTT:     {statistics.mean(health_times):.1f} ms")
        if mean_overhead > 50:
            print()
            print("  ⚠ HTTP overhead > 50ms — consider:")
            print("    - Using msgpack instead of JSON (smaller payload)")
            print("    - JPEG encoding instead of PNG (faster, smaller)")
            print("    - WebSocket persistent connection (skip TCP handshake)")
            print("    - gRPC with protobuf (binary, streaming)")
        elif mean_overhead > 20:
            print()
            print("  △ HTTP overhead 20-50ms — acceptable for most eval scenarios")
        else:
            print()
            print("  ✓ HTTP overhead < 20ms — negligible")


if __name__ == "__main__":
    main()
