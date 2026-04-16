#!/usr/bin/env python3
"""Stress-test HTTP overhead under various payload/concurrency scenarios.

Tests:
  A) Payload size: 1 cam → 2 cam → 4 cam → 4 cam + depth (float32 encoded)
  B) Resolution: 256 → 512 → 1024
  C) Concurrent requests: 1 → 2 → 4 → 8 simultaneous /act calls
  D) Network mode: host vs bridge (requires separate runs)

Each test uses --dummy mode on the server so we isolate HTTP overhead only.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import statistics
import struct
import time

import numpy as np
import requests
from PIL import Image


def encode_image_png(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def encode_image_jpeg(img: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def encode_depth_float32(depth: np.ndarray) -> str:
    """Encode HxW float32 depth map as raw base64 (no image format)."""
    return base64.b64encode(depth.astype(np.float32).tobytes()).decode()


def make_payload(
    n_cams: int = 1,
    resolution: int = 256,
    include_depth: bool = False,
    use_jpeg: bool = False,
    include_state: bool = True,
) -> dict:
    payload = {"task": "pick up the black bowl and place it on the plate"}
    cam_names = ["static", "wrist", "overhead", "side"]
    for i in range(n_cams):
        img = (np.random.rand(resolution, resolution, 3) * 255).astype(np.uint8)
        if use_jpeg:
            payload[f"observation.images.{cam_names[i]}"] = encode_image_jpeg(img)
        else:
            payload[f"observation.images.{cam_names[i]}"] = encode_image_png(img)
    if include_depth:
        depth = np.random.rand(resolution, resolution).astype(np.float32)
        payload["observation.images.depth_static"] = encode_depth_float32(depth)
    if include_state:
        payload["observation.state.eef_pos"] = [0.1, 0.2, 0.3]
        payload["observation.state.eef_quat"] = [0.0, 0.0, 0.0, 1.0]
        payload["observation.state.gripper_qpos"] = [0.04, -0.04]
        payload["observation.state.joint_pos"] = [0.0] * 7
    return payload


def measure_single(url: str, payload: dict, rounds: int, warmup: int) -> dict:
    for _ in range(warmup):
        requests.post(f"{url}/act", json=payload, timeout=30).raise_for_status()

    wall_times = []
    server_times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        r = requests.post(f"{url}/act", json=payload, timeout=30)
        r.raise_for_status()
        wall_ms = (time.perf_counter() - t0) * 1000
        wall_times.append(wall_ms)
        result = r.json()
        if "latency_ms" in result:
            server_times.append(result["latency_ms"])

    overhead = [w - s for w, s in zip(wall_times, server_times)] if server_times else wall_times
    payload_kb = len(json.dumps(payload).encode()) / 1024
    return {
        "wall_mean": statistics.mean(wall_times),
        "wall_p50": sorted(wall_times)[len(wall_times) // 2],
        "wall_p95": sorted(wall_times)[int(len(wall_times) * 0.95)] if len(wall_times) >= 20 else max(wall_times),
        "server_mean": statistics.mean(server_times) if server_times else 0,
        "overhead_mean": statistics.mean(overhead),
        "overhead_p95": sorted(overhead)[int(len(overhead) * 0.95)] if len(overhead) >= 20 else max(overhead),
        "payload_kb": payload_kb,
    }


def measure_concurrent(url: str, payload: dict, n_concurrent: int, rounds: int) -> dict:
    """Fire n_concurrent requests simultaneously, measure total throughput + per-request latency."""
    # warmup
    for _ in range(3):
        requests.post(f"{url}/act", json=payload, timeout=30)

    all_wall = []
    batch_times = []

    for _ in range(rounds):
        def _one_request(_):
            t0 = time.perf_counter()
            r = requests.post(f"{url}/act", json=payload, timeout=60)
            r.raise_for_status()
            return (time.perf_counter() - t0) * 1000, r.json().get("latency_ms", 0)

        batch_t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
            futures = [ex.submit(_one_request, i) for i in range(n_concurrent)]
            results = [f.result() for f in futures]
        batch_ms = (time.perf_counter() - batch_t0) * 1000
        batch_times.append(batch_ms)
        for wall, server in results:
            all_wall.append(wall)

    return {
        "n_concurrent": n_concurrent,
        "per_request_wall_mean": statistics.mean(all_wall),
        "per_request_wall_p95": sorted(all_wall)[int(len(all_wall) * 0.95)] if len(all_wall) >= 20 else max(all_wall),
        "batch_mean": statistics.mean(batch_times),
        "throughput_rps": n_concurrent / (statistics.mean(batch_times) / 1000),
    }


def fmt(val, unit="ms"):
    return f"{val:.1f}{unit}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8600")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    url = args.url
    rounds = args.rounds
    warmup = args.warmup

    info = requests.get(f"{url}/health", timeout=5).json()
    print(f"Server: {info.get('model')} (status={info.get('status')})")
    print(f"Rounds: {rounds}, Warmup: {warmup}")
    print()

    # ================================================================
    # TEST A: Payload size (number of cameras)
    # ================================================================
    print("=" * 70)
    print("TEST A: Number of cameras (256x256, PNG)")
    print("=" * 70)
    print(f"{'Cameras':>10} {'Payload':>10} {'Wall':>10} {'Server':>10} {'Overhead':>10} {'OH p95':>10}")
    print("-" * 70)
    for n_cams in [1, 2, 4]:
        p = make_payload(n_cams=n_cams, resolution=256)
        r = measure_single(url, p, rounds, warmup)
        print(f"{n_cams:>10} {fmt(r['payload_kb'], 'KB'):>10} {fmt(r['wall_mean']):>10} "
              f"{fmt(r['server_mean']):>10} {fmt(r['overhead_mean']):>10} {fmt(r['overhead_p95']):>10}")

    # with depth
    p = make_payload(n_cams=2, resolution=256, include_depth=True)
    r = measure_single(url, p, rounds, warmup)
    print(f"{'2+depth':>10} {fmt(r['payload_kb'], 'KB'):>10} {fmt(r['wall_mean']):>10} "
          f"{fmt(r['server_mean']):>10} {fmt(r['overhead_mean']):>10} {fmt(r['overhead_p95']):>10}")
    print()

    # ================================================================
    # TEST B: Resolution scaling
    # ================================================================
    print("=" * 70)
    print("TEST B: Resolution scaling (1 camera, PNG)")
    print("=" * 70)
    print(f"{'Resolution':>10} {'Payload':>10} {'Wall':>10} {'Server':>10} {'Overhead':>10} {'OH p95':>10}")
    print("-" * 70)
    for res in [128, 256, 512, 1024]:
        p = make_payload(n_cams=1, resolution=res)
        r = measure_single(url, p, rounds, warmup)
        print(f"{f'{res}x{res}':>10} {fmt(r['payload_kb'], 'KB'):>10} {fmt(r['wall_mean']):>10} "
              f"{fmt(r['server_mean']):>10} {fmt(r['overhead_mean']):>10} {fmt(r['overhead_p95']):>10}")
    print()

    # ================================================================
    # TEST C: PNG vs JPEG encoding
    # ================================================================
    print("=" * 70)
    print("TEST C: PNG vs JPEG (2 cameras, 256x256)")
    print("=" * 70)
    print(f"{'Format':>10} {'Payload':>10} {'Wall':>10} {'Server':>10} {'Overhead':>10}")
    print("-" * 70)
    for use_jpeg, label in [(False, "PNG"), (True, "JPEG")]:
        p = make_payload(n_cams=2, resolution=256, use_jpeg=use_jpeg)
        r = measure_single(url, p, rounds, warmup)
        print(f"{label:>10} {fmt(r['payload_kb'], 'KB'):>10} {fmt(r['wall_mean']):>10} "
              f"{fmt(r['server_mean']):>10} {fmt(r['overhead_mean']):>10}")
    print()

    # ================================================================
    # TEST D: Concurrent requests
    # ================================================================
    print("=" * 70)
    print("TEST D: Concurrent requests (1 cam, 256x256)")
    print("=" * 70)
    print(f"{'Concurrent':>10} {'Per-req':>10} {'Per-req p95':>12} {'Batch':>10} {'Throughput':>12}")
    print("-" * 70)
    p = make_payload(n_cams=1, resolution=256)
    for n in [1, 2, 4, 8]:
        r = measure_concurrent(url, p, n, rounds=max(5, rounds // 2))
        print(f"{n:>10} {fmt(r['per_request_wall_mean']):>10} {fmt(r['per_request_wall_p95']):>12} "
              f"{fmt(r['batch_mean']):>10} {fmt(r['throughput_rps'], ' rps'):>12}")
    print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)


if __name__ == "__main__":
    main()
