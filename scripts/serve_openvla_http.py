"""FastAPI HTTP server exposing OpenVLA (LIBERO finetune) via the unified VLA protocol.

Implements the three-endpoint contract defined in VLA_COMMUNICATION_PROTOCOL.md:

    GET  /health  →  {status, model, action_type, action_keys, n_action_steps}
    POST /reset   →  {status}
    POST /act     →  {action.eef_pos, action.eef_euler, action.gripper, latency_ms}

The server accepts generic observations keyed by `observation.images.{cam}` (base64 PNG)
and `observation.state.{field}` (flat list), plus a `task` string. It then performs the
OpenVLA-LIBERO specific preprocessing (180° rotation, lanczos3 resize to 224x224,
center crop, gripper binarize + invert) so that every benchmark can talk to this server
without knowing anything about OpenVLA internals.

Run:
    python scripts/serve_openvla_http.py --port 8600 --dummy
    python scripts/serve_openvla_http.py --port 8600 \
        --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --unnorm-key libero_spatial
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image

# Make `experiments.robot.*` importable when the server is launched from /app.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("serve_openvla_http")

# ----------------------------------------------------------------------------- #
# Globals populated at startup
# ----------------------------------------------------------------------------- #
vla = None               # OpenVLAForActionPrediction
processor = None         # PrismaticProcessor
base_vla_name: str = ""  # checkpoint name — used to detect OpenVLA v0.1 prompt
unnorm_key: str = "libero_spatial"
center_crop: bool = True
is_dummy: bool = False
action_dim: int = 7
image_resolution: int = 224

app = FastAPI()


# ----------------------------------------------------------------------------- #
# Image utilities
# ----------------------------------------------------------------------------- #
def _b64_to_numpy(b64_str: str) -> np.ndarray:
    """base64-encoded PNG → HxWx3 uint8 numpy (RGB)."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))


def _libero_resize_224(img: np.ndarray) -> np.ndarray:
    """Lanczos3 resize via tf.image, matching the Octo dataloader Moo Jin uses.

    Duplicated (not imported) from `experiments/robot/libero/libero_utils.py` so
    that this server never touches any LIBERO-side code path that other experiments
    in this workspace may be depending on.
    """
    import tensorflow as tf

    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (image_resolution, image_resolution), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def _prepare_libero_image(b64_str: str) -> np.ndarray:
    """base64 PNG → 224x224 uint8 image, preprocessed like OpenVLA's LIBERO eval.

    Mirrors `libero_utils.get_libero_image` but inlined on purpose:
      1. 180° rotate (`img[::-1, ::-1]`) — LIBERO train preprocessing.
      2. tf.image JPEG encode/decode round-trip (RLDS dataset builder compat).
      3. lanczos3 resize to 224x224 (Octo-style).
    """
    raw = _b64_to_numpy(b64_str)
    raw = np.ascontiguousarray(raw[::-1, ::-1])
    return _libero_resize_224(raw)


# ----------------------------------------------------------------------------- #
# Model loading
# ----------------------------------------------------------------------------- #
def _load_model(checkpoint: str, attn_impl: str, load_in_8bit: bool, load_in_4bit: bool):
    """Load an OpenVLA policy from an HF Hub name or local path.

    Mirrors `experiments.robot.openvla_utils.get_vla` but with configurable
    `attn_implementation` so the server can fall back to `sdpa`/`eager` on GPUs
    without flash-attn. norm_stats for HF Hub checkpoints is populated via
    config (see prismatic/extern/hf/modeling_prismatic.py:497).
    """
    import json
    import os

    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    logger.info("Loading OpenVLA checkpoint %s (attn=%s, 8bit=%s, 4bit=%s)",
                checkpoint, attn_impl, load_in_8bit, load_in_4bit)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not load_in_8bit and not load_in_4bit:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

    # Local-checkpoint fallback for norm_stats, matching get_vla().
    if os.path.isdir(checkpoint):
        stats_path = os.path.join(checkpoint, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r") as f:
                model.norm_stats = json.load(f)

    proc = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    return model, proc


# ----------------------------------------------------------------------------- #
# HTTP endpoints
# ----------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {
        "status": "ok" if (vla is not None or is_dummy) else "loading",
        "model": "openvla_libero_dummy" if is_dummy else f"openvla:{base_vla_name}",
        "action_type": "relative",
        "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
        "n_action_steps": 1,
    }


@app.post("/reset")
async def reset():
    # OpenVLA is stateless per-step; nothing to do.
    return {"status": "reset"}


@app.post("/act")
async def act(payload: dict):
    t0 = time.time()

    # Pick the static/agentview camera. Falls back to the first available camera.
    b64_img = payload.get("observation.images.static")
    if b64_img is None:
        for k, v in payload.items():
            if k.startswith("observation.images."):
                b64_img = v
                break

    task = str(payload.get("task", "")).lower()

    if is_dummy:
        action = np.zeros(action_dim, dtype=np.float32)
        action[-1] = -1.0  # gripper open
    else:
        if b64_img is None:
            # No image → zero action (gripper open). Keeps the server crash-free
            # on degenerate payloads.
            action = np.zeros(action_dim, dtype=np.float32)
            action[-1] = -1.0
        else:
            img224 = _prepare_libero_image(b64_img)
            observation = {"full_image": img224}

            from experiments.robot.openvla_utils import get_vla_action
            from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

            action = get_vla_action(
                vla, processor, base_vla_name, observation, task, unnorm_key,
                center_crop=center_crop,
            )
            action = np.asarray(action, dtype=np.float32)
            assert action.shape == (action_dim,), f"unexpected action shape {action.shape}"
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

    # Split into unified sub-keys. Protocol requires 2-D lists, shape [N_steps, D].
    pos = action[np.newaxis, 0:3].tolist()
    euler = action[np.newaxis, 3:6].tolist()
    grip = action[np.newaxis, 6:7].tolist()
    return {
        "action.eef_pos": pos,
        "action.eef_euler": euler,
        "action.gripper": grip,
        "latency_ms": (time.time() - t0) * 1000.0,
    }


# ----------------------------------------------------------------------------- #
# Entry point
# ----------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument(
        "--checkpoint",
        default="openvla/openvla-7b-finetuned-libero-spatial",
        help="HF Hub repo id or local checkpoint directory.",
    )
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Falls back to sdpa/eager on GPUs without flash-attn built.",
    )
    parser.add_argument("--no-center-crop", action="store_true",
                        help="Disable center crop (default: enabled, matches aug-trained LIBERO ckpts).")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dummy", action="store_true",
                        help="Skip model loading; return zero actions. Useful for protocol smoke tests.")
    args = parser.parse_args()

    global vla, processor, base_vla_name, unnorm_key, center_crop, is_dummy
    is_dummy = args.dummy
    base_vla_name = args.checkpoint
    unnorm_key = args.unnorm_key
    center_crop = not args.no_center_crop

    if args.dummy:
        logger.warning("Running in --dummy mode (zero actions, no checkpoint).")
    else:
        vla, processor = _load_model(
            args.checkpoint, args.attn_impl, args.load_in_8bit, args.load_in_4bit,
        )
        # Auto-fall back to the `_no_noops` variant when the suite key isn't
        # present, matching run_libero_eval.py.
        if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
            logger.info("unnorm_key %s not found; falling back to %s_no_noops", unnorm_key, unnorm_key)
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in vla.norm_stats, (
            f"unnorm_key {unnorm_key!r} not in norm_stats keys {list(vla.norm_stats.keys())}"
        )
        logger.info("OpenVLA loaded. unnorm_key=%s  center_crop=%s", unnorm_key, center_crop)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
