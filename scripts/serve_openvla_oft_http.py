"""FastAPI HTTP server exposing OpenVLA-OFT (LIBERO fine-tune) via the unified VLA protocol.

Implements the three-endpoint contract defined in VLA_COMMUNICATION_PROTOCOL.md:

    GET  /health  →  {status, model, action_type, action_keys, n_action_steps}
    POST /reset   →  {status}
    POST /act     →  {action.eef_pos, action.eef_euler, action.gripper, latency_ms}

Differences vs the vanilla OpenVLA server (scripts/serve_openvla_http.py in the
openvla repo):
- Uses **2 input images** (static + wrist) instead of 1.
- Uses **proprioceptive state** (8D: eef_pos + axis_angle + gripper_qpos(2)).
- Requires loading **action_head** and **proprio_projector** from the HF checkpoint.
- Returns an **8-step action chunk** per /act call (n_action_steps=8).
- Uses the forked `moojink/transformers-openvla-oft` for bidirectional attention.

Run:
    python scripts/serve_openvla_oft_http.py --port 8700 --dummy
    python scripts/serve_openvla_oft_http.py --port 8700 \
        --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
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

# Make `experiments.robot.*` and `prismatic.*` importable when launched from /app.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("serve_openvla_oft_http")

# ----------------------------------------------------------------------------- #
# Globals populated at startup
# ----------------------------------------------------------------------------- #
vla = None
processor = None
action_head = None
proprio_projector = None
cfg_obj = None           # SimpleNamespace with the fields openvla_utils expects
unnorm_key: str = "libero_spatial"
center_crop: bool = True
is_dummy: bool = False
action_dim: int = 7       # LIBERO OSC_POSE
proprio_dim: int = 8      # eef_pos(3) + axis_angle(3) + gripper_qpos(2)
num_action_steps: int = 8 # NUM_ACTIONS_CHUNK for LIBERO
image_resolution: int = 224

app = FastAPI()


# ----------------------------------------------------------------------------- #
# Image utilities
# ----------------------------------------------------------------------------- #
def _b64_to_numpy(b64_str: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))


def _libero_resize_224(img: np.ndarray) -> np.ndarray:
    """Lanczos3 resize via tf.image — matches openvla-oft's resize_image_for_policy."""
    import tensorflow as tf

    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (image_resolution, image_resolution), method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def _prepare_image(b64_str: str) -> np.ndarray:
    """base64 PNG → 224x224 uint8 RGB, with 180° rotation + lanczos3 resize.

    Mirrors openvla-oft's `libero_utils.get_libero_image` + `resize_image_for_policy`.
    Kept inline to avoid touching upstream LIBERO helpers.
    """
    raw = _b64_to_numpy(b64_str)
    raw = np.ascontiguousarray(raw[::-1, ::-1])  # 180° rotation — LIBERO training preprocessing
    return _libero_resize_224(raw)


# ----------------------------------------------------------------------------- #
# Model loading
# ----------------------------------------------------------------------------- #
def _build_cfg(checkpoint: str, center_crop_flag: bool) -> object:
    """Minimal SimpleNamespace cfg matching what openvla_utils expects."""
    from types import SimpleNamespace

    return SimpleNamespace(
        pretrained_checkpoint=checkpoint,
        model_family="openvla",
        load_in_8bit=False,
        load_in_4bit=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        use_proprio=True,
        num_images_in_input=2,
        num_open_loop_steps=num_action_steps,
        center_crop=center_crop_flag,
        unnorm_key=unnorm_key,
        lora_rank=32,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
    )


def _load_model(checkpoint: str):
    """Load OpenVLA-OFT model + processor + action_head + proprio_projector from HF Hub.

    Delegates to the upstream openvla_utils helpers so that any checkpoint-specific
    logic (HF hub registration, auto_map fixup, component path resolution) stays
    identical to the reference eval.
    """
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_proprio_projector,
        get_processor,
        get_vla,
    )

    cfg = _build_cfg(checkpoint, center_crop)

    logger.info("Loading OpenVLA-OFT VLA from %s ...", checkpoint)
    model = get_vla(cfg)
    logger.info("Loading action_head (L1 regression) ...")
    ah = get_action_head(cfg, llm_dim=model.llm_dim)
    logger.info("Loading proprio_projector ...")
    pp = get_proprio_projector(cfg, llm_dim=model.llm_dim, proprio_dim=proprio_dim)
    logger.info("Loading processor ...")
    proc = get_processor(cfg)
    return model, proc, ah, pp, cfg


# ----------------------------------------------------------------------------- #
# State assembly (mirrors run_libero_eval.prepare_observation)
# ----------------------------------------------------------------------------- #
def _assemble_state(payload: dict) -> np.ndarray:
    """Build OFT's 8D proprio vector from unified observation.state.* keys.

    Layout: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8
    Mirrors `prepare_observation` in openvla-oft's run_libero_eval.
    """
    import math

    if "observation.state.eef_pos" in payload:
        eef_pos = np.asarray(payload["observation.state.eef_pos"], dtype=np.float32).reshape(-1)[:3]
    else:
        eef_pos = np.zeros(3, dtype=np.float32)
    if eef_pos.shape[0] < 3:
        eef_pos = np.pad(eef_pos, (0, 3 - eef_pos.shape[0]))

    # axis-angle preferred; fall back to quaternion → axis-angle conversion
    if "observation.state.eef_axis_angle" in payload:
        axis_angle = np.asarray(payload["observation.state.eef_axis_angle"], dtype=np.float32).reshape(-1)[:3]
    elif "observation.state.eef_quat" in payload:
        q = np.asarray(payload["observation.state.eef_quat"], dtype=np.float32).reshape(-1)[:4]
        w = float(np.clip(q[3], -1.0, 1.0))
        den = math.sqrt(max(1.0 - w * w, 0.0))
        if math.isclose(den, 0.0):
            axis_angle = np.zeros(3, dtype=np.float32)
        else:
            axis_angle = (q[:3] * 2.0 * math.acos(w)) / den
    else:
        axis_angle = np.zeros(3, dtype=np.float32)

    if "observation.state.gripper_qpos" in payload:
        grip = np.asarray(payload["observation.state.gripper_qpos"], dtype=np.float32).reshape(-1)[:2]
    elif "observation.state.gripper_opening" in payload:
        g = float(np.asarray(payload["observation.state.gripper_opening"]).reshape(-1)[0])
        grip = np.array([g, -g], dtype=np.float32)
    else:
        grip = np.zeros(2, dtype=np.float32)
    if grip.shape[0] < 2:
        grip = np.pad(grip, (0, 2 - grip.shape[0]))

    return np.concatenate([eef_pos, axis_angle, grip]).astype(np.float32)


# ----------------------------------------------------------------------------- #
# HTTP endpoints
# ----------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return {
        "status": "ok" if (vla is not None or is_dummy) else "loading",
        "model": "openvla_oft_libero_dummy" if is_dummy else f"openvla-oft:{cfg_obj.pretrained_checkpoint}",
        "action_type": "relative",
        "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
        "n_action_steps": num_action_steps,
    }


@app.post("/reset")
async def reset():
    # OpenVLA-OFT is stateless per /act call (the 8-step chunk is re-predicted each call).
    # The benchmark's action_queue gives us the "replan" behavior.
    return {"status": "reset"}


@app.post("/act")
async def act(payload: dict):
    t0 = time.time()

    # --- Gather images -------------------------------------------------------- #
    b64_static = payload.get("observation.images.static")
    b64_wrist = payload.get("observation.images.wrist")
    if b64_static is None:
        for k, v in payload.items():
            if k.startswith("observation.images.") and "wrist" not in k:
                b64_static = v
                break

    task = str(payload.get("task", "")).lower()

    # --- Inference ------------------------------------------------------------ #
    if is_dummy:
        # Deterministic zero chunk (gripper open). Enough to smoke-test the protocol.
        action_chunk = np.zeros((num_action_steps, action_dim), dtype=np.float32)
        action_chunk[:, -1] = -1.0
    else:
        if b64_static is None:
            action_chunk = np.zeros((num_action_steps, action_dim), dtype=np.float32)
            action_chunk[:, -1] = -1.0
        else:
            static_img = _prepare_image(b64_static)
            wrist_img = _prepare_image(b64_wrist) if b64_wrist is not None else static_img.copy()
            state = _assemble_state(payload)

            observation = {
                "full_image": static_img,
                "wrist_image": wrist_img,
                "state": state,
            }

            from experiments.robot.openvla_utils import get_vla_action
            from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

            # Returns List[np.ndarray], length == NUM_ACTIONS_CHUNK (8)
            action_list = get_vla_action(
                cfg_obj,
                vla,
                processor,
                observation,
                task,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                use_film=False,
            )
            action_chunk = np.stack([np.asarray(a, dtype=np.float32) for a in action_list], axis=0)
            # Post-process each action's gripper (matches process_action in run_libero_eval)
            for i in range(len(action_chunk)):
                a = action_chunk[i].copy()
                a = normalize_gripper_action(a, binarize=True)
                a = invert_gripper_action(a)
                action_chunk[i] = a

    # --- Split into unified sub-keys ----------------------------------------- #
    return {
        "action.eef_pos": action_chunk[:, 0:3].tolist(),
        "action.eef_euler": action_chunk[:, 3:6].tolist(),
        "action.gripper": action_chunk[:, 6:7].tolist(),
        "latency_ms": (time.time() - t0) * 1000.0,
    }


# ----------------------------------------------------------------------------- #
# Entry point
# ----------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument(
        "--checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
        help="HF Hub repo id or local checkpoint directory.",
    )
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument("--dummy", action="store_true",
                        help="Skip model loading; return zero 8-step chunks.")
    args = parser.parse_args()

    global vla, processor, action_head, proprio_projector, cfg_obj
    global unnorm_key, center_crop, is_dummy
    is_dummy = args.dummy
    unnorm_key = args.unnorm_key
    center_crop = not args.no_center_crop

    if args.dummy:
        logger.warning("Running in --dummy mode (zero chunks, no checkpoint).")
        cfg_obj = _build_cfg(args.checkpoint, center_crop)
    else:
        vla, processor, action_head, proprio_projector, cfg_obj = _load_model(args.checkpoint)

        # Unnorm-key fallback (matches run_libero_eval)
        if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
            logger.info("unnorm_key %s missing; falling back to %s_no_noops", unnorm_key, unnorm_key)
            unnorm_key = f"{unnorm_key}_no_noops"
            cfg_obj.unnorm_key = unnorm_key
        assert unnorm_key in vla.norm_stats, (
            f"unnorm_key {unnorm_key!r} not in norm_stats keys {list(vla.norm_stats.keys())}"
        )
        logger.info(
            "OpenVLA-OFT loaded. unnorm_key=%s  center_crop=%s  chunk=%d",
            unnorm_key, center_crop, num_action_steps,
        )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
