# Dockerfile for serving OpenVLA over the unified VLA HTTP protocol.
#
# Deps-only image: the openvla source tree is bind-mounted at /app at runtime
# (see openvla_http_compose.yml), so `git pull/push` from the host works
# without rebuilding.
#
# Build (from the openvla repo root):
#   docker build -t openvla-http:latest -f scripts/docker/serve_openvla_http.Dockerfile .
#
# Run (HF cache mounted so the 7B checkpoint is downloaded only once):
#   docker run --rm -it \
#     --network host \
#     --gpus all \
#     -v "$(pwd):/app" \
#     -v "$HOME/.cache/huggingface:/hf_cache" \
#     -e HF_HOME=/hf_cache \
#     openvla-http:latest
#
# Protocol smoke-test without a checkpoint:
#   docker run --rm -it --network host -v "$(pwd):/app" \
#     -e OPENVLA_HTTP_ARGS="--dummy" openvla-http:latest

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential ninja-build curl ca-certificates \
        python3.10 python3.10-venv python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

# --- PyTorch + OpenVLA deps, pinned via PIP_CONSTRAINT ----------------------- #
# Subsequent pip installs must NOT upgrade torch — bitsandbytes / transformers
# happily pull the latest nightly if left unconstrained, which on this box ends
# up as torch 2.11+cu130 (driver too old → cuda_ok=False). Pinning via a global
# constraints file is the only way to make pip keep our torch 2.2.0+cu121.
RUN printf "torch==2.2.0\ntorchvision==0.17.0\ntorchaudio==2.2.0\n" > /etc/pip.constraints
ENV PIP_CONSTRAINT=/etc/pip.constraints

RUN python -m pip install \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# OpenVLA inference deps (subset of pyproject.toml). Intentionally skips
# tensorflow_graphics and dlimp — neither is used by the inference path.
RUN python -m pip install \
    "accelerate>=0.25.0" \
    einops \
    huggingface_hub \
    json-numpy \
    jsonlines \
    matplotlib \
    peft==0.11.1 \
    protobuf \
    rich \
    sentencepiece==0.1.99 \
    timm==0.9.10 \
    tokenizers==0.19.1 \
    transformers==4.40.1 \
    draccus==0.8.0 \
    bitsandbytes \
    tensorflow==2.15.0 \
    tensorflow_datasets==4.9.3

# dlimp — required by prismatic/__init__.py's eager import chain even though
# inference itself never touches the RLDS/training datasets. Pulling it from
# the openvla fork that Moo Jin maintains (same source as pyproject.toml).
RUN python -m pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# HTTP serving deps
RUN python -m pip install \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    pillow \
    requests

# --- flash-attn (non-fatal) -------------------------------------------------- #
# Heavy build (~10 min). Failure leaves the image usable with --attn-impl sdpa.
# NOTE: PIP_CONSTRAINT must be unset for this install because flash-attn's
# build backend re-resolves torch in a fresh env, and our cu121 wheel isn't
# visible on the default index — so the constraint would block it. We still
# tolerate the failure and fall back to sdpa at runtime.
RUN python -m pip install packaging && \
    (PIP_CONSTRAINT= python -m pip install flash-attn==2.5.5 --no-build-isolation \
        || echo "[warn] flash-attn install failed; server will fall back to --attn-impl sdpa")

ENV OPENVLA_HTTP_PORT=8600 \
    HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache
EXPOSE 8600

WORKDIR /app

# Default command: run the entrypoint script from the bind-mounted source tree.
# Nothing from the repo is baked into the image — all code lives in /app so the
# user can `git pull/push` from the host at will.
CMD ["/bin/bash", "/app/scripts/docker/openvla_http_entrypoint.sh"]
