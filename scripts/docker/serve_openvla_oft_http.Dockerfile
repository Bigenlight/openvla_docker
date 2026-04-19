# Dockerfile for serving OpenVLA-OFT over the unified VLA HTTP protocol.
#
# Deps-only image: the openvla-oft source tree is bind-mounted at /app at
# runtime (see openvla_oft_http_compose.yml). The HF checkpoint is cached in
# ~/.cache/huggingface on the host and mounted as /hf_cache.
#
# Build:
#   docker build -t openvla-oft-http:latest -f scripts/docker/serve_openvla_oft_http.Dockerfile .
#
# Run:
#   docker run --rm -it --network host --gpus all \
#     -v "$(pwd):/app" \
#     -v "$HOME/.cache/huggingface:/hf_cache" \
#     -e HF_HOME=/hf_cache \
#     openvla-oft-http:latest
#
# Protocol smoke-test without a checkpoint:
#   docker run --rm -it --network host -v "$(pwd):/app" \
#     -e OPENVLA_OFT_HTTP_ARGS="--dummy" openvla-oft-http:latest

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

# --- Pin torch 2.2.0+cu121 via PIP_CONSTRAINT so transitive deps don't upgrade it.
RUN printf "torch==2.2.0\ntorchvision==0.17.0\ntorchaudio==2.2.0\n" > /etc/pip.constraints
ENV PIP_CONSTRAINT=/etc/pip.constraints

RUN python -m pip install \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# --- OpenVLA-OFT inference deps.
# Key differences vs vanilla OpenVLA:
#   - transformers comes from `moojink/transformers-openvla-oft` (bidirectional attn)
#   - diffusers is new (for the DDIM action head)
#   - json-numpy / imageio used by openvla_utils
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
    draccus==0.8.0 \
    diffusers==0.30.3 \
    imageio \
    bitsandbytes \
    tensorflow==2.15.0 \
    tensorflow_datasets==4.9.3

# Forked transformers (with bidirectional attention for OFT parallel decoding).
RUN python -m pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"

# dlimp — required by prismatic's import chain even in inference mode.
RUN python -m pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# HTTP serving deps.
RUN python -m pip install \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    pillow \
    requests

# flash-attn (non-fatal — fall back to sdpa at runtime if build fails).
RUN python -m pip install packaging && \
    (PIP_CONSTRAINT= python -m pip install flash-attn==2.5.5 --no-build-isolation \
        || echo "[warn] flash-attn install failed; runtime will use default attention impl")

ENV OPENVLA_OFT_HTTP_PORT=8700 \
    HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache
EXPOSE 8700

WORKDIR /app

CMD ["/bin/bash", "/app/scripts/docker/openvla_oft_http_entrypoint.sh"]
