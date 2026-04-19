# OpenVLA-OFT Docker HTTP Inference Server

> **Fork**: [Bigenlight/openvla-oft_docker](https://github.com/Bigenlight/openvla-oft_docker) *(TBD — 저장소 수동 생성이 필요할 수 있음)*
> **Base**: [moojink/openvla-oft](https://github.com/moojink/openvla-oft)

---

## 목차

1. [개요 (Overview)](#1-개요-overview)
2. [모델 다운로드 (Model Download)](#2-모델-다운로드-model-download)
3. [Docker 이미지 (Docker Image)](#3-docker-이미지-docker-image)
4. [빠른 시작 (Quick Start)](#4-빠른-시작-quick-start)
5. [LIBERO 벤치마크 실행 (Running LIBERO Benchmark)](#5-libero-벤치마크-실행-running-libero-benchmark)
6. [벤치마크 결과 (Benchmark Results)](#6-벤치마크-결과-benchmark-results)
7. [서버 CLI 옵션](#7-서버-cli-옵션)
8. [환경 변수](#8-환경-변수)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. 개요 (Overview)

이 저장소는 **OpenVLA-OFT**(Optimized Fine-Tuning 계열 OpenVLA fork)를 Docker 컨테이너로 패키징하여 **FastAPI HTTP 서버**로 제공하는 포크입니다. OFT는 vanilla OpenVLA 대비 병렬 디코딩과 action chunk 예측으로 **step당 추론 비용을 대폭 줄인** 버전입니다.

핵심 설계 원칙:

- **모델과 벤치마크의 분리**: 모델 컨테이너 하나, 벤치마크 컨테이너 하나. HTTP를 통해서만 통신하므로 모델 교체 시 벤치마크 코드를 수정할 필요가 없습니다.
- **통합 VLA 통신 규약**: `VLA_COMMUNICATION_PROTOCOL.md`에 정의된 프로토콜을 따르는 FastAPI 서버. 같은 인터페이스로 LIBERO, LIBERO-PRO 등 여러 벤치마크에서 호출 가능합니다.
- **표준 포트**: OFT 서버는 포트 **8700**을 사용합니다 (참고: vanilla OpenVLA=8600, Pi0.5=8400).
- **간편한 배포**: Docker Compose 한 줄로 서버 기동, HuggingFace Hub에서 체크포인트 자동 다운로드.

### vanilla OpenVLA와의 차이점

OFT는 vanilla OpenVLA와 아키텍처·입출력 스펙이 다르기 때문에 서버 내부 구현도 달라집니다:

| 항목 | vanilla OpenVLA | OpenVLA-OFT |
|------|-----------------|-------------|
| **입력 이미지** | 1장 (static/third-person) | **2장** (static + wrist, 둘 다 필수) |
| **Proprio state** | 사용 안 함 | **8D** 벡터 — `eef_pos(3) + axis_angle(3) + gripper_qpos(2)` |
| **Action chunk** | 1 step | **8 steps** (한 번의 `/act` 호출당 8개 action 반환) |
| **Attention 구현** | standard causal | **bidirectional** (moojink/transformers-openvla-oft fork) |
| **Action head** | LLM logits → discretized action | **L1 regression** (연속값 직접 회귀) |
| **Image aug** | 없음 | `center_crop=True` **필수** (학습 시 90% random crop 증강) |
| **추가 컴포넌트** | 단일 VLA | **action_head + proprio_projector** 별도 로드 |

> 이 구현적 차이는 `scripts/serve_openvla_oft_http.py`에 캡슐화되어 있으며, 벤치마크에서 보는 HTTP 인터페이스는 동일한 `VLA_COMMUNICATION_PROTOCOL`을 따릅니다.

### 아키텍처

```
┌─────────────────────┐         HTTP (port 8700)         ┌─────────────────────┐
│                     │  ─────────────────────────────▶  │                     │
│  LIBERO Benchmark   │  POST /act (2 images + state +   │  OpenVLA-OFT HTTP   │
│  Container          │               task)              │  Server Container   │
│                     │  ◀─────────────────────────────  │  (GPU, bfloat16)    │
│                     │   response: 8-step action chunk  │                     │
└─────────────────────┘                                  └─────────────────────┘
```

한 번의 `/act` 호출로 8 step chunk를 받아오므로, 벤치마크 측은 내부 action queue에 쌓아두고 8 step마다 한 번씩만 서버를 호출합니다. 이것이 OFT의 평균 step latency를 vanilla 대비 크게 낮추는 핵심 메커니즘입니다.

---

## 2. 모델 다운로드 (Model Download)

### 자동 다운로드

서버를 처음 실행하면 HuggingFace Hub에서 체크포인트를 **자동으로 다운로드**합니다. 기본 체크포인트는 `moojink/openvla-7b-oft-finetuned-libero-spatial`입니다.

OFT 체크포인트는 세 개의 컴포넌트로 구성됩니다:

| 컴포넌트 | 크기 | 설명 |
|----------|------|------|
| VLA backbone (`model-*.safetensors`) | ~15.9 GB | 7B 파라미터 OpenVLA (bfloat16) |
| `action_head.pt` | 302 MB | L1 regression head |
| `proprio_projector.pt` | 67 MB | 8D proprio → LLM hidden dim |

세 파일 모두 같은 HF repo 안에 있어서 별도 경로 지정 없이 자동으로 다운로드됩니다.

### 수동 다운로드

네트워크 환경이 불안정하거나 HF hub에서 xet 백엔드 문제가 발생할 경우, 사전에 수동으로 다운로드할 수 있습니다:

```bash
# huggingface-cli 설치 (이미 설치되어 있지 않은 경우)
pip install huggingface_hub[cli]

# xet 대신 기존 (resumable) 다운로드 강제
export HF_HUB_DISABLE_XET=1

# 체크포인트 다운로드 (~16.3 GB)
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial
```

### LIBERO Suite 체크포인트 목록

각 LIBERO 태스크 스위트에 대응하는 fine-tuned 체크포인트가 있습니다:

| Suite | HuggingFace Checkpoint |
|-------|----------------------|
| LIBERO-Spatial | `moojink/openvla-7b-oft-finetuned-libero-spatial` |
| LIBERO-Object | `moojink/openvla-7b-oft-finetuned-libero-object` |
| LIBERO-Goal (50K step) | `moojink/openvla-7b-oft-finetuned-libero-goal` |
| LIBERO-10 (Long) | `moojink/openvla-7b-oft-finetuned-libero-10` |
| LIBERO-Spatial+Object+Goal+10 (multi-task) | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |

> 체크포인트를 교체할 때는 `--unnorm-key`도 해당 suite에 맞게 바꿔줘야 합니다 (예: `libero_spatial` → `libero_object`).

### 캐시 경로

- **호스트**: `~/.cache/huggingface` (HuggingFace 기본 캐시 디렉토리)
- **컨테이너**: `/hf_cache` (Docker Compose에서 자동 마운트)
- **총 체크포인트 크기**: 약 **16.3 GB** (VLA 15.9 GB + action_head 302 MB + proprio_projector 67 MB)

> **참고**: 한 번 다운로드하면 캐시에 저장되므로, 이후 실행 시 다시 다운로드하지 않습니다. vanilla OpenVLA와 캐시 디렉토리를 공유하므로, 양쪽 서버를 모두 돌려도 캐시가 중복되지 않습니다.

---

## 3. Docker 이미지 (Docker Image)

### 이미지 정보

| 항목 | 내용 |
|------|------|
| **Docker Hub** | `bigenlight/openvla-oft-http:latest` (이미 푸시됨) |
| **이미지 크기** | ~18 GB |
| **Base** | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |

### 포함된 의존성

- Python 3.10
- PyTorch 2.2.0+cu121 (PIP_CONSTRAINT로 고정)
- **transformers (forked)**: `moojink/transformers-openvla-oft` — bidirectional attention 패치 포함
- **dlimp (forked)**: `moojink/dlimp_openvla` — prismatic import chain 요구사항
- diffusers 0.30.3, peft 0.11.1, timm 0.9.10, tokenizers 0.19.1
- tensorflow 2.15.0 (이미지 전처리 `lanczos3` resize용)
- flash-attn 2.5.5 (빌드 실패 시 sdpa로 자동 fallback)
- FastAPI + Uvicorn + Pillow

### 포함되지 않은 것 (마운트 필요)

- **모델 체크포인트**: HuggingFace 캐시를 `/hf_cache`로 마운트
- **소스 코드**: 프로젝트 루트를 `/app`으로 bind mount (코드 수정이 이미지 재빌드 없이 즉시 반영됨)

### 이미지 Pull

```bash
docker pull bigenlight/openvla-oft-http:latest
```

---

## 4. 빠른 시작 (Quick Start)

### a) Docker Hub에서 이미지 Pull

```bash
docker pull bigenlight/openvla-oft-http:latest
```

### b) 소스 코드 Clone

```bash
git clone git@github.com:Bigenlight/openvla-oft_docker.git
cd openvla-oft_docker
```

### c) Docker Compose로 서버 시작 (Real Mode)

첫 실행 시 HuggingFace Hub에서 약 16.3 GB 체크포인트를 자동으로 다운로드합니다:

```bash
docker compose -f scripts/docker/openvla_oft_http_compose.yml up
```

서버가 정상적으로 시작되면 다음과 같은 로그가 출력됩니다:

```
serve_openvla_oft_http: Loading OpenVLA-OFT VLA from moojink/openvla-7b-oft-finetuned-libero-spatial ...
serve_openvla_oft_http: Loading action_head (L1 regression) ...
serve_openvla_oft_http: Loading proprio_projector ...
serve_openvla_oft_http: Loading processor ...
serve_openvla_oft_http: OpenVLA-OFT loaded. unnorm_key=libero_spatial  center_crop=True  chunk=8
INFO:     Uvicorn running on http://0.0.0.0:8700
```

### d) Dummy Mode (모델 없이 프로토콜 테스트)

GPU가 없거나 모델 다운로드 없이 통신 프로토콜만 테스트하고 싶을 때 사용합니다. `action_chunk`를 영벡터로 고정하여 반환합니다:

```bash
OPENVLA_OFT_HTTP_ARGS="--dummy" docker compose -f scripts/docker/openvla_oft_http_compose.yml up
```

### e) Health Check

서버가 정상 동작하는지 확인합니다:

```bash
curl http://localhost:8700/health
```

정상 응답 (OFT는 `n_action_steps=8`이 반환됨에 주의):

```json
{
  "status": "ok",
  "model": "openvla-oft:moojink/openvla-7b-oft-finetuned-libero-spatial",
  "action_type": "relative",
  "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
  "n_action_steps": 8
}
```

### 전체 워크플로우 요약

```bash
# 1. 이미지 pull
docker pull bigenlight/openvla-oft-http:latest

# 2. 소스 clone
git clone git@github.com:Bigenlight/openvla-oft_docker.git
cd openvla-oft_docker

# 3. 서버 시작 (첫 실행 시 HF에서 ~16.3 GB 체크포인트 자동 다운로드)
docker compose -f scripts/docker/openvla_oft_http_compose.yml up

# 4. (선택) 프로토콜 테스트만 — 모델 없이
OPENVLA_OFT_HTTP_ARGS="--dummy" docker compose -f scripts/docker/openvla_oft_http_compose.yml up

# 5. Health check
curl http://localhost:8700/health
```

---

## 5. LIBERO 벤치마크 실행 (Running LIBERO Benchmark)

### 사전 요구 사항

- OpenVLA-OFT HTTP 서버가 포트 8700에서 실행 중이어야 합니다 (위의 [빠른 시작](#4-빠른-시작-quick-start) 참고).
- LIBERO-PRO 벤치마크 컨테이너가 필요합니다: `bigenlight/libero-pro:latest`
- `Libero-pro_benchmark` 저장소는 이미지와 **별도**로 클론해야 합니다 (벤치마크 측 `CLAUDE.md` 참고).

### 평가 실행

```bash
# OpenVLA-OFT 서버가 8700에서 실행 중인 상태에서:
cd Libero-pro_benchmark

# 10 tasks × 5 trials (총 50 에피소드) — 아래 벤치마크 결과와 동일 구성
./run.sh --vla-eval libero_spatial --vla-url http://localhost:8700 \
    --vla-num-tasks 10 --vla-num-trials 5
```

### 다른 Suite 평가

체크포인트와 `--unnorm-key`를 해당 suite에 맞게 바꿔서 서버를 재시작한 뒤, 벤치마크의 `--vla-eval` 도 맞춰줍니다:

```bash
# 서버 측: 체크포인트 교체
OPENVLA_OFT_HTTP_ARGS="--checkpoint moojink/openvla-7b-oft-finetuned-libero-object --unnorm-key libero_object" \
    docker compose -f scripts/docker/openvla_oft_http_compose.yml up

# 벤치마크 측
./run.sh --vla-eval libero_object --vla-url http://localhost:8700 \
    --vla-num-tasks 10 --vla-num-trials 5

# LIBERO-Goal (50K step 체크포인트)
./run.sh --vla-eval libero_goal --vla-url http://localhost:8700

# LIBERO-10 (Long-Horizon)
./run.sh --vla-eval libero_10 --vla-url http://localhost:8700
```

### 결과 확인

- **요약 결과**: `test_outputs/eval_openvla_oft/libero_spatial_YYYYMMDD_HHMMSS/summary.json`
- **에피소드 영상**: `test_outputs/eval_openvla_oft/libero_spatial_YYYYMMDD_HHMMSS/videos/*.mp4`

---

## 6. 벤치마크 결과 (Benchmark Results)

### LIBERO-Spatial 결과 요약

| 항목 | 값 |
|------|-----|
| **Suite** | LIBERO-Spatial |
| **Checkpoint** | `moojink/openvla-7b-oft-finetuned-libero-spatial` |
| **태스크 수** | 10 |
| **Trial 수 (per task)** | 5 |
| **총 성공 / 총 trial** | **48 / 50** |
| **성공률** | **96.0%** |

### 성능 지표 (vanilla OpenVLA 대비)

| 지표 | OpenVLA (vanilla) | **OpenVLA-OFT** |
|------|-------------------|-----------------|
| **/act 호출당 latency** | ~260 ms | **~169 ms** |
| **1 호출에서 반환되는 action 수** | 1 | **8** |
| **유효 per-step latency** | ~260 ms | **~21 ms** (≈ 169 / 8) |
| **LIBERO-Spatial 성공률** | 74.0% (37/50) | **96.0% (48/50)** |
| **테스트 GPU** | NVIDIA RTX A6000 | NVIDIA RTX A6000 |

> **해설**: OFT는 한 번의 forward pass에서 8 step action chunk를 병렬로 예측하므로, 벤치마크 측에서는 8 step에 한 번씩만 서버를 호출하면 됩니다. 따라서 체감 step-latency는 vanilla OpenVLA 대비 **~12× 빠릅니다** (260 ms → 21 ms). 성공률 측면에서도 96% vs 74%로 큰 격차를 보였습니다.

### Per-Task Breakdown

```json
{
  "suite": "libero_spatial",
  "num_tasks": 10,
  "num_trials_per_task": 5,
  "total_episodes": 50,
  "total_successes": 48,
  "success_rate": 0.96,
  "tasks": [
    {"task_id": 0, "successes": 5, "trials": 5, "avg_latency_ms": 169.16},
    {"task_id": 1, "successes": 5, "trials": 5, "avg_latency_ms": 168.41},
    {"task_id": 2, "successes": 5, "trials": 5, "avg_latency_ms": 169.76},
    {"task_id": 3, "successes": 5, "trials": 5, "avg_latency_ms": 165.31},
    {"task_id": 4, "successes": 4, "trials": 5, "avg_latency_ms": 169.10},
    {"task_id": 5, "successes": 5, "trials": 5, "avg_latency_ms": 170.82},
    {"task_id": 6, "successes": 5, "trials": 5, "avg_latency_ms": 173.52},
    {"task_id": 7, "successes": 4, "trials": 5, "avg_latency_ms": 171.77},
    {"task_id": 8, "successes": 5, "trials": 5, "avg_latency_ms": 174.27},
    {"task_id": 9, "successes": 5, "trials": 5, "avg_latency_ms": 170.09}
  ]
}
```

실패한 2개 trial은 task 4 (`akita black bowl in the top layer of the wooden cabinet`)와 task 7 (`akita black bowl on the stove`)에서 각각 1건씩 발생했습니다. 두 태스크 모두 장시간 manipulation이 필요한 케이스로, OFT의 chunk replanning 타이밍이 미세하게 어긋나는 경우가 관찰됩니다.

---

## 7. 서버 CLI 옵션

서버 실행 시 사용할 수 있는 CLI 옵션입니다. Docker Compose 환경에서는 `OPENVLA_OFT_HTTP_ARGS` 환경 변수를 통해 전달합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--port` | `8700` | HTTP 서버 포트 |
| `--checkpoint` | `moojink/openvla-7b-oft-finetuned-libero-spatial` | HuggingFace Hub 모델 이름 또는 로컬 경로 |
| `--unnorm-key` | `libero_spatial` | Action unnormalization에 사용할 데이터셋 키 |
| `--no-center-crop` | (비활성) | 이미지 center crop 비활성화 (**권장하지 않음** — 학습 시 90% crop 증강을 적용했으므로 eval에서도 켜야 함) |
| `--dummy` | (비활성) | 모델 로드 없이 dummy 모드로 실행 (프로토콜 테스트용, 영벡터 chunk 반환) |

> vanilla OpenVLA 서버에서 지원하던 `--attn-impl`, `--load-in-8bit`, `--load-in-4bit` 같은 플래그는 OFT 서버에는 아직 노출되어 있지 않습니다. OFT는 `moojink/transformers-openvla-oft` fork에 의존하기 때문에 attention 구현 변경을 하려면 내부 `_build_cfg`에 필드를 추가해야 합니다.

### 사용 예시

```bash
# 다른 체크포인트 사용 (suite 전환)
OPENVLA_OFT_HTTP_ARGS="--checkpoint moojink/openvla-7b-oft-finetuned-libero-object --unnorm-key libero_object" \
    docker compose -f scripts/docker/openvla_oft_http_compose.yml up

# LIBERO-10
OPENVLA_OFT_HTTP_ARGS="--checkpoint moojink/openvla-7b-oft-finetuned-libero-10 --unnorm-key libero_10" \
    docker compose -f scripts/docker/openvla_oft_http_compose.yml up

# 포트 변경
OPENVLA_OFT_HTTP_ARGS="--port 9700" \
OPENVLA_OFT_HTTP_PORT=9700 \
    docker compose -f scripts/docker/openvla_oft_http_compose.yml up

# 프로토콜 smoke test (GPU 불필요)
OPENVLA_OFT_HTTP_ARGS="--dummy" \
    docker compose -f scripts/docker/openvla_oft_http_compose.yml up
```

---

## 8. 환경 변수

Docker Compose 실행 시 사용할 수 있는 환경 변수입니다:

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `OPENVLA_OFT_HTTP_PORT` | `8700` | 호스트에 노출할 서버 포트 |
| `OPENVLA_OFT_HTTP_ARGS` | (없음) | 서버에 전달할 추가 CLI 인자 (위 CLI 옵션 참고) |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace 캐시 디렉토리 (컨테이너 내 `/hf_cache`로 마운트) |
| `HF_HUB_DISABLE_XET` | `1` (compose에서 강제) | xet 백엔드 비활성화 — hang 방지용 (아래 트러블슈팅 참고) |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` (compose에서 강제) | `hf_transfer` 비활성화 — xet와 충돌 방지 |
| `IS_DOCKER` | `true` | 프리셋 플래그, `openvla_utils`가 확인 |

### 사용 예시

```bash
# 모든 환경 변수를 조합하여 실행
OPENVLA_OFT_HTTP_PORT=8700 \
HF_HOME=/data/hf_cache \
OPENVLA_OFT_HTTP_ARGS="--checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial" \
    docker compose -f scripts/docker/openvla_oft_http_compose.yml up
```

---

## 9. 트러블슈팅

### HF Hub 다운로드가 멈춘 경우 (xet 백엔드 hang)

**증상**: 서버가 `Loading OpenVLA-OFT VLA from ...` 로그 직후 멈춘 채 아무 진행도 없음. 네트워크 I/O는 0에 가까움.

**원인**: HuggingFace hub가 최근 도입한 **xet 백엔드**가 일부 환경(특히 기업 방화벽/프록시 뒤 또는 낮은 대역폭)에서 **무한 대기**하는 알려진 버그가 있습니다.

**해결**: `openvla_oft_http_compose.yml`은 이미 `HF_HUB_DISABLE_XET=1` 과 `HF_HUB_ENABLE_HF_TRANSFER=0` 을 기본으로 설정해 두었습니다. 수동으로 설정하려면:

```bash
# 컨테이너 안에서 직접 실행할 경우
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# 혹은 compose 레벨에서 덮어쓰기
HF_HUB_DISABLE_XET=1 docker compose -f scripts/docker/openvla_oft_http_compose.yml up
```

이미 다운로드가 멈춘 세션은 `Ctrl+C`로 죽이고, `~/.cache/huggingface/hub/` 안의 해당 repo의 `.incomplete` 파일을 지운 뒤 재시도하세요.

### 체크포인트 다운로드가 느리거나 hang 증상이 반복되는 경우

서버 시작 전에 미리 다운로드해 두면 디버깅이 단순해집니다:

```bash
# 사전 다운로드 (xet 비활성화)
export HF_HUB_DISABLE_XET=1
huggingface-cli download moojink/openvla-7b-oft-finetuned-libero-spatial

# 다운로드 완료 후 서버 시작 — 캐시에서 바로 로드됨
docker compose -f scripts/docker/openvla_oft_http_compose.yml up
```

### flash-attn 빌드/로드 실패

flash-attn은 특정 GPU 아키텍처(sm_70 미만 등)에서 호환성 문제가 발생할 수 있습니다. Dockerfile은 이미 flash-attn 빌드 실패 시 이미지를 성공적으로 빌드하고 런타임에서 기본 attention(`sdpa`)으로 fallback 하도록 되어 있습니다.

이미지 내부에서 flash-attn이 로드되지 않는지 확인:

```bash
docker run --rm --gpus all bigenlight/openvla-oft-http:latest \
    python -c "import flash_attn; print(flash_attn.__version__)"
```

import 실패해도 서버는 정상 작동합니다.

### `moojink/transformers-openvla-oft` import 에러

OFT는 **fork된 transformers**를 사용하므로, `pip install transformers`로 공식 릴리즈를 설치하면 bidirectional attention 관련 속성이 사라져서 작동하지 않습니다. Dockerfile이 이미 올바른 fork를 설치해 두었지만, 직접 가상환경을 꾸리는 경우에는:

```bash
pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
```

를 반드시 사용하세요.

### CUDA Out of Memory

7B 모델은 bfloat16 기준 약 14–16 GB VRAM이 필요합니다. 추가로 action_head / proprio_projector / 2장의 이미지 tokenization으로 약간의 오버헤드가 있습니다. OFT 서버에는 아직 8-bit / 4-bit 양자화 플래그가 노출되어 있지 않으므로, VRAM이 부족하면:

1. 다른 GPU 프로세스를 종료합니다.
2. `--dummy` 모드로 우선 프로토콜만 검증합니다.
3. `_build_cfg` 안의 `load_in_8bit=True`로 직접 코드 수정이 필요합니다 (추후 서버 CLI 옵션으로 노출 예정).

### center_crop 관련 성능 저하

OFT 학습 시 입력 이미지에 **90% 크기의 random crop 증강**을 적용했기 때문에, eval 시에도 `center_crop=True`가 기본값입니다. 만약 `--no-center-crop` 을 켜면 distribution shift로 인해 성공률이 급격히 떨어집니다. **권장하지 않습니다.**

### 컨테이너가 GPU를 인식하지 못하는 경우

NVIDIA Container Toolkit이 설치되어 있는지 확인하세요:

```bash
# NVIDIA Container Toolkit 설치 확인
nvidia-ctk --version

# Docker에서 GPU 접근 테스트
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 서버 로그 확인

```bash
# 실시간 로그 확인
docker compose -f scripts/docker/openvla_oft_http_compose.yml logs -f

# 최근 100줄만 확인
docker compose -f scripts/docker/openvla_oft_http_compose.yml logs --tail 100
```

### `unnorm_key ... not in norm_stats` assertion 실패

체크포인트마다 norm_stats 의 키 이름이 살짝 다를 수 있습니다 (`libero_spatial` vs `libero_spatial_no_noops`). 서버 코드는 자동으로 `{key}_no_noops` 로 fallback을 시도합니다. 그래도 실패한다면 로그에 출력되는 `norm_stats keys [...]` 를 보고 `--unnorm-key`를 직접 지정하세요:

```bash
OPENVLA_OFT_HTTP_ARGS="--unnorm-key libero_spatial_no_noops" \
    docker compose -f scripts/docker/openvla_oft_http_compose.yml up
```

---

## 라이선스

이 포크는 원본 [openvla-oft](https://github.com/moojink/openvla-oft) 저장소의 라이선스를 따릅니다.
