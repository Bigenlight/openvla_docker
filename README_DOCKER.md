# OpenVLA Docker HTTP Inference Server

> **Fork**: [Bigenlight/openvla_docker](https://github.com/Bigenlight/openvla_docker)
> **Base**: [openvla/openvla](https://github.com/openvla/openvla)

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

이 저장소는 **OpenVLA-7B**를 Docker 컨테이너로 패키징하여 **HTTP 서버(FastAPI)** 로 제공하는 포크입니다. 핵심 설계 원칙은 다음과 같습니다:

- **모델과 벤치마크의 완전한 분리**: 모델 컨테이너 하나, 벤치마크 컨테이너 하나로 구성됩니다. 두 컨테이너는 HTTP를 통해서만 통신하므로, 모델 교체 시 벤치마크 코드를 수정할 필요가 없습니다.
- **통합 VLA 통신 규약**: `VLA_COMMUNICATION_PROTOCOL.md`에 정의된 프로토콜을 따르는 FastAPI 서버를 제공합니다. 이 규약을 준수하는 모든 벤치마크(LIBERO, LIBERO-PRO 등)에서 동일한 인터페이스로 모델을 호출할 수 있습니다.
- **표준 포트**: OpenVLA 서버는 포트 **8600**을 사용합니다 (참고: Pi0.5는 포트 8400).
- **간편한 배포**: Docker Compose 한 줄로 서버를 기동할 수 있으며, HuggingFace Hub에서 체크포인트를 자동으로 다운로드합니다.

### 아키텍처

```
┌─────────────────────┐         HTTP (port 8600)         ┌─────────────────────┐
│                     │  ─────────────────────────────▶  │                     │
│  LIBERO Benchmark   │    POST /act  (image + text)     │  OpenVLA HTTP       │
│  Container          │  ◀─────────────────────────────  │  Server Container   │
│                     │    response: 7-dim action        │  (GPU, bfloat16)    │
└─────────────────────┘                                  └─────────────────────┘
```

---

## 2. 모델 다운로드 (Model Download)

### 자동 다운로드

서버를 처음 실행하면 HuggingFace Hub에서 체크포인트를 **자동으로 다운로드**합니다. 기본 체크포인트는 `openvla/openvla-7b-finetuned-libero-spatial`입니다.

### 수동 다운로드

네트워크 환경이 불안정한 경우, 사전에 수동으로 다운로드할 수 있습니다:

```bash
# huggingface-cli 설치 (이미 설치되어 있지 않은 경우)
pip install huggingface_hub[cli]

# 체크포인트 다운로드 (~15GB)
huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial
```

### LIBERO Suite 체크포인트 목록

각 LIBERO 태스크 스위트에 대응하는 fine-tuned 체크포인트가 있습니다:

| Suite | HuggingFace Checkpoint |
|-------|----------------------|
| LIBERO-Spatial | `openvla/openvla-7b-finetuned-libero-spatial` |
| LIBERO-Object | `openvla/openvla-7b-finetuned-libero-object` |
| LIBERO-Goal | `openvla/openvla-7b-finetuned-libero-goal` |
| LIBERO-10 (Long) | `openvla/openvla-7b-finetuned-libero-10` |

### 캐시 경로

- **호스트**: `~/.cache/huggingface` (HuggingFace 기본 캐시 디렉토리)
- **컨테이너**: `/hf_cache` (Docker Compose에서 자동 마운트)
- **모델 크기**: 약 **15GB** (7B parameters, bfloat16 precision)

> **참고**: 한 번 다운로드하면 캐시에 저장되므로, 이후 실행 시 다시 다운로드하지 않습니다.

---

## 3. Docker 이미지 (Docker Image)

### 이미지 정보

| 항목 | 내용 |
|------|------|
| **Docker Hub** | `bigenlight/openvla-http:latest` |
| **이미지 크기** | ~17GB |
| **Base** | CUDA 12.1 |

### 포함된 의존성

- Python 3.10
- PyTorch 2.2.0+cu121
- flash-attn 2.5.5
- transformers 4.40.1
- FastAPI + Uvicorn

### 포함되지 않은 것 (마운트 필요)

- **모델 체크포인트**: HuggingFace 캐시를 `/hf_cache`로 마운트
- **소스 코드**: 프로젝트 루트를 컨테이너 내부로 마운트

### 이미지 Pull

```bash
docker pull bigenlight/openvla-http:latest
```

---

## 4. 빠른 시작 (Quick Start)

### a) Docker Hub에서 이미지 Pull

```bash
docker pull bigenlight/openvla-http:latest
```

### b) 소스 코드 Clone

```bash
git clone git@github.com:Bigenlight/openvla_docker.git
cd openvla_docker
```

### c) Docker Compose로 서버 시작 (Real Mode)

첫 실행 시 HuggingFace Hub에서 약 15GB 체크포인트를 자동으로 다운로드합니다:

```bash
docker compose -f scripts/docker/openvla_http_compose.yml up
```

서버가 정상적으로 시작되면 다음과 같은 로그가 출력됩니다:

```
INFO:     Uvicorn running on http://0.0.0.0:8600
INFO:     Model loaded: openvla/openvla-7b-finetuned-libero-spatial
```

### d) Dummy Mode (모델 없이 프로토콜 테스트)

GPU가 없거나 모델 다운로드 없이 통신 프로토콜만 테스트하고 싶을 때 사용합니다. 랜덤 action 값을 반환합니다:

```bash
OPENVLA_HTTP_ARGS="--dummy" docker compose -f scripts/docker/openvla_http_compose.yml up
```

### e) Health Check

서버가 정상 동작하는지 확인합니다:

```bash
curl http://localhost:8600/health
```

정상 응답:

```json
{"status": "ok"}
```

### 전체 워크플로우 요약

```bash
# 1. 이미지 pull
docker pull bigenlight/openvla-http:latest

# 2. 소스 clone
git clone git@github.com:Bigenlight/openvla_docker.git
cd openvla_docker

# 3. 서버 시작 (첫 실행 시 HF에서 ~15GB 체크포인트 자동 다운로드)
docker compose -f scripts/docker/openvla_http_compose.yml up

# 4. (선택) 프로토콜 테스트만 — 모델 없이
OPENVLA_HTTP_ARGS="--dummy" docker compose -f scripts/docker/openvla_http_compose.yml up

# 5. Health check
curl http://localhost:8600/health
```

---

## 5. LIBERO 벤치마크 실행 (Running LIBERO Benchmark)

### 사전 요구 사항

- OpenVLA HTTP 서버가 포트 8600에서 실행 중이어야 합니다 (위의 [빠른 시작](#4-빠른-시작-quick-start) 참고).
- LIBERO 벤치마크 컨테이너가 필요합니다: `bigenlight/libero-pro:latest`

### 평가 실행

```bash
# OpenVLA 서버가 8600에서 실행 중인 상태에서:
cd Libero-pro_benchmark

# 기본 실행 (전체 태스크, 기본 trial 수)
./run.sh --vla-eval libero_spatial --vla-url http://localhost:8600

# 태스크 수와 trial 수 지정
./run.sh --vla-eval libero_spatial --vla-url http://localhost:8600 \
    --vla-num-tasks 10 --vla-num-trials 50
```

### 다른 Suite 평가

체크포인트와 `--unnorm-key`를 맞춰서 실행해야 합니다:

```bash
# LIBERO-Object
./run.sh --vla-eval libero_object --vla-url http://localhost:8600

# LIBERO-Goal
./run.sh --vla-eval libero_goal --vla-url http://localhost:8600

# LIBERO-10 (Long-Horizon)
./run.sh --vla-eval libero_10 --vla-url http://localhost:8600
```

### 결과 확인

- **요약 결과**: `test_outputs/eval/libero_spatial_YYYYMMDD_HHMMSS/summary.json`
- **에피소드 영상**: `test_outputs/eval/libero_spatial_YYYYMMDD_HHMMSS/videos/*.mp4`

---

## 6. 벤치마크 결과 (Benchmark Results)

### LIBERO-Spatial 결과 요약

| 항목 | 값 |
|------|-----|
| **Suite** | LIBERO-Spatial |
| **태스크 수** | 10 |
| **Trial 수 (per task)** | 5 |
| **총 성공 / 총 trial** | **37 / 50** |
| **성공률** | **74.0%** |
| **논문 보고 수치** | ~84.7% |

### 성능 지표

| 지표 | 값 |
|------|-----|
| **평균 inference latency** | ~260ms / step |
| **HTTP 통신 오버헤드** | 2.9ms (전체의 1.2%) |
| **테스트 GPU** | NVIDIA RTX A6000 |

> **참고**: 논문 보고 수치(~84.7%)와의 차이는 evaluation 환경 및 프레임워크 차이에 기인할 수 있습니다. HTTP 통신 오버헤드는 전체 latency의 약 1.2%로, 통신 프로토콜 도입에 따른 성능 저하는 무시할 수 있는 수준입니다.

### Per-Task Breakdown

```json
{
  "suite": "libero_spatial",
  "num_tasks": 10,
  "num_trials_per_task": 5,
  "total_successes": 37,
  "total_trials": 50,
  "overall_success_rate": 0.74,
  "per_task_results": [
    {"task_id": 0, "task_name": "pick_up_the_black_bowl_on_the_left_and_place_it_on_the_plate",               "successes": 4, "trials": 5, "success_rate": 0.80},
    {"task_id": 1, "task_name": "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",        "successes": 5, "trials": 5, "success_rate": 1.00},
    {"task_id": 2, "task_name": "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",          "successes": 3, "trials": 5, "success_rate": 0.60},
    {"task_id": 3, "task_name": "pick_up_the_black_bowl_on_the_right_and_place_it_on_the_plate",               "successes": 3, "trials": 5, "success_rate": 0.60},
    {"task_id": 4, "task_name": "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", "successes": 2, "trials": 5, "success_rate": 0.40},
    {"task_id": 5, "task_name": "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",               "successes": 5, "trials": 5, "success_rate": 1.00},
    {"task_id": 6, "task_name": "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",          "successes": 4, "trials": 5, "success_rate": 0.80},
    {"task_id": 7, "task_name": "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",     "successes": 4, "trials": 5, "success_rate": 0.80},
    {"task_id": 8, "task_name": "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",      "successes": 3, "trials": 5, "success_rate": 0.60},
    {"task_id": 9, "task_name": "pick_up_the_black_bowl_next_to_the_wine_bottle_and_place_it_on_the_plate",    "successes": 4, "trials": 5, "success_rate": 0.80}
  ]
}
```

---

## 7. 서버 CLI 옵션

서버 실행 시 사용할 수 있는 CLI 옵션입니다. Docker Compose 환경에서는 `OPENVLA_HTTP_ARGS` 환경 변수를 통해 전달합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--port` | `8600` | HTTP 서버 포트 |
| `--checkpoint` | `openvla/openvla-7b-finetuned-libero-spatial` | HuggingFace Hub 모델 이름 또는 로컬 경로 |
| `--unnorm-key` | `libero_spatial` | Action unnormalization에 사용할 데이터셋 키 |
| `--attn-impl` | `flash_attention_2` | Attention 구현 방식 (`flash_attention_2` / `sdpa` / `eager`) |
| `--no-center-crop` | (비활성) | 이미지 center crop 비활성화 |
| `--load-in-8bit` | (비활성) | 8-bit 양자화로 모델 로드 (VRAM 절약) |
| `--load-in-4bit` | (비활성) | 4-bit 양자화로 모델 로드 (VRAM 추가 절약) |
| `--dummy` | (비활성) | 모델 로드 없이 dummy 모드로 실행 (프로토콜 테스트용) |

### 사용 예시

```bash
# 8-bit 양자화 + SDPA attention으로 실행
OPENVLA_HTTP_ARGS="--load-in-8bit --attn-impl sdpa" \
    docker compose -f scripts/docker/openvla_http_compose.yml up

# 다른 체크포인트 사용
OPENVLA_HTTP_ARGS="--checkpoint openvla/openvla-7b-finetuned-libero-object --unnorm-key libero_object" \
    docker compose -f scripts/docker/openvla_http_compose.yml up

# 포트 변경
OPENVLA_HTTP_ARGS="--port 9000" \
    docker compose -f scripts/docker/openvla_http_compose.yml up
```

---

## 8. 환경 변수

Docker Compose 실행 시 사용할 수 있는 환경 변수입니다:

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `OPENVLA_HTTP_PORT` | `8600` | 호스트에 노출할 서버 포트 |
| `OPENVLA_HTTP_ARGS` | (없음) | 서버에 전달할 추가 CLI 인자 (위의 CLI 옵션 참고) |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace 캐시 디렉토리 (컨테이너 내 `/hf_cache`로 마운트) |

### 사용 예시

```bash
# 모든 환경 변수를 조합하여 실행
OPENVLA_HTTP_PORT=8600 \
HF_HOME=/data/hf_cache \
OPENVLA_HTTP_ARGS="--checkpoint openvla/openvla-7b-finetuned-libero-spatial --load-in-8bit" \
    docker compose -f scripts/docker/openvla_http_compose.yml up
```

---

## 9. 트러블슈팅

### flash-attn 빌드/로드 실패

flash-attn은 특정 GPU 아키텍처에서 호환성 문제가 발생할 수 있습니다. 이 경우 PyTorch 내장 SDPA를 사용하세요:

```bash
OPENVLA_HTTP_ARGS="--attn-impl sdpa" \
    docker compose -f scripts/docker/openvla_http_compose.yml up
```

> `sdpa` (Scaled Dot-Product Attention)는 flash-attn과 유사한 성능을 제공하며, 별도 설치가 필요 없습니다.

### CUDA Out of Memory

7B 모델은 bfloat16 기준 약 14GB VRAM이 필요합니다. VRAM이 부족한 경우 8-bit 양자화를 사용하세요:

```bash
# 8-bit 양자화 (~7GB VRAM)
OPENVLA_HTTP_ARGS="--load-in-8bit" \
    docker compose -f scripts/docker/openvla_http_compose.yml up

# 4-bit 양자화 (~4GB VRAM, 정확도 저하 가능)
OPENVLA_HTTP_ARGS="--load-in-4bit" \
    docker compose -f scripts/docker/openvla_http_compose.yml up
```

### 체크포인트 다운로드가 느린 경우

서버 시작 전에 미리 다운로드해 두면 시작 시간을 단축할 수 있습니다:

```bash
# 사전 다운로드
huggingface-cli download openvla/openvla-7b-finetuned-libero-spatial

# 다운로드 완료 후 서버 시작
docker compose -f scripts/docker/openvla_http_compose.yml up
```

> **Tip**: `HF_HUB_ENABLE_HF_TRANSFER=1` 환경 변수를 설정하면 다운로드 속도가 크게 향상될 수 있습니다 (`pip install hf_transfer` 필요).

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
docker compose -f scripts/docker/openvla_http_compose.yml logs -f

# 최근 100줄만 확인
docker compose -f scripts/docker/openvla_http_compose.yml logs --tail 100
```

---

## 라이선스

이 포크는 원본 [OpenVLA](https://github.com/openvla/openvla) 저장소의 라이선스를 따릅니다.
