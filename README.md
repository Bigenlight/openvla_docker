# OpenVLA-OFT Docker — Dockerized HTTP Inference Server for LIBERO Benchmarks

> **⚠️ 주의: 이 브랜치(`oft`)는 [moojink/openvla-oft](https://github.com/moojink/openvla-oft) 포크입니다.**
> Bigenlight/openvla_docker 레포는 서로 다른 upstream의 두 코드베이스를 **브랜치로 분리**해서 관리합니다:
>
> | 브랜치 | Upstream | 모델 | Docker 이미지 | 포트 | LIBERO-Spatial 성공률 |
> |--------|---------|------|---------------|------|----------------------|
> | [`main`](../../tree/main) | [openvla/openvla](https://github.com/openvla/openvla) | OpenVLA (vanilla, 7B) | `bigenlight/openvla-http` | 8600 | 74% (37/50) |
> | **`oft`** (이 브랜치) | [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | OpenVLA-OFT (7B + OFT recipe) | `bigenlight/openvla-oft-http` | 8700 | **96% (48/50)** |
>
> 두 브랜치는 **unrelated histories**입니다 — 코드베이스가 다르므로 `git merge`하면 안 됩니다.
> Docker 컨테이너 + HTTP 서버로 패키징하여 LIBERO / LIBERO-PRO 벤치마크와 통합 프로토콜로 통신합니다.
> 원본 README는 아래에 그대로 유지되어 있습니다.

### Fork 추가 문서

| 문서 | 내용 |
|------|------|
| [README_DOCKER.md](README_DOCKER.md) | Docker 설치 가이드, 모델 다운로드, Quick Start, LIBERO 벤치마크 실행, 결과(96%), 트러블슈팅 (xet hang 포함) |
| [PROTOCOL_AND_ARCHITECTURE.md](PROTOCOL_AND_ARCHITECTURE.md) | 통신 규약 상세, JSON 스키마, 아키텍처 다이어그램, 11단계 데이터 흐름, 코드 경로 매핑, OpenVLA/OFT/Pi0.5 3자 비교, 지연 분석 |

### Fork 추가 파일

| 경로 | 설명 |
|------|------|
| `scripts/serve_openvla_oft_http.py` | FastAPI HTTP 서버 (포트 **8700**) |
| `scripts/docker/serve_openvla_oft_http.Dockerfile` | deps-only Docker 이미지 (moojink transformers fork 포함) |
| `scripts/docker/openvla_oft_http_compose.yml` | Docker Compose (소스 + HF 캐시 마운트, xet 비활성화) |
| `scripts/docker/openvla_oft_http_entrypoint.sh` | 컨테이너 entrypoint |
| `prismatic/**/{__init__}.py` | lazy-import 패치 (inference 시 training deps 제거) |

### Quick Start

```bash
docker pull bigenlight/openvla-oft-http:latest
git clone git@github.com:Bigenlight/openvla-oft_docker.git && cd openvla-oft_docker
docker compose -f scripts/docker/openvla_oft_http_compose.yml up
curl http://localhost:8700/health
```

### 벤치마크 결과 (LIBERO-Spatial)

| 모델 | 포트 | 성공률 (10 tasks × 5 trials) | 평균 latency/call | Effective per-step |
|------|------|------------------------------|--------------------|---------------------|
| OpenVLA (vanilla) | 8600 | 74% (37/50) | 260ms | 260ms |
| **OpenVLA-OFT** | **8700** | **96% (48/50)** | **169ms** | **~21ms** (8 chunk) |

자세한 내용은 [README_DOCKER.md](README_DOCKER.md)를 참고하세요.

---

# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (Original README)

**Project website: https://openvla-oft.github.io/**

**Paper: https://arxiv.org/abs/2502.19645**

**Summary video: https://youtu.be/T3Zkkr_NTSA**

## System Requirements

Inference:
* 1 GPU with ~16 GB VRAM for LIBERO sim benchmark tasks
* 1 GPU with ~18 GB VRAM for ALOHA robot tasks

Training:
* Between 1-8 GPUs with 27-80 GB, depending on the desired training setup (with default bfloat16 data type). See [this FAQ on our project website](https://openvla-oft.github.io/#train-compute) for details.

## Quick Start

First, set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Then, run the Python script below to download a pretrained OpenVLA-OFT checkpoint and run inference to generate an action chunk:

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
print("Generated action chunk:")
for act in actions:
    print(act)
```

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the conda environment.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

See [ALOHA.md](ALOHA.md) for fine-tuning/evaluating on real-world ALOHA robot tasks.

## Support

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2502.19645):

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
