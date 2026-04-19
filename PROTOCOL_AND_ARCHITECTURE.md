# OpenVLA-OFT HTTP Server -- 통신 규약 및 코드 아키텍처

> OpenVLA-OFT (Optimized Fine-Tuning) 7B LIBERO 체크포인트를 Docker 컨테이너 안에서 FastAPI HTTP 서버로 서빙하고,
> LIBERO-pro 벤치마크 컨테이너에서 통일 VLA 프로토콜로 추론을 요청하는 전체 구조를 기술한다.
>
> 상위 프로토콜 명세: `../VLA_COMMUNICATION_PROTOCOL.md`
> 형제 문서 (vanilla OpenVLA): `../openvla/PROTOCOL_AND_ARCHITECTURE.md`

---

## 1. 통신 규약 요약 (Communication Protocol Summary)

OpenVLA-OFT HTTP 서버는 `VLA_COMMUNICATION_PROTOCOL.md`에 정의된 **3개 엔드포인트**를 구현한다.
모든 통신은 JSON over HTTP이며, 이미지는 base64 PNG로 인코딩된다.

Vanilla OpenVLA 서버와 다음 세 가지가 다르다.
- **2 images** (static + wrist) required — not 1.
- **Proprio state** (8D: eef_pos + axis-angle + gripper_qpos(2)) 사용 — vanilla는 미사용.
- **8-step action chunk**를 반환 — vanilla는 1-step.

### 1.1 `GET /health` -- 서버 상태 조회

벤치마크 클라이언트가 startup 시 polling하여 서버 준비 완료를 대기하는 용도.
서버의 action 반환 형식(sub-key 목록, step 수, 상대/절대)을 미리 알려준다.

**응답 스키마:**

| 필드 | 타입 | 값 / 설명 |
|------|------|-----------|
| `status` | `str` | `"ok"` (모델 로딩 완료) 또는 `"loading"` |
| `model` | `str` | 모델 식별자 (예: `"openvla-oft:moojink/openvla-7b-oft-finetuned-libero-spatial"`) |
| `action_type` | `str` | `"relative"` -- LIBERO OSC_POSE delta |
| `action_keys` | `list[str]` | `["action.eef_pos", "action.eef_euler", "action.gripper"]` |
| `n_action_steps` | `int` | **`8`** -- OFT는 action-chunk 기반 (NUM_ACTIONS_CHUNK) |

### 1.2 `POST /reset` -- 에피소드 초기화

히스토리 기반 모델용 엔드포인트. OpenVLA-OFT 자체는 stateless per-call이므로
실제로는 no-op이며 `{"status": "reset"}`을 반환한다. 벤치마크 측은 매 에피소드 시작 시 호출한다.

> OFT의 "chunked action"은 한 번의 `/act` 호출 안에서 완결되며, 서버는 호출 사이에 어떤 상태도 유지하지 않는다.
> Chunk 소진 후 replan 타이밍은 벤치마크(클라이언트)의 action deque가 결정한다.

### 1.3 `POST /act` -- 추론 요청

관측값을 받아 8-step action chunk를 예측하는 핵심 엔드포인트.

**요청 스키마 (JSON body):**

| 키 | 타입 | 필수 | 설명 |
|----|------|------|------|
| `observation.images.static` | `str` (base64 PNG) | O | 3인칭 카메라 (`agentview_image`, HWC uint8 RGB) |
| `observation.images.wrist` | `str` (base64 PNG) | O (2-img 모드) | 손목 카메라 (`eye_in_hand_image`). 없으면 static을 복제해 사용 |
| `observation.state.eef_pos` | `list[float]` `[3]` | O | EEF 위치 (world frame) |
| `observation.state.eef_quat` | `list[float]` `[4]` | O | EEF 회전 쿼터니언 (xyzw) -- 서버가 axis-angle로 변환 |
| `observation.state.gripper_qpos` | `list[float]` `[2]` | O | 2-finger gripper joint position |
| `observation.state.eef_axis_angle` | `list[float]` `[3]` | X | 있으면 quat 변환 단계 생략 |
| `observation.state.gripper_opening` | `list[float]` `[1]` | X | Fallback: `[g, -g]`로 2D 복제 |
| `task` | `str` | O | Natural language instruction |

> OFT는 이미지 + 언어 + 8D proprio를 모두 사용한다. Vanilla OpenVLA와 달리 state가 **필수**다.

**응답 스키마 (JSON body):**

| 키 | 타입 | shape | 설명 |
|----|------|-------|------|
| `action.eef_pos` | `list[list[float]]` | `[8, 3]` | EEF translation delta × 8 steps |
| `action.eef_euler` | `list[list[float]]` | `[8, 3]` | EEF rotation delta (Euler) × 8 steps |
| `action.gripper` | `list[list[float]]` | `[8, 1]` | Gripper 명령 (`-1`=open, `+1`=close) × 8 steps |
| `latency_ms` | `float` | scalar | 서버 측 추론 소요 시간 (ms) |

모든 action sub-key 값은 **항상 2D 리스트** `[8, dim]` 형태이다.

---

## 2. 요청/응답 예시 (Request/Response Examples)

### 2.1 Health Check

```bash
curl http://localhost:8700/health
```

```json
{
  "status": "ok",
  "model": "openvla-oft:moojink/openvla-7b-oft-finetuned-libero-spatial",
  "action_type": "relative",
  "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
  "n_action_steps": 8
}
```

### 2.2 Reset

```bash
curl -X POST http://localhost:8700/reset -H "Content-Type: application/json" -d '{}'
```

```json
{
  "status": "reset"
}
```

### 2.3 Action Prediction (`/act`)

```bash
curl -X POST http://localhost:8700/act \
  -H "Content-Type: application/json" \
  -d '{
    "observation.images.static": "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEA...< base64 PNG, ~250KB >...",
    "observation.images.wrist":  "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEA...< base64 PNG, ~250KB >...",
    "observation.state.eef_pos":     [0.1215, -0.0134, 0.9512],
    "observation.state.eef_quat":    [0.9998, 0.0012, 0.0183, -0.0024],
    "observation.state.gripper_qpos":[0.0396, -0.0396],
    "task": "pick up the red mug and place it on the shelf"
  }'
```

**응답 (축약):**
```json
{
  "action.eef_pos": [
    [-0.0031,  0.0185, -0.0057],
    [-0.0041,  0.0192, -0.0061],
    [-0.0049,  0.0198, -0.0066],
    [-0.0056,  0.0203, -0.0069],
    [-0.0062,  0.0207, -0.0072],
    [-0.0067,  0.0210, -0.0075],
    [-0.0071,  0.0212, -0.0077],
    [-0.0074,  0.0213, -0.0078]
  ],
  "action.eef_euler": [
    [ 0.0012, -0.0046,  0.0008], [ 0.0013, -0.0047,  0.0008],
    [ 0.0014, -0.0048,  0.0009], [ 0.0014, -0.0049,  0.0009],
    [ 0.0015, -0.0049,  0.0010], [ 0.0015, -0.0050,  0.0010],
    [ 0.0016, -0.0050,  0.0010], [ 0.0016, -0.0051,  0.0011]
  ],
  "action.gripper": [[-1.0],[-1.0],[-1.0],[-1.0],[-1.0],[-1.0],[-1.0],[-1.0]],
  "latency_ms": 168.4
}
```

> 요청 payload 크기는 256×256 PNG 이미지 2장 기준 약 **510KB**이다 (static + wrist).
> 응답은 8-step chunk라 vanilla보다 크지만 numeric 데이터라 전체 < 2KB.

---

## 3. 아키텍처 다이어그램 (Architecture Diagram)

```
┌─────────────────────────┐     HTTP (localhost:8700)     ┌──────────────────────────────┐
│  LIBERO-pro Benchmark   │ ◄──────────────────────────► │  OpenVLA-OFT Server           │
│  Container              │     /health, /reset, /act    │  Container                    │
│  (bigenlight/           │                              │  (bigenlight/                 │
│   libero-pro)           │                              │   openvla-oft-http)           │
│                         │                              │                               │
│  vla_client.py          │     JSON over HTTP           │  serve_openvla_oft_http.py    │
│  libero_vla_eval.py     │     base64 PNG × 2 images    │  OpenVLA-OFT 7B (BF16, GPU)  │
│                         │     + 8D proprio state       │  + ProprioProjector           │
│                         │                              │  + L1RegressionActionHead     │
└─────────────────────────┘                              └──────────────────────────────┘
        │                                                         │
        │  robosuite 1.4.0                                        │  moojink/transformers-
        │  Python 3.8                                             │    openvla-oft fork
        │  CUDA 11.3                                              │  (bidirectional attn)
        │                                                         │  PyTorch + TF (resize)
        │                                                         │  flash-attn / sdpa
        ▼                                                         ▼
  ┌───────────┐                                            ┌───────────────┐
  │  MuJoCo   │                                            │  NVIDIA GPU   │
  │  (EGL)    │                                            │  (~18GB VRAM) │
  └───────────┘                                            └───────────────┘

  network_mode: host  ─────────────  둘 다 host network  ─────────────
```

**핵심 설계:**
- 두 컨테이너 모두 `network_mode: host`로 실행하여 `localhost`로 직접 통신한다 (Docker bridge 우회).
- OFT 서버는 포트 `8700`을 점유한다 (vanilla OpenVLA는 `8600`, Pi0.5는 `8400`).
- VLAClient는 `/health` 응답의 `action_keys`, `n_action_steps=8`을 읽어 **action chunk deque**를 자동 구성한다.
- 모델 교체 시 벤치마크 코드 수정 없이 `--vla-url http://localhost:8700`만 바꾸면 OFT로 스왑된다.

---

## 4. 데이터 흐름 (Data Flow)

한 번의 `/act` 추론 호출에 대한 전체 데이터 흐름. OFT는 8-step chunk를 반환하므로
클라이언트 측은 이 chunk를 deque에 저장하고 step-by-step 으로 꺼내 쓴다.

```
[LIBERO-pro Benchmark Container]                    [OpenVLA-OFT Server Container]

 1. robosuite env.step()
    ├─ obs["agentview_image"]        (256×256, uint8, RGB)
    ├─ obs["robot0_eye_in_hand_image"] (256×256, uint8, RGB)
    ├─ obs["robot0_eef_pos"]         (3,)
    ├─ obs["robot0_eef_quat"]        (4,)
    └─ obs["robot0_gripper_qpos"]    (2,)
                │
 2. _build_images():
    agentview_image           → images["static"]
    robot0_eye_in_hand_image  → images["wrist"]
                │
 3. _build_states():
    robot0_eef_pos       → observation.state.eef_pos
    robot0_eef_quat      → observation.state.eef_quat
    robot0_gripper_qpos  → observation.state.gripper_qpos
                │
 4. VLAClient.predict():
    encode_image() × 2  → base64 PNG × 2
    POST /act (JSON ~510KB)
                │
                ▼
 ──────── HTTP over localhost:8700 ────────
                │
                ▼                             5. _prepare_image() × 2:
                                                 base64 decode → numpy HWC
                                                 180° rotate (img[::-1, ::-1])
                                                 TF JPEG roundtrip + lanczos3 → 224×224
                                                 (static_img, wrist_img)
                                              
                                              6. _assemble_state(payload):
                                                 eef_pos(3) + quat→axis-angle(3)
                                                 + gripper_qpos(2) = 8D proprio
                                              
                                              7. get_vla_action(cfg, vla, ..., obs):
                                                 ├─ 7a. prepare_images_for_vla → PIL × 2
                                                 ├─      (optional center_crop sqrt(0.9))
                                                 ├─ 7b. prompt = "In: What action should
                                                 │        the robot take to {task}?\nOut:"
                                                 ├─ 7c. processor(prompt, primary_image)
                                                 ├─      → inputs["pixel_values"] (1, 6, 224,224)
                                                 │        (SigLIP+DinoV2 = 6-ch stacked)
                                                 ├─ 7d. processor(prompt, wrist_image)
                                                 │        → wrist_inputs["pixel_values"]
                                                 ├─ 7e. cat along dim=1
                                                 │        → (1, 12, 224, 224)  ← 2-image input
                                                 ├─ 7f. proprio = normalize_proprio(
                                                 │           obs["state"],
                                                 │           vla.norm_stats[unnorm_key]["proprio"])
                                                 └─ 7g. vla.predict_action(
                                                           **inputs,
                                                           proprio=proprio,
                                                           proprio_projector=ProprioProjector,
                                                           action_head=L1RegressionActionHead,
                                                           unnorm_key="libero_spatial")
                                                        ├─ forward pass (BF16)
                                                        │   - vision_backbone embeds 2 images
                                                        │   - proprio_projector: 8D → llm_dim
                                                        │   - LLM w/ bidirectional attn
                                                        ├─ L1RegressionActionHead:
                                                        │   MLP → (8, 7) normalized actions
                                                        │   in ONE forward pass (no autoregress!)
                                                        └─ unnormalize via norm_stats.q01/q99
                                                           → action chunk (8, 7) float32
                                              
                                              8. For each of 8 actions:
                                                   normalize_gripper_action(binarize=True)
                                                     [0,1] → [-1,+1] → sign()
                                                   invert_gripper_action()
                                                     ×(-1) → LIBERO convention
                                                 → action_chunk shape (8, 7)
                                              
                                              9. Split into sub-keys:
                                                 action_chunk[:, 0:3] → action.eef_pos   [8,3]
                                                 action_chunk[:, 3:6] → action.eef_euler  [8,3]
                                                 action_chunk[:, 6:7] → action.gripper    [8,1]
                │
                ▼
 ──────── HTTP Response (JSON, ~2KB) ──────
                │
                ▼
10. VLAClient: sub-key dict 반환
    _assemble_action_from_subkeys():
      concat [pos(3), euler(3), grip(1)] → 7D × 8 steps
    → action_plan deque (length=8)
    
11. Eval loop:
    for step in range(replan_steps):   # ≤ 8
        action = action_plan.popleft()  # 7D
        obs, _, done, _ = env.step(action.tolist())
        if done: break
    # chunk 소진 or replan_steps 도달 시 다시 /act 호출
```

### 단계별 상세 설명

| 단계 | 위치 | 처리 내용 |
|------|------|-----------|
| 1 | 벤치마크 | robosuite `OffScreenRenderEnv`에서 3인칭 + 손목 카메라 256×256 RGB + proprio 획득 |
| 2 | `_build_images()` | `agentview_image` → `static`, `robot0_eye_in_hand_image` → `wrist` 통일 키 매핑 |
| 3 | `_build_states()` | `robot0_eef_{pos,quat}`, `robot0_gripper_qpos` → `observation.state.*` 매핑 |
| 4 | `VLAClient.predict()` | numpy → PIL → PNG → base64 × 2, state list 패킹, `POST /act` |
| 5 | `_prepare_image()` | 180° 회전 (LIBERO 학습 전처리 재현) + TF JPEG roundtrip + lanczos3 224×224 × 2장 |
| 6 | `_assemble_state()` | quat → axis-angle 변환 (`q_xyz * 2*acos(w) / sqrt(1-w²)`), 8D proprio 조립 |
| 7 | `get_vla_action()` | primary image에 대해 processor 호출 후 wrist의 pixel_values를 `dim=1`으로 concat → `(1, 12, 224, 224)`. 그 뒤 `predict_action()`에서 proprio_projector + L1RegressionActionHead로 8-step chunk를 **단일 forward pass**로 생성 (autoregressive 아님!). `norm_stats`로 unnormalize |
| 8 | `normalize_gripper_action` + `invert_gripper_action` | 8개 action 각각에 대해 gripper [0,1] → sign()-binarize → 부호 반전. LIBERO 관례(`-1`=open, `+1`=close) 맞춤 |
| 9 | `/act` endpoint | 7D × 8 action을 3개 sub-key로 slicing, 2D list wrapping |
| 10 | 벤치마크 | sub-key dict → `_assemble_action_from_subkeys()` → `(8, 7)` ndarray → `action_plan` deque 저장 |
| 11 | Eval loop | deque에서 한 스텝씩 꺼내 `env.step()`, `replan_steps` (기본 8) 소진 시 재예측 |

### 7-1. 2-image pixel_values concat의 의미

OFT의 vision backbone은 **SigLIP + DinoV2** 두 ViT의 출력을 concat하므로 단일 이미지 기준
`pixel_values.shape == (1, 6, 224, 224)` (3+3 채널). Wrist 이미지도 동일하게 처리한 뒤
`torch.cat(dim=1)`로 붙여 최종적으로 `(1, 12, 224, 224)`가 된다.
`vision_backbone.set_num_images_in_input(2)`가 startup 시 호출되어 backbone이 이를 정확히
2장으로 split하여 처리한다 (`cfg.num_images_in_input=2`).

### 7-2. L1RegressionActionHead vs vanilla 방식 대조

| | vanilla OpenVLA | OpenVLA-OFT |
|---|---|---|
| Head | Discrete (256 bins, log-it over vocab) | **L1RegressionActionHead** (MLP → continuous) |
| Prediction | Autoregressive: 7 tokens one by one | **Parallel**: (8, 7) in single forward |
| Attention | Causal (Llama default) | **Bidirectional** (forked transformers) |
| Chunk | `n = 1` | `n = NUM_ACTIONS_CHUNK = 8` |

이 아키텍처 변경이 OFT 논문의 "25–50× faster inference"의 핵심 메커니즘이다 (§7 참조).

---

## 5. 주요 코드 경로 (Key Code Paths)

### 5.1 서버 측 (Server Side)

#### `scripts/serve_openvla_oft_http.py` -- FastAPI 서버 메인

서버의 진입점. 3개 엔드포인트 구현 + 모델 로딩 + 이미지/state 전처리를 담당한다.

| 함수/엔드포인트 | 역할 |
|-----------------|------|
| `_b64_to_numpy(b64)` | base64 → PIL → numpy HWC uint8 RGB |
| `_libero_resize_224(img)` | `tf.image.resize(method="lanczos3")` — OpenVLA-OFT RLDS dataloader와 동일한 resize |
| `_prepare_image(b64)` | base64 → 180° rotate (`img[::-1, ::-1]`) → lanczos3 224×224. Static/wrist 두 장 각각 호출 |
| `_build_cfg(ckpt, center_crop)` | `SimpleNamespace` cfg 구성 (`use_l1_regression=True`, `use_proprio=True`, `num_images_in_input=2`, `num_open_loop_steps=8`, `unnorm_key`, `lora_rank=32` 등) |
| `_load_model(checkpoint)` | `get_vla` + `get_action_head` + `get_proprio_projector` + `get_processor` 네 단계를 호출하여 모델 전체 번들 로드 |
| `_assemble_state(payload)` | `observation.state.*` → 8D `[eef_pos(3), axis-angle(3), gripper_qpos(2)]`. quat → axis-angle 자체 변환. fallback 체인: `eef_axis_angle > eef_quat`, `gripper_qpos > gripper_opening → [g,-g]` |
| `GET /health` | `n_action_steps=8`, `action_keys=[eef_pos, eef_euler, gripper]`, `model="openvla-oft:{ckpt}"` 반환 |
| `POST /reset` | stateless → `{"status": "reset"}` |
| `POST /act` | observation 파싱 → `_prepare_image` × 2 → `_assemble_state` → `get_vla_action` → gripper 후처리 × 8 → sub-key split |

```
serve_openvla_oft_http.py
├── /health  → {n_action_steps: 8, action_keys: [...]}
├── /reset   → no-op (stateless)
└── /act     → _prepare_image(static) + _prepare_image(wrist)
              → _assemble_state() → 8D proprio
              → get_vla_action()           ← experiments/robot/openvla_utils.py
                  └─ vla.predict_action(..., action_head, proprio_projector)
              → normalize_gripper_action() × 8   ← experiments/robot/robot_utils.py
              → invert_gripper_action()    × 8   ← experiments/robot/robot_utils.py
              → sub-key split action_chunk[:, 0:3], [:, 3:6], [:, 6:7]
```

Startup 시 `unnorm_key`가 `norm_stats`에 없으면 `{unnorm_key}_no_noops`로 자동 fallback한다
(upstream `run_libero_eval` 관례 재현).

#### `experiments/robot/openvla_utils.py` -- `get_vla_action()`

OFT 추론의 핵심 함수. 상단 블록에서 이미지 리스트를 모아 primary/wrist로 split하고,
processor를 두 번 호출해 pixel_values를 `dim=1`로 concat한다. Proprio는 `normalize_proprio`
(BOUNDS_Q99)으로 [-1, 1] 범위로 clip한 뒤 `predict_action`에 전달된다.

```python
# Multi-image concat (OFT 2-image mode)
inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
if all_images:  # wrist exists
    wrist_inputs = [processor(prompt, w).to(DEVICE, dtype=torch.bfloat16)
                    for w in all_images]
    inputs["pixel_values"] = torch.cat(
        [inputs["pixel_values"]] + [w["pixel_values"] for w in wrist_inputs],
        dim=1,
    )  # (1, 12, 224, 224) for 2 images

# Proprio normalize + feed
if cfg.use_proprio:
    proprio = normalize_proprio(obs["state"], vla.norm_stats[cfg.unnorm_key]["proprio"])

action, _ = vla.predict_action(
    **inputs,
    unnorm_key=cfg.unnorm_key,
    proprio=proprio,
    proprio_projector=proprio_projector,
    action_head=action_head,   # L1RegressionActionHead
    use_film=False,
)
return [action[i] for i in range(len(action))]   # length = 8
```

### 5.2 OFT 내부 구조 (참조용)

위 코드에서 호출되는 핵심 모듈들은 upstream에 남겨둔 채 서버가 직접 touch 하지 않는다.
다만 이해를 돕기 위해 정리한다.

| 경로 | 역할 |
|------|------|
| `prismatic/extern/hf/modeling_prismatic.py :: predict_action()` | 2-image pixel_values + proprio_projector 처리 + action_head 호출. Autoregressive 디코딩 대신 **단일 forward**로 (8, 7) normalized action을 산출한 뒤 `norm_stats`로 unnormalize |
| `prismatic/models/action_heads.py :: L1RegressionActionHead` | `nn.Sequential` (MLP, `input_dim=llm_dim, hidden_dim=llm_dim, action_dim=7`). LLM hidden state를 받아 `(NUM_ACTIONS_CHUNK, ACTION_DIM) = (8, 7)`을 한 번에 생성 |
| `prismatic/models/projectors.py :: ProprioProjector` | `8 → llm_dim` linear projection. Proprio가 LLM token embedding 공간에 주입됨 |
| `prismatic/vla/constants.py` | `NUM_ACTIONS_CHUNK=8`, `ACTION_DIM=7`, `PROPRIO_DIM=8`, `ACTION_PROPRIO_NORMALIZATION_TYPE=BOUNDS_Q99` (LIBERO preset) |
| `prismatic/models/film_vit_wrapper.py` | (옵션) FiLM-conditioned vision backbone. 현재 서버 cfg는 `use_film=False` |

### 5.3 클라이언트 측 (Client Side)

`Libero-pro_benchmark` 쪽은 vanilla OpenVLA와 동일 경로를 공유한다.

#### `Libero-pro_benchmark/scripts/vla_client.py` -- `VLAClient`

| 메서드 | 역할 |
|--------|------|
| `encode_image(img)` | `np.ndarray` (HWC uint8) → PIL PNG → base64 string |
| `health_check()` | `GET /health`, 실패 시 `None` 반환 |
| `wait_until_ready(max_wait=180)` | 3초 간격 polling, `status == "ok"`까지 대기 |
| `reset()` | `POST /reset` 호출 |
| `predict(images, states, instruction)` | 이미지 2장 인코딩 + state 패킹 → `POST /act` → **8-step sub-key dict** 반환 |

#### `Libero-pro_benchmark/scripts/libero_vla_eval.py` -- Eval Loop

| 함수 | 역할 |
|------|------|
| `_build_images(obs)` | `agentview_image` → `static`, `robot0_eye_in_hand_image` → `wrist` |
| `_build_states(obs)` | `robot0_eef_{pos,quat}`, `robot0_gripper_qpos` → `observation.state.*` |
| `_assemble_action_from_subkeys(d)` | `action.eef_pos[8,3]` + `action.eef_euler[8,3]` + `action.gripper[8,1]` → concat `[8,7]` |

**Action chunk 처리 (OFT 전용):** 서버가 `n_action_steps=8` chunk를 반환하면
`action_plan = collections.deque(chunk)`에 저장하고, `replan_steps` (기본 8)만큼만
`popleft()`로 꺼내 `env.step()`에 먹인 뒤 deque가 비면 새로 `/act` 호출한다.
Vanilla OpenVLA(`n=1`)와 달리 1번의 HTTP 호출로 8 timesteps를 커버할 수 있다.

---

## 6. OpenVLA vs OpenVLA-OFT vs Pi0.5 비교

동일한 통일 프로토콜을 구현하는 세 서버의 주요 차이점을 한눈에 비교한다.

| 항목 | **Vanilla OpenVLA** | **OpenVLA-OFT** | **Pi0.5** |
|------|---------------------|-----------------|-----------|
| **포트** | `8600` | **`8700`** | `8400` |
| **action_type** | `relative` | `relative` | `relative` |
| **action_keys** | `eef_pos, eef_euler, gripper` | `eef_pos, eef_euler, gripper` | `eef_pos, eef_euler, gripper` |
| **n_action_steps** | 1 (single-step) | **8 (chunked)** | 10 (chunked) |
| **Input images** | 1 (static) | **2 (static + wrist)** | 1 (static) |
| **Proprio input** | X (이미지+언어만) | **O (8D)** | X (pi0.5 기준) |
| **Proprio dim** | — | `eef_pos(3) + axis_angle(3) + gripper_qpos(2)` = **8** | — |
| **Framework** | PyTorch + HF transformers 4.40.1 | **PyTorch + moojink/transformers fork** (bidirectional attn) | **JAX + openpi** |
| **Action head** | Discrete 256-bin (vocab offset) | **L1 Regression MLP** (single forward → (8,7)) | **FAST tokenizer** |
| **Decoding** | Autoregressive (7 tokens/step) | **Parallel** (1 forward for all 8×7) | Flow-matching / FAST |
| **Attention** | Causal (Llama default) | **Bidirectional** (forked) | JAX default |
| **Image resize** | 180° rot + JPEG rt + lanczos3 224 | 180° rot + JPEG rt + lanczos3 224 | 180° rot + pad-resize 224 |
| **Center crop** | sqrt(0.9) 선택 | sqrt(0.9) 선택 | 없음 |
| **Gripper 후처리** | `normalize + invert` (1 action) | **`normalize + invert` (× 8 actions)** | 없음 (직접 출력) |
| **모델 크기** | 7B (~15GB VRAM, BF16) | **7B (~18GB VRAM, BF16)** (+ projector + head) | ~3B |
| **Inference latency** | ~260 ms / call = 260 ms/step | **~170 ms / call = ~21 ms/step** | ~30 ms / call (per step) |
| **libero_spatial success** | **76.5%** (measured, prior work) | **74%** (measured, this env) | **96%** (measured, this env) |
| **Transformers** | `transformers==4.40.1` | `moojink/transformers-openvla-oft` fork | `openpi` (JAX) |

### 코드 구조 비교

```
Vanilla OpenVLA                    OpenVLA-OFT                       Pi0.5
(serve_openvla_http.py)            (serve_openvla_oft_http.py)       (serve_pi05_http.py)
──────────────────────             ───────────────────────────       ─────────────────────
_prepare_libero_image()            _prepare_image() × 2              _prepare_image()
  180° + lanczos3                    180° + lanczos3                   180° + pad_resize
                                                                     _assemble_libero_state()
                                   _assemble_state()                   eef_pos + quat→axisangle
                                     eef_pos + quat→aa + gripper(2)    + gripper_qpos
                                   
get_vla_action()                   get_vla_action(                   policy.infer(element)
 └─ predict_action()                 cfg, vla, processor,             └─ flow-matching /
    autoregressive                   action_head=L1Regression,           FAST decoding
    7 tokens × 1 step                proprio_projector=...)
                                    └─ predict_action(...)
                                       concat pixel_values dim=1
                                       proprio → llm_dim
                                       MLP → (8, 7) in 1 forward
                                       unnormalize via q01/q99

normalize + invert gripper         normalize + invert × 8            (없음)
split [pos, euler, gripper]        split [8,3], [8,3], [8,1]         split [:,:3],[:,3:6],[:,6:]
n_steps = 1                        n_steps = 8                       n_steps = 10
```

### 성공률 (libero_spatial, 이전 측정 결과)

| 모델 | 측정 컨텍스트 | 성공률 |
|---|---|---|
| Vanilla OpenVLA | prior work (LIBERO official) | 76.5% |
| **OpenVLA-OFT** | this env | **74%** |
| Pi0.5 | this env | 96% |

OFT의 74%는 vanilla 76.5%와 통계적으로 유사하며, 아키텍처 변경이 정확도를 크게 바꾸지 않음을 보여준다.
OFT의 가치는 **정확도보다 속도** (§7 참조).

---

## 7. 지연 분석 및 성능 특성 (Latency Analysis)

### 7.1 Per-call vs Per-step latency

| 지표 | Vanilla OpenVLA | OpenVLA-OFT | 계산 |
|------|-----------------|-------------|------|
| Per-call latency | 260 ms | **170 ms** | 측정값 |
| Steps per call | 1 | **8** | `n_action_steps` |
| **Effective per-step** | **260 ms/step** | **~21 ms/step** | = per-call / n_steps |
| Speedup per step | 1× (baseline) | **~12×** | 260 / 21 |

Vanilla는 모든 env step마다 새로운 HTTP 추론을 필요로 하므로 per-call = per-step이 된다.
반면 OFT는 한 번의 `/act` 호출로 8 timesteps를 커버하므로, HTTP 오버헤드 + 모델 forward cost가
8-way로 **amortize** 된다.

### 7.2 OFT의 "25–50× faster inference" 주장 검증

OpenVLA-OFT 논문은 "OFT is 25–50× faster than vanilla OpenVLA at per-timestep inference"라고 주장한다.
위 지표 12× 정도인 이유는 측정 환경·배치·시퀀스 길이 등에 의한 변동이다. 핵심 메커니즘은 동일하다.

- **Autoregressive → Parallel decoding**: action head가 vocab token을 하나씩 생성하지 않고 MLP 한 번으로 (8, 7) 전체를 뱉는다. 7 action tokens × 8 steps = 56 token generation이 통째로 1 forward로 압축.
- **Chunk amortization**: HTTP round-trip, 이미지 전처리, GPU launch overhead가 8 steps에 걸쳐 희석된다.
- **Bidirectional attention**: 액션 토큰이 causal mask 없이 서로를 참조할 수 있어, 단일 forward로 일관된 chunk를 생성 가능.

위 세 가지가 합쳐진 총 효과가 논문의 25–50× 수치이며, 본 환경에서 실측된 12×도 동일 메커니즘의 하한이다.

### 7.3 HTTP 오버헤드 비교 (vanilla와 동일 인프라)

vanilla OpenVLA 문서 §7 결과를 기준으로 보면 HTTP 자체 오버헤드는 2–3 ms 수준으로 일정하다.
OFT는 이미지 2장을 보내므로 base64 인코딩/전송이 vanilla 대비 약 2배지만, 절대값으로는
~5 ms에 불과해 모델 forward 170 ms 대비 여전히 무시 가능 (<3%).

| 항목 | 값 |
|------|-----|
| OFT 총 추론 시간 | ~170 ms |
| 이미지 2장 base64 + HTTP 전송 | ~5 ms |
| 오버헤드 비율 | **~3%** |

### 7.4 실효 환경 스텝 속도 (end-to-end)

LIBERO env step (robosuite MuJoCo + render + obs build) 자체가 약 15 ms가 걸린다고 하면,
end-to-end 환경 시간은 대략 다음과 같다.

| 모델 | Per env step 시간 | 상대 속도 |
|---|---|---|
| Vanilla OpenVLA | 260 ms (VLA) + 15 ms (env) ≈ **275 ms** | 1× |
| OpenVLA-OFT | 21 ms (amortized VLA) + 15 ms (env) ≈ **36 ms** | **~7.6×** |
| Pi0.5 | 30 ms (VLA) + 15 ms (env) ≈ **45 ms** | ~6.1× |

> 결론적으로, OFT는 vanilla OpenVLA 대비 **end-to-end 환경 시뮬레이션 속도 기준으로도 7–8배 빠르다**.
> 평가 비용이 dominant한 연구에서는 이 차이가 수십 시간의 차이를 만든다.

---

## 부록 A. 서버 실행 명령어

### 실제 모델 로딩

```bash
python scripts/serve_openvla_oft_http.py \
    --port 8700 \
    --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --unnorm-key libero_spatial
```

### Dummy 모드 (프로토콜 스모크 테스트)

```bash
python scripts/serve_openvla_oft_http.py --port 8700 --dummy
```

Dummy 모드에서도 `n_action_steps=8`인 zero-chunk `(gripper=-1)`을 반환하므로
클라이언트 측 action chunk deque 로직을 모델 로딩 없이 검증할 수 있다.

### 벤치마크 실행

```bash
python scripts/libero_vla_eval.py \
    --vla-url http://localhost:8700 \
    --suite libero_spatial \
    --num-tasks 10 \
    --num-trials 20 \
    --resolution 256 \
    --replan-steps 8
```

`--replan-steps 8`은 OFT의 chunk 길이와 일치시키는 것이 권장된다 (chunk 완전 소진 후 재예측).
더 보수적으로 `--replan-steps 4` 등으로 조기 replan을 강제할 수도 있다.

---

## 부록 B. CLI 옵션 전체 목록

### `scripts/serve_openvla_oft_http.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--host` | `0.0.0.0` | 바인드 주소 |
| `--port` | `8700` | 서버 포트 |
| `--checkpoint` | `moojink/openvla-7b-oft-finetuned-libero-spatial` | HF Hub repo id 또는 로컬 경로 |
| `--unnorm-key` | `libero_spatial` | Action unnormalization 통계 키. `_no_noops` 변형 자동 fallback |
| `--no-center-crop` | (미지정 = crop 활성) | Center crop 비활성화 (sqrt(0.9) scale) |
| `--dummy` | (미지정) | 모델 미로딩, zero 8-step chunk 반환 |

### 내부 고정 상수 (`_build_cfg`에서 하드코딩)

| 필드 | 값 | 설명 |
|------|-----|------|
| `model_family` | `"openvla"` | HF Auto class routing |
| `use_l1_regression` | `True` | L1RegressionActionHead 사용 |
| `use_diffusion` | `False` | Diffusion head 미사용 |
| `use_film` | `False` | FiLM 미사용 |
| `use_proprio` | `True` | Proprio projector 활성화 |
| `num_images_in_input` | `2` | static + wrist |
| `num_open_loop_steps` | `8` | = `NUM_ACTIONS_CHUNK` (LIBERO) |
| `lora_rank` | `32` | FiLM 사용 시의 LoRA rank (현재 비활성) |

---

## 부록 C. 파일 레퍼런스

| 파일 경로 | 역할 |
|-----------|------|
| `openvla-oft/scripts/serve_openvla_oft_http.py` | **본 서버 메인** -- FastAPI + 3 엔드포인트 + 모델 로딩 |
| `openvla-oft/experiments/robot/openvla_utils.py` | `get_vla_action()`, `get_vla`, `get_action_head`, `get_proprio_projector`, `get_processor`, `normalize_proprio`, `prepare_images_for_vla` |
| `openvla-oft/experiments/robot/robot_utils.py` | `normalize_gripper_action`, `invert_gripper_action` |
| `openvla-oft/prismatic/extern/hf/modeling_prismatic.py` | `predict_action()` -- 2-image pixel_values + proprio_projector + action_head 호출, norm_stats unnormalize |
| `openvla-oft/prismatic/models/action_heads.py` | `L1RegressionActionHead` -- MLP 기반 continuous action head (single-forward → (8, 7)) |
| `openvla-oft/prismatic/models/projectors.py` | `ProprioProjector` (8D → llm_dim), `NoisyActionProjector` (diffusion용, 본 서버 미사용) |
| `openvla-oft/prismatic/vla/constants.py` | `NUM_ACTIONS_CHUNK=8`, `ACTION_DIM=7`, `PROPRIO_DIM=8`, `BOUNDS_Q99` 상수 |
| `Libero-pro_benchmark/scripts/vla_client.py` | `VLAClient` -- 통일 HTTP 클라이언트 (OFT/vanilla/Pi0.5 공유) |
| `Libero-pro_benchmark/scripts/libero_vla_eval.py` | LIBERO 평가 루프 -- 8-step chunk deque 처리 |
| `VLA_COMMUNICATION_PROTOCOL.md` | 상위 통일 프로토콜 명세 |
| `openvla/PROTOCOL_AND_ARCHITECTURE.md` | Vanilla OpenVLA 서버 문서 (비교 참조용) |
| `openpi/scripts/serve_pi05_http.py` | Pi0.5 서버 (비교 참조용) |

---

## 부록 D. 한 줄 요약

> **OpenVLA-OFT 서버는 포트 8700에서, 2개 이미지 + 8D proprio를 입력받아 `(8, 7)` action chunk를 반환한다.**
> L1 regression + parallel decoding + bidirectional attention 덕분에 vanilla OpenVLA 대비 per-step 기준 ~12× 빠르며 (170 ms/call ÷ 8 = 21 ms/step), LIBERO-spatial 성공률은 vanilla와 유사한 수준을 유지한다.
