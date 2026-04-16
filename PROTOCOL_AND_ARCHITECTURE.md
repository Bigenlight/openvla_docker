# OpenVLA HTTP Server -- 통신 규약 및 코드 아키텍처

> OpenVLA (7B) LIBERO fine-tune 모델을 Docker 컨테이너 안에서 FastAPI HTTP 서버로 서빙하고,
> LIBERO 벤치마크 컨테이너에서 통일 VLA 프로토콜로 추론을 요청하는 전체 구조를 기술한다.
>
> 상위 프로토콜 명세: `../VLA_COMMUNICATION_PROTOCOL.md`

---

## 1. 통신 규약 요약 (Communication Protocol Summary)

OpenVLA HTTP 서버는 `VLA_COMMUNICATION_PROTOCOL.md`에 정의된 **3개 엔드포인트**를 구현한다.
모든 통신은 JSON over HTTP이며, 이미지는 base64 PNG로 인코딩된다.

### 1.1 `GET /health` -- 서버 상태 조회

벤치마크 클라이언트가 startup 시 polling하여 서버 준비 완료를 대기하는 용도.
서버의 action 반환 형식(sub-key 목록, step 수, 상대/절대)을 미리 알려준다.

**응답 스키마:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `status` | `str` | `"ok"` (모델 로딩 완료) 또는 `"loading"` |
| `model` | `str` | 모델 식별자 (예: `"openvla:openvla/openvla-7b-finetuned-libero-spatial"`) |
| `action_type` | `str` | `"relative"` -- OpenVLA LIBERO는 항상 relative delta |
| `action_keys` | `list[str]` | `["action.eef_pos", "action.eef_euler", "action.gripper"]` |
| `n_action_steps` | `int` | `1` -- OpenVLA는 single-step 예측 |

### 1.2 `POST /reset` -- 에피소드 초기화

히스토리 기반 모델용 엔드포인트. OpenVLA는 stateless이므로 아무 작업도 하지 않고
`{"status": "reset"}`을 반환한다. 벤치마크 측은 매 에피소드 시작 시 호출해야 한다.

### 1.3 `POST /act` -- 추론 요청

관측값을 받아 action을 예측하는 핵심 엔드포인트.

**요청 스키마 (JSON body):**

| 키 | 타입 | 필수 | 설명 |
|----|------|------|------|
| `observation.images.static` | `str` (base64 PNG) | O | 3인칭 카메라 이미지 (HWC uint8 RGB) |
| `observation.images.wrist` | `str` (base64 PNG) | X | 손목 카메라 (OpenVLA는 사용 안 함) |
| `observation.state.eef_pos` | `list[float]` `[3]` | X | EEF 위치 (OpenVLA는 사용 안 함) |
| `observation.state.eef_quat` | `list[float]` `[4]` | X | EEF 회전 quaternion (사용 안 함) |
| `observation.state.gripper_qpos` | `list[float]` `[2]` | X | Gripper joint pos (사용 안 함) |
| `observation.state.joint_pos` | `list[float]` `[7]` | X | Arm joint pos (사용 안 함) |
| `task` | `str` | O | Natural language instruction |

> **참고:** OpenVLA는 이미지와 언어 지시만 사용하며, proprioceptive state는 무시한다.
> 벤치마크 측은 모든 state를 보내지만 서버가 필요한 키만 꺼내 쓰는 것이 프로토콜의 핵심 설계.

**응답 스키마 (JSON body):**

| 키 | 타입 | shape | 설명 |
|----|------|-------|------|
| `action.eef_pos` | `list[list[float]]` | `[1, 3]` | EEF translation delta |
| `action.eef_euler` | `list[list[float]]` | `[1, 3]` | EEF rotation delta (Euler) |
| `action.gripper` | `list[list[float]]` | `[1, 1]` | Gripper 명령 (`-1`=open, `+1`=close) |
| `latency_ms` | `float` | scalar | 서버 측 추론 소요 시간 (ms) |

모든 action sub-key 값은 **항상 2D 리스트** `[N_steps, dim]` 형태이다.
OpenVLA는 `N_steps = 1`이므로 outer list의 길이는 항상 1이다.

---

## 2. 요청/응답 예시 (Request/Response Examples)

### 2.1 Health Check

```bash
curl http://localhost:8600/health
```

```json
{
  "status": "ok",
  "model": "openvla:openvla/openvla-7b-finetuned-libero-spatial",
  "action_type": "relative",
  "action_keys": ["action.eef_pos", "action.eef_euler", "action.gripper"],
  "n_action_steps": 1
}
```

### 2.2 Reset

```bash
curl -X POST http://localhost:8600/reset -H "Content-Type: application/json" -d '{}'
```

```json
{
  "status": "reset"
}
```

### 2.3 Action Prediction (`/act`)

```bash
curl -X POST http://localhost:8600/act \
  -H "Content-Type: application/json" \
  -d '{
    "observation.images.static": "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEA...< base64 PNG, ~250KB >...",
    "observation.state.eef_pos": [0.1215, -0.0134, 0.9512],
    "observation.state.eef_quat": [0.9998, 0.0012, 0.0183, -0.0024],
    "observation.state.gripper_qpos": [0.0396, -0.0396],
    "observation.state.joint_pos": [-0.1, 0.3, 0.0, -2.2, 0.0, 2.5, 0.8],
    "observation.state.joint_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "task": "pick up the red mug and place it on the shelf"
  }'
```

**응답:**
```json
{
  "action.eef_pos":   [[-0.00312, 0.01845, -0.00567]],
  "action.eef_euler": [[0.00123, -0.00456, 0.00078]],
  "action.gripper":   [[-1.0]],
  "latency_ms": 248.3
}
```

> 요청 payload 크기는 256x256 PNG 이미지 1장 기준 약 **257KB**이다.
> `observation.state.*` 키들은 OpenVLA 서버에서 무시되지만, 프로토콜 호환성을 위해 벤치마크가 일괄 전송한다.

---

## 3. 아키텍처 다이어그램 (Architecture Diagram)

```
┌─────────────────────────┐     HTTP (localhost:8600)     ┌──────────────────────────┐
│  LIBERO Benchmark       │ ◄──────────────────────────► │   OpenVLA Server          │
│  Container              │     /health, /reset, /act     │   Container               │
│  (bigenlight/           │                               │   (bigenlight/            │
│   libero-pro)           │                               │    openvla-http)          │
│                         │                               │                           │
│  vla_client.py          │     JSON over HTTP            │  serve_openvla_http.py    │
│  libero_vla_eval.py     │     base64 PNG images         │  OpenVLA 7B (BF16, GPU)  │
└─────────────────────────┘                               └──────────────────────────┘
        │                                                          │
        │  robosuite 1.4.0                                         │  HF transformers
        │  Python 3.8                                              │  PyTorch + TF (resize)
        │  CUDA 11.3                                               │  flash-attn / sdpa
        │                                                          │
        ▼                                                          ▼
  ┌───────────┐                                            ┌───────────────┐
  │  MuJoCo   │                                            │  NVIDIA GPU   │
  │  (EGL)    │                                            │  (~15GB VRAM) │
  └───────────┘                                            └───────────────┘

  network_mode: host  ─────────────  둘 다 host network  ─────────────
```

**핵심 설계:**
- 두 컨테이너 모두 `network_mode: host`로 실행하여 `localhost`로 직접 통신한다.
- 모델 교체 시 벤치마크 코드 수정 없이 `--vla-url` 파라미터만 변경하면 된다.
- VLAClient는 `/health` 응답의 `action_keys`, `n_action_steps`를 읽어 모델별 action 형식에 자동 적응한다.

---

## 4. 데이터 흐름 (Data Flow)

한 번의 `/act` 추론 호출에 대한 전체 데이터 흐름:

```
[LIBERO Benchmark Container]                        [OpenVLA Server Container]

 1. robosuite env.step()
    └─ obs["agentview_image"]
       (256x256, uint8, RGB)
                │
 2. _build_images():
    agentview_image → images["static"]
                │
 3. _build_states():
    robot0_eef_pos   → observation.state.eef_pos
    robot0_eef_quat  → observation.state.eef_quat
    robot0_gripper_qpos → observation.state.gripper_qpos
    robot0_joint_pos → observation.state.joint_pos
    robot0_joint_vel → observation.state.joint_vel
                │
 4. VLAClient.predict():
    encode_image() → base64 PNG
    POST /act (JSON ~257KB)
                │
                ▼
 ──────── HTTP over localhost:8600 ────────
                │
                ▼                             5. _prepare_libero_image():
                                                 base64 decode → numpy HWC
                                                 180° rotate (img[::-1, ::-1])
                                                 JPEG encode/decode roundtrip (TF)
                                                 lanczos3 resize → 224x224
                                              
                                              6. Build prompt:
                                                 "In: What action should the robot
                                                  take to {task}?\nOut:"
                                              
                                              7. OpenVLA forward pass:
                                                 BF16, flash-attn
                                                 generate() → 7 action tokens
                                                 token→bin_center→unnormalize
                                                 → 7D float32 action
                                              
                                              8. Post-process:
                                                 normalize_gripper_action(binarize=True)
                                                   [0,1] → [-1,+1] → sign()
                                                 invert_gripper_action()
                                                   ×(-1) → LIBERO convention
                                              
                                              9. Split into sub-keys:
                                                 action[0:3] → action.eef_pos   [1,3]
                                                 action[3:6] → action.eef_euler  [1,3]
                                                 action[6:7] → action.gripper    [1,1]
                │
                ▼
 ──────── HTTP Response (JSON) ────────────
                │
                ▼
10. VLAClient: sub-key dict 반환
    _assemble_action_from_subkeys():
      concat [pos(3), euler(3), grip(1)] → 7D
    env.step(action.tolist())
```

### 단계별 상세 설명

| 단계 | 위치 | 처리 내용 |
|------|------|-----------|
| 1 | 벤치마크 | robosuite `OffScreenRenderEnv`에서 `agentview_image` (256x256 RGB uint8) 획득 |
| 2 | `_build_images()` | robosuite key `agentview_image` → 통일 카메라명 `static`으로 매핑 |
| 3 | `_build_states()` | `robot0_eef_pos`, `robot0_eef_quat` 등 → `observation.state.*` 통일 키로 매핑 |
| 4 | `VLAClient.predict()` | numpy → PIL → PNG → base64 인코딩 후 JSON payload 조립, `POST /act` |
| 5 | `_prepare_libero_image()` | 180도 회전 (LIBERO 학습 전처리 재현) + TF JPEG roundtrip + lanczos3 resize 224x224 |
| 6 | `/act` endpoint | `"In: What action should the robot take to {task}?\nOut:"` 형태의 VLA prompt 구성 |
| 7 | `get_vla_action()` → `predict_action()` | BF16 forward pass, autoregressive로 7개 action token 생성, bin center 역매핑 + unnormalize |
| 8 | `normalize_gripper_action()` + `invert_gripper_action()` | gripper [0,1] → [-1,+1] binarize → 부호 반전 (LIBERO: -1=open, +1=close) |
| 9 | `/act` endpoint | 7D action을 3개 sub-key로 분할, 2D list wrapping |
| 10 | 벤치마크 | sub-key dict → `_assemble_action_from_subkeys()` → 7D concat → `env.step()` |

---

## 5. 주요 코드 경로 (Key Code Paths)

### 5.1 서버 측 (Server Side)

#### `scripts/serve_openvla_http.py` -- FastAPI 서버 메인

서버의 진입점. 3개 엔드포인트를 구현하며, 모델 로딩부터 추론, 응답 분할까지 전 과정을 담당한다.

| 함수/엔드포인트 | 역할 |
|-----------------|------|
| `_prepare_libero_image(b64_str)` | base64 PNG → 180도 회전 → JPEG roundtrip → lanczos3 resize 224x224. LIBERO 학습 시 전처리를 정확히 재현 |
| `_libero_resize_224(img)` | `tf.image.resize(method="lanczos3")` 호출. Octo dataloader와 동일한 resize 로직 |
| `_b64_to_numpy(b64_str)` | base64 디코딩 → PIL → numpy HWC uint8 RGB |
| `_load_model(checkpoint, ...)` | HF `AutoModelForVision2Seq.from_pretrained()` + `AutoProcessor`. `flash_attention_2`/`sdpa`/`eager` 선택 가능 |
| `GET /health` | 서버 메타데이터 반환. `is_dummy` 모드 지원 |
| `POST /reset` | stateless이므로 no-op |
| `POST /act` | observation 파싱 → `_prepare_libero_image()` → `get_vla_action()` → gripper 후처리 → sub-key 분할 반환 |

```
serve_openvla_http.py
├── /health  → 서버 상태 + action 형식 메타
├── /reset   → no-op (stateless)
└── /act     → _prepare_libero_image()
              → get_vla_action()           ← experiments/robot/openvla_utils.py
              → normalize_gripper_action() ← experiments/robot/robot_utils.py
              → invert_gripper_action()    ← experiments/robot/robot_utils.py
              → sub-key split [pos, euler, gripper]
```

#### `experiments/robot/openvla_utils.py` -- `get_vla_action()`

모델 추론의 핵심 함수. 이미지를 PIL Image로 변환하고, `center_crop` 옵션이 켜져 있으면
`crop_scale=0.9`로 center crop 후 원래 크기로 resize한다.
VLA prompt를 구성한 뒤 `model.predict_action()`을 호출하여 7D continuous action을 반환한다.

```python
def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    image = Image.fromarray(obs["full_image"]).convert("RGB")
    # optional center crop (sqrt(0.9) scale)
    prompt = f"In: What action should the robot take to {task_label}?\nOut:"
    inputs = processor(prompt, image).to(vla.device, dtype=torch.bfloat16)
    action = vla.predict_action(inputs, unnorm_key=unnorm_key)
    return action  # shape (7,)
```

#### `prismatic/extern/hf/modeling_prismatic.py` -- `predict_action()`

OpenVLA의 토큰 디코딩 → continuous action 변환 핵심. Autoregressive로 `action_dim`개의
토큰을 생성한 뒤, vocabulary offset으로 discretized action bin을 역산하고,
`norm_stats`의 q01/q99 통계로 unnormalize한다.

```python
def predict_action(self, input_ids, unnorm_key=None, **kwargs):
    generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key))
    predicted_action_token_ids = generated_ids[0, -action_dim:]
    discretized_actions = self.vocab_size - predicted_action_token_ids
    normalized_actions = self.bin_centers[clip(discretized_actions - 1)]
    actions = 0.5 * (normalized_actions + 1) * (q99 - q01) + q01  # unnormalize
    return actions  # shape (7,)
```

#### `experiments/robot/robot_utils.py` -- Gripper 후처리

| 함수 | 변환 | 설명 |
|------|------|------|
| `normalize_gripper_action(action, binarize=True)` | `[0,1] → [-1,+1]`, `binarize` 시 `sign()` 적용 | OpenVLA 출력의 gripper가 [0,1] 범위인데, 환경은 {-1,+1} 이산값을 기대 |
| `invert_gripper_action(action)` | `×(-1)` | RLDS 데이터 로더가 0=close, 1=open으로 정렬하는데, LIBERO는 -1=open, +1=close이므로 부호 반전 |

### 5.2 클라이언트 측 (Client Side)

#### `Libero-pro_benchmark/scripts/vla_client.py` -- `VLAClient` class

모든 벤치마크가 공유하는 통일 HTTP 클라이언트. stdlib + numpy + Pillow + requests만 사용하여
LIBERO 컨테이너(Python 3.8)에서 추가 의존성 없이 동작한다.

| 메서드 | 역할 |
|--------|------|
| `encode_image(img)` | `np.ndarray` (HWC uint8) → PIL PNG → base64 string |
| `health_check()` | `GET /health`, 실패 시 `None` 반환 |
| `wait_until_ready(max_wait=180)` | 3초 간격 polling, `status == "ok"`까지 대기 |
| `reset()` | `POST /reset` 호출 |
| `predict(images, states, instruction)` | 이미지 인코딩 + state 패킹 → `POST /act` → sub-key dict 또는 flat ndarray 반환 |

`predict()`의 반환값 처리 로직:
- 응답에 `action.*` 키가 하나라도 있으면 → **sub-key dict** `{str: ndarray[N,D]}` 반환
- 없으면 → `action` 키의 값을 flat 2D ndarray로 반환 (레거시 호환)
- 1D 응답은 자동으로 `np.newaxis` 추가하여 2D로 승격

#### `Libero-pro_benchmark/scripts/libero_vla_eval.py` -- Eval Loop

LIBERO suite 평가 드라이버. 모델에 대한 지식 없이 VLAClient와 통일 프로토콜만으로 동작한다.

| 함수 | 역할 |
|------|------|
| `_build_images(obs)` | robosuite `agentview_image` → `images["static"]`, `robot0_eye_in_hand_image` → `images["wrist"]` |
| `_build_states(obs)` | `robot0_eef_pos` → `observation.state.eef_pos`, `robot0_eef_quat` → `observation.state.eef_quat` 등 6개 키 매핑 |
| `_assemble_action_from_subkeys(dict)` | `action.eef_pos[N,3]` + `action.eef_euler[N,3]` + `action.gripper[N,1]` → concat `[N,7]` |
| `_flatten_action(action_or_dict)` | sub-key dict이면 `_assemble_action_from_subkeys()`, flat이면 그대로 2D 변환 |

**Action chunk 처리:**
서버가 `n_action_steps > 1`인 chunk를 반환하면 `action_plan` deque에 저장하고,
`replan_steps` (기본값 5)만큼만 사용한 뒤 새로 예측을 요청한다.
OpenVLA는 `n_action_steps = 1`이므로 매 step 재예측한다.

---

## 6. OpenVLA vs Pi0.5 비교 (Comparison)

동일한 통일 프로토콜을 구현하는 두 서버의 주요 차이점:

| 항목 | OpenVLA | Pi0.5 |
|------|---------|-------|
| **포트** | 8600 | 8400 |
| **action_type** | `relative` | `relative` |
| **action_keys** | `eef_pos`, `eef_euler`, `gripper` | `eef_pos`, `eef_euler`, `gripper` |
| **n_action_steps** | 1 (single-step) | 10 (chunked) |
| **이미지 전처리** | 180 deg rot + JPEG roundtrip + lanczos3 resize 224 | 180 deg rot + pad-resize 224 |
| **프레임워크** | PyTorch + HF transformers (+ TF for resize) | JAX + openpi |
| **모델 크기** | 7B (~15GB VRAM, BF16) | ~3B |
| **Inference 속도** | ~260ms / step | ~30ms / step |
| **Gripper 후처리** | `normalize_gripper_action(binarize=True)` + `invert_gripper_action()` | 직접 출력 (후처리 없음) |
| **Proprio input** | 사용 안 함 (이미지 + 언어만) | 사용함 (state 8D: eef_pos + axis_angle + gripper_qpos) |
| **Attention** | `flash_attention_2` / `sdpa` / `eager` 선택 | JAX default |
| **State 조립** | 불필요 | `_assemble_libero_state()`: eef_pos(3) + quat→axis_angle(3) + gripper_qpos(2) = 8D |
| **카메라** | `static` 1개만 사용 | `static` + `wrist` 2개 |
| **Center crop** | 선택적 (sqrt(0.9) scale, `--no-center-crop`) | 없음 |
| **History** | Stateless (매 step 독립) | Stateless (pi0.5 기준) |

### 코드 구조 비교

```
OpenVLA Server (serve_openvla_http.py)          Pi0.5 Server (serve_pi05_http.py)
──────────────────────────────────               ────────────────────────────────
_prepare_libero_image()                          _prepare_image()
  └─ 180° rot → JPEG roundtrip → lanczos3         └─ 180° rot → pad_resize (openpi_client)
                                                 _assemble_libero_state()
                                                   └─ eef_pos + quat→axisangle + gripper_qpos
get_vla_action() → predict_action()              policy.infer(element)
  └─ autoregressive token generation               └─ diffusion-based action generation
normalize_gripper + invert_gripper               (직접 출력)
split [0:3], [3:6], [6:7]                       split [:, 0:3], [:, 3:6], [:, 6:7]
```

---

## 7. HTTP 지연 측정 결과 (Latency Benchmark)

통일 프로토콜의 HTTP 오버헤드가 모델 추론 대비 무시할 수 있는 수준인지 검증한 결과:

### 7.1 HTTP 오버헤드

| 측정 항목 | 값 |
|-----------|-----|
| 총 추론 시간 (OpenVLA) | 248ms |
| HTTP 전송 오버헤드 | 2.9ms |
| 오버헤드 비율 | **1.2%** |

> HTTP 오버헤드는 전체 추론 시간의 ~1%로, 실질적으로 무시 가능하다.
> `network_mode: host` 사용으로 Docker 네트워크 오버헤드도 최소화되어 있다.

### 7.2 카메라 수 스케일링

이미지 수 증가에 따른 base64 인코딩 + 전송 오버헤드:

| 카메라 수 | HTTP 전송 시간 |
|-----------|---------------|
| 1 cam | 2.8ms |
| 2 cam | 4.1ms |
| 4 cam | 6.5ms |

카메라 수에 대해 거의 선형 스케일링. 4개 카메라에서도 6.5ms로 충분히 낮다.

### 7.3 해상도 스케일링

이미지 해상도 증가에 따른 오버헤드 (PNG 인코딩 크기에 비례):

| 해상도 | HTTP 전송 시간 |
|--------|---------------|
| 256x256 | 2.7ms |
| 512x512 | 7.0ms |
| 1024x1024 | 22.2ms |

> 1024x1024에서도 22ms로, GPU 추론 시간(~248ms) 대비 약 9%에 불과하다.
> LIBERO 기본 해상도인 256x256에서는 사실상 무시할 수 있는 수준이다.

### 7.4 동시 요청 처리

| 동시 요청 수 | 처리량 |
|-------------|--------|
| 1 (sequential) | ~4 rps |
| 2 (concurrent) | ~4 rps |
| 4 (concurrent) | ~4 rps |

> **GPU가 병목**이다. 동시 요청을 늘려도 처리량은 ~4 rps로 동일하며,
> HTTP 서버 자체가 병목이 되는 구간은 관찰되지 않았다.
> 즉, HTTP 프로토콜 도입으로 인한 성능 손실은 실질적으로 없다.

---

## 부록 A. 서버 실행 명령어

### 실제 모델 로딩

```bash
python scripts/serve_openvla_http.py \
    --port 8600 \
    --checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --unnorm-key libero_spatial \
    --attn-impl flash_attention_2
```

### Dummy 모드 (프로토콜 스모크 테스트)

```bash
python scripts/serve_openvla_http.py --port 8600 --dummy
```

### 벤치마크 실행

```bash
python scripts/libero_vla_eval.py \
    --vla-url http://localhost:8600 \
    --suite libero_spatial \
    --num-tasks 10 \
    --num-trials 20 \
    --resolution 256
```

## 부록 B. CLI 옵션 전체 목록

### `serve_openvla_http.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--host` | `0.0.0.0` | 바인드 주소 |
| `--port` | `8600` | 서버 포트 |
| `--checkpoint` | `openvla/openvla-7b-finetuned-libero-spatial` | HF Hub repo id 또는 로컬 경로 |
| `--unnorm-key` | `libero_spatial` | Action unnormalization 통계 키. `_no_noops` 변형 자동 fallback |
| `--attn-impl` | `flash_attention_2` | Attention 구현체: `flash_attention_2`, `sdpa`, `eager` |
| `--no-center-crop` | (미지정 = crop 활성) | Center crop 비활성화 |
| `--load-in-8bit` | (미지정) | 8-bit 양자화 로딩 |
| `--load-in-4bit` | (미지정) | 4-bit 양자화 로딩 |
| `--dummy` | (미지정) | 모델 미로딩, zero action 반환 |

### `libero_vla_eval.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--vla-url` | `http://localhost:8400` (또는 `VLA_SERVER_URL` 환경변수) | VLA 서버 URL |
| `--suite` | `libero_spatial` | LIBERO suite 이름 |
| `--num-tasks` | `2` | 평가할 task 수 (0 = 전체) |
| `--num-trials` | `1` | Task당 trial 수 |
| `--num-steps-wait` | `10` | 에피소드 시작 시 no-op step 수 |
| `--resolution` | `256` | 렌더링 해상도 |
| `--replan-steps` | `5` | Action chunk에서 사용할 최대 step 수 |
| `--output-dir` | `/workspace/LIBERO-PRO/test_outputs/eval` | 결과 저장 경로 |
| `--seed` | `7` | Random seed |
| `--max-steps` | (suite별 기본값) | 에피소드 최대 step 수 |
| `--no-video` | (미지정) | 비디오 저장 비활성화 |
| `--task-ids` | (미지정) | 쉼표 구분 task id 목록 (`--num-tasks` 무시) |
| `--shard-tag` | (미지정) | 병렬 실행 시 출력 디렉토리 구분 태그 |

## 부록 C. 파일 레퍼런스

| 파일 경로 | 역할 |
|-----------|------|
| `openvla/scripts/serve_openvla_http.py` | OpenVLA FastAPI 서버 메인 |
| `openvla/experiments/robot/openvla_utils.py` | `get_vla_action()` -- 모델 추론 핵심 |
| `openvla/experiments/robot/robot_utils.py` | `normalize_gripper_action()`, `invert_gripper_action()` |
| `openvla/prismatic/extern/hf/modeling_prismatic.py` | `predict_action()` -- 토큰 디코딩 → continuous action |
| `Libero-pro_benchmark/scripts/vla_client.py` | `VLAClient` -- 통일 HTTP 클라이언트 |
| `Libero-pro_benchmark/scripts/libero_vla_eval.py` | LIBERO 평가 루프 |
| `VLA_COMMUNICATION_PROTOCOL.md` | 상위 통일 프로토콜 명세 |
| `openpi/scripts/serve_pi05_http.py` | Pi0.5 서버 (비교 참조용) |
