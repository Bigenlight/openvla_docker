# OpenVLA-OFT libero-10 Perturbation (OOD) 평가 노트

## 1. 개요

OpenVLA-OFT libero-10 fine-tuned 체크포인트의 **OOD(Out-of-Distribution) 강건성**을 LIBERO-PRO perturbation suite 4종에 대해 측정한다. 기준은 앞서 측정한 baseline libero_10 **100% (15/15)**.

## 2. 설정

- **체크포인트**: `moojink/openvla-7b-oft-finetuned-libero-10` (변경 없음)
- **서버**: `bigenlight/openvla-oft-http:latest` 포트 8700
- **평가 규모**: 각 perturbation suite 당 **2 tasks × 3 trials = 6 episodes**, 총 **24 episodes**
- **대상 task**: task 0 (alphabet soup + tomato sauce → basket), task 1 (cream cheese + butter → basket)
- **실행일**: 2026-04-19
- **코드 변경**: 없음 (체크포인트 스왑 없이 suite 이름만 변경)

## 3. Perturbation 종류

LIBERO-PRO는 BDDL + init 파일을 **사전 생성**하여 `ood_data/bddl_files/libero_10_<type>/`에 저장. 런타임에 perturbator가 재적용되지 않으므로 결정적.

| Suite | Perturbator | 변경 내용 |
|-------|-------------|-----------|
| `libero_10_swap` | `SwapPerturbator` | 물체 **초기 위치 교환** (의미 역할 유지, 공간만 바꿈) |
| `libero_10_object` | `ObjectReplacePerturbator` | 물체 **identity 교체** (같은 semantic role, 다른 visual) |
| `libero_10_lan` | `LanguagePerturbator` | 명령어 **paraphrase** ("put X in Y" → "place X into Y") |
| `libero_10_task` | `TaskPerturbator` | task **goal 자체를 변경** (다른 task로 대체) |

## 4. 결과 요약

| Suite | 성공 | 성공률 | baseline(100%) 대비 |
|-------|------|--------|---------------------|
| **baseline** (비OOD) | 15/15 | **100%** | — |
| `libero_10_lan` | 6/6 | **100%** | ±0%p |
| `libero_10_object` | 3/6 | **50%** | -50%p |
| `libero_10_swap` | 0/6 | **0%** | -100%p |
| `libero_10_task` | 0/6 | **0%** | -100%p |

### 관측된 강건성 순서
**언어 >> 물체 identity >> (공간 위치 ≈ task 변경)**

## 5. 상세 결과

### 5.1 libero_10_lan — 6/6 (100%) ✅
언어 paraphrase에 완전히 적응. 원본 대비 명령어 변화:
- 원본: "put both the alphabet soup and the tomato sauce in the basket"
- OOD:  "place alphabet soup and tomato sauce into basket"

두 task 모두 3/3 성공. 평균 262~276 steps로 baseline보다 약간 더 걸렸지만 모두 성공.

**해석**: LLM backbone의 언어 일반화 능력이 task instruction의 어휘/구문 변화에 강건함을 시사. VLA가 단어 수준이 아니라 semantic 수준에서 task를 이해한다는 증거.

### 5.2 libero_10_object — 3/6 (50%) ⚠️
물체 visual identity 교체.

| Task | 성공 |
|------|------|
| 0: alphabet soup + tomato sauce → basket | **0/3** |
| 1: cream cheese + butter → basket | **3/3** |

**해석**: task 1은 visual 변경에도 "두 박스를 바구니에 넣는다"는 공간 패턴이 안정적이라 성공. Task 0은 caneed 형태가 교체되면서 모델이 혼란한 것으로 추정.

### 5.3 libero_10_swap — 0/6 (0%) ❌
물체 초기 위치 교환. **전 trial 520 steps에서 timeout**.

**해석**: OpenVLA-OFT는 학습 시 본 정확한 공간 배치에 강하게 overfit되어 있음. 위치가 바뀌면 어느 물체를 먼저 잡아야 할지, 어디로 가야 할지 전혀 판단 못 함. **시각적 geometry 변화가 가장 취약점**.

### 5.4 libero_10_task — 0/6 (0%) ❌
Task goal 자체가 변경되는 OOD.

| Task | 명령 | 성공 |
|------|------|------|
| 0 | "put both the **cream cheese** and the tomato sauce in the basket" | 0/3 |
| 1 | "put both the **alphabet soup** and the butter in the basket" | 0/3 |

**해석**: task 목표가 섞인(swapped) 조합 — 모델은 원본 task 중 하나와 매칭하려 하지만 시각/명령 조합이 학습 분포 밖이라 실패. 이것도 전부 timeout.

## 6. 비교 & 시사점

### OFT 강건성 프로파일
이번 실험으로 OpenVLA-OFT의 **일반화 경계**가 드러났다:

| 변화 축 | 강건도 |
|---------|--------|
| 언어 표현 (문구 paraphrase) | 💪 매우 강건 (100%) |
| 물체 visual identity | 🤏 중간 (50%, task 의존적) |
| 공간 배치 (초기 위치) | 🚫 극도로 취약 (0%) |
| Task goal 재조합 | 🚫 극도로 취약 (0%) |

### VLA 연구 관점
- **LLM 능력 vs Visual grounding**: 언어 계층은 일반화 잘 됨, 시각-행동 매핑은 학습 분포에 고정됨
- LIBERO 학습 데이터는 초기 상태 분포가 좁아서, 초기 공간 배치가 바뀌면 무너짐
- OFT의 "parallel decoding + L1 regression"은 속도/정확도는 개선했지만 **공간 OOD에 대한 강건성은 별도 문제**

## 7. 영상 위치

```
Libero-pro_benchmark/test_outputs/eval_openvla_oft_perturbation/
├── libero_10_swap_20260419_151529/
│   ├── summary.json
│   └── videos/                     # 6개 mp4 (전부 *_failure.mp4)
├── libero_10_object_20260419_152529/
│   ├── summary.json
│   └── videos/                     # 6개 mp4 (task0 실패 3 + task1 성공 3)
├── libero_10_lan_20260419_153304/
│   ├── summary.json
│   └── videos/                     # 6개 mp4 (전부 *_success.mp4)
└── libero_10_task_20260419_153844/
    ├── summary.json
    └── videos/                     # 6개 mp4 (전부 *_failure.mp4)
```

총 24개 영상, 4개 하위 폴더로 perturbation 종류별 분리.

## 8. 특기사항

- **코드 변경 없음**: libero-pro benchmark는 OOD suite가 BDDL 파일로 pre-generate되어 있어 suite name만 바꾸면 됨 (`./run.sh --vla-eval libero_10_swap`).
- **Suite 자동 등록**: `ood_data/`가 run.sh startup 때 컨테이너에 복사되어 `get_benchmark_dict()`가 자동 발견.
- **환경 perturbation (`libero_10_env`)은 없음**: 상류에서 BDDL 미공개 (LIBERO-PRO Issue #9).
- **표본 크기 작음**: 각 suite 6 episodes는 통계적으로 노이즈 있음. 50 trials로 확대하면 중간 구간(50%) 수치가 더 정밀해질 것.
- **timeout 패턴 일관성**: swap/task 실패는 전부 max_steps=520에서 time-out (early fail 없음). 모델이 "아무것도 안 하는 것"보다는 "계속 시도하지만 달성 못 함"에 가까움.
