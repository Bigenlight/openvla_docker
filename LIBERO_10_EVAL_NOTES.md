# OpenVLA-OFT libero-10 평가 노트

## 1. 개요

OpenVLA-OFT(Optimized Fine-Tuning) 모델을 LIBERO-10 (long-horizon) 벤치마크에서 평가했다. libero-10 전용으로 파인튜닝된 공식 체크포인트를 사용하여, OFT 논문 수치와 우리 측정값을 비교하는 것이 목적이다.

## 2. 설정

- **체크포인트**: `moojink/openvla-7b-oft-finetuned-libero-10`
- **unnorm-key**: `libero_10`
- **평가 규모**: 5 tasks × 3 trials = 총 **15 episodes**
- **모델**: OpenVLA-7B + OFT (action chunking=8, parallel decoding, L1 regression, 2 images + 8D proprio)
- **실행 환경**: `bigenlight/libero-pro:latest` 컨테이너 (LIBERO-10 suite), 서버 `bigenlight/openvla-oft-http:latest` 포트 8700
- **하드웨어**: RTX A6000
- **실행일**: 2026-04-19

## 3. libero_10 특성

libero-10은 LIBERO 벤치마크 중 **long-horizon** 시나리오를 담당한다.

- **max_steps = 520** (libero_spatial의 220 대비 약 2.4배)
- 멀티스텝 조작 (multi-step manipulation) 필요
  - 예: 서랍 열기 → 물체 꺼내기 → 놓기 → 서랍 닫기
  - 예: 여러 물체를 순차적으로 재배치
- 단순 pick-and-place가 아니라 task 내에 여러 sub-goal이 연결됨
- 따라서 성공률은 모델의 **장기 계획 유지 능력**과 누적 오류 내성에 민감

## 4. 결과

### 성공률: **15/15 = 100%** 🎯

| Task ID | Task 설명 | 시도 | 성공 | 성공률 |
|---------|----------|------|------|--------|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 3 | 3 | 100% |
| 1 | put both the cream cheese box and the butter in the basket | 3 | 3 | 100% |
| 2 | turn on the stove and put the moka pot on it | 3 | 3 | 100% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 3 | 3 | 100% |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 3 | 3 | 100% |
| **전체** | — | **15** | **15** | **100%** |

### 지연 시간

- **평균 inference latency**: 173.1 ms/call (task별 168~179 ms 범위)
- **call 당 반환 action 수**: 8개 (chunk) → **effective per-step: ~21.6 ms**
- 에피소드당 평균 스텝 수: 148 (range 130~156)
- 에피소드당 평균 wall-clock: ~55초 (max_steps=520 상한의 ~28%에서 조기 완료)

## 5. 비교

### libero_spatial vs libero_10

| 항목 | libero_spatial | libero_10 |
|------|----------------|-----------|
| 평가 규모 | 10 tasks × 5 trials = 50 | 5 tasks × 3 trials = 15 |
| max_steps | 220 | 520 |
| 에피소드당 스텝 (평균) | ~128 | ~148 |
| 평균 latency | 169 ms | 173 ms |
| **성공률 (우리 측정)** | **96% (48/50)** | **100% (15/15)** |
| 성공률 (OFT 논문) | 97.1% (per-suite) | ~97% (per-suite) / 96.8% (combined) |

### 표본 크기 관점

libero_10은 15 episodes로 표본이 작으므로 진정한 성공률은 ~80%~100% 신뢰구간에 있을 수 있음 (Wilson score). 50~150 episodes로 확대하면 노이즈가 줄어듦. 그래도 **전부 성공한 것은 강한 양성 신호**.

### 논문 수치 재현성

- libero_spatial: 우리 96% vs 논문 97.1% → **1.1%p 차이** (재현 성공 수준)
- libero_10: 우리 100% vs 논문 ~97% → 표본 작아서 오히려 위로 편향됐을 가능성

## 6. 영상 위치

```
Libero-pro_benchmark/test_outputs/eval_openvla_oft/libero_10_20260419_142351/
├── summary.json
└── videos/   # 15개 mp4 (모두 *_success.mp4)
```

task별로 3개씩, 전부 success 케이스. 장기 horizon에서의 행동 자연스러움을 정성 분석하기 좋음.

## 7. 특기사항

### 순조롭게 진행된 부분
- libero_spatial에서 빌드/푸시한 **동일 컨테이너 이미지 (`bigenlight/openvla-oft-http:latest`)** 재사용 — 코드 변경 0줄
- 체크포인트 스왑만으로 다른 suite 평가 가능: `OPENVLA_OFT_HTTP_ARGS="--checkpoint moojink/openvla-7b-oft-finetuned-libero-10 --unnorm-key libero_10"`
- 체크포인트 다운로드 시 `HF_HUB_DISABLE_XET=1` 환경변수가 compose.yml에 기본 주입되어 있어 hang 없이 정상 다운로드 (spatial에서 겪은 삽질 덕분)
- dataset_statistics.json에 `libero_10` 키가 이미 들어있어 unnorm_key 수동 매핑 불필요

### 이슈 없음
- 5 tasks × 3 trials 전부 첫 시도에 완료
- latency 분산도 작음 (stdev < 5ms across tasks)

### 추가로 해볼만한 것
- 전체 10 tasks × 50 trials로 확장 시 참 성공률 측정 (약 6~7시간 소요 예상)
- 나머지 suite (libero_object, libero_goal) 동일 방식으로 평가
- 통합 체크포인트 (`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`)를 한 번에 올려놓고 4개 suite 연속 평가
