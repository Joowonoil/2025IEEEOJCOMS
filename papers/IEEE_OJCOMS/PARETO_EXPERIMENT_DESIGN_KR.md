# Pareto Frontier 실험 설계
## PEFT 방법론의 파라미터 효율성 vs 성능 분석

**날짜**: 2025-11-02
**상태**: 계획됨 - 실행 준비 완료
**목적**: 채널 추정을 위한 PEFT 방법론의 과학적 비교

---

## 1. 개요

### 1.1 동기

**발견된 문제**:
초기 실험 설계에서는 파라미터 수가 크게 다른 PEFT 방법들을 비교했습니다:
- Adapter: 524,288 파라미터 (27.87%)
- LoRA: 98,304 파라미터 (7.18%)
- Hybrid: 61,952 파라미터 (4.53%)
- Prompt: 12,800 파라미터 (0.94%)

**문제점**: 파라미터 예산이 40배 차이날 때 성능을 공정하게 비교할 수 없습니다.
- Adapter > LoRA라면, 방법이 더 좋은 것인가 아니면 단순히 5배 많은 파라미터를 가져서인가?
- Prompt < LoRA라면, 본질적으로 나쁜 것인가 아니면 더 파라미터 효율적인 것인가?

**해결책**: **Pareto Frontier 분석** - 각 방법을 여러 파라미터 규모에서 테스트하여 파라미터-성능 곡선을 생성합니다.

---

## 2. 실험 설계

### 2.1 Pareto Frontier 접근법

단일 지점 비교가 아닌, 각 방법에 대한 **파라미터-성능 곡선**을 생성합니다:

```
성능 (NMSE)
      ↑
      │        Full Fine-tuning (1.35M 파라미터)
      │       /
      │      /  Adapter 곡선
      │     /  /
      │    /  /  LoRA 곡선
      │   /  /  /
      │  /  /  /  Hybrid 곡선
      │ /  /  /  /
      │/  /  /  /  Prompt 곡선
      └──────────────────────→ 학습 가능한 파라미터
```

**핵심 통찰**: **파라미터당 가장 높은 성능**을 달성하는 방법이 승리합니다.

### 2.2 구성 매트릭스

| 방법 | 구성 이름 | 하이퍼파라미터 | 학습 가능 파라미터 | 전체 모델 대비 % |
|--------|-------------|----------------|------------------|-----------------|
| **Adapter** | dim8 | bottleneck=8 | ~65,536 | 4.83% |
| | dim16 | bottleneck=16 | ~131,072 | 9.66% |
| | dim32 | bottleneck=32 | ~262,144 | 19.33% |
| | dim64 | bottleneck=64 | ~524,288 | 38.66% |
| **LoRA** | r4 | rank=4 | ~19,660 | 1.45% |
| | r8 | rank=8 | ~39,320 | 2.90% |
| | r16 | rank=16 | ~78,640 | 5.80% |
| | r20 | rank=20 | ~98,304 | 7.25% |
| **Prompt** | len50 | length=50 | 6,400 | 0.47% |
| | len100 | length=100 | 12,800 | 0.94% |
| | len200 | length=200 | 25,600 | 1.89% |
| **Hybrid (Base)** | p50_r5 | (prompt=50, rank=5) | ~30,230 | 2.23% |
| | p100_r10 | (prompt=100, rank=10) | ~61,952 | 4.57% |
| | p200_r20 | (prompt=200, rank=20) | ~123,904 | 9.14% |
| **Hybrid (Extra)** | p25_r2 | (prompt=25, rank=2) | ~12,752 | 0.94% |
| | p75_r8 | (prompt=75, rank=8) | ~48,896 | 3.61% |
| | p100_r5 | (prompt=100, rank=5) | ~36,630 | 2.70% |
| | p50_r10 | (prompt=50, rank=10) | ~55,552 | 4.10% |
| | p150_r15 | (prompt=150, rank=15) | ~92,672 | 6.83% |
| | p300_r25 | (prompt=300, rank=25) | ~161,280 | 11.89% |

**총 구성 개수**: 4 + 4 + 3 + (3+6) = **20개 구성**

**참고**: Hybrid는 더 풍부한 Pareto frontier를 만들기 위해 3개에서 9개 구성으로 확장:
- **Option 1** (중간 밀도): p75_r8, p150_r15 - 곡선 빈틈 채우기
- **Option 2** (극단값): p25_r2, p300_r25 - 경계 탐색
- **Option 3** (비대칭): p50_r10, p100_r5 - Prompt vs LoRA 기여도 분석

### 2.3 시나리오

모든 방법을 **5개 시나리오**에서 테스트:
1. **InH** (Indoor Hotspot): 실내 밀집, LoS/NLoS, 5-100m
2. **InF** (Indoor Factory): 실내 공장, LoS/NLoS, 10-100m
3. **UMi** (Urban Micro): 소형 셀 도시, LoS/NLoS, 10-500m
4. **UMa** (Urban Macro): 대형 셀 도시, LoS/NLoS, 10-10000m
5. **RMa** (Rural Macro): 확장 범위 농촌, LoS/NLoS, 10-10000m

### 2.4 총 실험 부하

```
20개 구성 × 5개 시나리오 = 100 runs

세부 내역:
- Adapter:       4개 구성 × 5개 시나리오 = 20 runs (~40시간)
- LoRA:          4개 구성 × 5개 시나리오 = 20 runs (~40시간)
- Prompt:        3개 구성 × 5개 시나리오 = 15 runs (~30시간)
- Hybrid (Base): 3개 구성 × 5개 시나리오 = 15 runs (~30시간)
- Hybrid (Extra):6개 구성 × 5개 시나리오 = 30 runs (~60시간)

총 예상 시간: 200-400시간 (8-16일) 단일 GPU 사용 시
양방향 병렬 실행 (8개 인스턴스) 시: 20-40시간!
```

### 2.5 양방향 병렬 실행 전략

**과제**: 순차 실행 시 8-16일 소요

**해결책**: 양 끝에서 동시에 실험을 실행하는 8개 Vast.ai 인스턴스 배포:

**Set A (정방향 - 4개 인스턴스):**
```
Instance 1: Adapter    (InH-dim8 → InH-dim16 → ... → RMa-dim64)
Instance 2: LoRA       (InH-r4 → InH-r8 → ... → RMa-r20)
Instance 3: Prompt     (InH-len50 → ... → RMa-len200)
Instance 4: Hybrid     (InH-p50_r5 → ... → RMa-p200_r20)
```

**Set B (역방향 - 4개 인스턴스):**
```
Instance 5: Adapter    (RMa-dim64 → ... → InH-dim8)
Instance 6: LoRA       (RMa-r20 → ... → InH-r4)
Instance 7: Prompt     (RMa-len200 → ... → InH-len50)
Instance 8: Hybrid     (RMa-p200_r20 → ... → InH-p50_r5)
```

**Extra Hybrid (추가 2개 인스턴스):**
```
Instance 9:  Hybrid Extra 정방향  (InH-p75_r8 → ... → RMa-p300_r25)
Instance 10: Hybrid Extra 역방향  (RMa-p100_r5 → ... → InH-p25_r2)
```

**결과**: 세트가 중간에서 만날 때 모든 실험 완료 (~20-40시간)

---

## 3. 주요 설계 결정

### 3.1 고정 예산이 아닌 Pareto를 선택한 이유

**고려된 대안**: 모든 방법을 ~100K 파라미터로 맞춤
- Adapter-12, LoRA-20, Hybrid-조정됨, Prompt-781

**거부된 이유**:
1. 일부 방법(Prompt)은 **매우 적은 파라미터**로 작동하도록 설계됨 - 100K를 강제하는 것은 현실적이지 않을 수 있음
2. **파라미터 효율성 스토리**를 놓침 - Prompt가 13K로 95% 성능 달성 vs LoRA가 98K 필요
3. **스케일링 동작**을 보여주지 못함 - Adapter가 더 많은 파라미터로 이득을 보는가? LoRA가 포화되는가?

**Pareto 장점**:
- 파라미터-성능 트레이드오프의 **전체 그림** 표시
- 각 방법의 **최적 작동점** 식별 가능
- **다중 통찰** 추출 가능:
  - 낮은 예산에서 어떤 방법이 가장 파라미터 효율적인가?
  - 어떤 방법이 가장 빨리 포화되는가?
  - 어떤 방법이 추가 파라미터로부터 가장 많은 이득을 얻는가?

### 3.2 Hybrid 구성: 대각선 vs 그리드

**현재 설계 (대각선)**:
```
(Prompt, LoRA): (50,5), (100,10), (200,20)
→ 균형 잡힌 스케일링: 둘 다 함께 증가
```

**대각선으로 시작한 이유**:
1. **효율성**: 전체 그리드의 9개 대신 3개 구성
2. **명확한 해석**: "균형 잡힌 하이브리드" 스케일링 테스트
3. **베이스라인 확립**: 비율 탐색 전에 하이브리드가 작동하는지 증명 필요

**향후 확장 (전체 그리드)**:
```
        LoRA rank
         5     10    20
Prompt
50      ×     ×     ×     ← LoRA 지배적 (가중치 적응)
100     ×     ×     ×     ← 균형
200     ×     ×     ×     ← Prompt 지배적 (입력 적응)
```

**그리드에 대한 연구 질문**:
- 입력 레벨(Prompt) 또는 가중치 레벨(LoRA) 적응이 더 중요한가?
- 주어진 파라미터 예산에 대한 최적 할당 비율은?
- 두 적응이 곱셈 효과를 가지는가 아니면 덧셈 효과를 가지는가?

**결정**: 대각선으로 시작, 다음 경우 그리드로 확장:
1. Hybrid가 단일 방법에 비해 명확한 이점을 보임
2. 리뷰어가 더 철저한 분석 요청
3. 더 깊은 조사가 필요한 흥미로운 동작 관찰

### 3.3 학습 구성

모든 실험은 **동일한 학습 설정** 사용:
- **반복 횟수**: 100,000 (100K)
- **체크포인트**: 20,000마다 (20K, 40K, 60K, 80K, 100K)
- **배치 크기**: 32
- **학습률**: 1e-4, cosine annealing + warmup
- **그래디언트 클리핑**: 최대 norm 1.0
- **베이스 모델**: `Large_estimator_v4_base_final` (1M 반복)

**근거**:
- 100K 반복: 수렴과 시간 간의 균형
- 20K 체크포인트: 부드러운 플롯을 위한 곡선당 5개 점
- 동일한 LR: 하이퍼파라미터 튜닝을 교란 요인으로 제거

---

## 4. 예상 결과 및 분석

### 4.1 가설

**H1 (Prompt 효율성)**:
Prompt Learning은 다른 방법보다 **10-50배 적은 파라미터**로 경쟁력 있는 NMSE를 달성할 것입니다.

**H2 (LoRA 효과성)**:
LoRA는 중간 범위(50-100K 파라미터)에서 **파라미터당 최고 성능**을 제공할 것입니다.

**H3 (Adapter 포화)**:
Adapter는 dim=32-64에서 포화되어 ~260K 파라미터 이후 수익 체감을 보일 것입니다.

**H4 (Hybrid 시너지)**:
Hybrid는 파라미터 범위에서 **Pareto frontier를 지배**하여, 동일한 파라미터 수에서 Prompt-only 및 LoRA-only를 능가할 것입니다.

**H5 (스케일링 법칙)**:
각 방법은 **로그-선형 스케일링 법칙**을 따를 것입니다: NMSE ∝ log(params), 하지만 다른 기울기로.

### 4.2 주요 메트릭

각 (방법, 구성, 시나리오) 튜플에 대해:

**주요 메트릭**:
- **최종 NMSE** (100K 반복에서)

**보조 메트릭**:
- **수렴 속도** (최종 성능의 90%까지 반복 횟수)
- **파라미터 효율성** (NMSE / trainable_params)
- **상대 성능** (전체 미세 조정 성능의 %)

**시각화**:
1. **Pareto 곡선**: 각 방법에 대한 NMSE vs 파라미터 (5개 곡선, 시나리오당 1개)
2. **효율성 히트맵**: 방법 및 시나리오 전반의 성능/파라미터
3. **수렴 플롯**: 대표 구성에 대한 학습 곡선
4. **스케일링 법칙**: 멱법칙 피팅을 보여주는 로그-로그 플롯

### 4.3 통계 분석

**유의성 테스트**:
- 유사한 파라미터 수를 가진 구성에 대한 쌍체 t-검정
- 예: Hybrid (62K) vs LoRA (78K) vs Adapter (66K) 비교

**효과 크기**:
- 의미 있는 차이에 대한 Cohen's d 보고
- 예: "Hybrid는 LoRA보다 0.5 표준편차 더 나은 NMSE를 달성 (d=0.8, p<0.01)"

**신뢰 구간**:
- 각 Pareto 점에 대한 부트스트랩 95% CI (체크포인트를 유사 복제본으로 사용)

---

## 5. 구현 세부사항

### 5.1 파일 구조

**구성 파일**:
```
config/
├── config_pareto_adapter.yaml    # 4개 어댑터 구성
├── config_pareto_lora.yaml       # 4개 LoRA 구성
├── config_pareto_prompt.yaml     # 3개 Prompt 구성
└── config_pareto_hybrid.yaml     # 3개 Hybrid 구성
```

**실행 스크립트**:
```
Transfer_Pareto_Adapter.py   # 20 runs (4×5)
Transfer_Pareto_LoRA.py      # 20 runs (4×5)
Transfer_Pareto_Prompt.py    # 15 runs (3×5)
Transfer_Pareto_Hybrid.py    # 15 runs (3×5)
```

**출력 구조**:
```
saved_model/pareto/
├── Large_estimator_v3_to_InH_adapter_dim8.pt
├── Large_estimator_v3_to_InH_adapter_dim8_iter_20000.pt
├── ... (70개 최종 모델 + 350개 체크포인트 = 420개 파일)
```

### 5.2 실행 순서 (권장)

**Phase 1: 빠른 방법 우선** (조기 결과 획득)
1. ✅ Prompt (15 runs, ~30시간)
2. ✅ LoRA (20 runs, ~40시간)
3. ✅ Hybrid (15 runs, ~30시간)

**Phase 2: 느린 방법** (GPU 사용 가능 시 병렬 실행)
4. ✅ Adapter (20 runs, ~40시간)

**근거**:
- 빠른 방법의 조기 결과로 예비 분석 가능
- 예상치 못한 패턴 발생 시 전략 조정 가능
- Adapter 실행 중 논문 섹션 작성 시작 가능

### 5.3 모니터링 및 체크포인트

**WandB 프로젝트** (각 방법별로 분리):
```
DNN_channel_estimation_InH_Adapter_Pareto
DNN_channel_estimation_InH_LoRA_Pareto
DNN_channel_estimation_InH_Prompt_Pareto
DNN_channel_estimation_InH_Hybrid_Pareto
... (총 20개 프로젝트)
```

**로그되는 주요 메트릭**:
- `ch_nmse`: 주요 성능 메트릭
- `ch_loss`: 학습 목적 함수
- `learning_rate`: LR 스케줄 검증
- `trainable_params`: 파라미터 수 확인
- `iteration`: 학습 진행률

**재개 기능**:
- 각 스크립트는 20K마다 체크포인트 저장
- 중단 시 마지막 체크포인트에서 재시작 가능
- 독립적인 run으로 여러 GPU에서 병렬 실행 가능

---

## 6. 분석 계획

### 6.1 데이터 수집

모든 run 완료 후, 구조화된 형식으로 수집:

```python
results_df = pd.DataFrame({
    'method': [...],
    'config': [...],
    'scenario': [...],
    'params': [...],
    'nmse': [...],
    'mse': [...],
    'convergence_iter': [...]
})
```

### 6.2 시각화 스크립트

`plot_pareto_curves.py` 생성:

```python
# 1. Pareto Frontier (시나리오당 1개, 5개 플롯)
for scenario in scenarios:
    plt.figure()
    for method in methods:
        plot_curve(params, nmse, method, scenario)
    plt.xlabel('학습 가능 파라미터')
    plt.ylabel('NMSE (dB)')
    plt.legend()

# 2. 정규화된 효율성 히트맵
efficiency = nmse / params  # 낮을수록 좋음
sns.heatmap(pivot_table(scenario, method, efficiency))

# 3. 스케일링 법칙 (로그-로그)
for method in methods:
    fit_power_law(log(params), log(nmse))
    plot_fit_and_data()
```

### 6.3 논문 섹션

**Section IV: 실험 설계**
- 하위 섹션: Pareto Frontier 방법론
- Table 3: 구성 매트릭스
- 파라미터 범위에 대한 정당화

**Section V: 결과**
- 하위 섹션 A: 시나리오별 Pareto 곡선
  - Figure 5: InH/InF Pareto Frontiers
  - Figure 6: UMi/UMa/RMa Pareto Frontiers
- 하위 섹션 B: 파라미터 효율성 분석
  - Table 4: 예산별 최고 구성
  - Figure 7: 효율성 히트맵
- 하위 섹션 C: Hybrid 시너지
  - Figure 8: 동일 파라미터 비교
  - 통계 테스트: 동일 파라미터에서 Hybrid vs 단일 방법

**Section VI: 논의**
- 배포 시나리오에 대한 최적 작동점
- 파라미터 예산 권장 사항
- 스케일링 법칙 해석

---

## 7. 위험 완화

### 7.1 잠재적 문제

**문제 1: 극단 구성에서 비수렴**
- 예: Adapter dim=8이 너무 작거나, Prompt len=200이 너무 클 수 있음
- **완화**: 처음 몇 시나리오를 모니터링, 필요 시 범위 조정

**문제 2: 시나리오 전반에 걸친 일관성 없는 경향**
- 예: Prompt가 InH에서 최고이지만 RMa에서 최악
- **완화**: 이것은 실제로 흥미롭습니다! 시나리오별 권장 사항 보고

**문제 3: Hybrid가 시너지를 보이지 않음**
- 예: Hybrid(50,5) ≈ Prompt(50)과 LoRA(5)의 평균
- **완화**: 정직한 보고, 하이브리드가 유익하지 않은 경우 논의 (여전히 기여)

**문제 4: 시간/리소스 제약**
- 예: 실험 중간에 연구실 GPU 다운
- **완화**:
  - 독립적인 run을 별도로 실행 가능
  - 먼저 1-2개 시나리오 우선, 유망하면 5개 모두로 확장
  - 마감 접근 시 부분 결과(3/5 시나리오) 제시 가능

### 7.2 비상 계획

**Plan A (이상적)**: 70개 run 모두 완료 → 전체 Pareto 분석

**Plan B (시간 제약)**: 2-3개 시나리오 완료 → 부분 Pareto, 외삽

**Plan C (최소)**: 1개 시나리오(RMa) 모든 구성 완료 → 케이스 스터디 분석

**Plan D (최종 대비)**: 기존 결과 + 1-2개 새 제어 실험

---

## 8. 타임라인

### 8.1 실행 일정

**Week 1**: Prompt + LoRA (35 runs, ~70시간)
- Day 1-3: Prompt (15 runs)
- Day 4-6: LoRA (20 runs)
- Day 7: 예비 결과 분석

**Week 2**: Hybrid + Adapter (35 runs, ~70시간)
- Day 1-3: Hybrid (15 runs)
- Day 4-7: Adapter (20 runs)

**Week 3**: 분석 및 작성
- Day 1-2: 모든 Pareto 플롯 생성
- Day 3-4: 통계 분석
- Day 5-7: 결과 섹션 작성

### 8.2 마일스톤

- [ ] **M1**: Prompt 실험 완료 (파라미터 효율성에 대한 첫 통찰)
- [ ] **M2**: LoRA 실험 완료 (중간 범위 베이스라인 확립)
- [ ] **M3**: Hybrid 실험 완료 (시너지 가설 테스트)
- [ ] **M4**: Adapter 실험 완료 (고파라미터 영역 채우기)
- [ ] **M5**: 모든 Pareto 곡선 생성
- [ ] **M6**: 그림과 함께 결과 섹션 초안
- [ ] **M7**: 논의 및 결론 완료

---

## 9. 성공 기준

**최소 성공**:
- Pareto 곡선이 방법 간 명확한 차별화 표시
- 최소 하나의 방법이 강력한 파라미터 효율성 표시 (NMSE < 임계값, <50K 파라미터)
- 주요 비교에 대한 통계적 유의성

**목표 성공**:
- Hybrid가 시너지 입증 (동일 예산에서 단일 방법 능가)
- 파라미터 예산에 대한 명확한 권장 사항 (예: "<20K는 Prompt, 20-100K는 LoRA, >100K는 Adapter 사용")
- 스케일링 법칙이 잘 맞음 (모든 방법에 대해 R²>0.9)

**도전 성공**:
- Hybrid 그리드 탐색이 최적 Prompt:LoRA 비율 발견
- 시나리오 전반의 일반화 분석 (각 방법이 뛰어난 시기 식별)
- 관찰된 스케일링 동작에 대한 이론적 설명

---

## 10. 미해결 질문 (향후 연구)

### 10.1 Hybrid 그리드 탐색

대각선 결과 후, 전체 3×3 그리드 탐색하여 답변:
- 최적 Prompt:LoRA 비율은?
- 최적 비율이 총 파라미터 예산에 따라 달라지는가?
- Prompt와 LoRA 기여가 독립적(덧셈)인가 곱셈인가?

### 10.2 다목적 최적화

NMSE 외에 고려:
- **추론 지연**: Prompt가 오버헤드를 추가하는가? LoRA는?
- **메모리 풋프린트**: 런타임 메모리 vs 파라미터 수
- **적응 속도**: 수렴까지 반복 횟수 (온라인 학습에 중요)

### 10.3 테스트 범위를 넘어선 스케일링

- **초저 영역**: Prompt len=10-20, LoRA r=1-2
- **고 영역**: Adapter dim=128, LoRA r=64
- 멱법칙이 외삽되는가? 어디서 깨지는가?

### 10.4 시나리오별 튜닝

- 시나리오별 최적 구성을 **학습**할 수 있는가?
- 메타 학습: 시나리오 특성에서 최고 (방법, 구성) 예측

---

## 11. 결론

이 Pareto Frontier 설계는 다음을 제공합니다:
1. ✅ 파라미터 차이에도 불구하고 방법 간 **공정한 비교**
2. ✅ 체계적인 파라미터 스케일링을 통한 **과학적 엄격성**
3. ✅ 배포에 대한 **실용적 통찰** (어떤 예산에 어떤 방법?)
4. ✅ 향후 연구로의 **확장성** (하이브리드 그리드, 다목적)
5. ✅ **출판 강도** (공정성에 대한 리뷰어 우려 해결)

**70개 실험 run**은 채널 추정 작업에 대한 PEFT 방법 선택에 대한 결정적인 답변을 제공할 포괄적인 조사를 나타냅니다.

---

**문서 상태**: ✅ 완료 및 실행 준비 완료
**다음 단계**: `python Transfer_Pareto_Prompt.py`로 실행 시작
**연락처**: 질문 또는 문제는 프로젝트 README 참조
