# Pareto Frontier Analysis Results

본 문서는 5G NR 채널 추정을 위한 PEFT(Parameter-Efficient Fine-Tuning) 방법들의 성능 및 효율성 분석 결과를 정리한다.

## 실험 설정

- **Base Model**: Large_estimator_v3 (1,356,000 parameters)
- **PEFT Methods**: Adapter, LoRA, Prompt Learning, Hybrid
- **Scenarios**: InH, InF, UMi, UMa, RMa
- **Training Iterations**: 20K, 40K, 60K, 80K, 100K
- **Evaluation Metric**: NMSE (dB, negative = better)

### PEFT Configurations

| Method | Configurations | Parameter Range |
|--------|---------------|-----------------|
| Adapter | dim8, dim16, dim32, dim64 | 0.3% - 38.66% of base |
| LoRA | r4, r8, r16, r20 | 0.5% - 5.80% of base |
| Prompt | len50, len100, len200 | 0.47% - 2.23% of base |
| Hybrid | p50_r5, p100_r10, p200_r20 | 2.23% - 9.14% of base |

---

## 1. PEFT Performance Heatmap

### 요약
PEFT 방법별로 파라미터 크기를 증가시키면서 5개 시나리오(InH, InF, UMi, UMa, RMa)에 대한 cross-environment 성능을 비교. 상단 값은 Base 대비 개선도(dB), 하단 값은 실제 NMSE(dB)를 나타냄.

### 주요 결과

1. **In-domain 성능 우수**: 대각선 방향(train=test 환경)에서 가장 짙은 녹색을 보이며, 모든 PEFT 방법이 Base 모델 대비 유의미한 개선(+0.5~2dB)을 달성

2. **파라미터 효율성**: Adapter와 LoRA는 소형 구성(dim8, r4)에서도 indoor 시나리오(InH, InF)에서 강력한 성능을 보이며, 파라미터 증가가 항상 성능 향상으로 이어지지 않음 (diminishing returns)

3. **Cross-domain 한계**: UMa와 RMa 열에서 붉은색이 지배적이며, 특히 indoor에서 학습한 모델들이 outdoor 환경에서 Base보다 못한 성능(-0.5~-2dB)을 보여 domain shift의 심각성을 확인

4. **Prompt의 안정성**: Prompt 방법은 파라미터 크기(len50~200)와 무관하게 일관된 성능을 보이며, cross-domain degradation이 상대적으로 적음

5. **Hybrid의 균형**: Hybrid 방법이 LoRA와 Prompt의 장점을 결합하여 대부분의 시나리오에서 안정적인 in-domain 성능과 비교적 나은 cross-domain robustness를 동시에 제공

---

## 2. Pareto Frontier: Parameter Efficiency vs Performance

### 요약
각 시나리오별 in-domain 성능에서 파라미터 효율성(% of base model) 대비 Base 모델 대비 개선도(dB)를 비교. Y=0선이 Base 성능이며, 음수 값이 더 나은 성능을 의미.

### 주요 결과

#### 1. 환경 복잡도와 최적 파라미터 규모의 상관관계

**Indoor (InH, InF, 거리 범위 소)**:
- 가장 작은 구성(Adapter dim8 0.3%, LoRA r4 0.5%)이 최고 성능(+0.6~0.7dB)
- 파라미터를 늘릴수록 오히려 성능 저하(-0.5dB at dim64)

**Outdoor (UMa, RMa, 거리 범위 대)**:
- 상대적으로 큰 파라미터 구성이 더 나은 성능
- 작은 파라미터는 복잡한 채널 패턴을 충분히 표현하지 못함

**원인**:
- 동일한 iteration(100K)에서 파라미터가 많을수록 개별 파라미터당 업데이트 횟수 감소
- Indoor의 단순한 패턴은 소형 구성으로도 빠르게 수렴하지만, 대형 구성은 undertraining으로 과소적합
- Outdoor는 높은 복잡도로 인해 더 많은 파라미터와 iteration이 필요

#### 2. Urban Micro의 중간 특성 (UMi)
- Indoor와 Outdoor의 전환점에 해당
- Adapter는 여전히 파라미터 증가에 따른 성능 향상을 보이지만(dim8: -0.9dB → dim64: -0.3dB), Indoor만큼 극단적이지는 않음
- LoRA/Prompt/Hybrid는 Base 수준(±0.1dB)에 안착하여 안정적

#### 3. 방법론적 차이
- **Prompt**: 파라미터 크기에 불변한 안정성(같은 iteration에 덜 민감)
- **Adapter/LoRA**: 시나리오별 최적화 필요
- **Hybrid**: 중간 균형점

---

## 3. In-domain Performance: Parameter Configuration Comparison

### 요약
각 PEFT 방법별로 파라미터 configuration이 in-domain 성능(train=test 환경)에 미치는 영향을 5개 시나리오에서 비교. Y=0선이 Base 성능, 음수가 더 나은 성능을 의미. 대부분의 경우 적절한 configuration 선택 시 Base 대비 개선 또는 동등 수준 달성.

### 주요 결과

#### 1. Pareto Frontier 패턴의 재확인

**Indoor (InH, InF)**:
- 작은 파라미터 configuration이 최적
- Adapter dim8과 LoRA r4가 명확한 개선(-0.3~-0.5dB)
- 큰 파라미터는 overtraining으로 악화(+0.7~+1.5dB)

**UMi (중간 복잡도)**:
- 중대형 파라미터가 유리
- Adapter는 dim8에서 -0.9dB 저하(Base보다 나쁨), dim64에서 Base 수준 회복

**Outdoor (UMa, RMa)**:
- 모든 방법이 일관된 개선
- 특히 RMa에서 Adapter/LoRA/Hybrid가 -0.5~-1.0dB의 명확한 성능 향상
- Hybrid가 가장 우수(-0.9~-1.0dB)

#### 2. 방법별 파라미터 민감도 차이

- **Adapter**: 가장 높은 민감도. 환경별 최적 configuration이 극단적으로 다름 (Indoor: dim8 vs UMi: dim64)
- **LoRA**: 중간 수준 민감도. 작은 rank(r4~r8) 선호, 큰 rank는 약간 악화
- **Prompt**: 파라미터 크기 완전 불변 (±0.05dB 이내). Configuration 선택 부담 없음
- **Hybrid**: Indoor에서 파라미터 증가 시 약간 악화, Outdoor에서는 모든 config 일관 개선

#### 3. 실무적 함의
적절한 파라미터 선택이 critical. Indoor는 극소 파라미터(0.3~0.5%)로 충분하며 과도한 파라미터는 역효과. 복잡도가 높아질수록 더 많은 파라미터 필요하나, Outdoor는 PEFT 자체의 표현력 한계 직면.

---

## 4. In-domain Performance Comparison by Scenario

### 요약
각 PEFT 방법의 최고 성능 configuration을 선택하여 시나리오별 in-domain 최적 성능을 비교. 각 방법의 best-case 시나리오를 제시. 막대 위 파라미터 비율(% of base model) 표시.

### 주요 결과

#### 1. Indoor 시나리오의 Adapter 우위 (InH, InF)
- Adapter가 유일하게 명확한 개선(-0.2~-0.5dB), 특히 InH에서 가장 우수
- LoRA/Prompt/Hybrid는 모두 Base 수준(±0.05dB)
- Indoor의 단순한 패턴에서 Adapter의 bottleneck 구조가 가장 효과적
- **파라미터 비율**: Adapter 4.83%, LoRA 1.45%, Prompt 0.47%, Hybrid 2.23%

#### 2. UMi에서 Adapter의 압도적 우위
- Adapter가 -0.9dB로 가장 큰 개선, 다른 방법들은 -0.1dB 수준
- 중간 복잡도 환경에서 Adapter의 적응력이 특히 뛰어남
- **파라미터 비율**: Adapter 4.83%, LoRA 1.45%, Prompt 0.94%, Hybrid 2.23%

#### 3. Outdoor 시나리오의 균형 (UMa, RMa)

**UMa**:
- 모든 방법이 근소 개선(-0.1~-0.2dB)으로 비슷한 수준
- Prompt가 상대적으로 선전
- **파라미터 비율**: Adapter 38.66% (가장 큼!), LoRA 5.80%, Prompt 1.89%, Hybrid 9.14%

**RMa**:
- Hybrid가 최우수(-0.9dB), Adapter/LoRA도 좋은 성능(-0.6dB)
- Prompt는 상대적으로 약함(-0.2dB)
- 높은 복잡도에서 LoRA+Prompt 결합(Hybrid)의 강점이 드러남
- **파라미터 비율**: Adapter 9.67%, LoRA 5.80%, Prompt 1.89%, Hybrid 9.14%

#### 4. 방법론 선택 가이드라인
- **Indoor/UMi**: Adapter 우선
- **Outdoor (특히 RMa)**: Hybrid 우선
- **안정성 중시**: Prompt (모든 환경에서 Base 수준 이상 보장)
- **LoRA**: 범용적이지만 돋보이는 시나리오 없음

#### 5. 파라미터 효율성 관찰
- UMa에서 Adapter가 38.66%로 가장 큰 파라미터를 사용하면서도 -0.2dB 정도의 근소한 개선만 보임
- Prompt는 1.89%로 비슷한 성능을 내어 효율성 측면에서 우수

---

## 5. Cross-environment Performance Comparison by Test Scenario

### 요약
각 테스트 환경별로 모든 training 환경에서 학습된 모델의 성능을 비교. 각 PEFT 방법의 best configuration 사용. Train ≠ Test 조합에서 domain mismatch의 영향을 확인.

### 주요 결과

#### 1. Indoor 학습 → Outdoor 테스트 시 극심한 성능 저하
- **RMa 테스트에서 InH 학습 모델**: Adapter +2.6dB, LoRA +2.4dB, **Hybrid +2.7dB** - 최악의 악화
- **UMa 테스트에서 InH 학습 모델**: Adapter +1.7dB, Hybrid +1.1dB
- Indoor 패턴에 학습된 PEFT가 Outdoor 복잡도와 맞지 않음

#### 2. UMa 학습 → Indoor 테스트 시 예상 밖 개선
- **InH 테스트에서 UMa 학습**: Adapter -1.7dB, LoRA -1.0~-1.2dB로 오히려 **in-domain보다 더 좋음**
- UMa의 높은 복잡도가 다양한 패턴을 학습하여 Indoor에도 적용 가능
- 단, 이는 Adapter/LoRA에만 해당, Prompt/Hybrid는 이런 효과 없음

#### 3. RMa 학습 모델의 cross-domain 성능 저하
- UMi 테스트: +1.6~+2.3dB (Prompt/Hybrid가 최악 +2.3dB)
- InH/InF 테스트: 0~+1.1dB
- **RMa는 in-domain에서만 효과적**, 다른 환경에서는 성능 크게 저하

#### 4. Indoor 간 상호 호환성
- InH ↔ InF 조합에서 Adapter/LoRA -0.5~-0.7dB 개선
- 유사한 거리 범위와 채널 특성으로 인한 긍정적 결과

#### 5. Prompt의 극단적 안정성
- **모든 test-train 조합에서 ±0.3dB 이내** 유지
- Domain mismatch에도 거의 degradation 없지만, 큰 개선도 없음
- Domain-agnostic한 특성

#### 6. Site-specific 배포의 중요성
- **Adapter/LoRA/Hybrid**: 높은 in-domain 성능, 하지만 domain mismatch 시 +2.7dB까지 악화 위험
- **Prompt**: Domain mismatch에 안정적이지만 개선 폭 제한적
- **실무 권장**: 각 환경에 맞는 site-specific 모델 배포 필수. 환경 불확실성이 높다면 Prompt, 확실하다면 Adapter 선택

---

## 6. Training Convergence Curves

### 요약
각 시나리오별로 PEFT 방법들의 iteration별 수렴 속도를 비교. 각 방법의 best performing configuration을 선택하여 20K~100K iteration 동안의 NMSE 변화를 추적.

### 주요 결과

#### 1. Indoor 시나리오의 조기 수렴 (InH, InF)
- **대부분 20-40K iteration에서 이미 수렴 완료**
- InH에서 Adapter: 20K (-19.8dB) → 40K (-19.6dB) 이후 평탄
- LoRA는 40K에서 peak (-18.9dB) 이후 오히려 약간 악화 → overtraining 징후
- **결론**: Indoor는 단순한 패턴으로 인해 빠른 학습 가능, 100K iteration은 과도함

#### 2. Outdoor 시나리오의 지속적 개선 (UMa, RMa)
- **100K까지 꾸준히 성능 향상 지속**
- UMa에서 Adapter: 20K (-12.05dB) → 100K (-12.16dB)
- RMa에서 Prompt: 20K (-15.0dB) → 100K (-15.3dB)
- **결론**: 높은 복잡도로 인해 더 많은 iteration 필요, 100K에서도 아직 수렴 중

#### 3. UMi의 즉각적 수렴
- **거의 모든 방법이 20K 이내에 최종 성능 달성**
- Adapter/LoRA/Prompt/Hybrid 모두 20K 이후 거의 변화 없음 (±0.1dB)
- 중간 복잡도 환경에서 빠른 수렴

#### 4. Prompt의 극단적 안정성
- **모든 시나리오에서 가장 평탄한 곡선**
- InH/InF: 완전 수평선 (20K~100K 동안 변화 없음)
- UMa/RMa: 약간의 개선 있지만 다른 방법 대비 매우 완만
- 파라미터 크기가 학습에 덜 민감함을 재확인

#### 5. 방법별 수렴 특성
- **Adapter**: 시나리오별로 가장 다양한 수렴 패턴 (InH 조기 수렴 vs RMa 지속 개선)
- **LoRA**: Adapter와 유사하지만 일부 시나리오에서 overtraining (InH 40K peak)
- **Hybrid**: 대부분 40K 이내 수렴, 이후 안정
- **Prompt**: 거의 20K 이내 수렴 또는 처음부터 평탄

#### 6. Pareto 그래프와의 연관성
- Indoor에서 큰 파라미터 configuration이 성능 저하를 보인 이유: **조기 수렴 환경에서 100K iteration은 큰 파라미터를 충분히 최적화하지 못함**
- Outdoor에서 큰 파라미터가 유리한 이유: **100K까지도 계속 개선되며, 복잡도에 맞는 표현력 제공**

#### 7. 실무적 함의
- **Indoor 배포**: 20-40K iteration으로 충분, early stopping 권장
- **Outdoor 배포**: 100K 이상 iteration 필요, 더 긴 학습 시간 확보
- **Training budget 최적화**: 환경별 맞춤 iteration 설정으로 효율성 향상

---

## 종합 결론

### 1. 환경별 최적 PEFT 전략

| Scenario | Best Method | Optimal Config | Parameter Ratio | Key Insight |
|----------|-------------|----------------|-----------------|-------------|
| InH | Adapter | dim8 | 4.83% | 조기 수렴, 소형 파라미터 최적 |
| InF | Adapter | dim8 | 4.83% | 조기 수렴, 소형 파라미터 최적 |
| UMi | Adapter | dim64 | 4.83% | 중형 파라미터 필요 |
| UMa | Prompt | len50-200 | 1.89% | 안정성 중시, 파라미터 불변 |
| RMa | Hybrid | p50_r5 | 9.14% | 지속적 개선, 큰 파라미터 유리 |

### 2. 핵심 발견사항

1. **파라미터-복잡도 트레이드오프**:
   - Indoor는 작은 파라미터로 빠른 수렴
   - Outdoor는 큰 파라미터로 지속적 개선
   - 동일 iteration에서 파라미터 크기 ↑ → 개별 파라미터 최적화 ↓

2. **Domain Mismatch의 심각성**:
   - Cross-domain 성능 저하 최대 +2.7dB
   - Site-specific 모델 배포 필수
   - Prompt만 domain-agnostic 특성

3. **Training Iteration 최적화**:
   - Indoor: 20-40K 충분
   - Outdoor: 100K+ 필요
   - 환경별 맞춤 설정으로 효율성 향상

4. **방법론별 특성**:
   - **Adapter**: 최고 in-domain 성능, 높은 파라미터 민감도
   - **LoRA**: Adapter와 유사, 약간 더 안정적
   - **Prompt**: 극단적 안정성, 제한적 개선
   - **Hybrid**: 균형잡힌 성능, RMa에서 우수

### 3. 실무 권장사항

1. **확실한 환경 매칭**: Adapter (InH/InF/UMi) 또는 Hybrid (RMa)
2. **환경 불확실성**: Prompt (모든 환경에서 안정)
3. **파라미터 예산 제한**: Prompt (0.5-2%) 또는 LoRA (1-2%)
4. **최대 성능 추구**: 환경별 최적 방법 선택 + 맞춤 iteration

---

## 생성된 그래프 목록

1. `pareto_heatmap.png` - PEFT 성능 히트맵
2. `pareto_curves.png` - Pareto Frontier 곡선
3. `pareto_transfer_effectiveness.png` - 파라미터별 성능 비교
4. `pareto_domain_indomain.png` - 시나리오별 in-domain 비교
5. `pareto_domain_crossenv.png` - 시나리오별 cross-environment 비교
6. `pareto_convergence.png` - 수렴 곡선

---

**작성일**: 2025-11-06
**실험 코드**: `pareto_analysis.py`
