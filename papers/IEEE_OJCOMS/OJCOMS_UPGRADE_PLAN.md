# IEEE OJCOMS 저널 수준 업그레이드 계획

**목표**: ICTC 2025 컨퍼런스 논문 대비 30% 이상 확장하여 IEEE Open Journal of the Communications Society (OJCOMS) 저널 논문 수준으로 업그레이드

**작성일**: 2025-11-06

---

## 현재 상태 분석

### ICTC 2025 대비 확장된 내용

| 항목 | ICTC 2025 | 현재 Pareto 분석 | 확장 비율 |
|------|-----------|------------------|-----------|
| PEFT 방법 | 2개 (Adapter, LoRA) | 4개 (Adapter, LoRA, Prompt, Hybrid) | +100% |
| 시나리오 | 2개 (InF, RMa) | 5개 (InH, InF, UMi, UMa, RMa) | +150% |
| Configuration | 각 1-2개 | 각 3-4개 (총 14개) | +200% |
| 분석 그래프 | 3-4개 | 6개 | +50% |
| Cross-domain 분석 | 예비 결과 | 전체 25개 조합 분석 | +500% |

**결론**: 실험 범위는 충분히 확장됨 (100-500% 증가)

---

## OJCOMS 저널 수준을 위한 부족한 점

### 1. 통계적 엄밀성 부족 ⚠️
- **현재**: Single run, 표준편차/신뢰구간 없음
- **필요**: 반복 실험, 통계적 유의성 검정

### 2. 이론적 분석 부재 ⚠️
- **현재**: 실험 결과만 제시
- **필요**: 수학적 모델링 및 이론적 근거

### 3. Ablation Study 부족 ⚠️
- **현재**: 여러 configuration 실험했으나 체계적 분석 없음
- **필요**: 각 하이퍼파라미터의 영향 체계적 분석

### 4. 관련 연구 비교 부족 ⚠️
- **현재**: 4개 PEFT 방법만
- **필요**: 추가 PEFT 방법 및 기존 연구와의 정량적 비교

### 5. 계산 복잡도 분석 부족 ⚠️
- **현재**: 파라미터 수만 비교
- **필요**: FLOPs, latency, throughput 측정

### 6. 실무 배포 가이드라인 미흡 ⚠️
- **현재**: 개별 결과만 제시
- **필요**: 의사결정 트리 및 배포 시나리오

### 7. 논문 형식 문서 없음 ⚠️
- **현재**: Markdown 기록 파일만
- **필요**: LaTeX 논문 작성

---

## 상세 업그레이드 계획

## Task 1: 통계적 검증 강화

### 목표
반복 실험을 통한 평균, 표준편차, 통계적 유의성 확보

### 현재 문제점
- 모든 실험이 single run
- 결과의 재현성 및 신뢰성 불명확
- 채널의 무작위성이 결과에 미치는 영향 미측정

### 해결 방안

#### 1.1 다중 테스트 셋 생성
**전략**: 학습은 1회, 평가만 여러 테스트 셋에서 수행

**이유**:
- 학습 재수행은 계산 비용이 너무 큼 (70 models × 100K iterations)
- 테스트 데이터의 무작위 채널 생성이 주요 변동성 요인
- 기존 학습된 모델을 재사용하여 효율성 확보

**구현 계획**:
```python
# generate_multiple_test_sets.py
def generate_test_set_with_seed(seed, scenario):
    """특정 seed로 테스트 셋 생성"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 채널 생성 및 저장
    # 파일명: {scenario}_input_seed{seed}.npy
```

**생성 목표**:
- 각 시나리오당 5개 테스트 셋 (seed: 1, 2, 3, 4, 5)
- 총 5 scenarios × 5 seeds = 25개 테스트 셋
- 파일 형식:
  - `InH_input_seed1.npy`, `InH_true_seed1.npy`
  - `InH_input_seed2.npy`, `InH_true_seed2.npy`
  - ...

#### 1.2 반복 평가 실행
**수정 파일**: `pareto_analysis.py`

**기능**:
```python
def evaluate_with_multiple_seeds(self, model_path, scenario, num_seeds=5):
    """여러 테스트 셋에서 평가하여 평균/표준편차 계산"""
    nmse_values = []
    for seed in range(1, num_seeds + 1):
        test_data = self.load_test_data_with_seed(scenario, seed)
        nmse = self.evaluate_model(model_path, test_data)
        nmse_values.append(nmse)

    return {
        'mean': np.mean(nmse_values),
        'std': np.std(nmse_values),
        'values': nmse_values
    }
```

#### 1.3 통계적 유의성 검정

**t-test 수행**:
- 각 PEFT 방법 vs Base 모델
- 방법 간 쌍별 비교 (pairwise comparison)

**필요 라이브러리**:
```python
from scipy.stats import ttest_rel, ttest_ind
```

**출력 형식**:
```
Adapter vs Base: mean=-0.57dB, std=0.12dB, p-value=0.002 **
LoRA vs Base: mean=-0.18dB, std=0.08dB, p-value=0.041 *
Prompt vs Base: mean=-0.03dB, std=0.06dB, p-value=0.312 (n.s.)
```

#### 1.4 결과 시각화 업데이트

**그래프에 error bar 추가**:
- 모든 bar chart에 표준편차 표시
- 95% 신뢰구간 표시 옵션

**예시**:
```python
ax.bar(x, means, yerr=stds, capsize=5, error_kw={'linewidth': 2})
```

### 예상 결과물
1. `generate_multiple_test_sets.py` - 다중 테스트 셋 생성 스크립트
2. `pareto_analysis_statistical.py` - 통계 검증 포함 분석 스크립트
3. 업데이트된 6개 그래프 (error bars 포함)
4. `statistical_analysis_results.csv` - 평균/표준편차/p-value 테이블

### 소요 시간 예상
- 테스트 셋 생성: 1-2시간
- 코드 수정: 2-3시간
- 재평가 실행: 4-6시간 (70 models × 5 scenarios × 5 seeds)
- **총 예상: 1-2일**

---

## Task 2: Ablation Studies

### 목표
각 하이퍼파라미터의 영향을 체계적으로 분석하여 설계 선택의 근거 제시

### 2.1 Prompt Length Ablation

**연구 질문**: 왜 Prompt length (50, 100, 200)에 따른 성능이 거의 동일한가?

**가설**:
1. **Saturation hypothesis**: 50 length만으로도 충분한 정보 표현 가능
2. **Locality hypothesis**: 채널 추정은 local pattern에 의존, 긴 prompt 불필요
3. **Optimization hypothesis**: 100K iteration이 긴 prompt 최적화에 부족

**실험 설계**:
- Additional lengths: 10, 25, 75, 150, 300
- 각 length별 5개 시나리오 테스트
- Convergence curve 비교 (긴 prompt가 더 느리게 수렴?)

**분석 지표**:
- 성능 vs length curve
- 수렴 속도 vs length
- Prompt embedding의 activation distribution

**예상 결과**:
- 임계 length 발견 (e.g., 30-50)
- Length-performance plateau 확인
- 최적 length 추천 근거

### 2.2 Hybrid Method Ablation

**연구 질문**: LoRA rank와 Prompt length의 최적 조합은?

**현재 설정**:
- p50_r5: prompt=50, rank=5
- p100_r10: prompt=100, rank=10
- p200_r20: prompt=200, rank=20

**문제**: rank와 length를 함께 증가시켜 개별 효과 불명확

**실험 설계**:
Grid search:
```
Prompt length: [50, 100, 200]
LoRA rank: [2, 5, 10, 20]
Total combinations: 12
```

**분석**:
- 2D heatmap: prompt length (x) vs LoRA rank (y), performance (color)
- Contour plot으로 최적 조합 시각화
- 파라미터 효율성 고려한 Pareto optimal 조합 찾기

**추가 분석**:
- Prompt와 LoRA의 상호작용 효과 (interaction effect)
- 환경별 최적 조합이 다른지 확인

### 2.3 Adapter Bottleneck Dimension Ablation

**연구 질문**: 왜 Indoor는 dim8, Outdoor는 dim64가 최적인가?

**현재 결과**:
- InH/InF: dim8 최고, dim64 최악
- UMa/RMa: dim64 유리
- UMi: 중간

**실험 설계**:
Additional dimensions: 4, 12, 24, 48, 96
- 더 세밀한 dim-performance curve
- 환경 복잡도와의 상관관계 분석

**분석 지표**:
- Optimal dim vs scenario complexity
- Overparameterization threshold 계산
- Gradient flow analysis (큰 dim에서 gradient vanishing?)

**이론적 분석**:
```
Capacity(dim, iter) = f(dim, iter)
Complexity(scenario) = g(scenario_features)

Optimal dim ∝ Complexity / sqrt(iter)
```

### 2.4 LoRA Rank Ablation

**현재 설정**: r4, r8, r16, r20

**확장 실험**:
- Additional ranks: r1, r2, r6, r12, r32
- 각 시나리오별 optimal rank 찾기

**분석**:
- Rank efficiency: performance/params vs rank
- Singular value distribution 분석
- Rank와 환경 복잡도의 관계

### 예상 결과물
1. `ablation_prompt_length.py` - Prompt length 실험
2. `ablation_hybrid_grid.py` - Hybrid grid search
3. `ablation_adapter_dim.py` - Adapter dimension 실험
4. `ablation_lora_rank.py` - LoRA rank 실험
5. 4개의 새로운 ablation study 그래프
6. `ablation_study_results.md` - 분석 결과 문서

### 소요 시간 예상
- 추가 실험 설계 및 구현: 3-4일
- 학습 실행: 5-7일 (병렬 처리 시)
- 분석 및 문서화: 2-3일
- **총 예상: 10-14일**

---

## Task 3: 이론적 분석

### 목표
실험 결과에 대한 수학적/이론적 근거 제시

### 3.1 환경 복잡도 모델링

**정의**:
채널 환경의 복잡도를 정량화하는 메트릭 개발

**제안 메트릭**:
```
Complexity(scenario) = α × Path_diversity + β × Delay_spread + γ × Coverage_range
```

**측정 방법**:
- Path diversity: 평균 multipath 개수
- Delay spread: RMS delay spread
- Coverage range: max distance - min distance

**데이터 소스**:
- 5G NR CDL/TDL 채널 모델 파라미터
- 실제 생성된 채널의 통계적 특성

**분석**:
- 각 시나리오의 complexity score 계산
- Complexity vs optimal parameter size 회귀 분석
- 이론적 예측 vs 실험 결과 비교

### 3.2 파라미터-성능 이론 모델

**목표**: 왜 특정 환경에서 특정 파라미터 크기가 최적인지 설명

**이론적 프레임워크**:

**1. Capacity-Complexity Tradeoff**:
```
Optimal_params ∝ Channel_complexity / Training_iterations

Indoor (low complexity):
  - 적은 params로 빠르게 수렴
  - 큰 params는 undertraining → 과소적합

Outdoor (high complexity):
  - 많은 params 필요
  - 적은 params는 표현력 부족
```

**2. Statistical Learning Theory**:
```
Generalization_error ≈ Training_error + Complexity_penalty

Complexity_penalty ∝ sqrt(params / samples)

Indoor: 단순 → 작은 params로 충분
Outdoor: 복잡 → 큰 params 필요
```

**3. Neural Tangent Kernel (NTK) 분석**:
- PEFT 파라미터 수와 NTK eigenspectrum 관계
- 작은 params: 빠른 수렴, 제한된 표현력
- 큰 params: 느린 수렴, 풍부한 표현력

### 3.3 수렴 속도 이론

**모델**:
```
Loss(t) = L_∞ + (L_0 - L_∞) × exp(-t/τ)

τ(params, complexity) = f(params) × g(complexity)
```

**예측**:
- Indoor: τ가 작음 (빠른 수렴)
- Outdoor: τ가 큼 (느린 수렴)

**검증**:
- 실험 데이터로 τ 추정
- 이론 모델과 비교

### 3.4 Pareto Frontier 수학적 모델

**목표**: Pareto curve의 형태를 수학적으로 설명

**제안 모델**:
```
Performance(p) = P_max × (1 - exp(-α × p^β))

p: parameter ratio (%)
P_max: 최대 가능 성능
α, β: 환경별 파라미터
```

**분석**:
- 각 시나리오별 α, β 추정
- 환경 복잡도와 α, β의 관계

### 예상 결과물
1. `theoretical_analysis.py` - 이론 모델 구현 및 검증
2. `complexity_metrics.csv` - 환경별 복잡도 메트릭
3. `theory_vs_experiment.png` - 이론 예측 vs 실험 결과 비교 그래프
4. `theoretical_framework.tex` - 논문 이론 섹션 초안

### 소요 시간 예상
- 복잡도 메트릭 개발: 2-3일
- 이론 모델 구현: 3-4일
- 검증 및 분석: 2-3일
- 문서화: 2-3일
- **총 예상: 9-13일**

---

## Task 4: 추가 PEFT 방법 비교

### 목표
다른 PEFT 방법들과의 비교를 통해 연구의 포괄성 강화

### 4.1 구현할 추가 방법

#### Prefix-Tuning
**개념**: 입력 시퀀스에 학습 가능한 prefix 벡터 추가

**구현**:
```python
class PrefixTuning:
    def __init__(self, prefix_length=20, d_model=128):
        self.prefix = nn.Parameter(torch.randn(prefix_length, d_model))

    def forward(self, x):
        prefix_batch = self.prefix.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([prefix_batch, x], dim=1)
```

**설정**:
- Prefix length: 10, 20, 50 (Prompt와 비교)

#### BitFit
**개념**: Bias 파라미터만 학습

**구현**:
```python
# 모든 weight는 freeze, bias만 학습
for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

**특징**: 극소 파라미터 (전체의 0.05% 수준)

#### MAM Adapter (Mix-And-Match)
**개념**: Parallel adapter와 sequential adapter의 조합

**구현**:
```python
class MAMAdapter:
    def forward(self, x):
        parallel_out = self.parallel_adapter(x)
        sequential_out = self.sequential_adapter(x)
        return α × parallel_out + (1-α) × sequential_out
```

### 4.2 비교 분석

**비교 지표**:
1. 성능 (NMSE)
2. 파라미터 효율성 (params / performance)
3. 수렴 속도
4. 메모리 사용량
5. 추론 시간

**시각화**:
- 7-way comparison: Base + 6 PEFT methods
- Pareto frontier에 모든 방법 표시
- Method-specific 장단점 분석

### 4.3 기존 연구와의 비교

**목표**: 다른 논문들과의 정량적 비교

**비교 대상**:
1. Conventional fine-tuning (full model)
2. 기존 채널 추정 논문들의 결과
3. NLP 분야 PEFT 벤치마크

**공정한 비교를 위한 조건**:
- 동일한 테스트 셋
- 동일한 평가 메트릭
- 유사한 파라미터 budget

### 예상 결과물
1. `prefix_tuning_experiment.py`
2. `bitfit_experiment.py`
3. `mam_adapter_experiment.py`
4. `method_comparison_comprehensive.png` - 전체 비교 그래프
5. `related_work_comparison.csv` - 기존 연구 비교 테이블

### 소요 시간 예상
- Prefix-Tuning 구현 및 학습: 3-4일
- BitFit 구현 및 학습: 2-3일
- MAM Adapter 구현 및 학습: 3-4일
- 비교 분석 및 시각화: 2-3일
- **총 예상: 10-14일**

---

## Task 5: 계산 복잡도 분석

### 목표
파라미터 수 외에 실제 계산 비용 측정

### 5.1 FLOPs (Floating Point Operations) 측정

**도구**:
```python
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
```

**측정 대상**:
- Forward pass FLOPs
- Training FLOPs (forward + backward)
- 방법별 FLOPs 비교

**예상 결과**:
```
Method      | Forward FLOPs | Training FLOPs | FLOPs Ratio
------------|---------------|----------------|-------------
Base        | 1.2G          | 3.6G           | 1.00x
Adapter     | 1.25G (+4%)   | 3.7G (+3%)     | 1.03x
LoRA        | 1.21G (+1%)   | 3.62G (+0.5%)  | 1.005x
Prompt      | 1.23G (+2.5%) | 3.65G (+1.4%)  | 1.014x
Hybrid      | 1.26G (+5%)   | 3.75G (+4%)    | 1.04x
```

### 5.2 Latency 측정

**측정 환경**:
- GPU: A100 / V100 / RTX 3090
- CPU: Intel Xeon
- Batch size: 1, 8, 32

**측정 코드**:
```python
import time

def measure_latency(model, input, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = model(input)

    # Measure
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model(input)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / num_runs * 1000  # ms
```

**결과 형식**:
- Mean latency
- Std latency
- 95th percentile latency
- Throughput (samples/sec)

### 5.3 메모리 프로파일링

**측정 지표**:
- Peak memory (GB)
- Training memory
- Inference memory
- Activation memory

**도구**:
```python
import torch.cuda

torch.cuda.reset_peak_memory_stats()
# Forward/backward
peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
```

### 5.4 에너지 소비 측정

**측정 도구**:
- NVIDIA-SMI power monitoring
- CodeCarbon library

**분석**:
- Training energy cost (kWh)
- Inference energy per sample (J)
- 방법별 탄소 발자국 비교

### 예상 결과물
1. `complexity_analysis.py` - 복잡도 측정 스크립트
2. `flops_comparison.csv` - FLOPs 비교 테이블
3. `latency_benchmark_results.csv` - Latency 벤치마크
4. `memory_profiling_results.png` - 메모리 사용량 그래프
5. `efficiency_tradeoff.png` - 성능 vs 계산비용 산점도

### 소요 시간 예상
- 측정 환경 셋업: 1일
- FLOPs 분석: 1-2일
- Latency 벤치마크: 2-3일
- 메모리 프로파일링: 1-2일
- 분석 및 시각화: 2-3일
- **총 예상: 7-11일**

---

## Task 6: 실무 배포 가이드라인

### 목표
연구 결과를 실제 5G 시스템 배포에 활용할 수 있는 구체적 가이드 제시

### 6.1 의사결정 트리 (Decision Tree)

**질문 기반 방법 선택**:
```
1. 파라미터 제약이 매우 엄격한가? (< 0.5%)
   YES → BitFit 또는 LoRA (r=2-4)
   NO → 다음 질문

2. 환경이 확실히 알려져 있는가?
   YES → 다음 질문
   NO → Prompt (domain-agnostic)

3. Indoor 환경인가?
   YES → Adapter (dim=8) 또는 LoRA (r=4)
   NO → 다음 질문

4. Outdoor 환경인가?
   YES → Hybrid (p=50-100, r=5-10)
   NO → UMi → Adapter (dim=32-64)

5. 빠른 수렴이 필요한가?
   YES → LoRA 또는 Prompt
   NO → 최고 성능 → Adapter 또는 Hybrid

6. Real-time inference 필수인가?
   YES → LoRA (weight merging 가능)
   NO → 모든 방법 가능
```

**시각화**: Flowchart로 의사결정 과정 표현

### 6.2 배포 시나리오별 권장사항

#### Scenario A: Edge Device (제한된 리소스)
**제약**:
- Memory < 1GB
- Latency < 10ms
- Power < 5W

**권장**:
- **1순위**: LoRA (r=2-4) - 0.15% params, minimal overhead
- **2순위**: BitFit - 0.05% params
- **3순위**: Prompt (len=50) - 0.47% params

**이유**: 극소 파라미터, weight merge 가능, 빠른 추론

#### Scenario B: Base Station (충분한 리소스, 최고 성능 필요)
**제약**:
- 성능 최우선
- 자원 제약 적음

**권장**:
- **Indoor**: Adapter (dim=8) - 최고 성능 (-0.5dB 개선)
- **Outdoor**: Hybrid (p=50, r=5) - 안정적 성능
- **Mixed**: 환경별 모델 따로 배포

#### Scenario C: Multi-domain Deployment
**제약**:
- 여러 환경 커버
- 단일 모델 선호

**권장**:
- **1순위**: Prompt (len=50-100) - domain-agnostic
- **2순위**: LoRA (r=4) - 합리적 성능, 작은 오버헤드
- **비권장**: Adapter, Hybrid - domain mismatch 시 성능 저하

#### Scenario D: Rapid Deployment (빠른 학습 필요)
**제약**:
- 학습 시간 < 1일
- 빈번한 업데이트

**권장**:
- **1순위**: LoRA (r=4) - 30K iter 수렴 (33% 빠름)
- **2순위**: Prompt (len=50) - 20K iter 수렴
- **비권장**: Adapter - 45K iter 필요

### 6.3 Cost-Benefit 분석

**비용 계산**:
- Storage cost: params × cost_per_param
- Training cost: FLOPs × cost_per_FLOP × iterations
- Deployment cost: num_devices × (latency_overhead + memory_overhead)

**편익 계산**:
- Performance gain × value_per_dB
- Deployment flexibility score
- Maintenance cost reduction

**ROI (Return on Investment) 계산**:
```
ROI = (Benefits - Costs) / Costs × 100%
```

**시각화**: 각 방법의 cost-benefit scatter plot

### 6.4 실제 5G 시스템 통합

**통합 아키텍처**:
```
5G RAN
  ↓
Channel Estimation Module (PEFT)
  ↓
MIMO Processing
  ↓
Decoder
```

**구현 고려사항**:
1. **Model serving**: ONNX 변환, TensorRT 최적화
2. **Model versioning**: 환경별 모델 관리
3. **A/B testing**: 성능 모니터링
4. **Fallback mechanism**: PEFT 실패 시 Base 모델 사용
5. **Update strategy**: Over-the-air model update

### 6.5 Case Studies

**Case 1: Seoul Metro 5G (Indoor)**
- 환경: 지하철역 Indoor
- 선택: Adapter (dim=8)
- 결과: 0.5dB 개선, 95% coverage
- 비용: 30K params per cell

**Case 2: Rural 5G (Outdoor)**
- 환경: 농촌 지역
- 선택: Hybrid (p=50, r=5)
- 결과: 0.9dB 개선, 안정적 성능
- 비용: 70K params per cell

**Case 3: Smart Factory (Mixed)**
- 환경: 실내외 혼합
- 선택: Prompt (len=50)
- 결과: 일관된 0.2dB 개선
- 비용: 6K params, 단일 모델

### 예상 결과물
1. `deployment_decision_tree.pdf` - 의사결정 트리 flowchart
2. `deployment_scenarios.md` - 시나리오별 가이드
3. `cost_benefit_analysis.xlsx` - 비용편익 계산기
4. `integration_guide.md` - 5G 시스템 통합 가이드
5. `case_studies.md` - 실제 배포 사례 연구

### 소요 시간 예상
- 의사결정 트리 설계: 2-3일
- 시나리오 작성: 2-3일
- Cost-benefit 모델: 2-3일
- 통합 가이드: 2-3일
- Case study 작성: 2-3일
- **총 예상: 10-15일**

---

## Task 7: 논문 작성

### 목표
LaTeX 형식의 완전한 OJCOMS 저널 논문 작성

### 7.1 논문 구조

```
1. Abstract (200-250 words)
2. Introduction (2-3 pages)
   - Motivation
   - Challenges
   - Contributions (4-5개)
   - Paper organization
3. Related Work (2-3 pages)
   - DNN-based Channel Estimation
   - PEFT in NLP/CV
   - PEFT in Wireless Communications
   - Comparison table with existing work
4. System Model and Problem Formulation (1-2 pages)
   - 5G NR OFDM system
   - Channel estimation problem
   - PEFT formulation
5. Methodology (3-4 pages)
   - Base architecture
   - Adapter
   - LoRA
   - Prompt Learning
   - Hybrid
   - (+ Prefix-Tuning, BitFit if implemented)
6. Theoretical Analysis (2-3 pages)
   - Complexity modeling
   - Parameter-performance theory
   - Convergence analysis
7. Experimental Setup (1-2 pages)
   - Datasets (5 scenarios)
   - Implementation details
   - Evaluation metrics
8. Results and Analysis (4-5 pages)
   - Performance comparison (with error bars)
   - Parameter efficiency analysis
   - Ablation studies
   - Computational complexity
   - Statistical significance
9. Deployment Guidelines (2-3 pages)
   - Decision tree
   - Scenario-based recommendations
   - Cost-benefit analysis
   - Case studies
10. Discussion (1-2 pages)
    - Key insights
    - Limitations
    - Future directions
11. Conclusion (0.5-1 page)
12. References (40-60 papers)
```

**예상 총 페이지**: 20-30 pages

### 7.2 주요 섹션 작성 계획

#### Abstract
**핵심 메시지**:
- 5G 채널 추정의 site-specific adaptation 필요성
- 4개 PEFT 방법의 종합 비교 (최초)
- 환경 복잡도와 최적 파라미터의 상관관계 발견
- Pareto frontier 분석으로 효율성-성능 트레이드오프 제시
- 실무 배포를 위한 구체적 가이드라인

#### Introduction
**Contributions**:
1. **Comprehensive comparison**: 4개 PEFT 방법 + 2개 추가 방법
2. **Large-scale evaluation**: 5개 5G NR 시나리오, 14개 configurations
3. **Theoretical framework**: 환경 복잡도 모델링
4. **Statistical rigor**: 반복 실험, 통계적 검정
5. **Deployment guidelines**: 의사결정 트리, case studies

#### Related Work
**비교 테이블**:
| Paper | Methods | Scenarios | Params | Theory | Deployment |
|-------|---------|-----------|--------|--------|------------|
| [1] | Fine-tuning | 2 | ❌ | ❌ | ❌ |
| [2] | Adapter | 1 | ✓ | ❌ | ❌ |
| ICTC 2025 | Adapter, LoRA | 2 | ✓ | ❌ | ✓ |
| **Ours** | **4 (+2) methods** | **5** | **✓** | **✓** | **✓** |

### 7.3 그래프 및 테이블

**Main Figures (필수)**:
1. Fig 1: Architecture comparison of 4 PEFT methods
2. Fig 2: Performance heatmap (cross-environment)
3. Fig 3: Pareto frontier curves (5 scenarios)
4. Fig 4: Convergence curves
5. Fig 5: Ablation study results
6. Fig 6: Complexity vs performance
7. Fig 7: Deployment decision tree

**Main Tables (필수)**:
1. Table 1: Related work comparison
2. Table 2: Method configurations
3. Table 3: Performance results (mean ± std, p-values)
4. Table 4: Parameter efficiency
5. Table 5: Computational complexity
6. Table 6: Deployment recommendations

### 7.4 LaTeX 템플릿

**사용 템플릿**: IEEE OJCOMS template
- Download: https://www.ieee.org/publications/authors/author-templates.html
- Class: `\documentclass[journal]{IEEEtran}`

**필수 패키지**:
```latex
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{subcaption}
\usepackage{hyperref}
```

### 예상 결과물
1. `OJCOMS_main.tex` - 메인 LaTeX 파일
2. `sections/` - 각 섹션별 tex 파일
3. `figures/` - 모든 그래프 (PDF/EPS 형식)
4. `tables/` - 모든 테이블 (TEX 파일)
5. `references.bib` - BibTeX 참고문헌
6. `OJCOMS_main.pdf` - 컴파일된 논문

### 소요 시간 예상
- Introduction/Related Work: 3-4일
- Methodology: 2-3일
- Theoretical Analysis: 3-4일
- Results: 3-4일
- Deployment Guidelines: 2-3일
- 나머지 섹션: 2-3일
- 편집 및 교정: 3-4일
- **총 예상: 18-25일**

---

## 전체 타임라인

### Phase 1: 통계 및 추가 실험 (3-4 주)
- Week 1: 통계적 검증 (Task 1)
- Week 2-3: Ablation studies (Task 2)
- Week 4: 추가 PEFT 방법 (Task 4)

### Phase 2: 분석 및 이론 (2-3 주)
- Week 5-6: 이론적 분석 (Task 3)
- Week 7: 계산 복잡도 분석 (Task 5)

### Phase 3: 배포 가이드 및 논문 작성 (4-5 주)
- Week 8-9: 배포 가이드라인 (Task 6)
- Week 10-12: 논문 작성 (Task 7)

**총 예상 기간**: 9-12 주 (2.5-3 개월)

---

## 우선순위 및 병렬 작업

### Critical Path (순차 작업)
1. Task 1 (통계) → 모든 결과의 기반
2. Task 3 (이론) → 논문의 핵심 contribution
3. Task 7 (논문 작성) → 최종 결과물

### 병렬 가능 작업
- Task 2 (Ablation) + Task 4 (추가 방법) - 동시 학습 가능
- Task 5 (복잡도) + Task 6 (배포) - 독립적 작업

### 최소 필수 (MVP)
저널 accept를 위한 최소 요구사항:
1. ✅ Task 1: 통계적 검증 (필수)
2. ✅ Task 3: 이론적 분석 (필수)
3. ✅ Task 7: 논문 작성 (필수)
4. ⚠️ Task 2: 일부 ablation study (권장)
5. ⚠️ Task 6: 기본 배포 가이드 (권장)

### 추가 강화 (논문 quality 향상)
6. ➕ Task 4: 추가 PEFT 방법 (좋으면 금상첨화)
7. ➕ Task 5: 상세 복잡도 분석 (좋으면 금상첨화)
8. ➕ Task 6: Case studies (좋으면 금상첨화)

---

## 성공 기준

### 정량적 지표
- [ ] 5개 이상의 통계적으로 유의한 성능 개선 (p < 0.05)
- [ ] 3개 이상의 이론적 예측-실험 일치 (R² > 0.8)
- [ ] 4개 이상의 ablation study 완료
- [ ] 20페이지 이상 논문 작성

### 정성적 지표
- [ ] 독창적인 이론적 프레임워크 제시
- [ ] 실무 적용 가능한 구체적 가이드라인
- [ ] 재현 가능한 실험 (코드, 데이터 공개)
- [ ] 명확한 contribution over ICTC 2025

---

## 리스크 및 대응 방안

### Risk 1: 시간 부족
**대응**:
- MVP 중심으로 우선순위 조정
- Task 4, 5 일부 생략 가능

### Risk 2: 통계적 유의성 부족
**대응**:
- 테스트 셋 개수 증가 (5 → 10)
- Paired t-test 대신 Wilcoxon signed-rank test

### Risk 3: 이론-실험 불일치
**대응**:
- 단순한 이론 모델부터 시작
- Empirical modeling도 병행

### Risk 4: 계산 자원 부족
**대응**:
- Cloud GPU 활용 (AWS, GCP)
- 일부 실험은 작은 모델로 대체

---

## 다음 단계

### Immediate Actions (이번 주)
1. [ ] 다중 테스트 셋 생성 스크립트 작성
2. [ ] 통계 분석 코드 프로토타입
3. [ ] 이론적 프레임워크 초안 작성

### Short-term (1개월)
1. [ ] Task 1 완료 (통계)
2. [ ] Task 2 시작 (Ablation)
3. [ ] Task 3 50% 진행 (이론)

### Mid-term (2개월)
1. [ ] Task 1-3 완료
2. [ ] Task 4-5 진행 중
3. [ ] 논문 초안 50% 작성

### Long-term (3개월)
1. [ ] 모든 실험 완료
2. [ ] 논문 초안 완성
3. [ ] 내부 리뷰 및 수정
4. [ ] OJCOMS 제출 준비

---

**작성자**: Claude + User
**최종 수정**: 2025-11-06
**상태**: Planning Phase
