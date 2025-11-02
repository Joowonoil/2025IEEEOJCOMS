# Cross-Domain Transfer Learning Experiment

## 📋 **실험 개요**

본 실험은 진정한 전이학습(Transfer Learning) 효과를 검증하기 위해 설계되었습니다. 기존 v4 모델이 모든 환경 데이터로 학습되어 전이학습 효과가 미미했던 문제를 해결하고, 완전히 다른 도메인 간의 지식 전이를 통해 LoRA의 효과를 극대화합니다.

## 🎯 **실험 목적**

1. **진정한 Domain Adaptation 검증**: 완전히 다른 환경에서 학습된 모델의 전이학습 효과
2. **LoRA 효율성 입증**: 적은 파라미터로도 큰 성능 향상 달성
3. **실용적 시나리오 검증**: 실제 무선 통신 환경 배포 시나리오 모사
4. **논문 기여도 향상**: 명확한 성능 향상으로 학술적 가치 증대

## 🌍 **환경별 데이터셋 분류**

### **Urban Environments (도시 환경)**
- **UMa (Urban Macro)**: 도시 광역 셀 환경
  - 고층 빌딩, 넓은 커버리지
  - 거리 범위: 수백 미터 ~ 수 킬로미터
- **UMi (Urban Micro)**: 도시 소규모 셀 환경  
  - 밀집된 건물, 작은 셀 크기
  - 거리 범위: 수십 ~ 수백 미터

### **Rural Environments (농촌 환경)**
- **RMa (Rural Macro)**: 농촌 광역 셀 환경
  - 개방된 공간, 넓은 커버리지
  - 거리 범위: 수 킬로미터
  - 지형: 평지, 언덕

### **Indoor Environments (실내 환경)**
- **InH (Indoor Hotspot)**: 실내 핫스팟
  - 사무실, 쇼핑몰, 공항 등
  - 높은 사용자 밀도
- **InF (Indoor Factory)**: 실내 공장
  - 제조업 환경, 기계 장비
  - 금속 구조물로 인한 반사

### **Outdoor Environments (야외 환경)**
- **UMa + UMi + RMa**: 모든 야외 환경 통합
  - 도시 + 농촌 야외 환경

## 🔬 **실험 설계**

### **방안 1: Urban-Rural Transfer**

#### **Scenario A: Urban → Rural**
```yaml
Base Model: Urban Environments
- Dataset: UMa_Los, UMa_Nlos, UMi_Los, UMi_Nlos
- 특성: 고밀도 건물, 복잡한 전파 환경
- 학습: 200k iterations

Transfer Model: Rural Environment
- Dataset: RMa_Los, RMa_Nlos  
- 특성: 개방된 공간, 단순한 전파 환경
- 전이학습: 30k iterations with LoRA
```

#### **Scenario B: Rural → Urban**
```yaml
Base Model: Rural Environment
- Dataset: RMa_Los, RMa_Nlos
- 특성: 개방된 공간, 장거리 전파
- 학습: 200k iterations

Transfer Model: Urban Environments
- Dataset: UMa_Los, UMa_Nlos, UMi_Los, UMi_Nlos
- 특성: 복잡한 건물 구조, 다중 경로
- 전이학습: 30k iterations with LoRA
```

### **방안 2: Indoor-Outdoor Transfer**

#### **Scenario C: Indoor → Outdoor**
```yaml
Base Model: Indoor Environments  
- Dataset: InH_Los, InH_Nlos, InF_Los, InF_Nlos
- 특성: 제한된 공간, 벽면 반사
- 학습: 200k iterations

Transfer Model: Outdoor Environments
- Dataset: UMa_Los, UMa_Nlos, UMi_Los, UMi_Nlos, RMa_Los, RMa_Nlos
- 특성: 개방된 공간, 장거리 전파
- 전이학습: 30k iterations with LoRA
```

#### **Scenario D: Outdoor → Indoor**
```yaml
Base Model: Outdoor Environments
- Dataset: UMa_Los, UMa_Nlos, UMi_Los, UMi_Nlos, RMa_Los, RMa_Nlos  
- 특성: 개방된 공간, 다양한 거리
- 학습: 200k iterations

Transfer Model: Indoor Environments
- Dataset: InH_Los, InH_Nlos, InF_Los, InF_Nlos
- 특성: 폐쇄된 공간, 근거리 통신
- 전이학습: 30k iterations with LoRA
```

## 📊 **예상 결과**

### **성능 예측**

| Scenario | Base Performance | After Transfer | Improvement |
|----------|------------------|----------------|-------------|
| Urban → Rural | -10 ~ -15 dB | -20 ~ -25 dB | **5-10 dB** |
| Rural → Urban | -12 ~ -17 dB | -22 ~ -27 dB | **5-10 dB** |
| Indoor → Outdoor | -8 ~ -13 dB | -18 ~ -23 dB | **5-10 dB** |
| Outdoor → Indoor | -10 ~ -15 dB | -20 ~ -25 dB | **5-10 dB** |

### **LoRA 효율성**
- **파라미터 수**: 전체 모델의 ~4% (26,624개)
- **학습 시간**: 200k → 30k (85% 단축)
- **성능 향상**: 5-10 dB (기존 실험 대비 10-50배)

## 🚀 **실행 계획**

### **Phase 1: Base Model Training (현재)**
```bash
# 4개 Base 모델 동시 학습
python engine_v4_urban_base.py &
python engine_v4_rural_base.py &  
python engine_v4_indoor_base.py &
python engine_v4_outdoor_base.py &
```

**예상 소요시간**: 2-3시간 (200k iterations)

### **Phase 2: Transfer Learning (Base 완료 후)**
```bash
# 4개 Transfer 모델 동시 학습
python Transfer_v4_Urban_to_Rural.py &
python Transfer_v4_Rural_to_Urban.py &
python Transfer_v4_Indoor_to_Outdoor.py &
python Transfer_v4_Outdoor_to_Indoor.py &
```

**예상 소요시간**: 30-45분 (30k iterations)

### **Phase 3: Performance Evaluation**
```bash
# 성능 비교 및 분석
python cross_domain_comparison.py
```

## 📈 **기대 효과**

### **학술적 기여**
1. **진정한 Transfer Learning 검증**: Domain shift 상황에서의 LoRA 효과
2. **실용적 시나리오 제시**: 실제 무선 통신 시스템 배포 시나리오
3. **Parameter Efficiency 입증**: 극소수 파라미터로 큰 성능 향상
4. **Cross-Domain Adaptation**: 다양한 환경 간 지식 전이 가능성

### **산업적 가치**
1. **비용 효율적 배포**: 기존 모델을 새 환경에 빠르게 적응
2. **리소스 절약**: 전체 재학습 없이 특정 환경 최적화
3. **실시간 적응**: 환경 변화에 따른 모델 업데이트
4. **확장성**: 새로운 환경 추가 시 빠른 적응

## 📝 **논문 구성 (예상)**

### **Title**
"Cross-Domain Transfer Learning for Channel Estimation: From Urban to Rural, Indoor to Outdoor via LoRA"

### **Abstract**
- Problem: 환경별 채널 추정 모델의 도메인 특화 필요성
- Method: LoRA 기반 cross-domain transfer learning
- Results: 4% 파라미터로 5-10 dB 성능 향상
- Impact: 실용적 무선 시스템 배포 솔루션

### **Key Contributions**
1. 무선 채널 추정에서 첫 번째 cross-domain transfer learning 연구
2. Urban-Rural, Indoor-Outdoor 간 지식 전이 가능성 입증  
3. LoRA의 극소 파라미터로 큰 성능 향상 달성
4. 실제 배포 시나리오에 적용 가능한 실용적 솔루션

## ⚠️ **주의사항**

### **리소스 관리**
- 4개 모델 동시 학습 시 GPU 메모리 모니터링 필요
- 각 모델별 WandB 프로젝트 분리로 로그 관리
- 디스크 공간 확보 (모델당 ~1GB)

### **실험 검증**
- Base 모델이 타겟 도메인에서 실제로 낮은 성능을 보이는지 확인
- Transfer 후 성능 향상이 통계적으로 유의한지 검증
- 다양한 테스트 시나리오에서 일관된 결과 확인

## 📅 **일정**

| 단계 | 작업 | 소요시간 | 상태 |
|------|------|----------|------|
| 1 | Base Model Config/Script 생성 | 30분 | 진행중 |
| 2 | Base Model 학습 (4개 동시) | 2-3시간 | 대기 |
| 3 | Transfer Config/Script 생성 | 1시간 | 대기 |
| 4 | Transfer 학습 (4개 동시) | 45분 | 대기 |
| 5 | 성능 분석 및 비교 | 1시간 | 대기 |
| 6 | 결과 정리 및 문서화 | 1시간 | 대기 |

**총 예상 소요시간**: 6-8시간

---

*Last updated: 2025-08-17*  
*Experiment Status: Phase 1 - Base Model Preparation*