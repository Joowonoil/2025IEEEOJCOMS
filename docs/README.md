# 프로젝트 문서 - DNN 채널 추정 시스템

> 5G/6G 통신을 위한 딥러닝 기반 DMRS 채널 추정 전이학습 시스템

## 📚 문서 구조

### 핵심 문서
- **[TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md)** - 기술 아키텍처 및 모델 구조
- **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** - 훈련 실행 및 설정 가이드
- **[MODELS_COMPARISON.md](./MODELS_COMPARISON.md)** - v3 Adapter vs v4 LoRA 비교

## 🎯 프로젝트 개요

### 핵심 기술
- **전이학습**: 베이스 모델에서 특정 환경으로 적응
- **v3 Adapter**: 병렬 모듈 방식 (~5% 추가 파라미터)
- **v4 LoRA**: Low-Rank Adaptation (~1% 추가 파라미터)
- **DMRS**: 5G/6G 참조신호 기반 채널 추정

### 지원 환경
- **InF**: Indoor Factory (50K samples)
- **RMa**: Rural Macro (50K samples)
- **InH**: Indoor Hotspot
- **UMa/UMi**: Urban Macro/Micro

## 🏗️ 시스템 구조

```
DNN_channel_estimation_training/
├── 실행 파일
│   ├── engine_v3.py           # v3 Adapter 베이스 모델
│   ├── engine_v4.py           # v4 LoRA 베이스 모델
│   ├── Transfer_v3_*.py       # Adapter 전이학습
│   └── Transfer_v4_*.py       # LoRA 전이학습
├── model/                     # 모델 아키텍처
├── config/                    # 설정 파일
├── dataset/                   # 채널 데이터
└── docs/                      # 문서 (현재 위치)
```

## 📈 성능 요약

| 메트릭 | v3 Adapter | v4 LoRA |
|--------|------------|---------|
| **InF NMSE** | -25.2 dB | -26.4 dB |
| **RMa NMSE** | -24.8 dB | -25.9 dB |
| **파라미터** | 524K (~5%) | 98K (~1%) |
| **추론 속도** | 14.8 ms | 12.3 ms |
| **메모리** | 8.2 GB | 6.8 GB |

## 🚀 빠른 시작

### 1. 환경 설정
```bash
git clone https://github.com/joowonoil/channel-estimation-training.git
cd channel-estimation-training
```

### 2. 베이스 모델 훈련
```bash
python engine_v4.py  # LoRA 지원 (권장)
# 또는
python engine_v3.py  # Adapter 지원
```

### 3. 전이학습
```bash
# InF 환경 적응
python Transfer_v4_InF.py  # LoRA
python Transfer_v3_InF.py  # Adapter
```

## 📖 자세한 가이드

### 🔧 기술 이해
**[TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md)**에서 확인:
- 시스템 아키텍처 상세
- v3/v4 모델 구조
- 데이터 처리 파이프라인
- 최적화 기법

### 🎓 훈련 실행
**[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)**에서 확인:
- 단계별 훈련 과정
- 설정 파일 구성
- 트러블슈팅 가이드
- 성능 모니터링

### ⚖️ 모델 선택
**[MODELS_COMPARISON.md](./MODELS_COMPARISON.md)**에서 확인:
- v3 vs v4 상세 비교
- 성능 벤치마크
- 사용 시나리오별 권장사항
- 마이그레이션 가이드

## 🔬 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **프레임워크** | PyTorch 2.4.1, CUDA 12.1 |
| **전이학습** | Adapter (v3), LoRA (v4) |
| **최적화** | TensorRT, ONNX |
| **실험 관리** | Weights & Biases |
| **배포** | Docker, Vast AI |

## 🎯 사용 가이드라인

### v4 LoRA 선택 시
- ✅ 최고의 파라미터 효율성 필요
- ✅ 빠른 수렴과 높은 성능 우선
- ✅ 메모리 제약이 있는 환경

### v3 Adapter 선택 시
- ✅ 모듈식 확장성 중요
- ✅ 다중 도메인 동시 지원
- ✅ 구현 단순성 우선

## 📄 참고사항

- 모든 실험은 RTX 4080 Super 기준으로 측정
- 성능은 InF/RMa 환경에서 검증
- Weights & Biases로 훈련 과정 모니터링

---

*최종 업데이트: 2025-08-13*