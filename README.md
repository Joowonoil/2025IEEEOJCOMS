# DNN Channel Estimation Training

> 5G/6G 통신을 위한 딥러닝 기반 DMRS 채널 추정 전이학습 시스템

## 📚 문서 구조

모든 상세 문서는 **[docs/](./docs/)** 폴더에서 확인하세요:

- **[📖 프로젝트 개요](./docs/README.md)** - 전체 프로젝트 소개
- **[🔧 기술 가이드](./docs/TECHNICAL_GUIDE.md)** - 아키텍처 및 모델 구조  
- **[🎓 훈련 가이드](./docs/TRAINING_GUIDE.md)** - 실행 및 설정 방법
- **[⚖️ 모델 비교](./docs/MODELS_COMPARISON.md)** - v3 Adapter vs v4 LoRA

## 🚀 빠른 시작

### 환경 설정
```bash
# Vast AI 자동 설치
curl -sSL https://raw.githubusercontent.com/joowonoil/channel-estimation/main/setup_vast_ai.sh | bash

# 또는 Docker 환경
docker pull joowonoil/channel-estimation-env:latest
docker run --gpus all -it joowonoil/channel-estimation-env:latest
```

### 훈련 실행
```bash
# 1단계: 베이스 모델 훈련
python engine_v4.py  # LoRA 지원 (권장)
python engine_v3.py  # Adapter 지원

# 2단계: 전이학습 (InF 환경)
python Transfer_v4_InF.py  # v4 LoRA 방식
python Transfer_v3_InF.py  # v3 Adapter 방식

# 3단계: 성능 테스트
python simple_model_test.py
```

## 🎯 핵심 기능

### 전이학습 방법론
- **v4 LoRA**: Low-Rank Adaptation, 1% 추가 파라미터
- **v3 Adapter**: 병렬 모듈 방식, 5% 추가 파라미터
- **지원 환경**: InF, RMa, InH, UMa, UMi

### 기술 스택
- **PyTorch 2.4.1** + CUDA 12.1
- **Transformers** + PEFT (LoRA)
- **TensorRT** 최적화
- **Weights & Biases** 실험 관리

## 📈 성능 비교

| 메트릭 | v3 Adapter | v4 LoRA | 개선율 |
|--------|------------|---------|--------|
| **InF NMSE** | -25.2 dB | **-26.4 dB** | +1.2 dB |
| **RMa NMSE** | -24.8 dB | **-25.9 dB** | +1.1 dB |
| **파라미터** | 524K | **98K** | 81% 감소 |
| **추론 속도** | 14.8 ms | **12.3 ms** | 17% 향상 |
| **메모리 사용** | 8.2 GB | **6.8 GB** | 17% 절약 |

## 🏗️ 프로젝트 구조

```
DNN_channel_estimation_training/
├── 📄 README.md               # 이 파일
├── 📁 docs/                   # 📚 상세 문서
│   ├── README.md             # 프로젝트 개요
│   ├── TECHNICAL_GUIDE.md    # 기술 가이드
│   ├── TRAINING_GUIDE.md     # 훈련 가이드
│   └── MODELS_COMPARISON.md  # 모델 비교
├── 🎯 실행 파일
│   ├── engine_v3.py          # v3 Adapter 베이스
│   ├── engine_v4.py          # v4 LoRA 베이스
│   ├── Transfer_v3_*.py      # v3 전이학습
│   └── Transfer_v4_*.py      # v4 전이학습
├── 🧠 model/                 # 모델 구현
├── ⚙️ config/                # 설정 파일
└── 📊 dataset/               # 채널 데이터
```

## 💡 권장 사용법

### v4 LoRA 선택 (권장)
```bash
# 최고 효율성과 성능
python engine_v4.py
python Transfer_v4_InF.py
```
- ✅ 최소 파라미터로 최고 성능
- ✅ 빠른 수렴과 메모리 효율
- ✅ 실시간 추론 최적화

### v3 Adapter 선택
```bash
# 모듈식 확장성 중요시
python engine_v3.py  
python Transfer_v3_InF.py
```
- ✅ 다중 도메인 동시 지원
- ✅ 모듈별 독립적 관리
- ✅ 구현 단순성

## 🛠️ 주요 명령어

```bash
# TensorRT 최적화
python tensorrt_conversion_v4.py

# 모델 성능 비교
python v3_adapter_comparison.py
python simple_model_test.py

# 체크포인트 확인
ls saved_model/
```


## 📄 라이선스

MIT License

---

자세한 기술 정보는 **[docs/](./docs/)** 폴더의 문서들을 참조.