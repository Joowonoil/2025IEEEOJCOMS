# 훈련 가이드 - 실행 및 설정

## 🚀 빠른 시작

### 환경 설정
```bash
# Vast AI 자동 설치
curl -sSL https://raw.githubusercontent.com/joowonoil/channel-estimation/main/setup_vast_ai.sh | bash

# Docker 환경
docker pull joowonoil/channel-estimation-env:latest
docker run --gpus all -it joowonoil/channel-estimation-env:latest
```

### 필수 패키지
```
pytorch==2.4.1
transformers==4.36.0
peft==0.7.0  # LoRA support
tensorrt
wandb
```

## 📋 훈련 워크플로우

### 1단계: 베이스 모델 훈련

#### v3 Adapter 모델
```bash
# config_v3.yaml 설정
python engine_v3.py
# 출력: saved_model/Large_estimator_v3_base_final.pt
```

#### v4 LoRA 모델
```bash
# config.yaml 설정
python engine_v4.py
# 출력: saved_model/Large_estimator_v4_base.pt
```

### 2단계: 전이학습

#### InF 환경
```bash
# v3 Adapter
python Transfer_v3_InF.py
# 출력: Large_estimator_v3_to_InF_adapter.pt

# v4 LoRA
python Transfer_v4_InF.py
# 출력: Large_estimator_v4_to_InF_*.pt
```

#### RMa 환경
```bash
# v3 Adapter
python Transfer_v3_RMa.py

# v4 LoRA
python Transfer_v4_RMa.py
```

### 3단계: 성능 테스트
```bash
# 모델 비교
python simple_model_test.py
python v3_adapter_comparison.py
```

## ⚙️ 설정 파일 구조

### 베이스 모델 설정
```yaml
# config_v3.yaml / config.yaml
dataset:
  channel_type: ["InF_Los", "InF_Nlos", ...]  # 모든 채널 타입
  batch_size: 32
  distance_range: [10.0, 500.0]  # 넓은 거리 범위

training:
  lr: 0.0001                      # 베이스 학습률
  num_iter: 100000                # 충분한 훈련
  optimizer: 'Adam'
  use_scheduler: true             # Cosine annealing
  saved_model_name: 'Large_estimator_v3_base_final'

ch_estimation:
  transformer:
    num_layers: 4
    d_model: 128
    n_head: 8
    dim_feedforward: 1024
```

### 전이학습 설정
```yaml
# config_transfer_v3_InF.yaml (Adapter)
dataset:
  channel_type: ["InF_Los_50000", "InF_Nlos_50000"]
  distance_range: [40.0, 60.0]   # 특정 환경 범위

training:
  lr: 0.00001                     # 낮은 학습률
  num_iter: 200000
  pretrained_model_name: 'Large_estimator_v3_base_final'
  num_freeze_layers: 4            # 모든 레이어 동결

ch_estimation:
  adapter:
    bottleneck_dim: 256
    dropout: 0.1
```

```yaml
# config_transfer_v4_InF.yaml (LoRA)
ch_estimation:
  peft:
    peft_type: LORA
    r: 8                          # LoRA rank
    lora_alpha: 8                 # 스케일링
    target_modules: ["mha_q_proj", "mha_k_proj", "mha_v_proj"]
    lora_dropout: 0.1
```

## 🔄 훈련 파라미터 가이드

### 학습률 선택
| 단계 | v3 Adapter | v4 LoRA |
|------|------------|---------|
| 베이스 훈련 | 0.0001 | 0.0001 |
| 전이학습 | 0.00001 | 0.00001 |
| Fine-tuning | 0.000001 | 0.000001 |

### 배치 크기 권장
| GPU Memory | Batch Size | 설명 |
|------------|------------|------|
| 8GB | 16 | 기본 훈련 |
| 16GB | 32 | 권장 설정 |
| 24GB+ | 64 | 빠른 수렴 |

### Early Stopping 설정
```yaml
training:
  use_early_stopping: true
  patience: 500              # 개선 없이 대기할 스텝
  delta: 0.0001             # 최소 개선량
  checkpoint_path: 'saved_model/checkpoint.pt'
```

## 📊 모니터링

### Weights & Biases 설정
```yaml
training:
  use_wandb: true
  wandb_proj: 'DNN_channel_estimation'
```

### 로깅 항목
- `train/ch_nmse`: 채널 추정 손실
- `train/lr`: 현재 학습률
- `val/ch_nmse`: 검증 손실
- `train/grad_norm`: 그래디언트 노름

## 🛠️ 트러블슈팅

### CUDA 메모리 부족
```yaml
# 배치 크기 감소
batch_size: 16  # 32 → 16

# Gradient accumulation
accumulation_steps: 2
```

### 수렴 문제
```python
# 학습률 조정
lr: 0.00005  # 증가 또는 감소

# Gradient clipping
max_norm: 0.5  # 1.0 → 0.5
```

### 모델 로드 오류
```python
# strict=False로 부분 로드
model.load_state_dict(checkpoint, strict=False)

# 키 매핑 확인
for key in checkpoint.keys():
    print(key)
```

## 💡 최적화 팁

### 1. 데이터 로더 최적화
```python
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # CPU 코어 수에 맞게
    pin_memory=True,    # GPU 전송 가속
    persistent_workers=True
)
```

### 2. Mixed Precision Training
```python
# 자동 혼합 정밀도
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
```

### 3. 체크포인트 저장
```python
# 주기적 저장
if iteration % save_step == 0:
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_{iteration}.pt')
```

## 🚀 프로덕션 배포

### TensorRT 변환
```bash
python tensorrt_conversion_v4.py
# 출력: *.engine 파일
```

### ONNX 내보내기
```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11,
    do_constant_folding=True
)
```

### 추론 서버
```python
# FastAPI 예제
from fastapi import FastAPI
app = FastAPI()

@app.post("/estimate")
async def estimate_channel(data: np.ndarray):
    with torch.no_grad():
        result = model(torch.from_numpy(data))
    return result.numpy()
```