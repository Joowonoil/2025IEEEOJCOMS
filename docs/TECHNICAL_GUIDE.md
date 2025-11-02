# 기술 가이드 - DNN 채널 추정 시스템

## 🏗️ 시스템 아키텍처

### 프로젝트 구조
```
DNN_channel_estimation_training/
├── 실행 파일
│   ├── engine_v3.py           # v3 Adapter 베이스 모델
│   ├── engine_v4.py           # v4 LoRA 베이스 모델
│   ├── Transfer_v3_*.py       # Adapter 전이학습
│   └── Transfer_v4_*.py       # LoRA 전이학습
├── model/
│   ├── adapter.py             # Adapter 모듈 구현
│   ├── estimator_v3.py        # v3 채널 추정기
│   ├── estimator_v4.py        # v4 채널 추정기
│   ├── transformer_v3.py      # v3 Transformer
│   └── transformer_v4.py      # v4 Transformer
└── config/
    ├── config_v3.yaml         # v3 베이스 설정
    ├── config_v4.yaml         # v4 베이스 설정
    └── config_transfer_*.yaml # 전이학습 설정
```

## 🧠 모델 아키텍처

### 공통 구조
- **입력**: (batch, 14_symbols, 3072_subcarriers, 2_channels)
- **Transformer**: 4 layers, 8 heads, d_model=128
- **FFN**: dim_feedforward=1024
- **출력**: 추정 채널 응답 + 보상된 수신 신호

### v3 Adapter 아키텍처
```python
class Adapter(nn.Module):
    """병렬 Adapter 모듈 - 각 Transformer 레이어에 추가"""
    def __init__(self, input_dim=128, bottleneck_dim=64, dropout=0.1):
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)  # Down-projection
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck_dim, input_dim)  # Up-projection
    
    def forward(self, x):
        residual = x
        x = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return residual + x  # Residual connection
```

**특징**:
- 추가 파라미터: ~1.3% (131,072개)
- 모듈식 확장 가능
- 추론 시 추가 연산 필요

### v4 LoRA 아키텍처
```python
# 분리된 Projection Layers (LoRA 타겟)
self.mha_q_proj = Linear(d_model, d_model)  # Query
self.mha_k_proj = Linear(d_model, d_model)  # Key  
self.mha_v_proj = Linear(d_model, d_model)  # Value
self.out_proj = Linear(d_model, d_model)    # Output
self.ffnn_linear1 = Linear(d_model, dim_feedforward)
self.ffnn_linear2 = Linear(dim_feedforward, d_model)
```

**LoRA 분해**:
```
W' = W + ΔW = W + BA
- B: (d×r), A: (r×k) where r=8
- 파라미터: r×(d+k) << d×k
```

**특징**:
- 추가 파라미터: ~1% (98,304개)
- 병합 후 원본과 동일한 추론 속도
- 메모리 효율적

## 📊 데이터 처리

### 채널 데이터 구조
```python
# DMRS 참조 신호 설정
ref_conf_dict = {
    'dmrs': [0, 3072, 6]  # [시작, 끝, 간격]
}

# 채널 타입
channel_types = [
    "InF_Los", "InF_Nlos",    # Indoor Factory
    "RMa_Los", "RMa_Nlos",    # Rural Macro
    "InH_Los", "InH_Nlos",    # Indoor Hotspot
    "UMa_Los", "UMa_Nlos",    # Urban Macro
    "UMi_Los", "UMi_Nlos"     # Urban Micro
]
```

### 데이터 전처리
1. **PDP 로드**: `.mat` 파일에서 Power Delay Profile 읽기
2. **복소수 분리**: 실수부/허수부로 분리 (2채널)
3. **정규화**: 채널별 정규화
4. **DMRS 추출**: 참조 신호 위치에서 샘플링

## ⚡ 최적화 기법

### 메모리 최적화
```python
# Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### 계산 최적화
```python
# PyTorch 2.0 Compile
model = torch.compile(model)

# Efficient Attention
F.scaled_dot_product_attention(q, k, v)
```

### TensorRT 변환
```python
# TensorRT 최적화 (추론용)
import torch_tensorrt

trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch.randn(batch_size, 14, 3072, 2)],
    enabled_precisions={torch.float16}
)
```

## 🔧 핵심 클래스

### Engine 클래스 구조
```python
class Engine_v4:
    def __init__(self, conf_file):
        self._conf = yaml.safe_load(conf_file)
        self._estimator = Estimator_v4(conf_file)
        self._dataset, self._dataloader = get_dataset_and_dataloader()
        self.set_optimizer()
    
    def train(self):
        for data in self._dataloader:
            # Forward pass
            ch_est, _ = self._estimator(rx_signal)
            # NMSE loss
            loss = self.calculate_nmse_loss(ch_est, ch_true)
            # Backward pass
            loss.backward()
            self._optimizer.step()
```

### TransferLearning 클래스
```python
class TransferLearningEngine:
    def load_model(self):
        # v3: Adapter 추가
        self._estimator = Estimator_v3(conf_file)
        # 베이스 파라미터 동결
        for name, param in self._estimator.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
        
        # v4: LoRA 적용
        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(r=8, lora_alpha=8, ...)
        self._estimator = get_peft_model(self._estimator, lora_config)
```

## 📈 성능 메트릭

### NMSE (Normalized Mean Square Error)
```python
def calculate_nmse(pred, target):
    mse = torch.mean(torch.abs(pred - target) ** 2)
    norm = torch.mean(torch.abs(target) ** 2)
    nmse = mse / norm
    return 10 * torch.log10(nmse)  # dB scale
```

### 평가 지표
- **채널 추정 정확도**: NMSE in dB
- **수렴 속도**: iterations to convergence
- **추론 시간**: ms per batch
- **메모리 사용량**: GB (training/inference)