# 새 노트북 개발 환경 설정 가이드 (RTX 5070)

이 문서는 RTX 5070 GPU가 탑재된 새 노트북에서 DNN Channel Estimation 프로젝트를 실행하기 위한 환경 설정 가이드입니다.

## 1. 시스템 요구사항

- **GPU**: NVIDIA RTX 5070 (Laptop)
- **OS**: Windows 11 권장
- **RAM**: 16GB 이상 권장
- **Storage**: SSD 50GB 이상 여유 공간

## 2. 기본 소프트웨어 설치

### 2.1 Python 설치
1. Python 3.11.6 다운로드 및 설치
   - https://www.python.org/downloads/release/python-3116/
   - 설치 시 "Add Python to PATH" 체크 필수
   - 설치 확인: `python --version`

### 2.2 PyCharm 설치
1. PyCharm Professional 또는 Community Edition 다운로드
   - https://www.jetbrains.com/pycharm/download/
   - Professional 버전 추천 (학생 라이선스 무료)

### 2.3 Git 설치
1. Git for Windows 다운로드 및 설치
   - https://git-scm.com/download/win
   - 기본 설정으로 설치
   - 설치 확인: `git --version`

## 3. NVIDIA 드라이버 및 CUDA 설정

### ⚠️ 중요: RTX 5070 호환성 정보
- **RTX 5070은 Blackwell 아키텍처(sm_120)로 CUDA 12.8 이상이 필요합니다**
- **현재 프로젝트의 PyTorch 2.4.1 + CUDA 12.1은 RTX 5070과 호환되지 않습니다**
- **PyTorch nightly 빌드 + CUDA 12.8 설치가 필수입니다**

### 3.1 NVIDIA 드라이버 설치
1. NVIDIA GeForce Experience 또는 수동 다운로드
   - https://www.nvidia.com/download/index.aspx
   - RTX 5070 Laptop GPU 선택
   - **드라이버 버전 570 이상 필수** (2025년 기준)
   - Open Kernel 드라이버 권장 (nvidia-driver-570-open)

### 3.2 CUDA Toolkit 설치 (RTX 5070 필수)
1. **CUDA 12.8 이상 설치** (RTX 5070 최소 요구사항)
   - https://developer.nvidia.com/cuda-downloads
   - Windows > x86_64 > 11 > exe (local) 선택
   - 기본 설정으로 설치
   - 환경 변수 자동 설정됨
   
   **주의**: CUDA 12.1은 RTX 5070을 지원하지 않음 (sm_120 미지원)

### 3.3 cuDNN 설치
1. NVIDIA Developer 계정 생성 (무료)
2. cuDNN 9.x for CUDA 12.8 다운로드
   - https://developer.nvidia.com/cudnn
   - Windows용 ZIP 파일 다운로드
3. 압축 해제 후 파일 복사:
   ```
   cudnn-windows-x86_64-9.x.x_cuda12-archive\bin\*.dll → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\
   cudnn-windows-x86_64-9.x.x_cuda12-archive\include\*.h → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include\
   cudnn-windows-x86_64-9.x.x_cuda12-archive\lib\x64\*.lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64\
   ```

### 3.4 환경 변수 확인
시스템 환경 변수에 다음 경로가 포함되어 있는지 확인:
- `CUDA_PATH`: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
- `PATH`에 포함:
  - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
  - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp

## 4. 프로젝트 설정

### 4.1 프로젝트 클론
```bash
git clone [your-repository-url]
cd DNN_channel_estimation_training
```

### 4.2 가상 환경 생성
```bash
# 가상 환경 생성
python -m venv venv

# 활성화 (Windows)
venv\Scripts\activate

# 활성화 (PowerShell에서 오류 시)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\activate
```

### 4.3 PyTorch 및 의존성 설치

#### ⚠️ RTX 5070용 PyTorch 설치 (필수)
```bash
# PyTorch Nightly with CUDA 12.8 설치 (RTX 5070 필수)
# 약 3GB 다운로드 필요
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'sm_120 지원: {torch.cuda.get_device_capability(0)}')"
```

#### 나머지 패키지 설치
```bash
# 기본 패키지 설치
pip install numpy>=1.26.0
pip install scipy>=1.14.0
pip install matplotlib>=3.9.0
pip install PyYAML>=6.0.2
pip install einops>=0.8.0
pip install transformers>=4.44.0
pip install peft>=0.7.0
pip install wandb>=0.17.8
pip install torchsummary>=1.5.1
pip install tqdm>=4.66.5

# TensorRT 관련 (선택사항 - 고급 최적화용)
# RTX 5070의 경우 최신 버전 권장
pip install tensorrt>=10.1.0
pip install onnx>=1.14.0
pip install onnxruntime-gpu>=1.18.0
pip install pycuda>=2024.1.0
pip install logzero>=1.7.0
```

#### ⚠️ 호환성 주의사항
- RTX 5070은 stable PyTorch 버전을 지원하지 않음
- 반드시 PyTorch nightly + CUDA 12.8 조합 사용
- 프로젝트 코드가 nightly 버전과 호환되는지 테스트 필요

### 4.4 설치 확인
```python
# Python 실행 후 다음 코드로 확인
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## 5. PyCharm 프로젝트 설정

### 5.1 프로젝트 열기
1. PyCharm 실행
2. File > Open > 프로젝트 폴더 선택

### 5.2 인터프리터 설정
1. File > Settings (Ctrl+Alt+S)
2. Project > Python Interpreter
3. 톱니바퀴 아이콘 > Add
4. Existing environment 선택
5. `...\DNN_channel_estimation_training\venv\Scripts\python.exe` 선택

### 5.3 실행 구성 추가
1. Run > Edit Configurations
2. + 버튼 > Python
3. 주요 스크립트 설정:
   - Name: Transfer_v4_RMa
   - Script path: Transfer_v4_RMa.py
   - Working directory: 프로젝트 루트 폴더

## 6. 테스트 실행

### 6.1 간단한 테스트
```bash
# 가상 환경 활성화 후
python -c "import torch; print(torch.cuda.is_available())"
```

### 6.2 프로젝트 테스트
```bash
# 작은 데이터셋으로 테스트
python Transfer_v4_RMa.py
```

## 7. 문제 해결

### 7.1 RTX 5070 관련 오류
- **"sm_120 is not compatible"** 오류:
  - PyTorch stable 버전 제거 후 nightly 재설치
  - `pip uninstall torch torchvision torchaudio`
  - `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`

- **"no kernel image is available"** 오류:
  - CUDA 12.8 설치 확인
  - 드라이버 570 이상 확인

### 7.2 CUDA 오류
- "CUDA out of memory" 에러 시:
  - batch_size 줄이기
  - GPU 메모리 정리: `torch.cuda.empty_cache()`

### 7.3 cuDNN 오류
- cuDNN 버전 불일치 시:
  - CUDA 12.8용 cuDNN 9.x 재설치
  - 환경 변수 확인

### 7.4 PyCharm에서 GPU 인식 안 될 때
- PyCharm 재시작
- 시스템 환경 변수 확인
- PyCharm 터미널에서 `nvidia-smi` 실행 확인

## 8. 성능 최적화 팁

### 8.1 RTX 5070 특화 설정
```python
# Mixed Precision Training 활용
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Flash Attention 활용 (RTX 5070 지원)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 8.2 메모리 최적화
- RTX 5070의 VRAM에 맞춰 batch_size 조정
- Gradient checkpointing 사용 고려

## 9. 추가 도구 (선택사항)

### 9.1 NVIDIA Tools
- NVIDIA Nsight Systems (프로파일링)
- NVIDIA Visual Profiler

### 9.2 모니터링 도구
- GPU-Z
- HWiNFO64
- MSI Afterburner

## 10. 주의사항

1. **드라이버 업데이트**: RTX 5070은 최신 GPU이므로 드라이버를 자주 업데이트
2. **온도 관리**: 노트북 환경이므로 쿨링 패드 사용 권장
3. **전원 설정**: Windows 전원 옵션을 "고성능"으로 설정
4. **백업**: 코드 변경 전 Git commit 습관화

## 문의사항
프로젝트 관련 문의는 GitHub Issues 또는 팀 채널로 연락주세요.