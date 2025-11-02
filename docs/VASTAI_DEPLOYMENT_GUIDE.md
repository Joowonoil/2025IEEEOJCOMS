# Vast.ai 병렬 실행 가이드
Pareto Frontier 실험을 4개의 Vast.ai 인스턴스에서 병렬로 실행하는 방법

**작성일**: 2025-11-02
**예상 시간**: 35-70시간 (병렬 실행)
**예상 비용**: $60-100 (RTX 3090/4090 기준)

---

## 📋 목차
1. [사전 준비](#1-사전-준비)
2. [Vast.ai 인스턴스 설정](#2-vastai-인스턴스-설정)
3. [각 실험 실행](#3-각-실험-실행)
4. [모니터링](#4-모니터링)
5. [결과 수집](#5-결과-수집)
6. [트러블슈팅](#6-트러블슈팅)

---

## 1. 사전 준비

### 1.1 GitHub에 코드 푸시

```bash
cd C:\Users\Ramster\Documents\Files\SKKU\Project\DNN_channel_estimation_training
git add .
git commit -m "Pareto experiments ready for Vast.ai"
git push origin main
```

**GitHub 레포지토리 URL 복사** (나중에 사용)

### 1.2 데이터셋 준비

필요한 파일:
- `saved_model/Large_estimator_v3_base_extended_final.pt` (~1.7GB)
- `saved_model/Large_estimator_v4_base_extended_final.pt` (~1.7GB)
- `dataset/PDP_processed/` 폴더

**옵션 A: Google Drive (추천)**

1. Google Drive에 폴더 생성
2. 파일 업로드
3. 공유 링크 생성 (누구나 다운로드 가능)
4. 파일 ID 추출:
   ```
   https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
   ```

**옵션 B: Vast.ai 스토리지**
- Vast.ai 계정에 직접 업로드 (빠르지만 비쌈)

**옵션 C: SCP 직접 전송**
- 각 인스턴스에 개별 업로드 (시간 소요)

### 1.3 WandB API Key (선택)

```bash
# WandB 로그인 후
wandb login --relogin
# API key 복사
```

---

## 2. Vast.ai 인스턴스 설정

### 2.1 인스턴스 스펙 선택

**추천 GPU:**
- RTX 3090 (24GB VRAM): ~$0.3/hour
- RTX 4090 (24GB VRAM): ~$0.5/hour

**추천 설정:**
- **Image**: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`
- **Disk**: 100GB+
- **RAM**: 32GB+
- **Upload Speed**: 100Mbps+ (데이터 다운로드 속도)

**왜 Docker 이미지를 직접 만들지 않나?**
- Vast.ai는 이미 Docker 기반으로 실행됨
- PyTorch 이미지에 모든 것이 포함됨
- 추가 패키지는 `pip install`로 설치
- 직접 Docker 만들면 빌드 시간 낭비 + 복잡도 증가

### 2.2 인스턴스 4개 생성

각 실험마다 하나씩:
1. **Instance 1**: Prompt (15 runs, ~30-60h)
2. **Instance 2**: LoRA (20 runs, ~40-80h)
3. **Instance 3**: Hybrid (15 runs, ~30-60h)
4. **Instance 4**: Adapter (20 runs, ~40-80h)

**인스턴스 이름 설정:**
- `pareto-prompt`
- `pareto-lora`
- `pareto-hybrid`
- `pareto-adapter`

---

## 3. 각 실험 실행

### 3.1 기본 Setup (4개 인스턴스 모두 동일)

**Step 1: SSH 접속**
```bash
ssh root@X.X.X.X -p XXXXX
```

**Step 2: 코드 클론**
```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/YOUR_REPO DNN_channel_estimation_training
cd DNN_channel_estimation_training
```

**Step 3: 의존성 설치**
```bash
# Vast.ai용 간소화된 requirements 사용 (중요!)
pip install -r requirements_vastai.txt

# 또는 수동 설치
pip install transformers peft wandb einops pyyaml scipy h5py gdown torch-tensorrt
```

**주의**: `requirements.txt`는 Windows 환경용이므로 Vast.ai(Linux)에서는 `requirements_vastai.txt` 사용!

**Step 4: CUDA 확인**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

**Step 5: 데이터셋 다운로드**

**방법 1: 로컬에서 압축 후 Jupyter Lab 업로드 (강력 추천!)**

Google Drive gdown이 권한 문제로 실패할 수 있으므로, 직접 업로드가 가장 확실합니다.

*로컬 컴퓨터 (Windows PowerShell):*
```powershell
cd C:\Users\YOUR_PATH\DNN_channel_estimation_training

# 압축
Compress-Archive -Path dataset -DestinationPath dataset.zip -Force
Compress-Archive -Path saved_model -DestinationPath saved_model.zip -Force
```

*Vast.ai Jupyter Lab에서:*
1. Jupyter Terminal → Launch Application
2. 좌측 파일 브라우저
3. Upload 버튼 (↑) 클릭
4. `dataset.zip`, `saved_model.zip` 선택
5. 업로드 대기 (10-30분)

*Vast.ai 터미널에서 압축 해제:*
```bash
cd /workspace/DNN_channel_estimation_training
unzip dataset.zip
unzip saved_model.zip

# 확인
ls dataset/PDP_processed/*.mat | head -5
ls saved_model/Large_estimator_v3_base_final.pt
ls saved_model/Large_estimator_v4_base_final.pt
```

**방법 2: Google Drive (gdown) - 권한 문제 가능**

```bash
# Google Drive 폴더 다운로드
gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID

# 실패 시 wget으로 개별 파일
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=FILE_ID' -O file.zip
unzip file.zip
```

**주의**: Google Drive 공유 설정을 "링크가 있는 모든 사용자"로 변경해도 gdown이 실패할 수 있습니다. **방법 1 추천!**

**Step 6: WandB 로그인 (선택)**
```bash
wandb login YOUR_API_KEY
```

### 3.2 실험별 실행 명령

**추천 방법: nohup 사용 (tmux보다 간단)**

**Instance 1: Adapter**
```bash
cd /workspace/2025IEEEOJCOMS

# 백그라운드 실행
nohup python Transfer_Pareto_Adapter.py > adapter.log 2>&1 &

# 로그 실시간 확인
tail -f adapter.log

# GPU 사용률 확인
watch -n 1 nvidia-smi

# Ctrl+C로 로그 확인 중단 (실험은 계속 실행됨)
```

**대안: tmux 사용**
```bash
# tmux 세션 생성 (연결 끊겨도 계속 실행)
tmux new -s adapter

# 실행
python Transfer_Pareto_Adapter.py

# Ctrl+B, D로 detach (백그라운드 실행)
```

**Instance 2: LoRA**
```bash
tmux new -s lora
python Transfer_Pareto_LoRA.py
# Ctrl+B, D
```

**Instance 3: Hybrid**
```bash
tmux new -s hybrid
python Transfer_Pareto_Hybrid.py
# Ctrl+B, D
```

**Instance 4: Adapter**
```bash
tmux new -s adapter
python Transfer_Pareto_Adapter.py
# Ctrl+B, D
```

### 3.3 tmux 명령어

```bash
# 세션 재접속
tmux attach -t prompt

# 세션 목록 확인
tmux ls

# 세션 종료 (재접속 후)
exit

# detach (Ctrl+B, D)
```

---

## 4. 모니터링

### 4.1 로컬에서 WandB 모니터링

**프로젝트별 확인:**
- Prompt: `DNN_channel_estimation_InH_Prompt_Pareto` (외 4개)
- LoRA: `DNN_channel_estimation_InH_LoRA_Pareto` (외 4개)
- Hybrid: `DNN_channel_estimation_InH_Hybrid_Pareto` (외 4개)
- Adapter: `DNN_channel_estimation_InH_Adapter_Pareto` (외 4개)

총 20개 프로젝트 (4 methods × 5 scenarios)

### 4.2 인스턴스 직접 확인

```bash
# SSH 재접속
ssh root@X.X.X.X -p XXXXX

# tmux 세션 확인
tmux attach -t prompt

# GPU 사용률 확인
nvidia-smi

# 로그 확인 (출력 스크롤)
# tmux 안에서 Ctrl+B, [ 후 방향키
```

### 4.3 예상 소요 시간

| Method | Runs | Time/Run | Total Time |
|--------|------|----------|------------|
| Prompt | 15 | 2-4h | 30-60h |
| LoRA | 20 | 2-4h | 40-80h |
| Hybrid | 15 | 2-4h | 30-60h |
| Adapter | 20 | 2-4h | 40-80h |

**병렬 실행 시**: 최대 40-80시간 (LoRA/Adapter 기준)

---

## 5. 결과 수집

### 5.1 각 인스턴스에서 결과 압축

```bash
cd /workspace/DNN_channel_estimation_training

# Prompt 결과
tar -czf pareto_prompt_results.tar.gz saved_model/pareto/*prompt*

# LoRA 결과
tar -czf pareto_lora_results.tar.gz saved_model/pareto/*lora*

# Hybrid 결과
tar -czf pareto_hybrid_results.tar.gz saved_model/pareto/*hybrid*

# Adapter 결과
tar -czf pareto_adapter_results.tar.gz saved_model/pareto/*adapter*
```

### 5.2 로컬로 다운로드

```bash
# 로컬 터미널에서
scp -P PORT root@PROMPT_IP:/workspace/.../pareto_prompt_results.tar.gz .
scp -P PORT root@LORA_IP:/workspace/.../pareto_lora_results.tar.gz .
scp -P PORT root@HYBRID_IP:/workspace/.../pareto_hybrid_results.tar.gz .
scp -P PORT root@ADAPTER_IP:/workspace/.../pareto_adapter_results.tar.gz .
```

### 5.3 압축 해제 및 정리

```bash
# 로컬에서
tar -xzf pareto_prompt_results.tar.gz -C saved_model/pareto/
tar -xzf pareto_lora_results.tar.gz -C saved_model/pareto/
tar -xzf pareto_hybrid_results.tar.gz -C saved_model/pareto/
tar -xzf pareto_adapter_results.tar.gz -C saved_model/pareto/
```

---

## 6. 트러블슈팅

### 6.1 CUDA Out of Memory

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
```python
# config 파일에서 batch_size 줄이기
batch_size: 16  # 원래 32
```

### 6.2 데이터셋 다운로드 실패

**Google Drive 직접 다운로드 제한:**
```bash
# gdown이 안 되면 wget 사용
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILE_ID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILE_ID" -O filename && rm -rf /tmp/cookies.txt
```

### 6.3 WandB 로그인 안 됨

```bash
# 수동 로그인
wandb login

# 또는 config에서 비활성화
use_wandb: False
```

### 6.4 tmux 세션 끊김

```bash
# 세션 확인
tmux ls

# 재접속
tmux attach -t prompt

# 세션이 없으면 다시 시작
python Transfer_Pareto_Prompt.py
```

### 6.5 디스크 용량 부족

```bash
# 용량 확인
df -h

# 불필요한 파일 삭제
rm -rf /workspace/.cache
rm -rf /tmp/*

# 체크포인트만 남기고 삭제 (완료 후)
cd saved_model/pareto
rm *_iter_*.pt  # 중간 체크포인트 삭제
```

---

## 7. 비용 절감 팁

### 7.1 Interruptible 인스턴스 사용
- On-demand보다 50-70% 저렴
- 중단될 수 있으므로 checkpoint 저장 필수

### 7.2 완료 즉시 종료
```bash
# 스크립트 마지막에 자동 종료 추가
echo "shutdown -h now" >> run_script.sh
```

### 7.3 저렴한 GPU 선택
- RTX 3090: $0.2-0.4/hour
- 필터: "DLPerf > 80" + "Reliability > 0.95"

---

## 8. 체크리스트

### 실행 전
- [ ] GitHub에 코드 푸시 완료
- [ ] Google Drive에 모델 파일 업로드
- [ ] WandB API key 준비
- [ ] Vast.ai 계정에 크레딧 충전

### 각 인스턴스마다
- [ ] SSH 접속 확인
- [ ] 코드 클론 완료
- [ ] 의존성 설치 완료
- [ ] 데이터셋 다운로드 완료
- [ ] CUDA 작동 확인
- [ ] tmux 세션에서 실행 시작
- [ ] WandB에서 로그 확인

### 완료 후
- [ ] 4개 실험 모두 완료 확인
- [ ] 결과 파일 압축
- [ ] 로컬로 다운로드
- [ ] 인스턴스 종료 (비용 절감)
- [ ] WandB 로그 백업

---

## 9. 참고 명령어 모음

```bash
# 시스템 정보
nvidia-smi
df -h
free -h
top

# Python 환경
python --version
pip list | grep torch
pip list | grep peft

# Git
git pull
git status

# tmux
tmux new -s NAME
tmux attach -t NAME
tmux ls
tmux kill-session -t NAME

# 파일 전송
scp -P PORT local_file root@IP:/workspace/
scp -P PORT root@IP:/workspace/file local_path/

# 압축
tar -czf archive.tar.gz folder/
tar -xzf archive.tar.gz
```

---

## 10. 예상 결과

### 10.1 파일 구조
```
saved_model/pareto/
├── Large_estimator_v4_to_InH_prompt_len50.pt
├── Large_estimator_v4_to_InH_prompt_len50_iter_20000.pt
├── ... (총 420개 파일: 70 final + 350 checkpoints)
```

### 10.2 WandB 프로젝트
- 총 20개 프로젝트 (4 methods × 5 scenarios)
- 각 프로젝트당 3-4개 run
- 총 70개 runs

---

**다음 단계**: [PARETO_EXPERIMENT_DESIGN.md](PARETO_EXPERIMENT_DESIGN.md) 참조
