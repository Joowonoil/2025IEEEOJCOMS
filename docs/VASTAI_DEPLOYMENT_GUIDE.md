# Vast.ai 병렬 실행 가이드 (간소화 버전)

Pareto Frontier 실험을 4개의 Vast.ai 인스턴스에서 병렬로 실행하는 방법

**작성일**: 2025-11-02 (최종 업데이트: 2025-11-03)
**예상 시간**: 35-70시간 (병렬 실행)
**예상 비용**: $60-100 (RTX 3090/4090 + 고성능 CPU 기준)

---

## ⚠️ 중요: 실전에서 배운 교훈

### CPU 성능이 가장 중요!
- **이 프로젝트는 데이터 로딩/전처리가 병목** → CPU 싱글코어 성능이 핵심
- GPU만 좋고 CPU가 나쁘면 오히려 로컬보다 느림!

### 추천 CPU (필수!)
- ✅ **AMD Ryzen 7900X / 7950X** (5.4GHz+)
- ✅ AMD Ryzen 9 7900X3D / 7950X3D
- ✅ Intel Core i9-13900K / 14900K
- ❌ AMD EPYC (서버용, 느린 싱글코어)
- ❌ 오래된 Xeon (느림)

### Config 설정
```yaml
batch_size: 32        # 절대 변경 금지 (결과 일관성)
num_workers: 0        # CPU 좋으면 0이 최적 (multiprocessing 오버헤드 없음)
```

---

## 📋 빠른 시작

### 1. 인스턴스 선택 (가장 중요!)

**Vast.ai 필터 설정:**
```
GPU: RTX 3090 / RTX 4090 (24GB)
CPU: Ryzen 7900 OR Ryzen 7950 OR i9-13900 OR i9-14900
RAM: 32GB+
Disk: 100GB+
```

**기본 설정 (4개 인스턴스):**
- Instance 1: Adapter (20 runs)
- Instance 2: LoRA (20 runs)
- Instance 3: Prompt (15 runs)
- Instance 4: Hybrid (15 runs)

---

### 💡 시간 절약 팁: 양방향 병렬 실행 (8개 인스턴스)

**시간을 절반으로 줄이고 싶다면 8개 인스턴스 사용!**

**전략:**
- **Set A (4개)**: InH부터 시작 → RMa로 진행 (정방향)
- **Set B (4개)**: RMa부터 시작 → InH로 진행 (역방향)
- **양쪽에서 만나므로 시간 절반!**

**Set A (기본):**
```bash
# InH-dim8 → InH-dim16 → ... → RMa-dim64
nohup python Transfer_Pareto_Adapter.py > adapter.log 2>&1 &
nohup python Transfer_Pareto_LoRA.py > lora.log 2>&1 &
nohup python Transfer_Pareto_Prompt.py > prompt.log 2>&1 &
nohup python Transfer_Pareto_Hybrid.py > hybrid.log 2>&1 &
```

**Set B (역순):**
```bash
# RMa-dim64 → RMa-dim32 → ... → InH-dim8
nohup python Transfer_Pareto_Adapter_Reverse.py > adapter.log 2>&1 &
nohup python Transfer_Pareto_LoRA_Reverse.py > lora.log 2>&1 &
nohup python Transfer_Pareto_Prompt_Reverse.py > prompt.log 2>&1 &
nohup python Transfer_Pareto_Hybrid_Reverse.py > hybrid.log 2>&1 &
```

**효과:**
- 4개 인스턴스: 40-80시간
- 8개 인스턴스: **20-40시간** (절반!)
- 비용: 2배이지만 시간은 절반

**주의:**
- 같은 run을 중복 실행하지 않도록 주의
- WandB에서 양쪽 진행상황 모니터링
- 중간에서 만나면 불필요한 인스턴스 종료

### 💡💡 추가: Hybrid Extra 실험 (2개 인스턴스 더)

**목적**: Hybrid Pareto frontier를 더 풍부하게 만들기 (3→9 configs)

**추가 6개 config:**
- **Option 1** (중간 밀도): p75_r8, p150_r15 - 곡선 빈틈 채우기
- **Option 2** (극단값): p25_r2, p300_r25 - 경계 탐색
- **Option 3** (비대칭): p50_r10, p100_r5 - Prompt vs LoRA 기여도 비교

**Set Extra-A (정방향):**
```bash
nohup python Transfer_Pareto_Hybrid_Extra.py > hybrid_extra.log 2>&1 &
# InH-p75_r8 → ... → RMa-p300_r25
```

**Set Extra-B (역방향):**
```bash
nohup python Transfer_Pareto_Hybrid_Extra_Reverse.py > hybrid_extra.log 2>&1 &
# RMa-p100_r5 → ... → InH-p25_r2
```

**결과:**
- 기존 Hybrid: 3 configs × 5 scenarios = 15 runs
- Extra Hybrid: 6 configs × 5 scenarios = 30 runs
- **총 Hybrid: 45 runs** (9 configs, 훨씬 정밀한 Pareto curve!)

**총 실험: 100 runs** (기존 70 + Extra 30)

---

### 2. 데이터 준비 (로컬에서 한 번만)

```powershell
# Windows PowerShell
cd C:\Users\YOUR_PATH\DNN_channel_estimation_training

# 압축
Compress-Archive -Path dataset -DestinationPath dataset.zip -Force
Compress-Archive -Path saved_model -DestinationPath saved_model.zip -Force
```

---

### 3. 각 인스턴스 Setup (4개 반복)

#### Step 1: SSH 접속
```bash
ssh root@X.X.X.X -p XXXXX
```

#### Step 2: 코드 클론
```bash
cd /workspace
git clone https://github.com/Joowonoil/2025IEEEOJCOMS
cd 2025IEEEOJCOMS
```

#### Step 3: 패키지 설치
```bash
pip install -r requirements_vastai.txt
```

#### Step 4: CUDA 확인
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

#### Step 5: 데이터 업로드

**Jupyter Lab 사용 (가장 확실!):**

1. Vast.ai → Jupyter Lab 열기
2. 좌측 파일 브라우저 → Upload 버튼 (↑)
3. `dataset.zip`, `saved_model.zip` 업로드 (10-30분)
4. 터미널에서 압축 해제:

```bash
cd /workspace/2025IEEEOJCOMS
unzip dataset.zip
unzip saved_model.zip

# 확인
ls dataset/PDP_processed/*.mat | head -5
ls saved_model/Large_estimator_v3_base_extended_final.pt
ls saved_model/Large_estimator_v4_base_extended_final.pt
```

#### Step 6: WandB 로그인
```bash
wandb login YOUR_API_KEY
```

#### Step 7: Config 확인
```bash
# num_workers가 0인지 확인 (중요!)
grep -E "batch_size|num_workers" config/config_pareto_adapter.yaml
# 출력: batch_size: 32, num_workers: 0
```

#### Step 8: 실행
```bash
# Instance 1 (Adapter)
nohup python Transfer_Pareto_Adapter.py > adapter.log 2>&1 &

# Instance 2 (LoRA)
nohup python Transfer_Pareto_LoRA.py > lora.log 2>&1 &

# Instance 3 (Prompt)
nohup python Transfer_Pareto_Prompt.py > prompt.log 2>&1 &

# Instance 4 (Hybrid)
nohup python Transfer_Pareto_Hybrid.py > hybrid.log 2>&1 &
```

#### Step 9: 모니터링
```bash
# 로그 실시간 확인
tail -f adapter.log  # (또는 lora.log, prompt.log, hybrid.log)

# GPU 사용률 확인 (별도 터미널)
watch -n 1 nvidia-smi

# Ctrl+C로 종료해도 실험은 계속 실행됨
```

---

## 🔍 모니터링

### GPU Utilization 확인
```bash
nvidia-smi
```

**정상 상태:**
- GPU Utilization: 50-90%
- Memory-Usage: 2-5GB (batch_size=32 기준)
- Power: 100-200W

**비정상 상태 (CPU 병목):**
- GPU Utilization: 0-10%
- Memory-Usage: 사용 중이지만 idle
- → **CPU를 더 좋은 것으로 교체!**

### WandB 확인
- Adapter: `DNN_channel_estimation_*_Adapter_Pareto` (5개 시나리오)
- LoRA: `DNN_channel_estimation_*_LoRA_Pareto` (5개 시나리오)
- Prompt: `DNN_channel_estimation_*_Prompt_Pareto` (5개 시나리오)
- Hybrid: `DNN_channel_estimation_*_Hybrid_Pareto` (5개 시나리오)

총 20개 프로젝트, 70개 runs

---

## 📥 결과 수집

### 각 인스턴스에서:
```bash
cd /workspace/2025IEEEOJCOMS

# 결과 압축
tar -czf pareto_adapter_results.tar.gz saved_model/pareto/*adapter*
# (또는 lora, prompt, hybrid)
```

### 로컬로 다운로드:
```bash
# 로컬 터미널
scp -P PORT root@IP:/workspace/2025IEEEOJCOMS/pareto_adapter_results.tar.gz .
```

### 압축 해제:
```bash
tar -xzf pareto_adapter_results.tar.gz -C saved_model/pareto/
```

---

## 🔧 트러블슈팅

### 1. 학습이 느림 (가장 흔한 문제!)

**증상:** nvidia-smi에서 GPU Utilization 0-10%

**원인:** CPU 성능 부족 (데이터 로딩 병목)

**해결:**
```bash
# CPU 확인
lscpu | grep "Model name"

# CPU가 EPYC, 오래된 Xeon이면 인스턴스 교체!
# Ryzen 7900X / 7950X / i9-13900K 이상으로 교체
```

### 2. CUDA Out of Memory

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
```bash
# config 파일 수정 (결과 달라질 수 있으니 주의!)
nano config/config_pareto_adapter.yaml
# batch_size: 32 → 16
```

### 3. 프로세스 확인/종료

**프로세스 확인:**
```bash
ps aux | grep Transfer_Pareto
```

**종료:**
```bash
pkill -f Transfer_Pareto_Adapter.py
# 또는
kill <PID>
```

### 4. 디스크 용량 부족

**확인:**
```bash
df -h
```

**정리:**
```bash
rm -rf /workspace/.cache
rm -rf /tmp/*
```

---

## ✅ 체크리스트

### 인스턴스 대여 시
- [ ] CPU: Ryzen 7900X/7950X 또는 i9-13900K 이상
- [ ] GPU: RTX 3090 / 4090 (24GB)
- [ ] RAM: 32GB+
- [ ] Disk: 100GB+

### 각 인스턴스 Setup
- [ ] 코드 클론
- [ ] requirements_vastai.txt 설치
- [ ] CUDA 작동 확인
- [ ] dataset + saved_model 업로드 & 압축 해제
- [ ] WandB 로그인
- [ ] config 확인 (batch_size: 32, num_workers: 0)
- [ ] nohup 실행
- [ ] nvidia-smi GPU 사용 확인 (50-90%)

### 완료 후
- [ ] 4개 실험 모두 완료
- [ ] 결과 압축 & 다운로드
- [ ] 인스턴스 종료 (비용 절감!)
- [ ] WandB 로그 확인

---

## 💡 추가 팁

### 비용 절감
- **Interruptible 인스턴스**: 50-70% 저렴하지만 중단될 수 있음
- checkpoint 저장 간격 확인: `model_save_step: 20000`

### 데이터 재사용
- 첫 번째 인스턴스에서 dataset.zip, saved_model.zip 다운로드
- 나머지 인스턴스에 재업로드 (시간 절약)

### 동시 작업
- 4개 터미널 열어서 동시에 setup하면 빠름
- Jupyter Lab은 여러 탭에서 동시 업로드 가능

---

## 📊 예상 결과

**파일 개수:**
- Adapter: 20 final + 100 checkpoints = 120개
- LoRA: 20 final + 100 checkpoints = 120개
- Prompt: 15 final + 75 checkpoints = 90개
- Hybrid (Base): 15 final + 75 checkpoints = 90개
- Hybrid (Extra): 30 final + 150 checkpoints = 180개
- **총 600개 파일**

**WandB Runs:**
- 총 100개 runs (기존 70 + Hybrid Extra 30)
- 4 methods × 5 scenarios × (3-9 configs)

---

## 🚨 알려진 이슈

### torch_tensorrt 에러 (이미 해결됨)
- `ModuleNotFoundError: No module named 'torch_tensorrt'`
- → estimator_v3.py, estimator_v4.py에서 import 제거됨
- → `git pull`로 최신 코드 받으면 해결

### Google Drive gdown 실패
- 권한 문제로 실패 가능성 높음
- → **Jupyter Lab 업로드 사용 권장**

### Windows multiprocessing 에러
- `num_workers > 0`이면 pickle 에러 발생
- → Vast.ai(Linux)에서는 문제 없음
- → 하지만 `num_workers: 0`이 더 빠름 (CPU 좋으면)

---

**다음 단계**: [PARETO_EXPERIMENT_DESIGN.md](PARETO_EXPERIMENT_DESIGN.md) 참조
