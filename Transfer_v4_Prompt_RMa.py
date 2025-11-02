import torch # PyTorch 라이브러리 임포트 (딥러닝 프레임워크)
import torch.nn.functional as F # PyTorch의 함수형 API 임포트
import yaml # YAML 파일 파싱을 위한 라이브러리 임포트
from pathlib import Path # 파일 경로 관리를 위한 Path 객체 임포트
import wandb # Weights & Biases 로깅 라이브러리 임포트
from dataset import get_dataset_and_dataloader # 데이터셋 및 데이터로더를 가져오는 함수 임포트
from model.prompt_estimator_v4 import PromptEstimator_v4 # 모델 PromptEstimator_v4 클래스 임포트
from transformers import get_cosine_schedule_with_warmup # get_cosine_schedule_with_warmup 스케줄러 임포트
import numpy as np # NumPy 라이브러리 임포트 (수치 연산용)
from utils.plot_signal import plot_signal # 신호 플롯팅 함수 임포트
from utils.auto_upload import auto_upload_models # 자동 모델 업로드 함수 임포트
# from peft import LoraConfig, get_peft_model # PEFT 라이브러리 임포트 (프롬프트 학습에서는 비활성화)
# from model.transformer_v2 import Transformer # v2 Transformer 모델 임포트 (v3에서는 사용 안 함, 주석 처리됨)

# EarlyStopping 클래스 제거됨 (불필요)

class PromptTransferLearningEngine: # 프롬프트 기반 전이 학습 엔진 클래스 정의
    def __init__(self, conf_file): # 초기화 메서드
        # 설정 파일 경로 설정
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file # 설정 파일 경로 생성
        # 설정 파일 로드
        with open(conf_path, encoding='utf-8') as f: # 설정 파일 열기
            self._conf = yaml.safe_load(f) # 설정 파일 로드
        self._conf_file = conf_file # 설정 파일 이름 저장

        # 설정 파일에서 기본 파라미터 로드
        self._device = self._conf['training'].get('device', 'cuda:0') # 사용할 디바이스 설정 (기본값 'cuda:0')
        self._use_wandb = self._conf['training'].get('use_wandb', True) # WandB 사용 여부 (기본값 True)
        self._wandb_proj = self._conf['training'].get('wandb_proj', 'DNN_channel_estimation') # WandB 프로젝트 이름 (기본값 'DNN_channel_estimation')


        # WandB 초기화
        if self._use_wandb: # WandB 사용 설정이 True이면
            wandb.init(project=self._wandb_proj, config=self._conf) # WandB 초기화 (프로젝트 이름 및 설정 전달)
            self._conf = wandb.config # WandB config로 업데이트 (하이퍼파라미터 스위프 등 고려)
        
        # 훈련 데이터셋 및 데이터로더 가져오기 (설정에 따라 RMa 데이터셋 로드)
        self._dataset, self._dataloader = get_dataset_and_dataloader(self._conf['dataset']) # 훈련 데이터셋 및 데이터로더 가져오기
        # 검증 데이터셋 및 데이터로더 가져오기 (검증 데이터가 없으므로 비활성화)
        self._val_dataset, self._val_dataloader = None, None # 검증 데이터셋 비활성화

        # 채널 및 위상 잡음 추정 네트워크 (모델은 나중에 로드) # 채널 및 위상 잡음 추정 네트워크 (모델은 load_model 메서드에서 로드)

        # 옵티마이저 및 스케줄러는 모델 로드 후 설정 # 옵티마이저 및 스케줄러는 load_model 메서드 호출 후 set_optimizer 메서드에서 설정

    def load_model(self): # 모델 로드 및 초기화 메서드
        # PromptEstimator_v4 클래스는 내부에서 모델 로딩 및 프롬프트 설정을 처리합니다.
        self._estimator = PromptEstimator_v4(self._conf_file).to(self._device) # PromptEstimator_v4 인스턴스 생성 및 지정된 디바이스로 이동

        # Prompt Learning 설정 확인 및 출력
        prompt_config_dict = self._conf['ch_estimation'].get('prompt', {}) # 설정 파일에서 Prompt 설정 가져오기
        if prompt_config_dict: # Prompt 설정이 존재하면
            prompt_length = prompt_config_dict.get('prompt_length', 5) # 프롬프트 토큰 수 가져오기 (기본값 5)
            print(f"Prompt Learning enabled with {prompt_length} prompt tokens.") # 프롬프트 학습 활성화 메시지 출력
        else:
            print("No Prompt Learning configuration found. Using default prompt length.")
        
        # 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self._estimator.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self._estimator.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.3f}%)") # 학습 가능한 파라미터 수 및 비율 출력

        model_load_mode = self._conf['training'].get('model_load_mode', 'pretrained') # 모델 로드 모드 설정 파일에서 로드 (기본값 'pretrained')

        if model_load_mode == 'finetune': # 모델 로드 모드가 'finetune'이면
            load_path = self._conf['training'].get('load_model_path') # 로드할 모델 경로 설정 파일에서 로드
            if load_path: # 로드 경로가 설정되어 있으면
                try:
                    model_path = Path(load_path) # 로드 경로를 Path 객체로 변환
                    if model_path.exists(): # 모델 파일이 존재하는지 확인
                        # PEFT 모델의 LoRA 가중치만 로드
                        self._estimator.load_state_dict(torch.load(model_path), strict=False) # PEFT 모델에 상태 사전 로드 (strict=False로 부분 로드 허용)
                        print(f"PEFT LoRA weights loaded successfully from {model_path} for finetuning.") # LoRA 가중치 로드 성공 메시지 출력
                    else: # 모델 파일이 존재하지 않으면
                        print(f"Error: Model file not found at {model_path} for finetuning.") # 오류 메시지 출력
                        print("Please check 'load_model_path' in the config file.") # 설정 파일 확인 요청 메시지 출력
                        raise FileNotFoundError(f"Model file not found at {model_path}") # FileNotFoundError 발생
                except Exception as e: # 로드 중 예외 발생 시
                    print(f"Error loading model from {load_path}: {e}") # 오류 메시지 출력
                    print("Please check 'load_model_path' and the model file.") # 경로 및 파일 확인 요청 메시지 출력
                    raise RuntimeError(f"Failed to load model from {load_path}") # RuntimeError 발생
            else: # 로드 경로가 설정되지 않았으면
                print("Error: 'load_model_path' is not specified in the config file for 'finetune' mode.") # 오류 메시지 출력
                print("Please specify the path to the saved model.") # 경로 지정 요청 메시지 출력
                raise ValueError("'load_model_path' must be specified for 'finetune' mode.") # ValueError 발생
        elif model_load_mode == 'pretrained': # 모델 로드 모드가 'pretrained'이면
            print(f"Estimator_v4 model initialized and pretrained weights loaded.") # 사전 학습 가중치 로드 메시지 출력
        else: # 알 수 없는 모델 로드 모드이면
            print(f"Warning: Unknown model_load_mode '{model_load_mode}'. Initializing with pretrained weights.") # 경고 메시지 출력
            print(f"Estimator_v4 model initialized and pretrained weights loaded.") # 사전 학습 가중치 로드 메시지 출력


        self.set_optimizer() # 옵티마이저 설정 메서드 호출

        # Early Stopping 제거됨

    def set_optimizer(self): # 옵티마이저 설정 메서드
        # 전이 학습을 위해 훈련 가능한 파라미터만 가져와 옵티마이저 설정
        # Estimator_v4에서는 LoRA 파라미터만 훈련 가능하도록 설정되어 있습니다.
        ch_params = [p for n, p in self._estimator.named_parameters() if p.requires_grad] # 훈련 가능한 파라미터 가져오기 (PEFT가 자동으로 설정)
        self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._conf['training']['lr']) # Adam 옵티마이저 설정 (LoRA 파라미터에만 적용)

        # 스케줄러 사용 여부 확인
        if self._conf['training'].get('use_scheduler', False):  # 'use_scheduler'가 True일 때만 사용'
            num_warmup_steps = self._conf['training'].get('num_warmup_steps', 0) # warm-up 단계 수 설정 파일에서 로드
            self._ch_scheduler = get_cosine_schedule_with_warmup( # Cosine Annealing with Warmup 스케줄러 생성
                self._ch_optimizer, # 옵티마이저 전달
                num_warmup_steps=num_warmup_steps, # warm-up 단계 수 전달
                num_training_steps=self._conf['training']['num_iter'] # 총 훈련 단계 수 전달
            )
        else: # 스케줄러 사용 설정이 False이면
            self._ch_scheduler = None # 스케줄러 사용 안 함

    def train(self): # 훈련 메서드
        ch_loss_weight = self._conf['training'].get('ch_loss_weight', 1) # 채널 손실 가중치 (설정 파일에서 로드, 기본값 1)

        # 훈련 데이터로더를 순회하며 훈련
        for it, data in enumerate(self._dataloader): # 훈련 데이터로더 순회 (이터레이션 번호와 데이터 가져오기)
            self._estimator.train() # 모델을 훈련 모드로 설정 (드롭아웃 등 활성화)
            rx_signal = data['ref_comp_rx_signal'] # 수신 신호 데이터 가져오기
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1) # 복소수 신호를 실수부와 허수부로 분리하여 NumPy 배열 생성
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device) # NumPy 배열을 PyTorch 텐서로 변환 및 지정된 디바이스에 할당

            ch_est, _ = self._estimator(rx_signal) # 모델을 통해 채널 추정 (위상 잡음 추정 결과는 무시)

            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device) # 실제 채널 데이터 가져오기 (복소수 형태)
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1) # 복소수 채널을 실수부와 허수부로 분리
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1] # 채널 MSE 계산 (배치별)
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1] # 실제 채널의 분산 계산 (배치별)
            ch_nmse = torch.mean(ch_mse / ch_var) # 채널 NMSE 계산 (배치 평균)
            ch_mse = torch.mean(ch_mse) # 채널 MSE 평균 계산 (배치 평균)
            ch_loss = ch_nmse * ch_loss_weight # 채널 손실 계산 (NMSE에 가중치 적용)

            self._ch_optimizer.zero_grad() # 옵티마이저의 그래디언트 초기화
            
            # 역전파 전 학습 가능한 파라미터 확인
            trainable_params = [p for p in self._estimator.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                print(f"ERROR: No trainable parameters found at iteration {it + 1}")
                print("This usually happens after model saving. Checking parameter states...")
                for name, param in self._estimator.named_parameters():
                    if '_prompt_tokens' in name or '_prompt_pos_embed' in name:
                        print(f"Prompt param {name}: requires_grad={param.requires_grad}")
                break  # 훈련 중단
            
            try:
                ch_loss.backward() # 역전파를 통해 그래디언트 계산
                # 학습 가능한 파라미터만 클리핑 (prompt tokens 또는 전체 모델)
                torch.nn.utils.clip_grad_norm_(self._estimator.parameters(), max_norm=self._conf['training']['max_norm']) # 그래디언트 클리핑 (PEFT 모델의 모든 학습 가능한 파라미터에 적용)
                self._ch_optimizer.step() # 옵티마이저 스텝 (파라미터 업데이트)
            except RuntimeError as e:
                print(f"ERROR during backward pass at iteration {it + 1}: {e}")
                print(f"ch_loss requires_grad: {ch_loss.requires_grad}")
                print(f"ch_loss grad_fn: {ch_loss.grad_fn}")
                print("Checking model parameters:")
                trainable_count = 0
                for name, param in self._estimator.named_parameters():
                    if param.requires_grad:
                        trainable_count += 1
                        print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
                print(f"Total trainable parameters: {trainable_count}")
                if trainable_count == 0:
                    print("No trainable parameters found - this is the root cause of the error")
                break  # 훈련 중단

            # 스케줄러 사용 시 학습률 업데이트
            if self._ch_scheduler: # 스케줄러 사용 설정이 True이면
                self._ch_scheduler.step() # 스케줄러 스텝 (학습률 업데이트)

            # 로깅 스텝마다 정보 출력 및 로깅
            if (it + 1) % self._conf['training']['logging_step'] == 0: # 현재 이터레이션이 로깅 스텝의 배수이면
                current_lr = self._ch_scheduler.get_last_lr()[0] if self._ch_scheduler else self._conf['training']['lr'] # 현재 학습률 가져오기 (스케줄러 사용 시 스케줄러 학습률, 아니면 초기 학습률)
                print(f"iteration: {it + 1}, ch_nmse: {ch_nmse}, lr: {current_lr}") # 훈련 상태 출력 (이터레이션, 채널 NMSE, 학습률)
                self._logging(it, ch_nmse, ch_est, ch_true) # 로깅 함수 호출

            # Early Stopping 제거됨


            # 설정된 최대 이터레이션에 도달하면 훈련 중단
            if it >= self._conf['training']['num_iter'] - 1: # 현재 이터레이션이 최대 이터레이션 수에 도달하면
                break # 훈련 루프 중단

        # Early Stopping 제거됨

        # 훈련 완료 후 Prompt 모델 저장
        self.save_prompt_model(self._conf['training'].get('saved_model_name', 'final_prompt_model')) # 최종 프롬프트 모델 저장

    @torch.no_grad() # 그래디언트 계산 비활성화 (평가 모드)
    def evaluate(self): # 평가 메서드
        if self._val_dataloader is None: # 검증 데이터가 없으면
            print("No validation data available, skipping evaluation.") # 검증 생략 메시지 출력
            return 0.0 # 기본값 반환
        
        self._estimator.eval() # 모델을 평가 모드로 설정 (드롭아웃 등 비활성화)
        total_nmse = 0.0 # 총 NMSE 초기화
        num_batches = 0 # 배치 카운터 초기화
        # 검증 데이터로더를 순회하며 평가
        for data in self._val_dataloader: # 검증 데이터로더 순회
            rx_signal = data['ref_comp_rx_signal'] # 수신 신호 데이터 가져오기
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1) # 복소수 신호를 실수부와 허수부로 분리하여 NumPy 배열 생성
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device) # NumPy 배열을 PyTorch 텐서로 변환 및 지정된 디바이스에 할당

            ch_est, _ = self._estimator(rx_signal) # 모델을 통해 채널 추정 (위상 잡음 추정 결과는 무시)

            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device) # 실제 채널 데이터 가져오기 (복소수 형태)
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1) # 복소수 채널을 실수부와 허수부로 분리
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1] # 채널 MSE 계산 (배치별)
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1] # 실제 채널의 분산 계산 (배치별)
            ch_nmse = torch.mean(ch_mse / ch_var) # 채널 NMSE 계산 (배치 평균)

            total_nmse += ch_nmse.item() # 총 NMSE에 현재 배치의 NMSE 추가
            num_batches += 1 # 배치 카운터 증가

        avg_nmse = total_nmse / num_batches # 평균 NMSE 계산
        print(f"Validation NMSE: {avg_nmse}") # 검증 NMSE 출력
        if self._use_wandb: # WandB 사용 설정이 True이면
            wandb.log({'val_ch_nmse': avg_nmse}) # 검증 NMSE 로깅
        return avg_nmse # 평균 NMSE 반환


    @torch.no_grad() # 그래디언트 계산 비활성화 (로깅 모드)
    def _logging(self, it, ch_nmse, ch_est, ch_true): # 로깅 메서드
        log = {'ch_nmse': ch_nmse} # 훈련 NMSE 로깅 데이터 딕셔너리 생성
        if self._use_wandb: # WandB 사용 설정이 True이면
            wandb.log(log) # 훈련 NMSE 로깅
        if (it + 1) % self._conf['training']['evaluation_step'] == 0: # 현재 이터레이션이 평가 스텝의 배수이면 (평가와 동일한 간격으로 플롯 로깅)
            show_batch_size = self._conf['training']['evaluation_batch_size'] # 플롯팅할 배치 크기 설정
            ch_true = ch_true[:, :, 0] + 1j * ch_true[:, :, 1] # 실제 채널 복소수 형태로 변환
            ch_true = ch_true[:show_batch_size].detach().cpu().numpy() # 플롯팅할 실제 채널 데이터 (지정된 배치 크기만큼, detach 후 CPU로 이동 및 NumPy 변환)
            ch_est = ch_est[:, :, 0] + 1j * ch_est[:, :, 1] # 추정 채널 복소수 형태로 변환
            ch_est = ch_est[:show_batch_size].detach().cpu().numpy() # 플롯팅할 추정 채널 데이터 (지정된 배치 크기만큼, detach 후 CPU로 이동 및 NumPy 변환)

            sig_dict = {} # 신호 플롯팅을 위한 딕셔너리 초기화
            sig_dict['ch_est_real'] = {'data': ch_est, 'type': 'real'} # 추정 채널 실수부 데이터 추가
            sig_dict['ch_true_real'] = {'data': ch_true, 'type': 'real'} # 실제 채널 실수부 데이터 추가
            sig_dict['ch_est_imag'] = {'data': ch_est, 'type': 'imag'} # 추정 채널 허수부 데이터 추가
            sig_dict['ch_true_imag'] = {'data': ch_true, 'type': 'imag'} # 실제 채널 허수부 데이터 추가 (ch_imag를 ch_true로 수정)

            f = plot_signal(sig_dict, shape=(3, 2)) # 신호 플롯 생성 (3행 2열 형태)
            f.show() # 플롯 표시
            if self._use_wandb: # WandB 사용 설정이 True이면
                wandb.log({'estimation': wandb.Image(f)}) # 플롯 이미지를 WandB에 로깅
            
            # 모델 저장 간격에 따라 모델 저장
            if (it + 1) % self._conf['training'].get('model_save_step', 100000) == 0: # model_save_step마다 모델 저장
                self.save_prompt_model(f"{self._conf['training'].get('saved_model_name', 'checkpoint')}_iter_{it + 1}") # 중간 Prompt 체크포인트 저장
    
    def save_model(self, file_name): # 모델 저장 메서드 (이 메서드는 현재 사용되지 않음)
        path = Path(__file__).parents[0].resolve() / 'saved_model' # 모델 저장 디렉토리 경로 생성
        path.mkdir(parents=True, exist_ok=True) # 모델 저장 디렉토리가 없으면 생성
        # LoRA 가중치만 저장 (이 메서드는 save_combined_model_as_pt로 대체됨)
        # self._estimator.save_pretrained(path / file_name) # PEFT 모델의 LoRA 가중치만 저장
        # print(f"PEFT LoRA weights saved to {path / file_name}") # 모델 저장 경로 출력

    def save_prompt_model(self, file_name): # Prompt 모델을 .pt 파일로 저장하는 메서드
        try:
            print(f"Starting prompt model save process for {file_name}")
            
            path = Path(__file__).parents[0].resolve() / 'saved_model' # 모델 저장 디렉토리 경로 생성
            path.mkdir(parents=True, exist_ok=True) # 모델 저장 디렉토리가 없으면 생성
            full_path = path / f"{file_name}.pt" # .pt 확장자를 포함한 전체 경로 생성
            
            # Prompt 모델 저장 (engine.py와 호환)
            torch.save(self._estimator, full_path)
            print(f"Prompt model saved to {full_path}")
                
        except Exception as e:
            print(f"Error during prompt model saving: {e}")
            print("Continuing training without saving...")
            import traceback
            traceback.print_exc()


if __name__ == "__main__": # RMa 환경 특화 전이학습 실행
    #torch.autograd.set_detect_anomaly(True) # 자동 미분 이상 감지 활성화 (주석 처리됨)
    
    # RMa 환경 전용 설정 파일 사용
    conf_file = 'config_transfer_v4_prompt_RMa.yaml' # RMa 환경 특화 Prompt 설정 파일 사용
    
    print("=" * 60)
    print("RMa (Rural Macro) Prompt Transfer Learning Start")
    print("=" * 60)
    print(f"Config file: {conf_file}")
    print(f"Target environment: Rural Macro (Los/Nlos)")
    print(f"Transfer method: Prompt Learning")
    print("=" * 60)
    
    engine = PromptTransferLearningEngine(conf_file) # PromptTransferLearningEngine 객체 생성
    engine.load_model() # v4 베이스 모델 로드 및 Prompt 설정
    
    print("RMa prompt transfer learning starting...")
    engine.train() # RMa 데이터로 Prompt 전이학습 수행
    
    print("=" * 60)
    print("RMa Prompt transfer learning completed!")
    print(f"Saved model: check in saved_model/ directory")
    print("Training results available in WandB")
    print("=" * 60)
    
    # 자동 모델 업로드 (설정에서 활성화된 경우)
    print("\n" + "="*50)
    print("Training completed! Checking auto-upload...")
    try:
        final_model_name = engine._conf['training'].get('saved_model_name', 'Large_estimator_v4_to_RMa_prompt')
        auto_upload_models(engine._conf, final_model_name)
    except Exception as e:
        print(f"Warning: Auto-upload failed: {str(e)}")
        print("Models are saved locally in saved_model/ folder")
    print("="*50)