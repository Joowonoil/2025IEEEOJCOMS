#%%
from dataset import get_dataset_and_dataloader
import torch
import torch.nn.functional as F
import torch_tensorrt
import math
import numpy as np
from pathlib import Path
import yaml
from einops import repeat, rearrange
from model.estimator_v3 import Estimator_v3
from utils.plot_signal import plot_signal
from utils.auto_upload import auto_upload_models
from timeit import default_timer as timer
from torchsummary import summary
import wandb
from torch.optim.lr_scheduler import CyclicLR
from transformers import get_cosine_schedule_with_warmup


# Estimator_v3를 그대로 사용하되 베이스 모델용으로 활용


class Engine_v3:
    def __init__(self, conf_file, device=None, use_wandb=None, wandb_proj=None):
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file
        with open(conf_path, encoding='utf-8') as f:
            self._conf = yaml.safe_load(f)
        
        # 설정 파일에서 기본값 읽기, 파라미터로 override 가능
        self._device = device or self._conf['training'].get('device', 'cuda:0')
        self._use_wandb = use_wandb if use_wandb is not None else self._conf['training'].get('use_wandb', True)
        self._wandb_proj = wandb_proj or self._conf['training'].get('wandb_proj', 'DNN_channel_estimation_v3_base')
        if self._use_wandb:
            wandb.init(project=self._wandb_proj, config=self._conf)
            self._conf = wandb.config
        # Get dataset and dataloader
        self._dataset, self._dataloader = get_dataset_and_dataloader(self._conf['dataset'])
        # Channel and phase noise estimation network (v3 베이스 사용)
        self._estimator = Estimator_v3(conf_file).to(self._device)
        # Optimizer
        self._max_norm = self._conf['training']['max_norm']
        self._num_iter = self._conf['training']['num_iter']
        self._lr = self._conf['training']['lr']
        self._weight_decay = self._conf['training']['weight_decay']
        self._ch_optimizer = None
        self.set_optimizer()

    def set_optimizer(self):
        # v3에서는 모든 파라미터를 훈련 (Adapter 미적용 상태)
        ch_params = [p for n, p in self._estimator.ch_tf.named_parameters() if p.requires_grad]
        
        # 옵티마이저 타입을 설정에서 읽기
        optimizer_type = self._conf['training'].get('optimizer', 'Adam').lower()
        
        if optimizer_type == 'adam':
            self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._lr, weight_decay=self._weight_decay)
        elif optimizer_type == 'adamw':
            self._ch_optimizer = torch.optim.AdamW([{"params": ch_params}], lr=self._lr, weight_decay=self._weight_decay)
        elif optimizer_type == 'sgd':
            momentum = self._conf['training'].get('momentum', 0.9)
            self._ch_optimizer = torch.optim.SGD([{"params": ch_params}], lr=self._lr, weight_decay=self._weight_decay, momentum=momentum)
        else:
            print(f"Warning: Unknown optimizer {optimizer_type}, using Adam as default")
            self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._lr, weight_decay=self._weight_decay)

        # 학습률 스케줄러 사용 여부를 설정 (None이면 사용 안함)
        if self._conf['training'].get('use_scheduler', False):  # 'use_scheduler'가 True일 때만 사용'
            num_warmup_steps = self._conf['training'].get('num_warmup_steps', 0)
            self._ch_scheduler = get_cosine_schedule_with_warmup(
                self._ch_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self._num_iter
            )
        else:
            self._ch_scheduler = None  # 스케줄러 사용 안함

    def train(self):

        # Loss weight 설정
        ch_loss_weight = self._conf['training'].get('ch_loss_weight', 1)  # 채널 추정 손실 가중치

        for it, data in enumerate(self._dataloader):
            # Forward estimator
            self._estimator.train()
            rx_signal = data['ref_comp_rx_signal']
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device)
            ch_est, _ = self._estimator(rx_signal)
            
            # Channel training
            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device)
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1)  # batch, data, re/im
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1]
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1]
            ch_nmse = torch.mean(ch_mse / ch_var)
            ch_mse = torch.mean(ch_mse)
            ch_loss = ch_nmse * ch_loss_weight  # 채널 추정 손실에 가중치 적용
            
            self._ch_optimizer.zero_grad()
            ch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._estimator.ch_tf.parameters(), max_norm=self._max_norm)
            self._ch_optimizer.step()

            # 학습률 업데이트 (스케줄러가 있을 때만 실행)
            if self._ch_scheduler:
                self._ch_scheduler.step()

            # Logging
            if (it + 1) % self._conf['training']['logging_step'] == 0:
                current_lr = self._ch_scheduler.get_last_lr()[0] if self._ch_scheduler else self._lr
                print(f"iteration: {it + 1}, ch_nmse: {ch_nmse}, lr: {current_lr}")
                self._logging(it, ch_nmse, ch_est, ch_true)

            if it >= self._num_iter - 1:
                break

    @torch.no_grad()
    def _logging(self, it, ch_nmse, ch_est, ch_true):
        log = {'ch_nmse': ch_nmse}
        if self._use_wandb:
            wandb.log(log)
        if (it + 1) % self._conf['training']['evaluation_step'] == 0:
            show_batch_size = self._conf['training']['evaluation_batch_size']
            ch_true = ch_true[:, :, 0] + 1j * ch_true[:, :, 1]
            ch_true = ch_true[:show_batch_size].detach().cpu().numpy()
            ch_est = ch_est[:, :, 0] + 1j * ch_est[:, :, 1]
            ch_est = ch_est[:show_batch_size].detach().cpu().numpy()

            sig_dict = {}
            sig_dict['ch_est_real'] = {'data': ch_est, 'type': 'real'}
            sig_dict['ch_true_real'] = {'data': ch_true, 'type': 'real'}
            sig_dict['ch_est_imag'] = {'data': ch_est, 'type': 'imag'}
            sig_dict['ch_true_imag'] = {'data': ch_true, 'type': 'imag'}
            
            f = plot_signal(sig_dict, shape=(3, 2))
            f.show()
            if self._use_wandb:
                wandb.log({'estimation': wandb.Image(f)})
            
            # v3 베이스 모델 저장 (Adapter 미적용 상태)
            save_name = self._conf['training'].get('saved_model_name', 'Large_estimator_v3_base')
            if (it + 1) % self._conf['training'].get('model_save_step', 100000) == 0:
                self.save_model(f"{save_name}_iter_{it + 1}")
            else:
                self.save_model(save_name)

    def save_model(self, file_name):
        """v3 베이스 모델을 저장 - Adapter 미적용 순수 v3 구조"""
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        path.mkdir(parents=True, exist_ok=True)
        
        # v3 구조 그대로 저장 (Adapter 미적용)
        torch.save(self._estimator, path / (file_name + '.pt'))
        print(f"v3 base model saved to {path / (file_name + '.pt')}")

    def load_model(self, file_name):
        """v3 베이스 모델 로드"""
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._estimator = torch.load(path / (file_name + '.pt'), weights_only=False)
        self._estimator = self._estimator.to(self._device)
        self.set_optimizer()
        print(f"v3 base model loaded from {path / (file_name + '.pt')}")

    def num_params(self):
        n_params = sum([p.numel() for p in self._estimator.parameters() if p.requires_grad])
        print(f'total_params: {n_params}')


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    # Training v3 base model - 모든 설정을 config_v3.yaml에서 읽음
    conf_file = 'config_v3.yaml'  # v3 기본 config 사용
    engine = Engine_v3(conf_file)  # 설정 파일에서 모든 파라미터 읽음
    
    # 기존 모델 로드 (필요한 경우)
    load_model_name = engine._conf['training'].get('load_model_path', None)
    if load_model_name:
        print(f"Loading existing model: {load_model_name}")
        engine.load_model(load_model_name)
    
    # 베이스 모델 훈련 시작
    print("Starting v3 base model training...")
    print(f"Device: {engine._device}")
    print(f"Use WandB: {engine._use_wandb}")
    print(f"WandB Project: {engine._wandb_proj}")
    print(f"Learning Rate: {engine._lr}")
    print(f"Max Iterations: {engine._num_iter}")
    print("Architecture: Transformer v3 (Adapter 미적용 순수 구조)")
    
    engine.train()
    
    # 최종 모델 저장
    final_model_name = engine._conf['training'].get('saved_model_name', 'Large_estimator_v3_base')
    engine.save_model(f"{final_model_name}_final")
    print(f"v3 base model training completed and saved as {final_model_name}_final.pt")

    # 파라미터 수 출력
    engine.num_params()
    
    # 자동 모델 업로드 (설정에서 활성화된 경우)
    print("\n" + "="*50)
    print("Training completed! Checking auto-upload...")
    try:
        auto_upload_models(engine._conf, f"{final_model_name}_final")
    except Exception as e:
        print(f"Warning: Auto-upload failed: {str(e)}")
        print("Models are saved locally in saved_model/ folder")
    print("="*50)

# %%