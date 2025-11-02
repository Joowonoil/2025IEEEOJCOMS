"""
Urban Base Model Training Engine
도시 환경(UMa + UMi) 전용 베이스 모델 학습

이 스크립트는 Cross-Domain Transfer Learning 실험의 일부로,
도시 환경에서만 학습된 베이스 모델을 생성합니다.
"""

import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import wandb
from dataset import get_dataset_and_dataloader
from model.estimator_v4 import Estimator_v4
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from utils.plot_signal import plot_signal
from utils.auto_upload import auto_upload_models

class TrainingEngine:
    def __init__(self, conf_file='config_v4_urban_base.yaml'):
        # Configuration file path
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file
        
        # Load configuration
        with open(conf_path, encoding='utf-8') as f:
            self._conf = yaml.safe_load(f)
        
        # Basic parameters
        self._device = self._conf['training'].get('device', 'cuda:0')
        self._use_wandb = self._conf['training'].get('use_wandb', True)
        self._wandb_proj = self._conf['training'].get('wandb_proj', 'DNN_channel_estimation_Urban_Base')
        
        # Initialize WandB
        if self._use_wandb:
            wandb.init(project=self._wandb_proj, config=self._conf)
            self._conf = wandb.config
        
        # Get dataset and dataloader
        self._dataset, self._dataloader = get_dataset_and_dataloader(self._conf['dataset'])
        
        # Channel estimation network
        self._estimator = Estimator_v4(conf_file).to(self._device)
        self.set_optimizer()
    
    def set_optimizer(self):
        # Set optimizer for channel estimation
        ch_params = list(self._estimator.parameters())
        self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._conf['training']['lr'])
        
        # Set scheduler if enabled
        if self._conf['training'].get('use_scheduler', False):
            num_warmup_steps = self._conf['training'].get('num_warmup_steps', 0)
            self._ch_scheduler = get_cosine_schedule_with_warmup(
                self._ch_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self._conf['training']['num_iter']
            )
        else:
            self._ch_scheduler = None
    
    def train(self):
        ch_loss_weight = self._conf['training'].get('ch_loss_weight', 1)
        
        for it, data in enumerate(self._dataloader):
            self._estimator.train()
            rx_signal = data['ref_comp_rx_signal']
            rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)
            rx_signal = torch.tensor(rx_signal, dtype=torch.float32).to(self._device)
            
            ch_est, _ = self._estimator(rx_signal)
            
            ch_true = torch.tensor(data['ch_freq'], dtype=torch.cfloat).to(self._device)
            ch_true = torch.stack((torch.real(ch_true), torch.imag(ch_true)), dim=-1)
            ch_mse = torch.sum(torch.square(ch_true - ch_est), dim=(1, 2)) / ch_true.shape[-1]
            ch_var = torch.sum(torch.square(ch_true), dim=(1, 2)) / ch_true.shape[-1]
            ch_nmse = torch.mean(ch_mse / ch_var)
            ch_mse = torch.mean(ch_mse)
            ch_loss = ch_nmse * ch_loss_weight
            
            self._ch_optimizer.zero_grad()
            ch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._estimator.parameters(), max_norm=self._conf['training']['max_norm'])
            self._ch_optimizer.step()
            
            if self._ch_scheduler:
                self._ch_scheduler.step()
            
            # Logging
            if (it + 1) % self._conf['training']['logging_step'] == 0:
                current_lr = self._ch_scheduler.get_last_lr()[0] if self._ch_scheduler else self._conf['training']['lr']
                print(f"[URBAN BASE] iteration: {it + 1}, ch_nmse: {ch_nmse}, lr: {current_lr}")
                self._logging(it, ch_nmse, ch_est, ch_true)
            
            # Model saving
            if (it + 1) % self._conf['training'].get('model_save_step', 10000) == 0:
                self.save_model(f"{self._conf['training'].get('saved_model_name', 'checkpoint')}_iter_{it + 1}")
            
            # Stop condition
            if it >= self._conf['training']['num_iter'] - 1:
                break
        
        # Save final model
        self.save_model(self._conf['training'].get('saved_model_name', 'final_model'))
    
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
    
    def save_model(self, file_name):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / f"{file_name}.pt"
        torch.save(self._estimator, full_path)
        print(f"[URBAN BASE] Model saved to {full_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("Urban Base Model Training Start")
    print("=" * 60)
    print("Dataset: UMa_Los, UMa_Nlos, UMi_Los, UMi_Nlos")
    print("Purpose: Cross-Domain Transfer Learning Base Model")
    print("Target: Urban environments (high-density buildings)")
    print("=" * 60)
    
    engine = TrainingEngine()
    engine.train()
    
    print("=" * 60)
    print("Urban Base Model Training Completed!")
    print("Model: Large_estimator_v4_urban_base.pt")
    print("Ready for Rural/Outdoor transfer learning")
    print("=" * 60)
    
    # Auto-upload (disabled in config)
    print("\n" + "="*50)
    print("Training completed! Checking auto-upload...")
    try:
        final_model_name = engine._conf['training'].get('saved_model_name', 'Large_estimator_v4_urban_base')
        auto_upload_models(engine._conf, f"{final_model_name}")
    except Exception as e:
        print(f"Note: Auto-upload disabled for cross-domain experiments")
        print("Models are saved locally in saved_model/ folder")
    print("="*50)