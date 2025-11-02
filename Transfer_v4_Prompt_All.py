import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import wandb
from dataset import get_dataset_and_dataloader
from model.prompt_estimator_v4 import PromptEstimator_v4
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from utils.plot_signal import plot_signal
from utils.auto_upload import auto_upload_models
import copy

class PromptTransferLearningEngine:
    """
    Prompt Learning Transfer Learning Engine
    Parameter-efficient transfer learning using Prompt Learning (Prefix Tuning)
    """
    def __init__(self, conf_file, scenario_name):
        # Load configuration file
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file
        with open(conf_path, encoding='utf-8') as f:
            base_conf = yaml.safe_load(f)

        # Create scenario-specific configuration
        self._conf = copy.deepcopy(base_conf)
        scenario_conf = base_conf['scenarios'][scenario_name]

        # Update dataset configuration with scenario-specific settings
        self._conf['dataset']['channel_type'] = scenario_conf['channel_type']
        self._conf['dataset']['distance_ranges'] = scenario_conf['distance_ranges']

        # Update training configuration with scenario-specific settings
        self._conf['training']['wandb_proj'] = scenario_conf['wandb_proj']
        self._conf['training']['saved_model_name'] = scenario_conf['saved_model_name']

        self._conf_file = conf_file
        self._scenario_name = scenario_name

        # Load parameters from configuration
        self._device = self._conf['training'].get('device', 'cuda:0')
        self._use_wandb = self._conf['training'].get('use_wandb', True)
        self._wandb_proj = self._conf['training']['wandb_proj']

        # Initialize WandB for this scenario
        if self._use_wandb:
            wandb.init(
                project=self._wandb_proj,
                name=f"{scenario_name}_Prompt",
                config=self._conf,
                reinit=True  # Allow multiple wandb.init() calls
            )
            self._conf = wandb.config

        # Load training and validation datasets
        self._dataset, self._dataloader = get_dataset_and_dataloader(self._conf['dataset'])
        self._val_dataset, self._val_dataloader = None, None

    def load_model(self):
        """Load model with Prompt Learning"""
        # Create a temporary config file for this scenario
        temp_conf = copy.deepcopy(self._conf)
        temp_conf_path = Path(__file__).parents[0].resolve() / 'config' / f'temp_{self._scenario_name}_prompt.yaml'
        with open(temp_conf_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_conf, f)

        # Initialize PromptEstimator_v4 (handles Prompt Learning automatically)
        self._estimator = PromptEstimator_v4(f'temp_{self._scenario_name}_prompt.yaml').to(self._device)

        print(f"\n[{self._scenario_name}] Prompt Learning applied to Estimator_v4.")

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self._estimator.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self._estimator.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.3f}%)")

        # Clean up temporary config file
        temp_conf_path.unlink()

        self.set_optimizer()

    def set_optimizer(self):
        """Set optimizer for trainable parameters (Prompt tokens)"""
        ch_params = [p for n, p in self._estimator.named_parameters() if p.requires_grad]
        self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._conf['training']['lr'])

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
        """Train the model"""
        ch_loss_weight = self._conf['training'].get('ch_loss_weight', 1)
        num_iter = self._conf['training']['num_iter']
        logging_step = self._conf['training']['logging_step']
        model_save_step = self._conf['training']['model_save_step']

        print(f"\n{'='*60}")
        print(f"Starting Prompt Transfer Learning: {self._scenario_name}")
        print(f"{'='*60}\n")

        for it, data in enumerate(self._dataloader):
            if it >= num_iter:
                break

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

            # Check trainable parameters before backward
            trainable_params = [p for p in self._estimator.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                print(f"ERROR: No trainable parameters found at iteration {it + 1}")
                break

            try:
                ch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._estimator.parameters(), max_norm=self._conf['training']['max_norm'])
                self._ch_optimizer.step()

                if self._ch_scheduler is not None:
                    self._ch_scheduler.step()

            except RuntimeError as e:
                print(f"Error during backward pass at iteration {it + 1}: {e}")
                break

            # Logging
            if (it + 1) % logging_step == 0:
                current_lr = self._ch_optimizer.param_groups[0]['lr']
                print(f"[{self._scenario_name}] Iter {it + 1}/{num_iter} | "
                      f"Ch Loss: {ch_loss.item():.6f} | Ch NMSE: {ch_nmse.item():.6f} | "
                      f"Ch MSE: {ch_mse.item():.6f} | LR: {current_lr:.6f}")

                if self._use_wandb:
                    wandb.log({
                        f'{self._scenario_name}/ch_loss': ch_loss.item(),
                        f'{self._scenario_name}/ch_nmse': ch_nmse.item(),
                        f'{self._scenario_name}/ch_mse': ch_mse.item(),
                        f'{self._scenario_name}/learning_rate': current_lr,
                        f'{self._scenario_name}/iteration': it + 1
                    })

            # Save model checkpoints
            if (it + 1) % model_save_step == 0 or (it + 1) == num_iter:
                self.save_model(iteration=it + 1)

        print(f"\n{'='*60}")
        print(f"Completed Prompt Transfer Learning: {self._scenario_name}")
        print(f"{'='*60}\n")

        # Finish WandB run for this scenario
        if self._use_wandb:
            wandb.finish()

    def save_model(self, iteration):
        """Save model checkpoint"""
        saved_model_dir = Path(__file__).parents[0].resolve() / 'saved_model'
        saved_model_dir.mkdir(exist_ok=True)

        base_name = self._conf['training']['saved_model_name']

        if iteration == self._conf['training']['num_iter']:
            # Final model
            save_path = saved_model_dir / f"{base_name}.pt"
        else:
            # Intermediate checkpoint
            save_path = saved_model_dir / f"{base_name}_iter_{iteration}.pt"

        # Save entire estimator (includes prompt tokens)
        torch.save(self._estimator, save_path)
        print(f"[{self._scenario_name}] Model saved: {save_path}")


def main():
    """Main function to run all 5 scenarios sequentially"""
    config_file = 'config_transfer_v4_prompt_all.yaml'
    scenarios = ['InH', 'InF', 'UMi', 'UMa', 'RMa']

    print("\n" + "="*70)
    print("Prompt Learning Transfer Learning - All Scenarios")
    print("="*70)
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print("="*70 + "\n")

    for idx, scenario in enumerate(scenarios, 1):
        print(f"\n{'#'*70}")
        print(f"# Scenario {idx}/{len(scenarios)}: {scenario}")
        print(f"{'#'*70}\n")

        try:
            engine = PromptTransferLearningEngine(config_file, scenario)
            engine.load_model()
            engine.train()

            # Clean up GPU memory
            del engine
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError in scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*70)
    print("All scenarios completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
