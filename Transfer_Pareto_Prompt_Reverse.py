"""
Prompt Learning Pareto Frontier Experiment
Tests multiple prompt lengths across all scenarios to create parameter-performance curve
Lengths: 50, 100, 200
Scenarios: InH, InF, UMi, UMa, RMa
Total: 3 lengths × 5 scenarios = 15 runs
"""

import torch
import yaml
from pathlib import Path
import wandb
from dataset import get_dataset_and_dataloader
from model.prompt_estimator_v4 import PromptEstimator_v4
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import copy

class PromptParetoEngine:
    def __init__(self, conf_file, scenario_name, length_config_name):
        conf_path = Path(__file__).parents[0].resolve() / 'config' / conf_file
        with open(conf_path, encoding='utf-8') as f:
            base_conf = yaml.safe_load(f)

        self._conf = copy.deepcopy(base_conf)
        scenario_conf = base_conf['scenarios'][scenario_name]
        length_conf = base_conf['prompt_configs'][length_config_name]

        # Update configurations
        self._conf['dataset']['channel_type'] = scenario_conf['channel_type']
        self._conf['dataset']['distance_ranges'] = scenario_conf['distance_ranges']
        self._conf['training']['wandb_proj'] = scenario_conf['wandb_proj']
        self._conf['training']['saved_model_name'] = f'Large_estimator_v4_to_{scenario_name}_prompt_{length_config_name}'

        # Update Prompt configuration
        self._conf['ch_estimation']['prompt']['prompt_length'] = length_conf['prompt_length']

        self._scenario_name = scenario_name
        self._length_config_name = length_config_name
        self._device = self._conf['training'].get('device', 'cuda:0')
        self._use_wandb = self._conf['training'].get('use_wandb', True)

        if self._use_wandb:
            wandb.init(
                project=self._conf['training']['wandb_proj'],
                name=f"{scenario_name}_Prompt_{length_config_name}",
                config=self._conf,
                reinit=True
            )
            self._conf = wandb.config

        self._dataset, self._dataloader = get_dataset_and_dataloader(self._conf['dataset'])

    def load_model(self):
        # Handle wandb.Config object (cannot deepcopy)
        if self._use_wandb and hasattr(self._conf, '__class__') and 'wandb' in str(type(self._conf)):
            temp_conf = dict(self._conf)
        else:
            temp_conf = copy.deepcopy(self._conf)

        temp_conf_path = Path(__file__).parents[0].resolve() / 'config' / f'temp_{self._scenario_name}_{self._length_config_name}_prompt.yaml'
        with open(temp_conf_path, 'w', encoding='utf-8') as f:
            yaml.dump(temp_conf, f)

        self._estimator = PromptEstimator_v4(f'temp_{self._scenario_name}_{self._length_config_name}_prompt.yaml').to(self._device)

        print(f"\n[{self._scenario_name}-{self._length_config_name}] Prompt Learning applied.")
        trainable_params = sum(p.numel() for p in self._estimator.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self._estimator.parameters())
        print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.3f}%)")

        temp_conf_path.unlink()
        self.set_optimizer()

    def set_optimizer(self):
        ch_params = [p for n, p in self._estimator.named_parameters() if p.requires_grad]
        self._ch_optimizer = torch.optim.Adam([{"params": ch_params}], lr=self._conf['training']['lr'])

        if self._conf['training'].get('use_scheduler', False):
            self._ch_scheduler = get_cosine_schedule_with_warmup(
                self._ch_optimizer,
                num_warmup_steps=self._conf['training'].get('num_warmup_steps', 0),
                num_training_steps=self._conf['training']['num_iter']
            )
        else:
            self._ch_scheduler = None

    def train(self):
        ch_loss_weight = self._conf['training'].get('ch_loss_weight', 1)
        num_iter = self._conf['training']['num_iter']
        logging_step = self._conf['training']['logging_step']
        model_save_step = self._conf['training']['model_save_step']

        print(f"\n{'='*70}")
        print(f"Prompt Pareto: {self._scenario_name} - {self._length_config_name}")
        print(f"{'='*70}\n")

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
            ch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._estimator.parameters(), max_norm=self._conf['training']['max_norm'])
            self._ch_optimizer.step()

            if self._ch_scheduler is not None:
                self._ch_scheduler.step()

            if (it + 1) % logging_step == 0:
                current_lr = self._ch_optimizer.param_groups[0]['lr']
                print(f"[{self._scenario_name}-{self._length_config_name}] Iter {it + 1}/{num_iter} | "
                      f"Loss: {ch_loss.item():.6f} | NMSE: {ch_nmse.item():.6f} | LR: {current_lr:.6f}")

                if self._use_wandb:
                    wandb.log({
                        'ch_loss': ch_loss.item(),
                        'ch_nmse': ch_nmse.item(),
                        'ch_mse': ch_mse.item(),
                        'learning_rate': current_lr,
                        'iteration': it + 1
                    })

            if (it + 1) % model_save_step == 0 or (it + 1) == num_iter:
                self.save_model(iteration=it + 1)

        if self._use_wandb:
            wandb.finish()

    def save_model(self, iteration):
        saved_model_dir = Path(__file__).parents[0].resolve() / 'saved_model' / 'pareto'
        saved_model_dir.mkdir(parents=True, exist_ok=True)

        base_name = self._conf['training']['saved_model_name']
        if iteration == self._conf['training']['num_iter']:
            save_path = saved_model_dir / f"{base_name}.pt"
        else:
            save_path = saved_model_dir / f"{base_name}_iter_{iteration}.pt"

        torch.save(self._estimator, save_path)
        print(f"[{self._scenario_name}-{self._length_config_name}] Saved: {save_path}")


def main():
    config_file = 'config_pareto_prompt.yaml'
    scenarios = ['RMa', 'UMa', 'UMi', 'InF', 'InH']  # REVERSED
    length_configs = ['len50', 'len100', 'len200']

    print("\n" + "="*70)
    print("Prompt Learning Pareto Frontier Experiment")
    print("="*70)
    print(f"Lengths: {', '.join(length_configs)}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Total runs: {len(length_configs)} × {len(scenarios)} = {len(length_configs) * len(scenarios)}")
    print("="*70 + "\n")

    for scenario_idx, scenario in enumerate(scenarios, 1):
        for length_idx, length_config in enumerate(length_configs, 1):
            run_num = (scenario_idx - 1) * len(length_configs) + length_idx
            total_runs = len(scenarios) * len(length_configs)

            print(f"\n{'#'*70}")
            print(f"# Run {run_num}/{total_runs}: {scenario} - {length_config}")
            print(f"{'#'*70}\n")

            try:
                engine = PromptParetoEngine(config_file, scenario, length_config)
                engine.load_model()
                engine.train()

                del engine
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError in {scenario}-{length_config}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "="*70)
    print("Prompt Pareto Frontier Experiment Completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
