"""
v3 Adapter Performance Comparison Tool

이 코드는 v3 Adapter 모델들의 성능을 비교 분석하는 도구입니다.

주요 기능:
1. Base v3 모델과 Adapter 전이학습 모델들의 성능 비교
2. InF/RMa 환경별 Adapter 전이학습 효과 비교
3. 파라미터 효율성 분석 (Adapter 파라미터 vs 전체 파라미터)
4. v3 아키텍처의 Adapter 방식 전이학습 효과 시각화

테스트 대상 모델:
- Base_v3: v3 베이스 모델 (engine_v3.py로 학습)
- InF_Adapter_v3: Adapter로 InF 전이학습된 모델
- RMa_Adapter_v3: Adapter로 RMa 전이학습된 모델

출력:
- v3_adapter_comparison.png: 모든 v3 모델 성능 비교 차트
- v3_adapter_efficiency.png: Adapter 파라미터 효율성 분석 차트
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from model.estimator_v3 import Estimator_v3

class V3AdapterTester:
    def __init__(self, device='cuda:0'):
        self.device = device
        
    def load_models(self):
        """학습된 v3 모델들 로드"""
        models = {}
        saved_model_dir = Path(__file__).parent / 'saved_model'
        
        # Base v3 모델 로드 (저장된 Estimator_v3 객체 직접 사용)
        base_path = saved_model_dir / 'Large_estimator_v3_base_final.pt'
        if base_path.exists():
            try:
                # 저장된 Estimator_v3 객체를 직접 로드
                base_model = torch.load(base_path, map_location=self.device)
                base_model.to(self.device)
                base_model.eval()
                
                models['Base_v3'] = base_model
                print("[OK] Loaded Base_v3 (Pure Transformer without Adapters)")
            except Exception as e:
                print(f"[ERROR] Failed to load Base_v3: {e}")
        else:
            print("[WARNING] Large_estimator_v3_base_final_iter_10000.pt not found")
            print("[INFO] Run engine_v3.py first to train the base model")
            
        # InF Adapter 모델 로드 - 기존 버전 (주석처리)
        # inf_adapter_path = saved_model_dir / 'Large_estimator_v3_to_InF_adapter_bottleneck10.pt'
        
        # 새로운 모델 경로 (saved_model/new에서 로드) - bottleneck=64로 훈련된 모델
        new_model_dir = saved_model_dir / 'new'
        inf_adapter_path = new_model_dir / 'Large_estimator_v3_to_InF_adapter_bottleneck10.pt'
        if inf_adapter_path.exists():
            try:
                # Adapter 모델은 state_dict로 저장되므로, 모델 구조를 만들고 로드
                from Transfer_v3_InF import TransferLearningEngine
                transfer_engine = TransferLearningEngine('config_transfer_v3_InF.yaml')
                transfer_engine.load_model()
                
                # 학습된 Adapter 가중치 로드
                adapter_state_dict = torch.load(inf_adapter_path, map_location=self.device)
                transfer_engine._estimator.load_state_dict(adapter_state_dict)
                transfer_engine._estimator.eval()
                
                models['InF_Adapter_v3'] = transfer_engine._estimator
                print("[OK] Loaded InF_Adapter_v3 (Transfer_v3_InF trained)")
                
                # Adapter 파라미터 수 확인
                adapter_params = sum(p.numel() for n, p in transfer_engine._estimator.named_parameters() 
                                   if p.requires_grad and 'adapter' in n)
                total_params = sum(p.numel() for p in transfer_engine._estimator.parameters())
                
                print(f"[INFO] InF Adapter parameters: {adapter_params:,}")
                print(f"[INFO] Total parameters: {total_params:,}")
                print(f"[INFO] Adapter ratio: {adapter_params/total_params:.1%}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load InF_Adapter_v3: {e}")
        else:
            print("[WARNING] Large_estimator_v3_to_InF_adapter.pt not found")
            print("[INFO] Run Transfer_v3_InF.py first to train the InF adapter model")
            
        # RMa Adapter 모델 로드 - 기존 버전 (주석처리)
        # rma_adapter_path = saved_model_dir / 'Large_estimator_v3_to_RMa_adapter_bottleneck10.pt'
        
        # 새로운 모델 경로 (saved_model/new에서 로드) - bottleneck=64로 훈련된 모델
        new_model_dir = saved_model_dir / 'new'
        rma_adapter_path = new_model_dir / 'Large_estimator_v3_to_RMa_adapter_bottleneck10.pt'
        if rma_adapter_path.exists():
            try:
                # Adapter 모델은 state_dict로 저장되므로, 모델 구조를 만들고 로드
                from Transfer_v3_RMa import TransferLearningEngine
                transfer_engine = TransferLearningEngine('config_transfer_v3_RMa.yaml')
                transfer_engine.load_model()
                
                # 학습된 Adapter 가중치 로드
                adapter_state_dict = torch.load(rma_adapter_path, map_location=self.device)
                transfer_engine._estimator.load_state_dict(adapter_state_dict)
                transfer_engine._estimator.eval()
                
                models['RMa_Adapter_v3'] = transfer_engine._estimator
                print("[OK] Loaded RMa_Adapter_v3 (Transfer_v3_RMa trained)")
                
                # Adapter 파라미터 수 확인
                adapter_params = sum(p.numel() for n, p in transfer_engine._estimator.named_parameters() 
                                   if p.requires_grad and 'adapter' in n)
                total_params = sum(p.numel() for p in transfer_engine._estimator.parameters())
                
                print(f"[INFO] RMa Adapter parameters: {adapter_params:,}")
                print(f"[INFO] Total parameters: {total_params:,}")
                print(f"[INFO] Adapter ratio: {adapter_params/total_params:.1%}")
                
            except Exception as e:
                print(f"[ERROR] Failed to load RMa_Adapter_v3: {e}")
        else:
            print("[WARNING] Large_estimator_v3_to_RMa_adapter.pt not found")
            print("[INFO] Run Transfer_v3_RMa.py first to train the RMa adapter model")
            
        return models
    
    def load_test_data(self):
        """실제 데이터셋 생성하여 테스트 데이터 생성"""
        from dataset import get_dataset_and_dataloader
        datasets = {}
        
        # InF 테스트 데이터 생성
        inf_params = {
            'channel_type': ["InF_Los", "InF_Nlos"],
            'batch_size': 8,
            'noise_spectral_density': -174.0,
            'subcarrier_spacing': 120.0,
            'transmit_power': 30.0,
            'distance_range': [40.0, 60.0],  # InF 거리 범위
            'carrier_freq': 28.0,
            'mod_order': 64,
            'ref_conf_dict': {'dmrs': (0, 3072, 6)},
            'fft_size': 4096,
            'num_guard_subcarriers': 1024,
            'num_symbol': 14,
            'cp_length': 590,
            'max_random_tap_delay_cp_proportion': 0.2,
            'rnd_seed': 0,
            'num_workers': 0,
            'is_phase_noise': False,
            'is_channel': True,
            'is_noise': True
        }
        
        try:
            inf_dataset, inf_dataloader = get_dataset_and_dataloader(params=inf_params)
            # 첫 번째 배치만 사용
            for data in inf_dataloader:
                rx_signal = data['ref_comp_rx_signal']
                true_channel = data['ch_freq']
                
                # 복소수를 실수/허수 분리
                rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)
                true_channel = np.stack((np.real(true_channel), np.imag(true_channel)), axis=-1)
                
                datasets['InF_50m'] = {
                    'input': torch.tensor(rx_signal, dtype=torch.float32).to(self.device),
                    'true': torch.tensor(true_channel, dtype=torch.float32).to(self.device)
                }
                print(f"[OK] Generated test dataset: InF_50m")
                break
        except Exception as e:
            print(f"[ERROR] Failed to generate InF test data: {e}")
            
        # RMa 테스트 데이터 생성  
        rma_params = {
            'channel_type': ["RMa_Los"],
            'batch_size': 8,
            'noise_spectral_density': -174.0,
            'subcarrier_spacing': 120.0,
            'transmit_power': 30.0,
            'distance_range': [300.0, 500.0],  # RMa 거리 범위
            'carrier_freq': 28.0,
            'mod_order': 64,
            'ref_conf_dict': {'dmrs': (0, 3072, 6)},
            'fft_size': 4096,
            'num_guard_subcarriers': 1024,
            'num_symbol': 14,
            'cp_length': 590,
            'max_random_tap_delay_cp_proportion': 0.2,
            'rnd_seed': 0,
            'num_workers': 0,
            'is_phase_noise': False,
            'is_channel': True,
            'is_noise': True
        }
        
        try:
            rma_dataset, rma_dataloader = get_dataset_and_dataloader(params=rma_params)
            # 첫 번째 배치만 사용
            for data in rma_dataloader:
                rx_signal = data['ref_comp_rx_signal']
                true_channel = data['ch_freq']
                
                # 복소수를 실수/허수 분리
                rx_signal = np.stack((np.real(rx_signal), np.imag(rx_signal)), axis=-1)
                true_channel = np.stack((np.real(true_channel), np.imag(true_channel)), axis=-1)
                
                datasets['RMa_300m'] = {
                    'input': torch.tensor(rx_signal, dtype=torch.float32).to(self.device),
                    'true': torch.tensor(true_channel, dtype=torch.float32).to(self.device)
                }
                print(f"[OK] Generated test dataset: RMa_300m")
                break
        except Exception as e:
            print(f"[ERROR] Failed to generate RMa test data: {e}")
                
        return datasets
    
    def evaluate_model(self, model, test_data):
        """모델 성능 평가 (NMSE 계산)"""
        with torch.no_grad():
            input_signal = test_data['input']
            true_channel = test_data['true']
            
            # 모델 추론
            estimated_channel, _ = model(input_signal)
            
            # NMSE 계산
            mse = torch.mean(torch.square(true_channel - estimated_channel))
            var = torch.mean(torch.square(true_channel))
            nmse = mse / var
            nmse_db = 10 * torch.log10(nmse)
            
            return nmse_db.item()
    
    def run_comparison(self):
        """v3 모델들 성능 비교 실행"""
        print("=" * 60)
        print("v3 Adapter Performance Comparison Started")
        print("=" * 60)
        
        # 모델 로드
        print("\nLoading v3 models...")
        models = self.load_models()
        
        if not models:
            print("[ERROR] No models loaded. Please train models first.")
            return
        
        # 테스트 데이터 로드
        print("\nLoading test datasets...")
        datasets = self.load_test_data()
        
        if not datasets:
            print("[ERROR] No test data found. Please prepare test data.")
            return
        
        # 성능 평가
        print("\nEvaluating model performance...")
        results = {}
        
        for model_name, model in models.items():
            results[model_name] = {}
            print(f"\nTesting {model_name}:")
            
            for dataset_name, test_data in datasets.items():
                try:
                    nmse_db = self.evaluate_model(model, test_data)
                    results[model_name][dataset_name] = nmse_db
                    print(f"  {dataset_name}: {nmse_db:.2f} dB")
                except Exception as e:
                    print(f"  {dataset_name}: ERROR - {e}")
                    results[model_name][dataset_name] = float('inf')
        
        # 결과 시각화
        print("\nGenerating comparison charts...")
        self.plot_results(results)
        
        # 요약 출력
        print("\n" + "=" * 60)
        print("v3 Adapter Performance Summary")
        print("=" * 60)
        for model_name, model_results in results.items():
            print(f"\n{model_name}:")
            for dataset_name, nmse_db in model_results.items():
                if nmse_db != float('inf'):
                    print(f"  {dataset_name}: {nmse_db:.2f} dB")
                else:
                    print(f"  {dataset_name}: ERROR")
        
        print("\nv3 Adapter comparison completed!")
        return results
    
    def plot_results(self, results):
        """결과 시각화 - v4 스타일과 동일하게"""
        # 데이터 준비
        model_names = list(results.keys())
        datasets = ['InF_50m', 'RMa_300m']
        
        # 첫 번째 플롯: 전체 모델 성능 비교
        plt.figure(figsize=(14, 7))
        
        bar_width = 0.25
        x_positions = np.arange(len(datasets))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, model in enumerate(model_names):
            values = []
            for dataset in datasets:
                value = results[model].get(dataset, np.nan)
                values.append(value)
            
            # NaN이 아닌 값만 플롯
            valid_values = []
            valid_positions = []
            for j, val in enumerate(values):
                if not np.isnan(val):
                    valid_values.append(val)
                    valid_positions.append(x_positions[j] + i * bar_width)
            
            if valid_values:
                bars = plt.bar(valid_positions, valid_values, bar_width, 
                              label=model, color=colors[i % len(colors)], alpha=0.8)
                
                # 막대 위에 값 표시
                for bar, val in zip(bars, valid_values):
                    plt.text(bar.get_x() + bar.get_width()/2., val + 0.3,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Test Environment', fontsize=12)
        plt.ylabel('NMSE (dB)', fontsize=12)
        plt.title('Adapter Transfer Learning Performance Comparison', fontsize=14)
        plt.xticks(x_positions + bar_width, datasets)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 첫 번째 플롯 저장
        save_path = 'v3_adapter_comparison_new_models.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.show()
        
        # 두 번째 플롯: 개선도 비교 (베이스 모델 대비)
        if 'Base_v3' in results:
            plt.figure(figsize=(12, 6))
            
            improvements = []
            model_labels = []
            
            for dataset in datasets:
                base_nmse = results['Base_v3'].get(dataset, np.nan)
                
                for model_name in model_names:
                    if 'Adapter' in model_name and not np.isnan(base_nmse):
                        model_nmse = results[model_name].get(dataset, np.nan)
                        if not np.isnan(model_nmse):
                            improvement = base_nmse - model_nmse
                            improvements.append(improvement)
                            
                            # 모델 이름과 환경 조합
                            env_name = 'InF' if 'InF' in model_name else 'RMa'
                            model_labels.append(f'{dataset}\n({env_name} Adapter)')
            
            if improvements:
                x_pos = np.arange(len(improvements))
                colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
                
                bars = plt.bar(x_pos, improvements, color=colors_imp, alpha=0.8)
                
                # 값 표시
                for bar, imp in zip(bars, improvements):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., 
                            height + 0.02 if height > 0 else height - 0.05,
                            f'{height:.2f}', ha='center', 
                            va='bottom' if height > 0 else 'top', fontsize=10)
                
                plt.xlabel('Dataset and Adapter Type', fontsize=12)
                plt.ylabel('NMSE Improvement vs Base (dB)', fontsize=12)
                plt.title('Adapter Performance Improvement\n(Compared to Base Model)', fontsize=14)
                plt.xticks(x_pos, model_labels)
                plt.grid(True, axis='y', alpha=0.3)
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                # 두 번째 플롯 저장
                save_path_imp = 'v3_adapter_improvement_new_models.png'
                plt.savefig(save_path_imp, dpi=300, bbox_inches='tight')
                print(f"[OK] Saved {save_path_imp}")
                plt.show()

def main():
    tester = V3AdapterTester()
    results = tester.run_comparison()
    return results

if __name__ == "__main__":
    main()