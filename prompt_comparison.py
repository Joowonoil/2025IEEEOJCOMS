"""
Prompt Learning Performance Comparison Tool

이 코드는 Prompt Learning 기법의 성능을 Base 모델과 비교 분석하는 도구입니다.

주요 기능:
1. Base v4 모델과 Prompt Learning 전이학습 모델들의 성능 비교
2. 파라미터 효율성 분석 (전체 모델의 ~0.01% 파라미터만 학습)
3. InF/RMa 환경별 Prompt 전이학습 효과 시각화

테스트 대상 모델:
- Base_v4_Final: 베이스 모델 (사전 학습된 v4 모델)
- Prompt_InF: Prompt Learning으로 InF 환경에 전이학습된 모델
- Prompt_RMa: Prompt Learning으로 RMa 환경에 전이학습된 모델

출력:
- prompt_vs_base_comparison.png: 모든 모델 성능 비교 차트
- prompt_improvement_vs_base.png: Base 모델 대비 개선도 차트
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from model.estimator_v4 import Estimator_v4
from model.prompt_estimator_v4 import PromptEstimator_v4

class SimpleModelTester:
    def __init__(self, device='cuda:0'):
        self.device = device
        
    def find_best_iteration_model(self, model_prefix, optimal_iter=None):
        """특정 모델의 모든 iteration 중 최적 모델 찾기"""
        saved_model_dir = Path(__file__).parent / 'saved_model'
        import glob
        import re
        
        # 모든 iteration 파일 찾기
        pattern = str(saved_model_dir / f"{model_prefix}_iter_*.pt")
        iter_files = glob.glob(pattern)
        
        # final 모델도 포함
        final_path = saved_model_dir / f"{model_prefix}.pt"
        if final_path.exists():
            iter_files.append(str(final_path))
        
        if not iter_files:
            return None, None, None
        
        best_model = None
        best_iter = None
        best_path = None
        
        # 지정된 최적 iteration 사용
        if optimal_iter:
            for file_path in iter_files:
                if f'_iter_{optimal_iter}.pt' in file_path:
                    best_path = file_path
                    best_iter = optimal_iter
                    break
        
        # 지정된 iteration이 없으면 final 사용
        if not best_path and final_path.exists():
            best_path = str(final_path)
            best_iter = 60000
        
        if best_path:
            try:
                best_model = torch.load(best_path, map_location=self.device, weights_only=False)
                best_model.eval()
                return best_model, best_iter, best_path
            except:
                pass
        
        return None, None, None
    
    def find_best_prompt_iteration_model(self, model_prefix, optimal_iter=None):
        """Prompt 모델의 최적 iteration 찾기"""
        saved_model_dir = Path(__file__).parent / 'saved_model'

        # 우선 순위: optimal_iter 지정된 경우 해당 iteration, 없으면 기본 순서
        iterations_to_check = []
        if optimal_iter:
            iterations_to_check.append(optimal_iter)

        # 다른 iteration들도 확인 (fallback)
        iterations_to_check.extend([60000, 50000, 40000, 30000, 20000, 10000])

        for iteration in iterations_to_check:
            if iteration == 60000:
                # Final 모델
                model_path = saved_model_dir / f'{model_prefix}.pt'
            else:
                # Iteration 모델
                model_path = saved_model_dir / f'{model_prefix}_iter_{iteration}.pt'

            if model_path.exists():
                try:
                    saved_model = torch.load(model_path, map_location=self.device, weights_only=False)

                    # 전체 모델 객체인지 state_dict인지 확인
                    if hasattr(saved_model, 'eval'):
                        # 전체 모델 객체인 경우
                        prompt_model = saved_model
                    else:
                        # state_dict인 경우 - 새로운 모델 구조에 로드
                        from model.prompt_estimator_v4 import PromptEstimator_v4

                        # 어떤 config를 사용할지 추정
                        if 'InF' in model_prefix:
                            config_file = 'config_transfer_v4_prompt_InF.yaml'
                        else:
                            config_file = 'config_transfer_v4_prompt_RMa.yaml'

                        prompt_model = PromptEstimator_v4(config_file).to(self.device)

                        # State dict 추출 및 로드
                        if hasattr(saved_model, 'state_dict'):
                            state_dict = saved_model.state_dict()
                        else:
                            state_dict = saved_model

                        prompt_model.load_state_dict(state_dict, strict=False)

                    prompt_model.eval()
                    return prompt_model, iteration, model_path
                except Exception as e:
                    print(f"[DEBUG] Failed to load {model_path}: {e}")
                    continue

        return None, None, None
    
    def load_models(self):
        """학습된 모델들 로드 - Base 모델과 Prompt 모델들"""
        models = {}
        saved_model_dir = Path(__file__).parent / 'saved_model'
        
        # Base v4 모델 로드 (Large_estimator_v4_base_final.pt)
        base_path = saved_model_dir / 'Large_estimator_v4_base_final.pt'
        if base_path.exists():
            try:
                base_model = torch.load(base_path, map_location=self.device, weights_only=False)
                base_model.eval()
                models['Base_v4_Final'] = base_model
                print("[OK] Loaded Base_v4_Final (Large_estimator_v4_base_final.pt)")
            except Exception as e:
                print(f"[ERROR] Failed to load Base_v4_Final: {e}")
        else:
            print("[WARNING] Large_estimator_v4_base_final.pt not found")
            
        # Prompt InF 모델 로드 (새로운 prompt_only 모델 우선)
        prompt_inf_model, prompt_inf_iter, prompt_inf_path = self.find_best_prompt_iteration_model('Large_estimator_v4_to_InF_prompt_only')
        if prompt_inf_model is not None:
            models['Prompt_InF_Only'] = prompt_inf_model
            iter_label = f"{prompt_inf_iter/1000:.0f}k" if prompt_inf_iter != 60000 else "final"
            print(f"[OK] Loaded Prompt_InF_Only (best @ {iter_label} iterations)")

            # 모델 타입 검증 (내부 로그)
            if hasattr(prompt_inf_model, 'ch_tf') and hasattr(prompt_inf_model.ch_tf, '_prompt_tokens'):
                prompt_shape = prompt_inf_model.ch_tf._prompt_tokens.shape
                print(f"[DEBUG] Prompt tokens detected: {prompt_shape}")
        else:
            print("[WARNING] Large_estimator_v4_to_InF_prompt_only.pt not found")

        # 기존 Prompt InF 모델도 로드 (비교용)
        prompt_inf_old_model, prompt_inf_old_iter, prompt_inf_old_path = self.find_best_prompt_iteration_model('Large_estimator_v4_to_InF_prompt')
        if prompt_inf_old_model is not None:
            models['Prompt_InF_Old'] = prompt_inf_old_model
            iter_label = f"{prompt_inf_old_iter/1000:.0f}k" if prompt_inf_old_iter != 60000 else "final"
            print(f"[OK] Loaded Prompt_InF_Old (best @ {iter_label} iterations)")

        # Prompt RMa 모델 로드 (새로운 prompt_only 모델 우선)
        prompt_rma_model, prompt_rma_iter, prompt_rma_path = self.find_best_prompt_iteration_model('Large_estimator_v4_to_RMa_prompt_only')
        if prompt_rma_model is not None:
            models['Prompt_RMa_Only'] = prompt_rma_model
            iter_label = f"{prompt_rma_iter/1000:.0f}k" if prompt_rma_iter != 60000 else "final"
            print(f"[OK] Loaded Prompt_RMa_Only (best @ {iter_label} iterations)")

            # 모델 타입 검증 (내부 로그)
            if hasattr(prompt_rma_model, 'ch_tf') and hasattr(prompt_rma_model.ch_tf, '_prompt_tokens'):
                prompt_shape = prompt_rma_model.ch_tf._prompt_tokens.shape
                print(f"[DEBUG] Prompt tokens detected: {prompt_shape}")
        else:
            print("[WARNING] Large_estimator_v4_to_RMa_prompt_only.pt not found")

        # 기존 Prompt RMa 모델도 로드 (비교용)
        prompt_rma_old_model, prompt_rma_old_iter, prompt_rma_old_path = self.find_best_prompt_iteration_model('Large_estimator_v4_to_RMa_prompt')
        if prompt_rma_old_model is not None:
            models['Prompt_RMa_Old'] = prompt_rma_old_model
            iter_label = f"{prompt_rma_old_iter/1000:.0f}k" if prompt_rma_old_iter != 60000 else "final"
            print(f"[OK] Loaded Prompt_RMa_Old (best @ {iter_label} iterations)")
            
        # 로드된 모델 개수 확인
        print(f"\n[INFO] Total loaded models: {len(models)}")
        for model_name in models.keys():
            print(f"[INFO] - {model_name}")
        
        return models
    
    def load_test_data(self):
        """간단한 테스트 데이터 로드"""
        test_data_dir = Path(__file__).parent / 'simple_test_data'
        datasets = {}
        
        for dataset_name in ['InF_50m', 'RMa_300m']:
            input_path = test_data_dir / f'{dataset_name}_input.npy'
            true_path = test_data_dir / f'{dataset_name}_true.npy'
            
            if input_path.exists() and true_path.exists():
                rx_input = np.load(input_path)
                ch_true = np.load(true_path)
                datasets[dataset_name] = (rx_input, ch_true)
                print(f"[OK] Loaded {dataset_name}: input {rx_input.shape}, true {ch_true.shape}")
            else:
                print(f"[WARNING] Test data for {dataset_name} not found")
        
        return datasets
    
    def calculate_nmse(self, ch_est, ch_true):
        """NMSE 계산 (학습과 동일한 방식)"""
        # 복소수를 실수부/허수부로 분리
        ch_true = np.stack((np.real(ch_true), np.imag(ch_true)), axis=-1)
        
        # NMSE 계산
        ch_mse = np.sum(np.square(ch_true - ch_est), axis=(1, 2)) / ch_true.shape[-1]
        ch_var = np.sum(np.square(ch_true), axis=(1, 2)) / ch_true.shape[-1]
        ch_nmse = np.mean(ch_mse / ch_var)
        
        return ch_nmse
    
    def test_models(self):
        """모델 테스트 실행"""
        models = self.load_models()
        datasets = self.load_test_data()
        
        if not models or not datasets:
            print("Models or datasets not loaded properly!")
            return
        
        results = {}
        
        print("\n" + "="*60)
        print("Simple Model Testing Results")
        print("="*60)
        
        for dataset_name, (rx_input, ch_true) in datasets.items():
            print(f"\nTesting on {dataset_name}:")
            results[dataset_name] = {}
            
            # 입력 데이터를 텐서로 변환
            rx_tensor = torch.tensor(rx_input, dtype=torch.float32).to(self.device)
            
            for model_name, model in models.items():
                try:
                    with torch.no_grad():
                        # 모델 추론
                        ch_est, _ = model(rx_tensor)
                        ch_est_np = ch_est.cpu().numpy()
                        
                        # NMSE 계산
                        nmse = self.calculate_nmse(ch_est_np, ch_true)
                        nmse_db = 10 * np.log10(nmse)
                        
                        results[dataset_name][model_name] = nmse_db
                        
                        print(f"  {model_name:<15}: {nmse_db:.2f} dB")
                        
                except Exception as e:
                    print(f"  {model_name:<15}: ERROR - {e}")
                    results[dataset_name][model_name] = np.nan
        
        # 결과 요약
        self.print_summary(results)
        
        # 플롯 그리기
        self.plot_results(results)
        
        return results
    
    def print_summary(self, results):
        """결과 요약 출력"""
        print("\n" + "="*100)
        print("SUMMARY - Prompt Learning Performance Comparison")
        print("="*100)

        # 새로운 모델 이름들
        print(f"{'Dataset':<15} {'Base_v4':<12} {'Prompt_InF_Only':<15} {'Prompt_RMa_Only':<15} {'Prompt_InF_Old':<15} {'Prompt_RMa_Old':<15}")
        print("-" * 90)

        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_Final', np.nan)
            prompt_inf_only_nmse = results[dataset_name].get('Prompt_InF_Only', np.nan)
            prompt_rma_only_nmse = results[dataset_name].get('Prompt_RMa_Only', np.nan)
            prompt_inf_old_nmse = results[dataset_name].get('Prompt_InF_Old', np.nan)
            prompt_rma_old_nmse = results[dataset_name].get('Prompt_RMa_Old', np.nan)

            # 포맷팅
            base_str = f"{base_nmse:.2f}" if not np.isnan(base_nmse) else "N/A"
            prompt_inf_only_str = f"{prompt_inf_only_nmse:.2f}" if not np.isnan(prompt_inf_only_nmse) else "N/A"
            prompt_rma_only_str = f"{prompt_rma_only_nmse:.2f}" if not np.isnan(prompt_rma_only_nmse) else "N/A"
            prompt_inf_old_str = f"{prompt_inf_old_nmse:.2f}" if not np.isnan(prompt_inf_old_nmse) else "N/A"
            prompt_rma_old_str = f"{prompt_rma_old_nmse:.2f}" if not np.isnan(prompt_rma_old_nmse) else "N/A"

            print(f"{dataset_name:<15} {base_str:<12} {prompt_inf_only_str:<15} {prompt_rma_only_str:<15} {prompt_inf_old_str:<15} {prompt_rma_old_str:<15}")

        print("="*100)

        # 개선량 분석
        print("\nPerformance Improvements (vs Base_v4_Final):")
        print("-" * 80)
        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_Final', np.nan)
            prompt_inf_only_nmse = results[dataset_name].get('Prompt_InF_Only', np.nan)
            prompt_rma_only_nmse = results[dataset_name].get('Prompt_RMa_Only', np.nan)
            prompt_inf_old_nmse = results[dataset_name].get('Prompt_InF_Old', np.nan)
            prompt_rma_old_nmse = results[dataset_name].get('Prompt_RMa_Old', np.nan)

            print(f"\n{dataset_name}:")
            if not np.isnan(base_nmse):
                if not np.isnan(prompt_inf_only_nmse):
                    improvement = base_nmse - prompt_inf_only_nmse
                    print(f"  Prompt InF Only (0.037% params): {improvement:+.2f} dB")
                if not np.isnan(prompt_rma_only_nmse):
                    improvement = base_nmse - prompt_rma_only_nmse
                    print(f"  Prompt RMa Only (0.037% params): {improvement:+.2f} dB")
                if not np.isnan(prompt_inf_old_nmse):
                    improvement = base_nmse - prompt_inf_old_nmse
                    print(f"  Prompt InF Old (full model): {improvement:+.2f} dB")
                if not np.isnan(prompt_rma_old_nmse):
                    improvement = base_nmse - prompt_rma_old_nmse
                    print(f"  Prompt RMa Old (full model): {improvement:+.2f} dB")

        print("="*70)
    
    def plot_results(self, results):
        """결과 시각화 - v3 스타일과 동일하게"""
        # 데이터 준비 - 새로운 prompt_only 모델들 우선 표시
        model_names = ['Base_v4_Final', 'Prompt_InF_Only', 'Prompt_RMa_Only', 'Prompt_InF_Old', 'Prompt_RMa_Old']
        datasets = ['InF_50m', 'RMa_300m']
        
        # 첫 번째 플롯: 전체 모델 성능 비교
        plt.figure(figsize=(14, 7))
        
        bar_width = 0.25
        x_positions = np.arange(len(datasets))
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, model in enumerate(model_names):
            values = []
            for dataset in datasets:
                value = results.get(dataset, {}).get(model, np.nan)
                values.append(value)
            
            # NaN이 아닌 값만 플롯
            valid_values = []
            valid_positions = []
            for j, val in enumerate(values):
                if not np.isnan(val):
                    valid_values.append(val)
                    valid_positions.append(x_positions[j] + i * bar_width)
            
            if valid_values:
                # 모델 이름 정리
                if model == 'Base_v4_Final':
                    label = 'Base v4'
                elif model == 'Prompt_InF_Only':
                    label = 'v4 + Prompt InF (0.037%)'
                elif model == 'Prompt_RMa_Only':
                    label = 'v4 + Prompt RMa (0.037%)'
                elif model == 'Prompt_InF_Old':
                    label = 'v4 + Prompt InF (Full)'
                elif model == 'Prompt_RMa_Old':
                    label = 'v4 + Prompt RMa (Full)'
                else:
                    label = model
                
                bars = plt.bar(valid_positions, valid_values, bar_width, 
                              label=label, color=colors[i % len(colors)], alpha=0.8)
                
                # 막대 위에 값 표시
                for bar, val in zip(bars, valid_values):
                    plt.text(bar.get_x() + bar.get_width()/2., val + 0.3,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Test Environment', fontsize=12)
        plt.ylabel('NMSE (dB)', fontsize=12)
        plt.title('Prompt Learning Performance Comparison', fontsize=14)
        plt.xticks(x_positions + bar_width, datasets)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 첫 번째 플롯 저장
        save_path = 'prompt_vs_base_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.show()
        
        # 두 번째 플롯: 개선도 비교 (베이스 모델 대비)
        if 'Base_v4_Final' in [m for d in results.values() for m in d.keys()]:
            plt.figure(figsize=(12, 6))
            
            improvements = []
            model_labels = []
            
            for dataset in datasets:
                base_nmse = results.get(dataset, {}).get('Base_v4_Final', np.nan)
                
                # Prompt InF Only 개선도
                if 'Prompt_InF_Only' in results.get(dataset, {}):
                    prompt_inf_nmse = results[dataset]['Prompt_InF_Only']
                    if not np.isnan(base_nmse) and not np.isnan(prompt_inf_nmse):
                        improvement = base_nmse - prompt_inf_nmse
                        improvements.append(improvement)
                        model_labels.append(f'{dataset}\n(Prompt InF Only)')

                # Prompt RMa Only 개선도
                if 'Prompt_RMa_Only' in results.get(dataset, {}):
                    prompt_rma_nmse = results[dataset]['Prompt_RMa_Only']
                    if not np.isnan(base_nmse) and not np.isnan(prompt_rma_nmse):
                        improvement = base_nmse - prompt_rma_nmse
                        improvements.append(improvement)
                        model_labels.append(f'{dataset}\n(Prompt RMa Only)')
            
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
                
                plt.xlabel('Dataset and Transfer Type', fontsize=12)
                plt.ylabel('NMSE Improvement vs Base (dB)', fontsize=12)
                plt.title('v4 Prompt Learning Performance Improvement\n(Compared to Base Model)', fontsize=14)
                plt.xticks(x_pos, model_labels)
                plt.grid(True, axis='y', alpha=0.3)
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                # 두 번째 플롯 저장
                save_path_imp = 'prompt_improvement_vs_base.png'
                plt.savefig(save_path_imp, dpi=300, bbox_inches='tight')
                print(f"[OK] Saved {save_path_imp}")
                plt.show()

if __name__ == "__main__":
    print("Prompt Learning Model Testing")
    print("=" * 40)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    tester = SimpleModelTester(device=device)
    results = tester.test_models()