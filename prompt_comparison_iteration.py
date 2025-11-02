"""
Prompt Learning Performance Comparison Tool with Iteration Analysis

이 코드는 Prompt Learning 기법의 성능을 Base 모델과 비교하고 iteration별 성능 변화를 분석합니다.

주요 기능:
1. Base v4 모델과 Prompt Learning 전이학습 모델들의 성능 비교
2. Iteration별 성능 변화 분석 (수렴 곡선)
3. 파라미터 효율성 분석

테스트 대상 모델:
- Base_v4_Final: 베이스 모델
- Prompt_InF_50_150k: Prompt 50, 150k iter로 InF 전이학습
- Prompt_RMa_50_150k: Prompt 50, 150k iter로 RMa 전이학습

출력:
- prompt_vs_base_comparison.png: 모든 모델 성능 비교
- prompt_improvement_vs_base.png: Base 대비 개선도
- prompt_iteration_analysis.png: Iteration별 성능 변화 (NEW)
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
        
    def find_all_iterations(self, model_prefix):
        """특정 모델의 모든 iteration 파일 찾기"""
        saved_model_dir = Path(__file__).parent / 'saved_model'
        import glob
        import re

        pattern = str(saved_model_dir / f"{model_prefix}_iter_*.pt")
        iter_files = glob.glob(pattern)

        # final 파일도 포함
        final_path = saved_model_dir / f"{model_prefix}.pt"
        if final_path.exists():
            iter_files.append(str(final_path))

        # iteration 번호 추출 및 정렬
        iteration_data = []
        for file_path in iter_files:
            if '_iter_' in file_path:
                match = re.search(r'_iter_(\d+)\.pt', file_path)
                if match:
                    iteration = int(match.group(1))
                    iteration_data.append((iteration, file_path))
            elif file_path.endswith(f'{model_prefix}.pt'):
                iteration_data.append((150000, file_path))  # final은 150k로 가정

        iteration_data.sort(key=lambda x: x[0])
        return iteration_data

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
            best_iter = 150000  # 새로운 기본값

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

        # 다른 iteration들도 확인 (fallback) - 150k 기준으로 수정
        iterations_to_check.extend([150000, 140000, 130000, 120000, 110000, 100000])

        for iteration in iterations_to_check:
            if iteration == 150000:
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
        """학습된 모델들 로드 - 새로운 prompt_only_50_150000 모델 우선"""
        models = {}
        saved_model_dir = Path(__file__).parent / 'saved_model'

        # Base v4 모델 로드
        base_path = saved_model_dir / 'Large_estimator_v4_base_final.pt'
        if base_path.exists():
            try:
                base_model = torch.load(base_path, map_location=self.device, weights_only=False)
                base_model.eval()
                models['Base_v4_Final'] = base_model
                print("[OK] Loaded Base_v4_Final")
            except Exception as e:
                print(f"[ERROR] Failed to load Base_v4_Final: {e}")
        else:
            print("[WARNING] Base model not found")

        # 새로운 Prompt InF 모델 로드 (100 tokens, 150k iter)
        prompt_inf_model, prompt_inf_iter, prompt_inf_path = self.find_best_prompt_iteration_model(
            'Large_estimator_v4_to_InF_prompt_only_100_150000'
        )
        if prompt_inf_model is not None:
            models['Prompt_InF_100_150k'] = prompt_inf_model
            iter_label = f"{prompt_inf_iter/1000:.0f}k"
            print(f"[OK] Loaded Prompt_InF_100_150k (@ {iter_label})")

            if hasattr(prompt_inf_model, 'ch_tf') and hasattr(prompt_inf_model.ch_tf, '_prompt_tokens'):
                prompt_shape = prompt_inf_model.ch_tf._prompt_tokens.shape
                print(f"[DEBUG] Prompt tokens: {prompt_shape}")
        else:
            print("[WARNING] Prompt InF 100_150k model not found")

        # 새로운 Prompt RMa 모델 로드 (100 tokens, 150k iter)
        prompt_rma_model, prompt_rma_iter, prompt_rma_path = self.find_best_prompt_iteration_model(
            'Large_estimator_v4_to_RMa_prompt_only_100_150000'
        )
        if prompt_rma_model is not None:
            models['Prompt_RMa_100_150k'] = prompt_rma_model
            iter_label = f"{prompt_rma_iter/1000:.0f}k"
            print(f"[OK] Loaded Prompt_RMa_100_150k (@ {iter_label})")

            if hasattr(prompt_rma_model, 'ch_tf') and hasattr(prompt_rma_model.ch_tf, '_prompt_tokens'):
                prompt_shape = prompt_rma_model.ch_tf._prompt_tokens.shape
                print(f"[DEBUG] Prompt tokens: {prompt_shape}")
        else:
            print("[WARNING] Prompt RMa 100_150k model not found")
            
        # 로드된 모델 개수 확인
        print(f"\n[INFO] Total loaded models: {len(models)}")
        for model_name in models.keys():
            print(f"[INFO] - {model_name}")
        
        return models
    
    def load_test_data(self):
        """간단한 테스트 데이터 로드"""
        test_data_dir = Path(__file__).parent / 'simple_test_data'
        datasets = {}
        
        for dataset_name in ['InF', 'RMa']:
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

    def test_model_at_iteration(self, model_path, config_file, rx_tensor, ch_true):
        """특정 iteration의 모델 테스트"""
        try:
            saved_model = torch.load(model_path, map_location=self.device, weights_only=False)

            if hasattr(saved_model, 'eval'):
                model = saved_model
            else:
                model = PromptEstimator_v4(config_file).to(self.device)
                if hasattr(saved_model, 'state_dict'):
                    state_dict = saved_model.state_dict()
                else:
                    state_dict = saved_model
                model.load_state_dict(state_dict, strict=False)

            model.eval()

            with torch.no_grad():
                ch_est, _ = model(rx_tensor)
                ch_est_np = ch_est.cpu().numpy()
                nmse = self.calculate_nmse(ch_est_np, ch_true)
                nmse_db = 10 * np.log10(nmse)

            return nmse_db
        except Exception as e:
            print(f"[ERROR] Failed at {model_path}: {e}")
            return np.nan

    def analyze_iterations(self, datasets):
        """Iteration별 성능 분석"""
        print("\n" + "="*60)
        print("Analyzing performance across iterations")
        print("="*60)

        iteration_results = {
            'InF': {'iterations': [], 'nmse': []},
            'RMa': {'iterations': [], 'nmse': []}
        }

        # InF 모델 iteration 분석
        inf_iterations = self.find_all_iterations('Large_estimator_v4_to_InF_prompt_only_100_150000')
        rma_iterations = self.find_all_iterations('Large_estimator_v4_to_RMa_prompt_only_100_150000')

        # InF 분석
        if 'InF' in datasets and inf_iterations:
            rx_input, ch_true = datasets['InF']
            rx_tensor = torch.tensor(rx_input, dtype=torch.float32).to(self.device)

            print("\nAnalyzing InF iterations:")
            for iteration, model_path in inf_iterations:
                nmse_db = self.test_model_at_iteration(
                    model_path,
                    'config_transfer_v4_prompt_InF.yaml',
                    rx_tensor,
                    ch_true
                )
                if not np.isnan(nmse_db):
                    iteration_results['InF']['iterations'].append(iteration)
                    iteration_results['InF']['nmse'].append(nmse_db)
                    print(f"  Iteration {iteration:6d}: {nmse_db:.2f} dB")

        # RMa 분석
        if 'RMa' in datasets and rma_iterations:
            rx_input, ch_true = datasets['RMa']
            rx_tensor = torch.tensor(rx_input, dtype=torch.float32).to(self.device)

            print("\nAnalyzing RMa iterations:")
            for iteration, model_path in rma_iterations:
                nmse_db = self.test_model_at_iteration(
                    model_path,
                    'config_transfer_v4_prompt_RMa.yaml',
                    rx_tensor,
                    ch_true
                )
                if not np.isnan(nmse_db):
                    iteration_results['RMa']['iterations'].append(iteration)
                    iteration_results['RMa']['nmse'].append(nmse_db)
                    print(f"  Iteration {iteration:6d}: {nmse_db:.2f} dB")

        return iteration_results

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
                    print(f"  {model_name:<20}: ERROR - {e}")
                    results[dataset_name][model_name] = np.nan

        # Iteration별 분석 추가
        iteration_results = self.analyze_iterations(datasets)

        # 결과 요약
        self.print_summary(results)

        # 플롯 그리기
        self.plot_results(results, iteration_results)

        return results, iteration_results
    
    def print_summary(self, results):
        """결과 요약 출력"""
        print("\n" + "="*80)
        print("SUMMARY - Prompt Learning Performance (100 tokens, 150k iterations)")
        print("="*80)

        print(f"{'Dataset':<15} {'Base_v4':<12} {'Prompt_InF_100_150k':<20} {'Prompt_RMa_100_150k':<20}")
        print("-" * 80)

        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_Final', np.nan)
            prompt_inf_nmse = results[dataset_name].get('Prompt_InF_100_150k', np.nan)
            prompt_rma_nmse = results[dataset_name].get('Prompt_RMa_100_150k', np.nan)

            base_str = f"{base_nmse:.2f}" if not np.isnan(base_nmse) else "N/A"
            prompt_inf_str = f"{prompt_inf_nmse:.2f}" if not np.isnan(prompt_inf_nmse) else "N/A"
            prompt_rma_str = f"{prompt_rma_nmse:.2f}" if not np.isnan(prompt_rma_nmse) else "N/A"

            print(f"{dataset_name:<15} {base_str:<12} {prompt_inf_str:<20} {prompt_rma_str:<20}")

        print("="*80)

        # 개선량 분석
        print("\nPerformance Improvements (vs Base_v4_Final):")
        print("-" * 60)
        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_Final', np.nan)
            prompt_inf_nmse = results[dataset_name].get('Prompt_InF_100_150k', np.nan)
            prompt_rma_nmse = results[dataset_name].get('Prompt_RMa_100_150k', np.nan)

            print(f"\n{dataset_name}:")
            if not np.isnan(base_nmse):
                if not np.isnan(prompt_inf_nmse):
                    improvement = base_nmse - prompt_inf_nmse
                    print(f"  Prompt InF (100 tokens, 150k): {improvement:+.2f} dB")
                if not np.isnan(prompt_rma_nmse):
                    improvement = base_nmse - prompt_rma_nmse
                    print(f"  Prompt RMa (100 tokens, 150k): {improvement:+.2f} dB")

        print("="*60)
    
    def plot_results(self, results, iteration_results):
        """결과 시각화 - 3개 그래프"""
        model_names = ['Base_v4_Final', 'Prompt_InF_100_150k', 'Prompt_RMa_100_150k']
        datasets = ['InF', 'RMa']
        
        # 첫 번째 플롯: 전체 모델 성능 비교
        plt.figure(figsize=(12, 6))

        bar_width = 0.25
        x_positions = np.arange(len(datasets))
        colors = ['#3498db', '#2ecc71', '#e74c3c']

        for i, model in enumerate(model_names):
            values = []
            for dataset in datasets:
                value = results.get(dataset, {}).get(model, np.nan)
                values.append(value)

            valid_values = []
            valid_positions = []
            for j, val in enumerate(values):
                if not np.isnan(val):
                    valid_values.append(val)
                    valid_positions.append(x_positions[j] + i * bar_width)

            if valid_values:
                if model == 'Base_v4_Final':
                    label = 'Base v4'
                elif model == 'Prompt_InF_100_150k':
                    label = 'v4 + Prompt InF (100, 150k)'
                elif model == 'Prompt_RMa_100_150k':
                    label = 'v4 + Prompt RMa (100, 150k)'
                else:
                    label = model

                bars = plt.bar(valid_positions, valid_values, bar_width,
                              label=label, color=colors[i], alpha=0.8)

                for bar, val in zip(bars, valid_values):
                    plt.text(bar.get_x() + bar.get_width()/2., val + 0.3,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        plt.xlabel('Test Environment', fontsize=12)
        plt.ylabel('NMSE (dB)', fontsize=12)
        plt.title('Prompt Learning Performance Comparison (100 tokens, 150k iterations)', fontsize=14)
        plt.xticks(x_positions + bar_width, datasets)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = 'prompt_vs_base_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.show()

        # 두 번째 플롯: 개선도 비교
        if 'Base_v4_Final' in [m for d in results.values() for m in d.keys()]:
            plt.figure(figsize=(10, 6))

            improvements = []
            model_labels = []

            for dataset in datasets:
                base_nmse = results.get(dataset, {}).get('Base_v4_Final', np.nan)

                if 'Prompt_InF_100_150k' in results.get(dataset, {}):
                    prompt_inf_nmse = results[dataset]['Prompt_InF_100_150k']
                    if not np.isnan(base_nmse) and not np.isnan(prompt_inf_nmse):
                        improvement = base_nmse - prompt_inf_nmse
                        improvements.append(improvement)
                        model_labels.append(f'{dataset}\n(Prompt InF Only)')

                if 'Prompt_RMa_100_150k' in results.get(dataset, {}):
                    prompt_rma_nmse = results[dataset]['Prompt_RMa_100_150k']
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

        # 세 번째 플롯: Iteration별 성능 변화 (NEW)
        if iteration_results:
            plt.figure(figsize=(12, 6))

            if iteration_results['InF']['iterations']:
                plt.plot(iteration_results['InF']['iterations'],
                        iteration_results['InF']['nmse'],
                        'o-', label='InF (Prompt InF)',
                        color='#2ecc71', linewidth=2, markersize=6)

            if iteration_results['RMa']['iterations']:
                plt.plot(iteration_results['RMa']['iterations'],
                        iteration_results['RMa']['nmse'],
                        's-', label='RMa (Prompt RMa)',
                        color='#e74c3c', linewidth=2, markersize=6)

            plt.xlabel('Training Iterations', fontsize=12)
            plt.ylabel('NMSE (dB)', fontsize=12)
            plt.title('Performance vs Training Iterations\n(Prompt Length: 100, Max Iterations: 150k)', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path_iter = 'prompt_iteration_analysis.png'
            plt.savefig(save_path_iter, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved {save_path_iter}")
            plt.show()

if __name__ == "__main__":
    print("Prompt Learning Model Testing with Iteration Analysis")
    print("=" * 60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    tester = SimpleModelTester(device=device)
    results, iteration_results = tester.test_models()