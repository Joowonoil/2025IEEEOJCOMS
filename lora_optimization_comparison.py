"""
LoRA Optimization Performance Comparison Tool

이 코드는 LoRA (Low-Rank Adaptation) 최적화 전후의 성능을 비교 분석하는 도구입니다.

주요 기능:
1. Base v4 모델과 LoRA 전이학습 모델들의 성능 비교
2. 최적화된 LoRA (rank=4, 3개 모듈) vs 기존 LoRA (rank=8, 6개 모듈) 성능 비교
3. 파라미터 효율성 분석 (114,688개 → 26,624개, 76.8% 감소)
4. InF/RMa 환경별 전이학습 효과 시각화

테스트 대상 모델:
- Base_v4_1k: 베이스 모델 (1k iteration 학습)
- InF_Transfer_Optimized: 최적화된 LoRA로 InF 전이학습된 모델
- RMa_Transfer_Optimized: 최적화된 LoRA로 RMa 전이학습된 모델
- InF_Transfer_Old: 기존 LoRA로 InF 전이학습된 모델 (비교용)
- RMa_Transfer_Old: 기존 LoRA로 RMa 전이학습된 모델 (비교용)

출력:
- optimized_lora_comparison.png: 모든 모델 성능 비교 차트
- lora_optimization_comparison.png: 최적화 전후 직접 비교 차트
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from model.estimator_v4 import Estimator_v4

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
                best_model = torch.load(best_path, map_location=self.device)
                best_model.eval()
                return best_model, best_iter, best_path
            except:
                pass
        
        return None, None, None
    
    def find_best_iteration_model_from_new(self, model_name, optimal_iter=None, model_dir=None):
        """새로운 모델 디렉토리에서 최적 iteration 모델 찾기"""
        if model_dir is None:
            model_dir = Path(__file__).parent / 'saved_model' / 'new'
            
        # 우선 순위: optimal_iter 지정된 경우 해당 iteration, 없으면 기본 순서
        iterations_to_check = []
        if optimal_iter:
            iterations_to_check.append(optimal_iter)
        
        # 다른 iteration들도 확인 (fallback)
        iterations_to_check.extend([10000, 20000, 30000, 40000, 50000, 60000])
        
        for iteration in iterations_to_check:
            if iteration == 60000:
                # Final 모델
                model_path = model_dir / f'{model_name}.pt'
            else:
                # Iteration 모델
                model_path = model_dir / f'{model_name}_iter_{iteration}.pt'
            
            if model_path.exists():
                try:
                    best_model = torch.load(model_path, map_location=self.device)
                    best_model.eval()
                    return best_model, iteration, model_path
                except:
                    pass
        
        return None, None, None
    
    def load_models(self):
        """학습된 모델들 로드"""
        models = {}
        # 기존 모델 경로 (주석처리)
        # saved_model_dir = Path(__file__).parent / 'saved_model'
        
        # 새로운 모델 경로 (saved_model/new에서 로드) - rank=20으로 훈련된 모델
        saved_model_dir = Path(__file__).parent / 'saved_model'
        new_model_dir = saved_model_dir / 'new'
        
        # Base v4 모델 로드 (Large_estimator_v4_base_final.pt)
        base_path = saved_model_dir / 'Large_estimator_v4_base_final.pt'
        if base_path.exists():
            try:
                base_model = torch.load(base_path, map_location=self.device)
                base_model.eval()
                models['Base_v4_Final'] = base_model
                print("[OK] Loaded Base_v4_Final (Large_estimator_v4_base_final.pt)")
            except Exception as e:
                print(f"[ERROR] Failed to load Base_v4_Final: {e}")
        else:
            print("[WARNING] Large_estimator_v4_base_final.pt not found")
            
        # InF Transfer 모델 - 20k iteration 사용 - new 폴더에서 로드
        inf_model, inf_iter, inf_path = self.find_best_iteration_model_from_new('Large_estimator_v4_to_InF_optimized', optimal_iter=20000, model_dir=new_model_dir)
        if inf_model is not None:
            models['InF_Transfer'] = inf_model
            iter_label = f"{inf_iter/1000:.0f}k" if inf_iter != 60000 else "final"
            print(f"[OK] Loaded InF_Transfer (best @ {iter_label} iterations)")
        else:
            # 폴백: 최종 모델 시도
            transfer_inf_path = saved_model_dir / 'Large_estimator_v4_to_InF_optimized.pt'
            if transfer_inf_path.exists():
                try:
                    transfer_inf_model = torch.load(transfer_inf_path, map_location=self.device)
                    transfer_inf_model.eval()
                    models['InF_Transfer'] = transfer_inf_model
                    print("[OK] Loaded InF_Transfer (final = 60k iterations)")
                except Exception as e:
                    print(f"[ERROR] Failed to load InF_Transfer: {e}")
            else:
                print("[WARNING] Large_estimator_v4_to_InF_optimized.pt not found")
            
        # RMa Transfer 모델 - 60k iteration 사용 - new 폴더에서 로드
        rma_model, rma_iter, rma_path = self.find_best_iteration_model_from_new('Large_estimator_v4_to_RMa_optimized', optimal_iter=60000, model_dir=new_model_dir)
        if rma_model is not None:
            models['RMa_Transfer'] = rma_model
            iter_label = f"{rma_iter/1000:.0f}k" if rma_iter != 60000 else "final"
            print(f"[OK] Loaded RMa_Transfer (best @ {iter_label} iterations)")
        else:
            # 폴백: 최종 모델 시도
            transfer_rma_path = saved_model_dir / 'Large_estimator_v4_to_RMa_optimized.pt'
            if transfer_rma_path.exists():
                try:
                    transfer_rma_model = torch.load(transfer_rma_path, map_location=self.device)
                    transfer_rma_model.eval()
                    models['RMa_Transfer'] = transfer_rma_model
                    print("[OK] Loaded RMa_Transfer (final = 60k iterations)")
                except Exception as e:
                    print(f"[ERROR] Failed to load RMa_Transfer: {e}")
            else:
                print("[WARNING] Large_estimator_v4_to_RMa_optimized.pt not found")
            
        # 기존 모델들도 로드 (비교용) - 현재는 사용하지 않음
        # old_inf_path = saved_model_dir / 'Large_estimator_v4_to_InF_test_v4.pt'
        # if old_inf_path.exists():
        #     try:
        #         old_inf_model = torch.load(old_inf_path, map_location=self.device)
        #         old_inf_model.eval()
        #         models['InF_Transfer_Old'] = old_inf_model
        #         print("[OK] Loaded InF_Transfer_Old (for comparison)")
        #     except Exception as e:
        #         print(f"[ERROR] Failed to load InF_Transfer_Old: {e}")
        # else:
        #     print("[WARNING] Large_estimator_v4_to_InF_test_v4.pt not found")
        #     
        # old_rma_path = saved_model_dir / 'Large_estimator_v4_to_RMa_test_v4.pt'
        # if old_rma_path.exists():
        #     try:
        #         old_rma_model = torch.load(old_rma_path, map_location=self.device)
        #         old_rma_model.eval()
        #         models['RMa_Transfer_Old'] = old_rma_model
        #         print("[OK] Loaded RMa_Transfer_Old (for comparison)")
        #     except Exception as e:
        #         print(f"[ERROR] Failed to load RMa_Transfer_Old: {e}")
        # else:
        #     print("[WARNING] Large_estimator_v4_to_RMa_test_v4.pt not found")
        
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
        print("SUMMARY - LoRA Transfer Learning Performance Comparison")
        print("="*100)
        
        # 업데이트된 모델 이름
        print(f"{'Dataset':<15} {'Base_v4_Final':<15} {'InF_Transfer':<15} {'RMa_Transfer':<15}")
        print("-" * 60)
        
        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_Final', np.nan)
            inf_nmse = results[dataset_name].get('InF_Transfer', np.nan)
            rma_nmse = results[dataset_name].get('RMa_Transfer', np.nan)
            
            # 포맷팅
            base_str = f"{base_nmse:.2f}" if not np.isnan(base_nmse) else "N/A"
            inf_str = f"{inf_nmse:.2f}" if not np.isnan(inf_nmse) else "N/A"
            rma_str = f"{rma_nmse:.2f}" if not np.isnan(rma_nmse) else "N/A"
            
            print(f"{dataset_name:<15} {base_str:<15} {inf_str:<15} {rma_str:<15}")
        
        print("="*100)
        
        # 개선량 분석
        print("\nPerformance Improvements (vs Base_v4_Final):")
        print("-" * 50)
        for dataset_name in results.keys():
            base_nmse = results[dataset_name].get('Base_v4_Final', np.nan)
            inf_nmse = results[dataset_name].get('InF_Transfer', np.nan)
            rma_nmse = results[dataset_name].get('RMa_Transfer', np.nan)
            
            print(f"\n{dataset_name}:")
            if not np.isnan(base_nmse):
                if not np.isnan(inf_nmse):
                    inf_improvement = base_nmse - inf_nmse
                    print(f"  InF Transfer: {inf_improvement:+.2f} dB")
                if not np.isnan(rma_nmse):
                    rma_improvement = base_nmse - rma_nmse
                    print(f"  RMa Transfer: {rma_improvement:+.2f} dB")
        
        print("="*70)
    
    def plot_results(self, results):
        """결과 시각화 - v3 스타일과 동일하게"""
        # 데이터 준비
        model_names = ['Base_v4_Final', 'InF_Transfer', 'RMa_Transfer']
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
                elif model == 'InF_Transfer':
                    label = 'v4 + LoRA (InF @20k)'
                elif model == 'RMa_Transfer':
                    label = 'v4 + LoRA (RMa @60k)'
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
        plt.title('LoRA Transfer Learning Performance Comparison', fontsize=14)
        plt.xticks(x_positions + bar_width, datasets)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # 첫 번째 플롯 저장
        save_path = 'v4_lora_comparison_new_models.png'
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
                
                # InF Transfer 개선도
                if 'InF_Transfer' in results.get(dataset, {}):
                    inf_nmse = results[dataset]['InF_Transfer']
                    if not np.isnan(base_nmse) and not np.isnan(inf_nmse):
                        improvement = base_nmse - inf_nmse
                        improvements.append(improvement)
                        model_labels.append(f'{dataset}\n(InF LoRA @20k)')
                
                # RMa Transfer 개선도
                if 'RMa_Transfer' in results.get(dataset, {}):
                    rma_nmse = results[dataset]['RMa_Transfer']
                    if not np.isnan(base_nmse) and not np.isnan(rma_nmse):
                        improvement = base_nmse - rma_nmse
                        improvements.append(improvement)
                        model_labels.append(f'{dataset}\n(RMa LoRA @60k)')
            
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
                plt.title('v4 LoRA Performance Improvement\n(Compared to Base Model)', fontsize=14)
                plt.xticks(x_pos, model_labels)
                plt.grid(True, axis='y', alpha=0.3)
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                # 두 번째 플롯 저장
                save_path_imp = 'v4_lora_improvement_new_models.png'
                plt.savefig(save_path_imp, dpi=300, bbox_inches='tight')
                print(f"[OK] Saved {save_path_imp}")
                plt.show()

if __name__ == "__main__":
    print("Simple LoRA Model Testing")
    print("=" * 40)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    tester = SimpleModelTester(device=device)
    results = tester.test_models()