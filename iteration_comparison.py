"""
Iteration-based Performance Comparison Tool for LoRA Transfer Learning

이 스크립트는 다양한 iteration에서 저장된 LoRA 전이학습 모델들의 성능을 비교 분석합니다.
최적의 학습 iteration을 찾고 과적합 시점을 탐지합니다.

주요 기능:
1. 5k, 10k, 15k, ..., 60k iteration 모델들의 성능 측정
2. 수렴 곡선 (Convergence Curve) 시각화
3. 최적 iteration 지점 자동 탐지
4. InF/RMa 환경별 수렴 패턴 분석
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from model.estimator_v4 import Estimator_v4
import glob
import re

class IterationComparisonTool:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.results = {}
        
    def find_iteration_models(self, model_prefix):
        """특정 prefix로 시작하는 iteration 모델들 찾기"""
        # 기존 모델 경로 (주석처리)
        # saved_model_dir = Path(__file__).parent / 'saved_model'
        
        # 새로운 모델 경로 (saved_model/new에서 로드)
        saved_model_dir = Path(__file__).parent / 'saved_model'
        new_model_dir = saved_model_dir / 'new'
        pattern = str(new_model_dir / f"{model_prefix}_iter_*.pt")
        model_files = glob.glob(pattern)
        
        # iteration 번호 추출 및 정렬
        models = []
        for file in model_files:
            match = re.search(r'_iter_(\d+)\.pt$', file)
            if match:
                iteration = int(match.group(1))
                models.append((iteration, file))
        
        # iteration 순서로 정렬
        models.sort(key=lambda x: x[0])
        
        # Base 모델을 iteration 0으로 추가
        base_model_path = saved_model_dir / 'Large_estimator_v4_base_final.pt'
        if base_model_path.exists():
            models.insert(0, (0, str(base_model_path)))
        
        # 최종 모델 파일은 추가하지 않음 (60k iteration까지만 분석)
            
        return models
    
    def load_test_data(self):
        """테스트 데이터 로드"""
        test_data_dir = Path(__file__).parent / 'simple_test_data'
        datasets = {}
        
        for dataset_name in ['InF_50m', 'RMa_300m']:
            input_path = test_data_dir / f'{dataset_name}_input.npy'
            true_path = test_data_dir / f'{dataset_name}_true.npy'
            
            if input_path.exists() and true_path.exists():
                rx_input = np.load(input_path)
                ch_true = np.load(true_path)
                datasets[dataset_name] = (rx_input, ch_true)
                print(f"[OK] Loaded {dataset_name} test data")
            else:
                print(f"[WARNING] Test data for {dataset_name} not found")
        
        return datasets
    
    def calculate_nmse(self, ch_est, ch_true):
        """NMSE 계산"""
        ch_true = np.stack((np.real(ch_true), np.imag(ch_true)), axis=-1)
        ch_mse = np.sum(np.square(ch_true - ch_est), axis=(1, 2)) / ch_true.shape[-1]
        ch_var = np.sum(np.square(ch_true), axis=(1, 2)) / ch_true.shape[-1]
        ch_nmse = np.mean(ch_mse / ch_var)
        return ch_nmse
    
    def test_iteration_models(self, model_prefix, model_name):
        """특정 모델의 모든 iteration 버전 테스트"""
        print(f"\n{'='*60}")
        print(f"Testing {model_name} at different iterations")
        print(f"{'='*60}")
        
        models = self.find_iteration_models(model_prefix)
        if not models:
            print(f"No iteration models found for {model_prefix}")
            return None
        
        datasets = self.load_test_data()
        if not datasets:
            print("Test data not loaded!")
            return None
        
        iteration_results = {}
        
        for iteration, model_path in models:
            if iteration == 0:
                iter_label = "base"
            elif iteration < 1000:
                iter_label = f"{iteration}"
            else:
                iter_label = f"{iteration//1000}k"
            print(f"\nTesting iteration {iter_label}:")
            
            try:
                # 모델 로드
                model = torch.load(model_path, map_location=self.device)
                model.eval()
                
                iter_results = {}
                
                # 각 데이터셋에 대해 테스트
                for dataset_name, (rx_input, ch_true) in datasets.items():
                    rx_tensor = torch.tensor(rx_input, dtype=torch.float32).to(self.device)
                    
                    with torch.no_grad():
                        ch_est, _ = model(rx_tensor)
                        ch_est_np = ch_est.cpu().numpy()
                        nmse = self.calculate_nmse(ch_est_np, ch_true)
                        nmse_db = 10 * np.log10(nmse)
                        
                        iter_results[dataset_name] = nmse_db
                        print(f"  {dataset_name}: {nmse_db:.2f} dB")
                
                iteration_results[iteration] = iter_results
                
            except Exception as e:
                print(f"  ERROR loading model at iteration {iter_label}: {e}")
                continue
        
        return iteration_results
    
    def compare_all_iterations(self):
        """InF와 RMa 모델들의 모든 iteration 비교"""
        
        # Base 모델은 이제 find_iteration_models에서 자동으로 포함됨
        
        # InF 모델 테스트
        inf_results = self.test_iteration_models(
            'Large_estimator_v4_to_InF_optimized',
            'InF Transfer Optimized'
        )
        
        # RMa 모델 테스트
        rma_results = self.test_iteration_models(
            'Large_estimator_v4_to_RMa_optimized',
            'RMa Transfer Optimized'
        )
        
        # 결과 저장
        self.results = {
            'InF': inf_results,
            'RMa': rma_results
        }
        
        # 결과 분석 및 시각화
        self.analyze_convergence()
        self.plot_convergence_curves()
        
        return self.results
    
    def test_base_model(self):
        """베이스 모델 테스트"""
        print(f"\n{'='*60}")
        print("Testing Base Model")
        print(f"{'='*60}")
        
        saved_model_dir = Path(__file__).parent / 'saved_model'
        base_path = saved_model_dir / 'Large_estimator_v4_base_final.pt'
        
        if not base_path.exists():
            print("[WARNING] Base model not found")
            return None
        
        datasets = self.load_test_data()
        base_results = {}
        
        try:
            model = torch.load(base_path, map_location=self.device)
            model.eval()
            
            for dataset_name, (rx_input, ch_true) in datasets.items():
                rx_tensor = torch.tensor(rx_input, dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    ch_est, _ = model(rx_tensor)
                    ch_est_np = ch_est.cpu().numpy()
                    nmse = self.calculate_nmse(ch_est_np, ch_true)
                    nmse_db = 10 * np.log10(nmse)
                    
                    base_results[dataset_name] = nmse_db
                    print(f"  {dataset_name}: {nmse_db:.2f} dB")
                    
        except Exception as e:
            print(f"ERROR testing base model: {e}")
            return None
        
        return base_results
    
    def analyze_convergence(self):
        """수렴 분석 및 최적 iteration 찾기"""
        print(f"\n{'='*60}")
        print("Convergence Analysis")
        print(f"{'='*60}")
        
        if not self.results or not self.results.get('InF') or not self.results.get('RMa'):
            print("Not enough data for analysis")
            return
        
        # 각 모델/데이터셋 조합에 대해 분석
        for model_type in ['InF', 'RMa']:
            if not self.results[model_type]:
                continue
                
            print(f"\n{model_type} Transfer Model Analysis:")
            
            for dataset_name in ['InF_50m', 'RMa_300m']:
                iterations = sorted([k for k in self.results[model_type].keys()])
                nmse_values = [self.results[model_type][iter].get(dataset_name, np.nan) 
                              for iter in iterations]
                
                # NaN이 아닌 값들만 필터링
                valid_data = [(i, v) for i, v in zip(iterations, nmse_values) if not np.isnan(v)]
                
                if len(valid_data) < 2:
                    continue
                
                valid_iters, valid_nmse = zip(*valid_data)
                
                # 최적 iteration 찾기 (최소 NMSE)
                best_idx = np.argmin(valid_nmse)
                best_iter = valid_iters[best_idx]
                best_nmse = valid_nmse[best_idx]
                
                print(f"  {dataset_name}:")
                print(f"    Best iteration: {best_iter}k")
                print(f"    Best NMSE: {best_nmse:.2f} dB")
                
                # 수렴 판단 (연속 3개 포인트의 변화가 0.1dB 미만)
                if len(valid_nmse) >= 3:
                    for i in range(len(valid_nmse) - 2):
                        diff1 = abs(valid_nmse[i] - valid_nmse[i+1])
                        diff2 = abs(valid_nmse[i+1] - valid_nmse[i+2])
                        if diff1 < 0.1 and diff2 < 0.1:
                            print(f"    Converged at: {valid_iters[i]}k iterations")
                            break
                
                # 과적합 검사 (NMSE가 증가하기 시작하는 지점)
                for i in range(1, len(valid_nmse)):
                    if valid_nmse[i] > valid_nmse[i-1] + 0.05:  # 0.05 dB 이상 증가
                        print(f"    Potential overfitting after: {valid_iters[i-1]}k iterations")
                        break
    
    def plot_convergence_curves(self):
        """수렴 곡선 플롯"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('LoRA Transfer Learning Convergence Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # 색상 설정 (더 선명한 색상)
        colors = {'InF': '#1f77b4', 'RMa': '#ff7f0e'}
        markers = {'InF': 'o', 'RMa': 's'}
        
        # 각 데이터셋에 대해 플롯
        for idx, dataset_name in enumerate(['InF_50m', 'RMa_300m']):
            for model_idx, model_type in enumerate(['InF', 'RMa']):
                ax = axes[idx, model_idx]
                
                if not self.results.get(model_type):
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
                    ax.set_title(f'{model_type} on {dataset_name}', fontsize=14)
                    continue
                
                # iteration과 NMSE 값 추출
                iterations = sorted([k for k in self.results[model_type].keys()])
                nmse_values = [self.results[model_type][iter].get(dataset_name, np.nan) 
                              for iter in iterations]
                
                # NaN이 아닌 값들만 플롯 (iteration 0은 그대로, 나머지는 k 단위로)
                valid_data = [(i/1000 if i > 0 else 0, v) for i, v in zip(iterations, nmse_values) 
                             if not np.isnan(v)]
                
                if valid_data:
                    iters, nmses = zip(*valid_data)
                    ax.plot(iters, nmses, marker=markers[model_type], 
                           color=colors[model_type], linewidth=3, markersize=10,
                           label=f'{model_type} Transfer', alpha=0.9)
                    
                    # Base 모델 성능은 이미 iteration 0에 포함되어 있으므로 수평선 제거
                    
                    # 최적점 표시 (더 크고 선명하게)
                    best_idx = np.argmin(nmses)
                    best_iter_label = "base" if iters[best_idx] == 0 else f"{iters[best_idx]:.0f}k"
                    ax.plot(iters[best_idx], nmses[best_idx], 'r*', markersize=20,
                           label=f'Best: {nmses[best_idx]:.2f} dB @ {best_iter_label}',
                           markeredgewidth=2, markeredgecolor='darkred')
                    
                    # Base 모델 포인트 강조
                    if 0 in [iter_val for iter_val in iterations]:
                        base_idx = next(i for i, x in enumerate(iters) if x == 0)
                        ax.plot(iters[base_idx], nmses[base_idx], 'ko', markersize=12,
                               markerfacecolor='white', markeredgewidth=3, markeredgecolor='black',
                               label=f'Base: {nmses[base_idx]:.2f} dB')
                    
                    # 각 포인트에 값 표시
                    for i, (x, y) in enumerate(zip(iters, nmses)):
                        if i == best_idx or x == 0:
                            continue
                        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                  xytext=(0,8), ha='center', fontsize=9, alpha=0.7)
                    
                    ax.set_xlabel('Iterations (k)', fontsize=14, fontweight='bold')
                    ax.set_ylabel('NMSE (dB)', fontsize=14, fontweight='bold')
                    ax.set_title(f'{model_type} Transfer on {dataset_name}', fontsize=15, fontweight='bold')
                    ax.grid(True, alpha=0.4, linestyle=':', linewidth=1.5)
                    ax.legend(loc='best', fontsize=11, framealpha=0.9)
                    
                    # 축 라벨 크기 증가
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    
                    # y축 범위 설정 (더 나은 비교를 위해)
                    y_min, y_max = min(nmses), max(nmses)
                    y_range = y_max - y_min
                    ax.set_ylim([y_min - 0.15*y_range, y_max + 0.15*y_range])
        
        plt.subplots_adjust(hspace=0.25, wspace=0.15)
        
        # 플롯 저장
        save_path = Path(__file__).parent / 'iteration_convergence_analysis_new_models.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nConvergence analysis plot saved to: {save_path}")
        
        plt.show()
        
        # 추가: 단일 그래프에 모든 곡선 표시
        self.plot_combined_convergence()
    
    def plot_combined_convergence(self):
        """모든 모델의 수렴 곡선을 하나의 그래프에 표시"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('Combined Convergence Curves', fontsize=18, fontweight='bold', y=0.95)
        
        # 색상 팔레트 설정
        model_colors = {'InF': '#2E86AB', 'RMa': '#E63946'}
        
        for idx, dataset_name in enumerate(['InF_50m', 'RMa_300m']):
            ax = axes[idx]
            
            # 배경색 설정 (약간의 강조)
            if dataset_name == 'InF_50m':
                ax.set_facecolor('#f0f8ff')
            else:
                ax.set_facecolor('#fff5f5')
            
            # 각 모델의 수렴 곡선 플롯
            for model_type in ['InF', 'RMa']:
                if not self.results.get(model_type):
                    continue
                
                iterations = sorted([k for k in self.results[model_type].keys()])
                nmse_values = [self.results[model_type][iter].get(dataset_name, np.nan) 
                              for iter in iterations]
                
                valid_data = [(i/1000 if i > 0 else 0, v) for i, v in zip(iterations, nmse_values) 
                             if not np.isnan(v)]
                
                if valid_data:
                    iters, nmses = zip(*valid_data)
                    ax.plot(iters, nmses, marker='o' if model_type == 'InF' else 's',
                           label=f'{model_type} Transfer', linewidth=3, markersize=9,
                           color=model_colors[model_type], alpha=0.9,
                           markeredgewidth=1.5, markeredgecolor='white')
                    
                    # 최적점 강조
                    best_idx = np.argmin(nmses)
                    ax.scatter(iters[best_idx], nmses[best_idx], s=200, 
                             color='gold', marker='*', edgecolors='black', 
                             linewidth=2, zorder=5)
                    
                    # 최적점에 텍스트 추가
                    ax.annotate(f'{nmses[best_idx]:.2f} dB', 
                              (iters[best_idx], nmses[best_idx]),
                              xytext=(5, -15), textcoords='offset points',
                              fontsize=10, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Base 모델 성능은 이미 iteration 0에 포함되어 있으므로 수평선 제거
            
            ax.set_xlabel('Iterations (k)', fontsize=14, fontweight='bold')
            ax.set_ylabel('NMSE (dB)', fontsize=14, fontweight='bold')
            ax.set_title(f'Performance on {dataset_name}', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.5, linestyle=':', linewidth=1.2)
            ax.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='black')
            
            # 축 라벨 크기 증가
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # 축 스타일 설정
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # 플롯 저장
        save_path = Path(__file__).parent / 'combined_convergence_curves_new_models.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined convergence plot saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self):
        """상세 분석 리포트 생성"""
        print(f"\n{'='*80}")
        print("DETAILED CONVERGENCE ANALYSIS REPORT")
        print(f"{'='*80}")
        
        if not self.results:
            print("No results available for report")
            return
        
        # 요약 테이블 생성
        print("\nSUMMARY TABLE")
        print("-" * 80)
        print(f"{'Model':<20} {'Dataset':<15} {'5k':<8} {'10k':<8} {'20k':<8} {'30k':<8} {'60k':<8} {'Best':<8}")
        print("-" * 80)
        
        for model_type in ['InF', 'RMa']:
            if not self.results.get(model_type):
                continue
                
            for dataset_name in ['InF_50m', 'RMa_300m']:
                row = f"{model_type:<20} {dataset_name:<15}"
                
                # 특정 iteration들의 성능
                for iter_k in [5, 10, 20, 30, 60]:
                    iter_val = iter_k * 1000
                    if iter_val in self.results[model_type]:
                        nmse = self.results[model_type][iter_val].get(dataset_name, np.nan)
                        row += f" {nmse:>7.2f}" if not np.isnan(nmse) else f" {'N/A':>7}"
                    else:
                        row += f" {'N/A':>7}"
                
                # 최적 성능
                all_nmse = [self.results[model_type][i].get(dataset_name, np.nan) 
                           for i in self.results[model_type].keys()]
                valid_nmse = [v for v in all_nmse if not np.isnan(v)]
                if valid_nmse:
                    best_nmse = min(valid_nmse)
                    row += f" {best_nmse:>7.2f}"
                else:
                    row += f" {'N/A':>7}"
                
                print(row)
        
        # Base 모델 성능
        if self.results.get('base'):
            print("-" * 80)
            for dataset_name in ['InF_50m', 'RMa_300m']:
                if dataset_name in self.results['base']:
                    nmse = self.results['base'][dataset_name]
                    print(f"{'Base Model':<20} {dataset_name:<15} {'-'*40} {nmse:>7.2f}")
        
        print("=" * 80)
        
        # 권장사항
        print("\nRECOMMENDATIONS")
        print("-" * 80)
        
        # 각 모델별 최적 iteration 찾기
        for model_type in ['InF', 'RMa']:
            if not self.results.get(model_type):
                continue
            
            print(f"\n{model_type} Transfer Model:")
            
            # 전체 데이터셋에서 평균 성능 계산
            best_avg_iter = None
            best_avg_nmse = float('inf')
            
            for iteration in self.results[model_type].keys():
                nmse_values = [self.results[model_type][iteration].get(ds, np.nan) 
                              for ds in ['InF_50m', 'RMa_300m']]
                valid_values = [v for v in nmse_values if not np.isnan(v)]
                
                if valid_values:
                    avg_nmse = np.mean(valid_values)
                    if avg_nmse < best_avg_nmse:
                        best_avg_nmse = avg_nmse
                        best_avg_iter = iteration
            
            if best_avg_iter:
                print(f"  Recommended iterations: {best_avg_iter/1000:.0f}k")
                print(f"  Average NMSE: {best_avg_nmse:.2f} dB")
                
                # 60k와 비교
                if 60000 in self.results[model_type]:
                    nmse_60k = np.mean([self.results[model_type][60000].get(ds, np.nan) 
                                       for ds in ['InF_50m', 'RMa_300m']])
                    if not np.isnan(nmse_60k):
                        diff = nmse_60k - best_avg_nmse
                        if diff > 0.05:
                            print(f"  Warning: 60k iterations shows {diff:.2f} dB degradation")
                            print(f"  -> Potential overfitting detected")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    print("="*60)
    print("LoRA Transfer Learning Iteration Comparison")
    print("="*60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # 비교 도구 실행
    tool = IterationComparisonTool(device=device)
    results = tool.compare_all_iterations()
    
    # 상세 리포트 생성
    tool.generate_report()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("Check the generated plots for visual analysis")
    print("="*60)