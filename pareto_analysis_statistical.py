"""
Pareto Analysis with Statistical Validation

다중 테스트 셋 (5 seeds)을 사용하여 통계적으로 엄밀한 분석 수행
- 평균 (mean), 표준편차 (std) 계산
- t-test로 통계적 유의성 검정
- Error bars 포함 그래프 생성
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

class StatisticalParetoAnalyzer:
    def __init__(self, device='cuda:0', num_seeds=5):
        self.device = device
        self.num_seeds = num_seeds
        self.pareto_dir = Path(__file__).parent / 'saved_model' / 'pareto'

        # 파라미터 수 (pareto_analysis.py와 동일)
        self.config_params = {
            'Adapter': {
                'dim8': 65536,
                'dim16': 131072,
                'dim32': 262144,
                'dim64': 524288
            },
            'LoRA': {
                'r4': 19660,
                'r8': 39320,
                'r16': 78640,
                'r20': 98304
            },
            'Prompt': {
                'len50': 6400,
                'len100': 12800,
                'len200': 25600
            },
            'Hybrid': {
                'p50_r5': 30230,
                'p100_r10': 61952,
                'p200_r20': 123904,
                'p25_r2': 12752,
                'p75_r8': 48896,
                'p100_r5': 36630,
                'p50_r10': 55552,
                'p150_r15': 92672,
                'p300_r25': 161280
            }
        }

        # 환경 순서: Indoor → Urban → Rural
        self.scenarios = ['InF', 'InH', 'UMi', 'UMa', 'RMa']

    def load_test_data_with_seed(self, scenario, seed):
        """특정 seed의 테스트 데이터 로드"""
        test_data_dir = Path(__file__).parent / 'simple_test_data'

        input_path = test_data_dir / f'{scenario}_input_seed{seed}.npy'
        true_path = test_data_dir / f'{scenario}_true_seed{seed}.npy'

        if not input_path.exists() or not true_path.exists():
            return None

        try:
            rx_input = np.load(input_path)  # (batch, 14, 3072, 2)
            ch_true = np.load(true_path)    # (batch, 3072) - complex64

            # ch_true를 실수/허수 분리
            ch_true_sep = np.stack((np.real(ch_true), np.imag(ch_true)), axis=-1)

            return (rx_input, ch_true_sep)
        except Exception as e:
            print(f"[ERROR] Failed to load seed{seed} for {scenario}: {e}")
            return None

    def evaluate_model(self, model_path, test_data):
        """모델 성능 평가"""
        rx_input, true_channel = test_data

        try:
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            model.eval()

            input_tensor = torch.tensor(rx_input, dtype=torch.float32).to(self.device)
            true_tensor = torch.tensor(true_channel, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                estimated_channel, _ = model(input_tensor)

                mse = torch.mean(torch.square(true_tensor - estimated_channel))
                var = torch.mean(torch.square(true_tensor))
                nmse = mse / var
                nmse_db = 10 * torch.log10(nmse)

            return nmse_db.item()
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_path.name}: {e}")
            return None

    def evaluate_with_multiple_seeds(self, model_path, scenario):
        """여러 seed로 평가하여 평균/표준편차 계산"""
        nmse_values = []

        for seed in range(1, self.num_seeds + 1):
            test_data = self.load_test_data_with_seed(scenario, seed)
            if test_data is None:
                continue

            nmse = self.evaluate_model(model_path, test_data)
            if nmse is not None:
                nmse_values.append(nmse)

        if len(nmse_values) == 0:
            return None

        return {
            'mean': np.mean(nmse_values),
            'std': np.std(nmse_values, ddof=1),  # Sample std (n-1)
            'values': nmse_values,
            'n': len(nmse_values)
        }

    def find_pareto_models(self):
        """Pareto 실험 모델 찾기"""
        print("="*70)
        print("Searching for Pareto experiment models...")
        print("="*70)

        models_info = {}

        # 각 방법별 폴더 확인
        for method in ['Adapter', 'LoRA', 'Prompt', 'Hybrid']:
            method_dir = self.pareto_dir / method
            if not method_dir.exists():
                continue

            for model_file in method_dir.glob('*.pt'):
                if '_iter_' not in model_file.name:  # 최종 모델만
                    self.parse_model_filename(model_file, method, models_info)

        print(f"\nFound {len(models_info)} final models")
        return models_info

    def parse_model_filename(self, model_file, method, models_info):
        """모델 파일 이름에서 정보 추출"""
        filename = model_file.stem
        parts = filename.split('_')

        try:
            to_idx = parts.index('to')
            scenario = parts[to_idx + 1]

            if method == 'Adapter':
                config = parts[-1]
            elif method == 'LoRA':
                config = parts[-1]
            elif method == 'Prompt':
                config = parts[-1]
            elif method == 'Hybrid':
                config = '_'.join(parts[-2:])

            params = self.config_params.get(method, {}).get(config, 0)

            if scenario in self.scenarios and params > 0:
                key = f"{method}_{scenario}_{config}"
                models_info[key] = {
                    'method': method,
                    'scenario': scenario,
                    'config': config,
                    'params': params,
                    'path': model_file
                }
        except Exception as e:
            print(f"[WARNING] Failed to parse {filename}: {e}")

    def run_statistical_analysis(self):
        """전체 통계 분석 실행"""
        print("\n" + "="*70)
        print("STATISTICAL PARETO ANALYSIS")
        print("="*70)
        print(f"Number of seeds per scenario: {self.num_seeds}")
        print("="*70)

        # 모델 찾기
        models_info = self.find_pareto_models()
        if not models_info:
            print("[ERROR] No models found")
            return None

        # Base 모델 추가
        saved_model_dir = Path(__file__).parent / 'saved_model'
        base_v3_path = saved_model_dir / 'Large_estimator_v3_base_extended_final.pt'
        base_v4_path = saved_model_dir / 'Large_estimator_v4_base_extended_final.pt'

        if base_v3_path.exists():
            models_info['Base_v3'] = {
                'method': 'Base_v3',
                'scenario': 'Base',
                'config': 'base',
                'params': 0,
                'path': base_v3_path
            }
        if base_v4_path.exists():
            models_info['Base_v4'] = {
                'method': 'Base_v4',
                'scenario': 'Base',
                'config': 'base',
                'params': 0,
                'path': base_v4_path
            }

        # 통계적 평가 실행
        print("\n" + "="*70)
        print("Evaluating models with multiple seeds...")
        print("="*70)

        results = []
        total_evaluations = len(models_info) * len(self.scenarios) * self.num_seeds
        current_eval = 0

        for model_key, model_info in models_info.items():
            model_name = f"{model_info['method']}"
            if model_info['scenario'] != 'Base':
                model_name += f"_{model_info['scenario']}"
            if model_info['config'] != 'base':
                model_name += f"_{model_info['config']}"

            print(f"\n{'='*70}")
            print(f"Evaluating {model_name}")
            print(f"{'='*70}")

            for test_env in self.scenarios:
                current_eval += self.num_seeds
                print(f"[{current_eval}/{total_evaluations}] {model_name} on {test_env}...", end=" ")

                stats_result = self.evaluate_with_multiple_seeds(model_info['path'], test_env)

                if stats_result is not None:
                    result = {
                        'model_name': model_name,
                        'method': model_info['method'],
                        'train_env': model_info['scenario'],
                        'test_env': test_env,
                        'config': model_info['config'],
                        'params': model_info['params'],
                        'nmse_mean': stats_result['mean'],
                        'nmse_std': stats_result['std'],
                        'nmse_values': stats_result['values'],
                        'n_seeds': stats_result['n']
                    }
                    results.append(result)
                    print(f"{stats_result['mean']:.2f} ± {stats_result['std']:.3f} dB (n={stats_result['n']})")
                else:
                    print("FAILED")

        # Base 모델 통합 (v3와 v4 평균)
        print("\n" + "="*70)
        print("Merging Base_v3 and Base_v4...")
        print("="*70)

        base_v3_results = [r for r in results if r['method'] == 'Base_v3']
        base_v4_results = [r for r in results if r['method'] == 'Base_v4']
        results = [r for r in results if r['method'] not in ['Base_v3', 'Base_v4']]

        for test_env in self.scenarios:
            v3_result = next((r for r in base_v3_results if r['test_env'] == test_env), None)
            v4_result = next((r for r in base_v4_results if r['test_env'] == test_env), None)

            if v3_result and v4_result:
                # 더 나쁜 값 (max) 선택
                worse_result = v3_result if v3_result['nmse_mean'] > v4_result['nmse_mean'] else v4_result
                print(f"  {test_env}: Using {'v3' if worse_result == v3_result else 'v4'} (worse)")

                results.append({
                    'model_name': 'Base',
                    'method': 'Base',
                    'train_env': 'Base',
                    'test_env': test_env,
                    'config': 'base',
                    'params': 0,
                    'nmse_mean': worse_result['nmse_mean'],
                    'nmse_std': worse_result['nmse_std'],
                    'nmse_values': worse_result['nmse_values'],
                    'n_seeds': worse_result['n_seeds']
                })

        # CSV로 저장
        self.save_statistical_results(results)

        # 통계적 유의성 검정
        self.perform_statistical_tests(results)

        # 시각화 (error bars 포함)
        self.plot_statistical_analysis(results)

        return results

    def save_statistical_results(self, results):
        """통계 결과를 CSV로 저장"""
        csv_path = Path(__file__).parent / 'pareto_statistical_results.csv'

        with open(csv_path, 'w') as f:
            f.write('model_name,method,train_env,test_env,config,params,nmse_mean,nmse_std,n_seeds\n')
            for r in results:
                f.write(f"{r['model_name']},{r['method']},{r['train_env']},{r['test_env']},"
                       f"{r['config']},{r['params']},{r['nmse_mean']:.4f},{r['nmse_std']:.4f},"
                       f"{r['n_seeds']}\n")

        print(f"\n[OK] Statistical results saved to {csv_path}")

    def perform_statistical_tests(self, results):
        """통계적 유의성 검정 (t-test)"""
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*70)

        # In-domain 성능에 대해 각 PEFT vs Base 비교
        for test_env in self.scenarios:
            print(f"\n[{test_env}] PEFT methods vs Base (In-domain):")
            print("-" * 70)

            # Base 결과
            base_result = next((r for r in results
                               if r['method'] == 'Base' and r['test_env'] == test_env), None)
            if not base_result:
                continue

            base_values = base_result['nmse_values']

            # 각 PEFT 방법 비교
            for method in ['Adapter', 'LoRA', 'Prompt', 'Hybrid']:
                # In-domain: train_env == test_env
                method_results = [r for r in results
                                 if r['method'] == method
                                 and r['train_env'] == test_env
                                 and r['test_env'] == test_env]

                if not method_results:
                    continue

                # 최고 성능 config
                best = min(method_results, key=lambda x: x['nmse_mean'])
                peft_values = best['nmse_values']

                # Paired t-test
                if len(peft_values) == len(base_values):
                    t_stat, p_value = stats.ttest_rel(peft_values, base_values)

                    # Improvement
                    improvement = best['nmse_mean'] - base_result['nmse_mean']

                    # Significance markers
                    if p_value < 0.001:
                        sig = "***"
                    elif p_value < 0.01:
                        sig = "**"
                    elif p_value < 0.05:
                        sig = "*"
                    else:
                        sig = "n.s."

                    print(f"  {method:8s} ({best['config']:10s}): "
                          f"Δ={improvement:+.3f}dB ± {best['nmse_std']:.3f}, "
                          f"t={t_stat:.2f}, p={p_value:.4f} {sig}")

        print("\n" + "="*70)
        print("Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
        print("="*70)

    def plot_statistical_analysis(self, results):
        """Error bars 포함 그래프 생성"""
        print("\n[1/2] Generating in-domain comparison with error bars...")
        self.plot_indomain_with_errorbars(results)

        print("[2/2] Generating Pareto curves with confidence bands...")
        self.plot_pareto_with_confidence(results)

    def plot_indomain_with_errorbars(self, results):
        """In-domain 성능 비교 (error bars 포함)"""
        methods = ['Adapter', 'LoRA', 'Prompt', 'Hybrid']
        base_total_params = 1_356_000

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        method_colors = {
            'Adapter': '#1E88E5',
            'LoRA': '#7B1FA2',
            'Prompt': '#E53935',
            'Hybrid': '#00897B'
        }

        for idx, test_env in enumerate(self.scenarios):
            ax = axes[idx]

            # Base 성능
            base_result = next((r for r in results
                               if r['method'] == 'Base' and r['test_env'] == test_env), None)
            if not base_result:
                continue

            base_mean = base_result['nmse_mean']

            improvements = []
            errors = []
            param_ratios = []
            labels = []
            colors = []

            for method in methods:
                # In-domain: train_env == test_env
                in_domain = [r for r in results
                            if r['method'] == method
                            and r['train_env'] == test_env
                            and r['test_env'] == test_env]

                if in_domain:
                    best = min(in_domain, key=lambda x: x['nmse_mean'])
                    improvement = best['nmse_mean'] - base_mean
                    param_ratio = (best['params'] / base_total_params) * 100

                    improvements.append(improvement)
                    errors.append(best['nmse_std'])
                    param_ratios.append(param_ratio)
                else:
                    improvements.append(0)
                    errors.append(0)
                    param_ratios.append(0)

                labels.append(method)
                colors.append(method_colors[method])

            # 막대 그래프 with error bars
            x = np.arange(len(methods))
            bars = ax.bar(x, improvements, color=colors, alpha=0.85,
                         edgecolor='black', linewidth=1.5,
                         yerr=errors, capsize=5, error_kw={'linewidth': 2})

            # 파라미터 비율 표시
            for i, (bar, ratio) in enumerate(zip(bars, param_ratios)):
                if ratio > 0:
                    height = bar.get_height()
                    y_offset = 0.15 if height >= 0 else -0.25
                    ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                           f'{ratio:.2f}%',
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=9, fontweight='bold', color='black')

            # 0선 강조
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8)

            ax.set_xlabel('PEFT Method', fontsize=12, fontweight='bold')
            ax.set_ylabel('Improvement vs Base (dB)', fontsize=12, fontweight='bold')
            ax.set_title(f'{test_env}\n(In-domain, n={self.num_seeds})',
                        fontsize=14, fontweight='bold', pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(-2.5, 2.5)

        plt.suptitle('In-domain Performance with Statistical Error Bars\n'
                    '(Best configuration per method | Error bars = ±1 std)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = Path(__file__).parent / 'pareto_indomain_statistical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()

    def plot_pareto_with_confidence(self, results):
        """Pareto curves with confidence bands"""
        base_total_params = 1_356_000

        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()

        colors = {
            'Adapter': '#3498db',
            'LoRA': '#2ecc71',
            'Prompt': '#e74c3c',
            'Hybrid': '#f39c12'
        }

        markers = {
            'Adapter': 'o',
            'LoRA': 's',
            'Prompt': '^',
            'Hybrid': 'D'
        }

        for idx, scenario in enumerate(self.scenarios):
            ax = axes[idx]

            # Base 성능
            base_result = next((r for r in results
                               if r['method'] == 'Base' and r['test_env'] == scenario), None)
            if not base_result:
                continue

            base_mean = base_result['nmse_mean']

            for method in ['Adapter', 'LoRA', 'Prompt', 'Hybrid']:
                # In-domain: train_env == test_env == scenario
                scenario_results = [r for r in results
                                   if r['method'] == method
                                   and r['train_env'] == scenario
                                   and r['test_env'] == scenario]
                scenario_results.sort(key=lambda x: x['params'])

                if scenario_results:
                    # X: 파라미터 비율
                    param_ratios = [(r['params'] / base_total_params) * 100 for r in scenario_results]

                    # Y: 개선도
                    improvements = [r['nmse_mean'] - base_mean for r in scenario_results]
                    stds = [r['nmse_std'] for r in scenario_results]

                    # Confidence band (±1 std)
                    improvements = np.array(improvements)
                    stds = np.array(stds)
                    upper = improvements + stds
                    lower = improvements - stds

                    # Plot line with confidence band
                    ax.plot(param_ratios, improvements,
                           marker=markers[method], color=colors[method],
                           label=method, linewidth=2.5, markersize=10, alpha=0.85)

                    ax.fill_between(param_ratios, lower, upper,
                                   color=colors[method], alpha=0.2)

            # Base 기준선
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Base')

            ax.set_xlabel('Trainable Parameter Ratio (%)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Improvement vs Base (dB)', fontsize=13, fontweight='bold')
            ax.set_title(f'{scenario} (In-Domain, n={self.num_seeds})\nBase: {base_mean:.2f} dB',
                        fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            ax.set_xlim(0.1, 50)

        # 마지막 subplot 제거
        fig.delaxes(axes[5])

        plt.suptitle('Pareto Frontier with Confidence Bands\n'
                    '(Shaded area = ±1 std | Negative = Better)',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()

        save_path = Path(__file__).parent / 'pareto_curves_statistical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()


def main():
    analyzer = StatisticalParetoAnalyzer(num_seeds=5)
    results = analyzer.run_statistical_analysis()

    if results:
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("="*70)
        print(f"Total models evaluated: {len(set(r['model_name'] for r in results))}")
        print(f"Total evaluations: {len(results) * analyzer.num_seeds}")
        print("="*70)


if __name__ == "__main__":
    main()
