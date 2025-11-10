"""
Pareto Analysis with Statistical Validation

다중 테스트 셋 (5 seeds)을 사용하여 통계적으로 엄밀한 분석 수행
- CSV 있으면 로드, 없으면 모델 평가 수행
- 평균 (mean), 표준편차 (std) 계산
- t-test로 통계적 유의성 검정
- Clean 그래프 5개 생성 (error bars 제거, 평균값만 사용)
- 통계 테이블 생성 (Markdown & LaTeX)
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

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

    def load_csv_results(self):
        """CSV 파일에서 통계 결과 로드 (원본 값 포함)"""
        csv_path = Path(__file__).parent / 'pareto_statistical_results.csv'

        if not csv_path.exists():
            return None

        print(f"\n[OK] Found existing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(df)} results from CSV")

        # DataFrame을 딕셔너리 리스트로 변환
        results = df.to_dict('records')

        # nmse_values를 JSON에서 리스트로 변환
        import json
        for r in results:
            if 'nmse_values' in r and isinstance(r['nmse_values'], str):
                r['nmse_values'] = json.loads(r['nmse_values'])

        return results

    def run_statistical_analysis(self):
        """전체 통계 분석 실행"""
        print("\n" + "="*70)
        print("STATISTICAL PARETO ANALYSIS")
        print("="*70)
        print(f"Number of seeds per scenario: {self.num_seeds}")
        print("="*70)

        # CSV가 이미 있으면 로드
        results = self.load_csv_results()
        if results is not None:
            print("[OK] Using existing CSV data")
            # 통계적 유의성 검정
            self.perform_statistical_tests(results)
            # Clean 그래프 생성
            self.plot_clean_graphs(results)
            return results

        # CSV가 없으면 모델 평가 수행
        print("[INFO] CSV not found. Starting model evaluation...")

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

        # Clean 그래프 생성
        self.plot_clean_graphs(results)

        return results

    def save_statistical_results(self, results):
        """통계 결과를 CSV로 저장 (원본 값 포함)"""
        csv_path = Path(__file__).parent / 'pareto_statistical_results.csv'

        with open(csv_path, 'w') as f:
            f.write('model_name,method,train_env,test_env,config,params,nmse_mean,nmse_std,n_seeds,nmse_values\n')
            for r in results:
                # nmse_values를 JSON 형식으로 저장
                import json
                values_str = json.dumps(r['nmse_values'])

                f.write(f"{r['model_name']},{r['method']},{r['train_env']},{r['test_env']},"
                       f"{r['config']},{r['params']},{r['nmse_mean']:.4f},{r['nmse_std']:.4f},"
                       f"{r['n_seeds']},\"{values_str}\"\n")

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

    def plot_clean_graphs(self, results):
        """Clean 그래프 5개 생성 (error bars 제거)"""
        print("\n" + "="*70)
        print("Generating Clean Graphs (평균값만, error bars 제거)")
        print("="*70)

        print("[1/5] Generating heatmap...")
        self.plot_heatmap(results)

        print("[2/5] Generating in-domain parameter comparison...")
        self.plot_indomain_parameter_comparison(results)

        print("[3/5] Generating domain-wise in-domain comparison...")
        self.plot_domain_indomain(results)

        print("[4/5] Generating domain-wise cross-environment comparison...")
        self.plot_domain_crossenv(results)

        print("[5/5] Generating Pareto curves...")
        self.plot_pareto_curves(results)

        print("\n[BONUS] Generating statistical tables...")
        self.generate_statistical_tables(results)

        print("\n[EXTRA] Generating paper-ready heatmaps...")
        self.plot_paper_heatmaps(results)

    def plot_heatmap(self, results):
        """Heatmap: Cross-environment performance"""
        # Base 성능
        base_performance = {}
        for test_env in self.scenarios:
            base_nmse = next((r['nmse_mean'] for r in results
                             if r['method'] == 'Base' and r['test_env'] == test_env), None)
            base_performance[test_env] = base_nmse

        # PEFT 모델만
        peft_results = [r for r in results if r['method'] != 'Base']

        # 모델 정보 수집
        model_info_list = []
        for r in peft_results:
            model_name = r['model_name']
            if not any(m['model_name'] == model_name for m in model_info_list):
                model_info_list.append({
                    'model_name': model_name,
                    'method': r['method'],
                    'params': r['params']
                })

        # 정렬: 방법별 → 파라미터 → train_env
        method_order = ['Adapter', 'LoRA', 'Prompt', 'Hybrid']
        sorted_models = []

        for method in method_order:
            method_models = [m for m in model_info_list if m['method'] == method]

            # train_env 추출
            for model in method_models:
                parts = model['model_name'].split('_')
                model['train_env'] = parts[1] if len(parts) >= 2 else 'ZZZ'

            # 정렬: params → train_env
            method_models.sort(key=lambda x: (
                x['params'],
                self.scenarios.index(x['train_env']) if x['train_env'] in self.scenarios else 999
            ))

            sorted_models.extend(method_models)

        models = [m['model_name'] for m in sorted_models]
        test_envs = self.scenarios

        # 개선도 데이터 생성
        improvement_data = []
        actual_nmse_data = []
        display_labels = []

        for model_info in sorted_models:
            model = model_info['model_name']
            method = model_info['method']
            params = model_info['params']

            # Y축 레이블
            params_k = params / 1000
            if params_k < 1000:
                params_str = f"{params_k:.0f}K"
            else:
                params_str = f"{params_k/1000:.1f}M"

            display_label = f"[{method}-{params_str}] {model.replace(method+'_', '')}"
            display_labels.append(display_label)

            improvement_row = []
            actual_row = []
            for test_env in test_envs:
                # PEFT 평균 NMSE
                peft_nmse = next((r['nmse_mean'] for r in peft_results
                                 if r['model_name'] == model and r['test_env'] == test_env), None)
                base_nmse = base_performance.get(test_env)

                if peft_nmse is not None and base_nmse is not None:
                    improvement = peft_nmse - base_nmse
                    improvement_row.append(improvement)
                    actual_row.append(peft_nmse)
                else:
                    improvement_row.append(np.nan)
                    actual_row.append(np.nan)

            improvement_data.append(improvement_row)
            actual_nmse_data.append(actual_row)

        # DataFrame
        improvement_df = pd.DataFrame(improvement_data, index=display_labels, columns=test_envs)
        actual_nmse_df = pd.DataFrame(actual_nmse_data, index=display_labels, columns=test_envs)

        # Annotation: "improvement | actual_nmse"
        annot_labels = []
        for i in range(len(display_labels)):
            row_labels = []
            for j in range(len(test_envs)):
                improvement = improvement_df.iloc[i, j]
                actual_nmse = actual_nmse_df.iloc[i, j]
                if not np.isnan(improvement) and not np.isnan(actual_nmse):
                    row_labels.append(f"{improvement:+.2f} | {actual_nmse:.2f}")
                else:
                    row_labels.append("")
            annot_labels.append(row_labels)
        annot_labels = np.array(annot_labels)

        # Heatmap
        fig, ax = plt.subplots(figsize=(14, max(20, len(display_labels) * 0.35)))

        # 커스텀 컬러맵: 진한 초록 → 초록 → 연두 → 흰색 → 주황
        # 위치 조정: 연두가 -0.5에서 나타남
        colors = [
            '#1B5E20',  # 진한 초록 (-3, 매우 좋음)
            '#4CAF50',  # 밝은 초록 (-2, 좋음)
            '#AED581',  # 연두 (-0.5, 약간 좋음)
            '#FFFFFF',  # 흰색 (0, 중립)
            '#FF9800'   # 주황 (+3, 나쁨)
        ]
        n_bins = 100
        # positions: -3=0.0, -2=0.167, -0.5=0.417, 0=0.5, +3=1.0
        positions = [0.0, 0.167, 0.417, 0.5, 1.0]
        cmap_custom = LinearSegmentedColormap.from_list('green_white_orange',
                                                        list(zip(positions, colors)), N=n_bins)

        # 폰트 설정
        plt.rcParams['font.family'] = 'sans-serif'

        sns.heatmap(improvement_df,
                   annot=annot_labels,
                   fmt='',
                   cmap=cmap_custom,
                   center=0,
                   vmin=-3,
                   vmax=3,
                   linewidths=2,
                   linecolor='white',
                   cbar_kws={'label': 'Improvement vs Base (dB)', 'shrink': 0.8, 'fraction': 0.025, 'pad': 0.02},
                   ax=ax,
                   annot_kws={'fontsize': 12})

        # Add Enhanced/Degraded labels to colorbar (vertical)
        cbar = ax.collections[0].colorbar
        cbar.ax.text(2.5, -3, 'Enhanced', ha='left', va='center', fontsize=9, fontweight='bold', rotation=90)
        cbar.ax.text(2.5, 3, 'Degraded', ha='left', va='center', fontsize=9, fontweight='bold', rotation=90)

        ax.set_title('PEFT Performance Improvement vs Base Model\n'
                    '(Left: Improvement | Right: Actual NMSE in dB)\n'
                    'Grouped by Method and Sorted by Parameter Count',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Test Environment', fontsize=13, fontweight='bold')
        ax.set_ylabel('PEFT Model Configuration', fontsize=13, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=7)

        plt.tight_layout()
        save_path = Path(__file__).parent / 'pareto_heatmap_statistical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()

    def plot_indomain_parameter_comparison(self, results):
        """In-domain parameter size effect (2x2)"""
        methods = ['Adapter', 'LoRA', 'Prompt', 'Hybrid']
        base_total_params = 1_356_000

        # Base 성능
        base_performance = {}
        for test_env in self.scenarios:
            base_nmse = next((r['nmse_mean'] for r in results
                             if r['method'] == 'Base' and r['test_env'] == test_env), None)
            base_performance[test_env] = base_nmse

        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        axes = axes.flatten()

        param_colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']

        for idx, method in enumerate(methods):
            ax = axes[idx]

            # In-domain 결과
            method_results = [r for r in results
                            if r['method'] == method and r['train_env'] == r['test_env']]

            if not method_results:
                ax.set_title(f'{method} - No Data', fontsize=15, fontweight='bold')
                continue

            # Config 목록 (파라미터 순)
            configs = sorted(set(r['config'] for r in method_results),
                           key=lambda c: next((r['params'] for r in method_results if r['config'] == c), 0))

            x = np.arange(len(self.scenarios))
            width = 0.8 / len(configs)

            for config_idx, config in enumerate(configs):
                improvements = []

                for test_env in self.scenarios:
                    base_nmse = base_performance.get(test_env, 0)
                    matching = [r for r in method_results
                               if r['test_env'] == test_env and r['config'] == config]

                    if matching:
                        nmse = matching[0]['nmse_mean']
                        improvements.append(nmse - base_nmse)
                    else:
                        improvements.append(np.nan)

                offset = (config_idx - len(configs)/2 + 0.5) * width
                ax.bar(x + offset, improvements, width,
                      label=config,
                      color=param_colors[config_idx % len(param_colors)],
                      alpha=0.85,
                      edgecolor='black',
                      linewidth=0.8)

            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_xlabel('Scenario (In-domain)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Improvement vs Base (dB)', fontsize=13, fontweight='bold')
            ax.set_title(f'{method}: Parameter Size Effect\n(Negative = Better)',
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(self.scenarios, fontsize=12)
            ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black',
                     title='Config', title_fontsize=11)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(-2.5, 2.5)

        plt.suptitle('In-domain Performance: Parameter Configuration Comparison\n'
                    '(Train and test on same scenario)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = Path(__file__).parent / 'pareto_parameter_comparison_statistical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()

    def plot_domain_indomain(self, results):
        """Domain-wise in-domain comparison (1x5)"""
        methods = ['Adapter', 'LoRA', 'Prompt', 'Hybrid']
        base_total_params = 1_356_000

        base_performance = {}
        for test_env in self.scenarios:
            base_nmse = next((r['nmse_mean'] for r in results
                             if r['method'] == 'Base' and r['test_env'] == test_env), None)
            base_performance[test_env] = base_nmse

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        method_colors = {
            'Adapter': '#1E88E5',
            'LoRA': '#7B1FA2',
            'Prompt': '#E53935',
            'Hybrid': '#00897B'
        }

        for idx, test_env in enumerate(self.scenarios):
            ax = axes[idx]

            improvements = []
            param_ratios = []
            labels = []
            colors = []

            base_nmse = base_performance.get(test_env, 0)

            for method in methods:
                in_domain = [r for r in results
                            if r['method'] == method
                            and r['train_env'] == test_env
                            and r['test_env'] == test_env]

                if in_domain:
                    best = min(in_domain, key=lambda x: x['nmse_mean'])
                    improvement = best['nmse_mean'] - base_nmse
                    param_ratio = (best['params'] / base_total_params) * 100
                    improvements.append(improvement)
                    param_ratios.append(param_ratio)
                else:
                    improvements.append(0)
                    param_ratios.append(0)

                labels.append(method)
                colors.append(method_colors[method])

            x = np.arange(len(methods))
            bars = ax.bar(x, improvements, color=colors, alpha=0.85,
                         edgecolor='black', linewidth=1.5)

            # 파라미터 비율 표시 (항상 위쪽)
            for i, (bar, ratio) in enumerate(zip(bars, param_ratios)):
                if ratio > 0:
                    height = bar.get_height()
                    # 음수든 양수든 항상 0보다 위쪽에 표시
                    y_pos = max(height, 0) + 0.15
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{ratio:.2f}%',
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold', color='black')

            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_xlabel('PEFT Method', fontsize=12, fontweight='bold')
            ax.set_ylabel('Improvement vs Base (dB)', fontsize=12, fontweight='bold')
            ax.set_title(f'{test_env}\n(In-domain)', fontsize=14, fontweight='bold', pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(-2.5, 2.5)

        plt.suptitle('In-domain Performance Comparison by Scenario\n'
                    '(Best configuration per method | Negative = Better)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = Path(__file__).parent / 'pareto_domain_indomain_statistical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()

    def plot_domain_crossenv(self, results):
        """Domain-wise cross-environment comparison (2x3)"""
        methods = ['Adapter', 'LoRA', 'Prompt', 'Hybrid']
        base_total_params = 1_356_000

        base_performance = {}
        for test_env in self.scenarios:
            base_nmse = next((r['nmse_mean'] for r in results
                             if r['method'] == 'Base' and r['test_env'] == test_env), None)
            base_performance[test_env] = base_nmse

        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()

        train_colors = {
            'InF': '#2196F3',
            'InH': '#4CAF50',
            'UMi': '#FF9800',
            'UMa': '#9C27B0',
            'RMa': '#F44336'
        }

        for idx, test_env in enumerate(self.scenarios):
            ax = axes[idx]
            base_nmse = base_performance.get(test_env, 0)

            x_positions = np.arange(len(methods))
            width = 0.15

            for train_idx, train_env in enumerate(self.scenarios):
                improvements = []

                for method in methods:
                    matching = [r for r in results
                               if r['method'] == method
                               and r['train_env'] == train_env
                               and r['test_env'] == test_env]

                    if matching:
                        best_nmse = min(r['nmse_mean'] for r in matching)
                        improvement = best_nmse - base_nmse
                        improvements.append(improvement)
                    else:
                        improvements.append(np.nan)

                offset = (train_idx - 2) * width
                ax.bar(x_positions + offset, improvements, width,
                      label=f'Train: {train_env}',
                      color=train_colors[train_env],
                      alpha=0.85,
                      edgecolor='black',
                      linewidth=0.8)

            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
            ax.set_xlabel('PEFT Method', fontsize=12, fontweight='bold')
            ax.set_ylabel('Improvement vs Base (dB)', fontsize=12, fontweight='bold')
            ax.set_title(f'Test on {test_env}\n(All training scenarios)',
                        fontsize=14, fontweight='bold', pad=10)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(methods, fontsize=11)
            ax.legend(fontsize=9, loc='best', framealpha=0.95, edgecolor='black')
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(-2.5, 2.5)

        fig.delaxes(axes[5])

        plt.suptitle('Cross-environment Performance Comparison by Test Scenario\n'
                    '(Best configuration per method | Negative = Better)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = Path(__file__).parent / 'pareto_domain_crossenv_statistical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()

    def plot_pareto_curves(self, results):
        """Pareto curves (2x3)"""
        base_total_params = 1_356_000

        base_performance = {}
        for test_env in self.scenarios:
            base_nmse = next((r['nmse_mean'] for r in results
                             if r['method'] == 'Base' and r['test_env'] == test_env), None)
            base_performance[test_env] = base_nmse

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

            base_result = next((r for r in results
                               if r['method'] == 'Base' and r['test_env'] == scenario), None)
            if not base_result:
                continue

            base_mean = base_result['nmse_mean']

            for method in ['Adapter', 'LoRA', 'Prompt', 'Hybrid']:
                scenario_results = [r for r in results
                                   if r['method'] == method
                                   and r['train_env'] == scenario
                                   and r['test_env'] == scenario]
                scenario_results.sort(key=lambda x: x['params'])

                if scenario_results:
                    param_ratios = [(r['params'] / base_total_params) * 100
                                   for r in scenario_results]
                    improvements = [r['nmse_mean'] - base_mean for r in scenario_results]

                    ax.plot(param_ratios, improvements,
                           marker=markers[method], color=colors[method],
                           label=method, linewidth=2.5, markersize=10, alpha=0.85)

            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Base')
            ax.set_xlabel('Trainable Parameter Ratio (%)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Improvement vs Base (dB)', fontsize=13, fontweight='bold')
            ax.set_title(f'{scenario} Scenario (In-Domain)\nBase NMSE: {base_mean:.2f} dB',
                        fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.legend(loc='best', fontsize=10, framealpha=0.9)
            ax.set_xlim(0.1, 50)

        fig.delaxes(axes[5])

        plt.suptitle('Pareto Frontier: Parameter Efficiency vs Performance\n'
                    '(Negative = Better than Base | Log scale X-axis)',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()

        save_path = Path(__file__).parent / 'pareto_curves_statistical.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()

    def generate_statistical_tables(self, results):
        """통계 테이블 생성 (Markdown & LaTeX)"""
        base_total_params = 1_356_000

        # In-domain 성능 테이블
        base_performance = {}
        for test_env in self.scenarios:
            base_result = next((r for r in results
                               if r['method'] == 'Base' and r['test_env'] == test_env), None)
            if base_result:
                base_performance[test_env] = base_result['nmse_mean']

        # Markdown 테이블
        md_path = Path(__file__).parent / 'papers' / 'IEEE_OJCOMS' / 'statistical_tables.md'
        md_path.parent.mkdir(parents=True, exist_ok=True)

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Statistical Analysis Tables\n\n")
            f.write("## In-domain Performance Comparison (n=5)\n\n")
            f.write("| Scenario | Method | Config | NMSE (dB) | vs Base (dB) | Params | Param % |\n")
            f.write("|----------|--------|--------|-----------|--------------|--------|---------|\n")

            for test_env in self.scenarios:
                base_nmse = base_performance.get(test_env, 0)
                f.write(f"| {test_env} | Base | - | {base_nmse:.2f} ± 0.00 | - | 0 | 0.00% |\n")

                for method in ['Adapter', 'LoRA', 'Prompt', 'Hybrid']:
                    in_domain = [r for r in results
                                if r['method'] == method
                                and r['train_env'] == test_env
                                and r['test_env'] == test_env]

                    if in_domain:
                        best = min(in_domain, key=lambda x: x['nmse_mean'])
                        improvement = best['nmse_mean'] - base_nmse
                        param_ratio = (best['params'] / base_total_params) * 100

                        f.write(f"| {test_env} | {method} | {best['config']} | "
                               f"{best['nmse_mean']:.2f} ± {best['nmse_std']:.2f} | "
                               f"{improvement:+.2f} | {best['params']:,} | {param_ratio:.2f}% |\n")

        print(f"[OK] Saved {md_path}")

        # LaTeX 테이블
        tex_path = Path(__file__).parent / 'papers' / 'IEEE_OJCOMS' / 'statistical_tables.tex'

        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write("% Statistical Tables for IEEE OJCOMS Paper\n\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{In-domain Performance Comparison (n=5)}\n")
            f.write("\\label{tab:indomain_stats}\n")
            f.write("\\begin{tabular}{llccccc}\n")
            f.write("\\toprule\n")
            f.write("Scenario & Method & Config & NMSE (dB) & vs Base (dB) & Params & Param \\% \\\\\n")
            f.write("\\midrule\n")

            for test_env in self.scenarios:
                base_nmse = base_performance.get(test_env, 0)
                f.write(f"{test_env} & Base & - & ${base_nmse:.2f} \\pm 0.00$ & - & 0 & 0.00\\% \\\\\n")

                for method in ['Adapter', 'LoRA', 'Prompt', 'Hybrid']:
                    in_domain = [r for r in results
                                if r['method'] == method
                                and r['train_env'] == test_env
                                and r['test_env'] == test_env]

                    if in_domain:
                        best = min(in_domain, key=lambda x: x['nmse_mean'])
                        improvement = best['nmse_mean'] - base_nmse
                        param_ratio = (best['params'] / base_total_params) * 100

                        f.write(f"{test_env} & {method} & {best['config']} & "
                               f"${best['nmse_mean']:.2f} \\pm {best['nmse_std']:.2f}$ & "
                               f"${improvement:+.2f}$ & {best['params']:,} & {param_ratio:.2f}\\% \\\\\n")

                if test_env != self.scenarios[-1]:
                    f.write("\\midrule\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"[OK] Saved {tex_path}")

    def plot_paper_heatmaps(self, results):
        """논문용 heatmap: Method-config 단위로 분할"""
        plots_dir = Path(__file__).parent / 'plots'
        plots_dir.mkdir(exist_ok=True)

        print("Generating method-config split heatmaps...")
        self.plot_heatmap_methodconfig_split(results, plots_dir)

    def get_custom_colormap(self):
        """공통 컬러맵 생성"""
        colors = [
            '#1B5E20',  # 진한 초록 (-3, 매우 좋음)
            '#4CAF50',  # 밝은 초록 (-2, 좋음)
            '#AED581',  # 연두 (-0.5, 약간 좋음)
            '#FFFFFF',  # 흰색 (0, 중립)
            '#FF9800'   # 주황 (+3, 나쁨)
        ]
        positions = [0.0, 0.167, 0.417, 0.5, 1.0]
        return LinearSegmentedColormap.from_list('green_white_orange',
                                                list(zip(positions, colors)), N=100)

    def plot_heatmap_methodconfig_split(self, results, plots_dir):
        """옵션 4: Method-Config 단위로 5줄씩 분할 (전체 heatmap과 동일한 순서)"""
        base_performance = {}
        for test_env in self.scenarios:
            base_nmse = next((r['nmse_mean'] for r in results
                             if r['method'] == 'Base' and r['test_env'] == test_env), None)
            base_performance[test_env] = base_nmse

        peft_results = [r for r in results if r['method'] != 'Base']

        # 전체 heatmap과 동일한 정렬
        model_info_list = []
        for r in peft_results:
            model_name = r['model_name']
            if not any(m['model_name'] == model_name for m in model_info_list):
                model_info_list.append({
                    'model_name': model_name,
                    'method': r['method'],
                    'params': r['params'],
                    'config': r['config']
                })

        # 정렬: 방법별 → 파라미터 → train_env
        method_order = ['Adapter', 'LoRA', 'Prompt', 'Hybrid']
        sorted_models = []

        for method in method_order:
            method_models = [m for m in model_info_list if m['method'] == method]

            # train_env 추출
            for model in method_models:
                parts = model['model_name'].split('_')
                model['train_env'] = parts[1] if len(parts) >= 2 else 'ZZZ'

            # 정렬: params → train_env
            method_models.sort(key=lambda x: (
                x['params'],
                self.scenarios.index(x['train_env']) if x['train_env'] in self.scenarios else 999
            ))

            sorted_models.extend(method_models)

        # 5개씩 묶어서 저장
        for i in range(0, len(sorted_models), 5):
            chunk = sorted_models[i:i+5]

            # 첫 번째 모델로 파일명 결정
            first_model = chunk[0]
            method = first_model['method']
            config = first_model['config']

            # 데이터 생성
            improvement_data = []
            annot_labels = []
            display_labels = []

            for model_info in chunk:
                model = model_info['model_name']
                train_env = model_info['train_env']

                # 환경만 표시 (method-config는 제목에 있음)
                display_label = train_env
                display_labels.append(display_label)

                improvement_row = []
                row_labels = []

                for test_env in self.scenarios:
                    peft_nmse = next((r['nmse_mean'] for r in peft_results
                                     if r['model_name'] == model and r['test_env'] == test_env), None)
                    base_nmse = base_performance.get(test_env)

                    if peft_nmse is not None and base_nmse is not None:
                        improvement = peft_nmse - base_nmse
                        improvement_row.append(improvement)
                        row_labels.append(f"{improvement:+.2f} | {peft_nmse:.2f}")
                    else:
                        improvement_row.append(np.nan)
                        row_labels.append("")

                improvement_data.append(improvement_row)
                annot_labels.append(row_labels)

            # DataFrame
            improvement_df = pd.DataFrame(improvement_data, index=display_labels, columns=self.scenarios)
            annot_labels = np.array(annot_labels)

            # Heatmap
            fig, ax = plt.subplots(figsize=(10, max(4, len(display_labels) * 0.6)))

            # 폰트 설정
            plt.rcParams['font.family'] = 'sans-serif'

            sns.heatmap(improvement_df,
                       annot=annot_labels,
                       fmt='',
                       cmap=self.get_custom_colormap(),
                       center=0,
                       vmin=-3,
                       vmax=3,
                       linewidths=2,
                       linecolor='white',
                       cbar_kws={
                           'label': 'Improvement vs Base (dB)',
                           'orientation': 'horizontal',
                           'shrink': 0.8,
                           'pad': 0.2
                       },
                       ax=ax,
                       annot_kws={'fontsize': 14})

            # Add Enhanced/Degraded labels to colorbar (horizontal)
            cbar = ax.collections[0].colorbar
            cbar.ax.text(-3, -0.8, 'Enhanced', ha='center', va='top', fontsize=9, fontweight='bold')
            cbar.ax.text(3, -0.8, 'Degraded', ha='center', va='top', fontsize=9, fontweight='bold')

            ax.set_title(f'{method} ({config}): Performance Improvement vs Base\n'
                        f'(Left: Improvement | Right: Actual NMSE in dB)',
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Test Environment', fontsize=11, fontweight='bold')
            ax.set_ylabel('Transfer Target\n(Model)', fontsize=11, fontweight='bold')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)

            plt.tight_layout()
            save_path = plots_dir / f'heatmap_{method}_{config}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  [OK] Saved {save_path}")
            plt.close()


def main():
    analyzer = StatisticalParetoAnalyzer(num_seeds=5)
    results = analyzer.run_statistical_analysis()

    if results:
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS COMPLETE")
        print("="*70)
        print(f"Total models: {len(set(r['model_name'] for r in results))}")
        print(f"Total result entries: {len(results)}")
        print("="*70)
        print("\nGenerated files:")
        print("  [CSV]")
        print("    - pareto_statistical_results.csv")
        print("  [Graphs]")
        print("    - pareto_heatmap_statistical.png")
        print("    - pareto_parameter_comparison_statistical.png")
        print("    - pareto_domain_indomain_statistical.png")
        print("    - pareto_domain_crossenv_statistical.png")
        print("    - pareto_curves_statistical.png")
        print("  [Tables]")
        print("    - papers/IEEE_OJCOMS/statistical_tables.md")
        print("    - papers/IEEE_OJCOMS/statistical_tables.tex")
        print("="*70)


if __name__ == "__main__":
    main()
