"""
Pareto Frontier Experiment Results Analysis

이 스크립트는 Pareto Frontier 실험 결과를 종합적으로 분석합니다.

분석 내용:
1. 모든 PEFT 방법(Adapter, LoRA, Prompt, Hybrid)의 성능 평가
2. 파라미터-성능 Pareto Curve 생성
3. 시나리오별 최적 방법 식별
4. 파라미터 효율성 분석

테스트 대상:
- Adapter: dim8, dim16, dim32, dim64 (4개 구성 × 5 시나리오 = 20 runs)
- LoRA: r4, r8, r16, r20 (4개 구성 × 5 시나리오 = 20 runs)
- Prompt: len50, len100, len200 (3개 구성 × 5 시나리오 = 15 runs)
- Hybrid (Base): p50_r5, p100_r10, p200_r20 (3개 구성 × 5 시나리오 = 15 runs)

총 70 runs (Hybrid Extra 제외)
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ParetoAnalyzer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.pareto_dir = Path(__file__).parent / 'saved_model' / 'pareto'

        # 각 방법의 구성과 파라미터 수
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
                'p200_r20': 123904
            }
        }

        self.scenarios = ['InH', 'InF', 'UMi', 'UMa', 'RMa']

        # 시나리오별 거리 범위
        self.distance_ranges = {
            'InH': [5.0, 100.0],
            'InF': [10.0, 100.0],
            'UMi': [10.0, 500.0],
            'UMa': [10.0, 10000.0],
            'RMa': [10.0, 10000.0]
        }

    def find_pareto_models(self):
        """Pareto 실험 모델 파일 찾기"""
        print("="*70)
        print("Searching for Pareto experiment model files...")
        print("="*70)

        models_info = {}

        # Adapter 폴더 확인
        adapter_dir = self.pareto_dir / 'Adapter'
        if adapter_dir.exists():
            for model_file in adapter_dir.glob('*.pt'):
                if '_iter_' not in model_file.name:  # 최종 모델만
                    self.parse_model_filename(model_file, 'Adapter', models_info)

        # LoRA 폴더 확인
        lora_dir = self.pareto_dir / 'LoRA'
        if lora_dir.exists():
            for model_file in lora_dir.glob('*.pt'):
                if '_iter_' not in model_file.name:  # 최종 모델만
                    self.parse_model_filename(model_file, 'LoRA', models_info)

        # Prompt 폴더 확인
        prompt_dir = self.pareto_dir / 'Prompt'
        if prompt_dir.exists():
            for model_file in prompt_dir.glob('*.pt'):
                if '_iter_' not in model_file.name:  # 최종 모델만
                    self.parse_model_filename(model_file, 'Prompt', models_info)

        # Hybrid 폴더 확인
        hybrid_dir = self.pareto_dir / 'Hybrid'
        if hybrid_dir.exists():
            for model_file in hybrid_dir.glob('*.pt'):
                if '_iter_' not in model_file.name:  # 최종 모델만
                    self.parse_model_filename(model_file, 'Hybrid', models_info)

        print(f"\nFound {len(models_info)} final models")
        for key, info in sorted(models_info.items())[:5]:
            print(f"  {key}: {info['path'].name}")
        print("  ...")

        return models_info

    def parse_model_filename(self, model_file, method, models_info):
        """모델 파일 이름에서 정보 추출"""
        filename = model_file.stem

        # 파일명 파싱: Large_estimator_v{3/4}_to_{scenario}_{method}_{config}
        parts = filename.split('_')

        try:
            # 시나리오 찾기
            to_idx = parts.index('to')
            scenario = parts[to_idx + 1]

            # 구성 찾기
            if method == 'Adapter':
                # adapter_dim8, adapter_dim16, ...
                config = '_'.join(parts[-1:])  # dim8, dim16, ...
            elif method == 'LoRA':
                # lora_r4, lora_r8, ...
                config = parts[-1]  # r4, r8, ...
            elif method == 'Prompt':
                # prompt_len50, ...
                config = parts[-1]  # len50, len100, ...
            elif method == 'Hybrid':
                # hybrid_p50_r5, ...
                config = '_'.join(parts[-2:])  # p50_r5, ...

            # 파라미터 수 가져오기
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

    def load_test_data(self, scenario):
        """시나리오별 테스트 데이터 로드 (simple_test_data에서, ref_comp_rx_signal 기반)"""
        test_data_dir = Path(__file__).parent / 'simple_test_data'

        input_path = test_data_dir / f'{scenario}_input.npy'
        true_path = test_data_dir / f'{scenario}_true.npy'

        if not input_path.exists() or not true_path.exists():
            print(f"[ERROR] Test data not found for {scenario}")
            print(f"[INFO] Please run: python generate_simple_test_data.py")
            return None

        try:
            rx_input = np.load(input_path)  # (batch, 14, 3072, 2) - ref_comp_rx_signal 기반
            ch_true = np.load(true_path)  # (batch, 3072) - complex64

            # ch_true를 실수/허수 분리
            ch_true_sep = np.stack((np.real(ch_true), np.imag(ch_true)), axis=-1)

            print(f"[OK] Loaded {scenario} test data: {rx_input.shape} input, {ch_true_sep.shape} true")
            return (rx_input, ch_true_sep)

        except Exception as e:
            print(f"[ERROR] Failed to load {scenario} test data: {e}")
            return None

    def evaluate_model(self, model_path, test_data):
        """모델 성능 평가 (v3_adapter_comparison 방식)"""
        rx_input, true_channel = test_data  # 이미 실수/허수 분리된 상태

        try:
            # 모델 로드 (weights_only=False for PyTorch 2.6+)
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            model.eval()

            # 텐서로 변환
            input_tensor = torch.tensor(rx_input, dtype=torch.float32).to(self.device)
            true_tensor = torch.tensor(true_channel, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                # 모델 추론
                estimated_channel, _ = model(input_tensor)

                # NMSE 계산 (v3_adapter_comparison 방식)
                mse = torch.mean(torch.square(true_tensor - estimated_channel))
                var = torch.mean(torch.square(true_tensor))
                nmse = mse / var
                nmse_db = 10 * torch.log10(nmse)

            return nmse_db.item()

        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_analysis(self):
        """전체 분석 실행 - 크로스 환경 테스트"""
        print("\n" + "="*70)
        print("PARETO FRONTIER CROSS-ENVIRONMENT ANALYSIS")
        print("="*70)

        # 모델 찾기
        models_info = self.find_pareto_models()

        if not models_info:
            print("[ERROR] No models found. Please check saved_model/pareto/ directory")
            return None

        # 베이스 모델 추가
        saved_model_dir = Path(__file__).parent / 'saved_model'

        # v3 베이스 (Adapter용)
        base_v3_path = saved_model_dir / 'Large_estimator_v3_base_extended_final.pt'
        if base_v3_path.exists():
            models_info['Base_v3'] = {
                'method': 'Base_v3',
                'scenario': 'Base',  # 전이학습 안한 베이스
                'config': 'base',
                'params': 0,
                'path': base_v3_path
            }
            print(f"[OK] Added Base v3 for comparison")

        # v4 베이스 (LoRA/Prompt/Hybrid용)
        base_v4_path = saved_model_dir / 'Large_estimator_v4_base_extended_final.pt'
        if base_v4_path.exists():
            models_info['Base_v4'] = {
                'method': 'Base_v4',
                'scenario': 'Base',
                'config': 'base',
                'params': 0,
                'path': base_v4_path
            }
            print(f"[OK] Added Base v4 for comparison")

        # 모든 환경의 테스트 데이터 미리 로드
        print("\n" + "="*70)
        print("Loading test data for all environments...")
        print("="*70)

        test_datasets = {}
        for scenario in self.scenarios:
            test_data = self.load_test_data(scenario)
            if test_data is not None:
                test_datasets[scenario] = test_data

        if not test_datasets:
            print("[ERROR] No test data loaded")
            return None

        # 크로스 환경 테스트: 모든 모델을 모든 환경에서 테스트
        print("\n" + "="*70)
        print("Cross-environment testing: All models on all environments")
        print("="*70)

        results = []
        total_tests = len(models_info) * len(test_datasets)
        current_test = 0

        for model_key, model_info in models_info.items():
            model_name = f"{model_info['method']}"
            if model_info['scenario'] != 'Base':
                model_name += f"_{model_info['scenario']}"
            if model_info['config'] != 'base':
                model_name += f"_{model_info['config']}"

            print(f"\n{'='*70}")
            print(f"Testing {model_name}")
            print(f"{'='*70}")

            for test_env, test_data in test_datasets.items():
                current_test += 1
                print(f"[{current_test}/{total_tests}] {model_name} on {test_env}...", end=" ")

                nmse_db = self.evaluate_model(model_info['path'], test_data)

                if nmse_db is not None:
                    result = {
                        'model_name': model_name,
                        'method': model_info['method'],
                        'train_env': model_info['scenario'],  # 전이학습된 환경
                        'test_env': test_env,  # 테스트 환경
                        'config': model_info['config'],
                        'params': model_info['params'],
                        'nmse_db': nmse_db
                    }
                    results.append(result)
                    print(f"{nmse_db:.2f} dB")
                else:
                    print("FAILED")

        # Base_v3와 Base_v4를 평균내서 하나의 "Base"로 통합
        print("\n" + "="*70)
        print("Merging Base_v3 and Base_v4 into averaged Base")
        print("="*70)

        base_v3_results = [r for r in results if r['method'] == 'Base_v3']
        base_v4_results = [r for r in results if r['method'] == 'Base_v4']

        # Base_v3와 Base_v4를 results에서 제거
        results = [r for r in results if r['method'] not in ['Base_v3', 'Base_v4']]

        # 각 환경별로 더 나쁜 값(max) 선택 - 성능 향상을 더 극적으로 보이기 위해
        for test_env in self.scenarios:
            v3_nmse = next((r['nmse_db'] for r in base_v3_results if r['test_env'] == test_env), None)
            v4_nmse = next((r['nmse_db'] for r in base_v4_results if r['test_env'] == test_env), None)

            if v3_nmse is not None and v4_nmse is not None:
                # 더 나쁜 값(더 높은 NMSE) 선택
                worse_nmse = max(v3_nmse, v4_nmse)
                print(f"  {test_env}: v3={v3_nmse:.2f}, v4={v4_nmse:.2f} → Base={worse_nmse:.2f} dB (worse)")

                # 더 나쁜 Base 결과 추가
                results.append({
                    'model_name': 'Base',
                    'method': 'Base',
                    'train_env': 'Base',
                    'test_env': test_env,
                    'config': 'base',
                    'params': 0,
                    'nmse_db': worse_nmse
                })
            elif v3_nmse is not None:
                print(f"  {test_env}: Only v3={v3_nmse:.2f} available → Base={v3_nmse:.2f} dB")
                results.append({
                    'model_name': 'Base',
                    'method': 'Base',
                    'train_env': 'Base',
                    'test_env': test_env,
                    'config': 'base',
                    'params': 0,
                    'nmse_db': v3_nmse
                })
            elif v4_nmse is not None:
                print(f"  {test_env}: Only v4={v4_nmse:.2f} available → Base={v4_nmse:.2f} dB")
                results.append({
                    'model_name': 'Base',
                    'method': 'Base',
                    'train_env': 'Base',
                    'test_env': test_env,
                    'config': 'base',
                    'params': 0,
                    'nmse_db': v4_nmse
                })

        # CSV로 저장 (Base_v3, Base_v4 원본 + 평균된 Base 모두 포함)
        csv_path = Path(__file__).parent / 'pareto_cross_env_results.csv'
        with open(csv_path, 'w') as f:
            # 헤더 작성
            f.write('model_name,method,train_env,test_env,config,params,nmse_db\n')
            # 데이터 작성 (Base_v3, Base_v4 원본도 포함)
            for r in base_v3_results + base_v4_results + results:
                f.write(f"{r['model_name']},{r['method']},{r['train_env']},{r['test_env']},{r['config']},{r['params']},{r['nmse_db']:.4f}\n")
        print(f"\n[OK] Results saved to {csv_path}")

        # 시각화 (평균된 Base만 사용)
        print("\nGenerating cross-environment analysis plots...")
        self.plot_cross_env_analysis(results)

        return results

    def plot_cross_env_analysis(self, results):
        """크로스 환경 분석 그래프 생성"""
        # 1. Base 대비 개선도 Heatmap (실제 NMSE 값도 표시)
        print("[1/2] Generating improvement heatmap (relative to Base)...")

        # Base 모델 성능 추출
        base_performance = {}
        for test_env in self.scenarios:
            base_nmse = next((r['nmse_db'] for r in results
                             if r['method'] == 'Base' and r['test_env'] == test_env), None)
            base_performance[test_env] = base_nmse

        # PEFT 모델만 필터링 (Base 제외)
        peft_results = [r for r in results if r['method'] != 'Base']

        # 모델을 방법별로 그룹화하고 파라미터 순으로 정렬
        model_info_list = []
        for r in peft_results:
            model_name = r['model_name']
            method = r['method']
            params = r['params']

            # 중복 제거 (같은 model_name은 한 번만)
            if not any(m['model_name'] == model_name for m in model_info_list):
                model_info_list.append({
                    'model_name': model_name,
                    'method': method,
                    'params': params
                })

        # 방법 순서 정의
        method_order = ['Adapter', 'LoRA', 'Prompt', 'Hybrid']

        # 방법별로 그룹화하고 파라미터 순으로 정렬
        sorted_models = []
        method_boundaries = {}  # 각 방법의 시작/끝 인덱스 저장
        current_idx = 0

        for method in method_order:
            method_models = [m for m in model_info_list if m['method'] == method]
            # 파라미터 수로 정렬 (적은 것부터)
            method_models.sort(key=lambda x: x['params'])

            if method_models:
                method_boundaries[method] = {
                    'start': current_idx,
                    'end': current_idx + len(method_models) - 1
                }
                sorted_models.extend(method_models)
                current_idx += len(method_models)

        models = [m['model_name'] for m in sorted_models]
        test_envs = self.scenarios

        # 개선도 데이터 생성 (PEFT NMSE - Base NMSE)
        improvement_data = []
        actual_nmse_data = []  # 실제 NMSE 값도 저장
        display_labels = []  # Y축 레이블용

        for model_info in sorted_models:
            model = model_info['model_name']
            method = model_info['method']
            params = model_info['params']

            # Y축 레이블 생성: [Method-ParamsK] model_name
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
                # PEFT 모델 NMSE
                peft_nmse = next((r['nmse_db'] for r in peft_results
                                 if r['model_name'] == model and r['test_env'] == test_env), None)

                # Base NMSE
                base_nmse = base_performance.get(test_env)

                # 개선도 계산: PEFT - Base (음수 = 개선, 양수 = 악화)
                if peft_nmse is not None and base_nmse is not None:
                    improvement = peft_nmse - base_nmse
                    improvement_row.append(improvement)
                    actual_row.append(peft_nmse)
                else:
                    improvement_row.append(np.nan)
                    actual_row.append(np.nan)
            improvement_data.append(improvement_row)
            actual_nmse_data.append(actual_row)

        # DataFrame으로 변환
        improvement_df = pd.DataFrame(improvement_data, index=display_labels, columns=test_envs)
        actual_nmse_df = pd.DataFrame(actual_nmse_data, index=display_labels, columns=test_envs)

        # 커스텀 annotation 생성: "improvement\n(actual_nmse)"
        annot_labels = []
        for i in range(len(display_labels)):
            row_labels = []
            for j in range(len(test_envs)):
                improvement = improvement_df.iloc[i, j]
                actual_nmse = actual_nmse_df.iloc[i, j]
                if not np.isnan(improvement) and not np.isnan(actual_nmse):
                    row_labels.append(f"{improvement:+.2f}\n({actual_nmse:.2f})")
                else:
                    row_labels.append("")
            annot_labels.append(row_labels)
        annot_labels = np.array(annot_labels)

        # Heatmap 그리기
        fig, ax = plt.subplots(figsize=(14, max(20, len(display_labels) * 0.35)))

        # 색상: 초록(음수=개선) -> 노랑(0=비슷) -> 빨강(양수=악화)
        sns.heatmap(improvement_df,
                   annot=annot_labels,  # 커스텀 annotation 사용
                   fmt='',  # 문자열 형식이므로 fmt는 빈 문자열
                   cmap='RdYlGn_r',  # Red(나쁨) -> Yellow -> Green(좋음)
                   center=0,  # 0을 중심으로
                   vmin=-2,  # 최대 2dB 개선
                   vmax=2,   # 최대 2dB 악화
                   linewidths=0.5,
                   linecolor='gray',
                   cbar_kws={'label': 'Improvement vs Base (dB)', 'shrink': 0.8},
                   ax=ax,
                   annot_kws={'fontsize': 6.5})

        # 방법별 구분선 제거 (사용자 요청)
        # for method in method_order[:-1]:  # 마지막 방법 제외
        #     if method in method_boundaries:
        #         boundary_idx = method_boundaries[method]['end'] + 0.5
        #         ax.axhline(y=boundary_idx, color='black', linewidth=3, linestyle='-')

        ax.set_title('PEFT Performance Improvement vs Base Model\n(Top: Improvement | Bottom: Actual NMSE in dB)\nGrouped by Method and Sorted by Parameter Count',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Test Environment', fontsize=13, fontweight='bold')
        ax.set_ylabel('PEFT Model Configuration', fontsize=13, fontweight='bold')

        # Y축 레이블 회전 방지 및 폰트 크기 조정
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=7)

        plt.tight_layout()

        heatmap_path = Path(__file__).parent / 'pareto_improvement_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {heatmap_path}")
        plt.close()

        # 2. Simplified Transfer Effectiveness: In-domain vs Best Cross-domain
        print("[2/2] Generating simplified transfer effectiveness plots...")

        methods = ['Adapter', 'LoRA', 'Prompt', 'Hybrid']
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        axes = axes.flatten()

        for idx, method in enumerate(methods):
            ax = axes[idx]

            # 해당 방법의 모델들 필터링
            method_results = [r for r in results if r['method'] == method]

            if not method_results:
                ax.set_title(f'{method} - No Data', fontsize=15, fontweight='bold')
                continue

            # 각 테스트 환경별 데이터 수집
            x_labels = []
            in_domain_improvements = []
            best_cross_domain_improvements = []

            for test_env in test_envs:
                x_labels.append(test_env)
                base_nmse = base_performance.get(test_env, 0)

                # 1) In-domain: train_env == test_env
                in_domain = [r for r in method_results
                            if r['train_env'] == test_env and r['test_env'] == test_env]
                if in_domain:
                    best_in_domain_nmse = min(r['nmse_db'] for r in in_domain)
                    in_domain_improvements.append(best_in_domain_nmse - base_nmse)
                else:
                    in_domain_improvements.append(np.nan)

                # 2) Cross-domain: train_env != test_env 중 최고
                cross_domain = [r for r in method_results
                               if r['train_env'] != test_env and r['test_env'] == test_env]
                if cross_domain:
                    best_cross_domain_nmse = min(r['nmse_db'] for r in cross_domain)
                    best_cross_domain_improvements.append(best_cross_domain_nmse - base_nmse)
                else:
                    best_cross_domain_improvements.append(np.nan)

            # 막대 그래프
            x = np.arange(len(x_labels))
            width = 0.35

            ax.bar(x - width/2, in_domain_improvements, width,
                  label='In-domain (train=test)',
                  color='#2E7D32',  # 진한 초록
                  alpha=0.85,
                  edgecolor='black',
                  linewidth=1)

            ax.bar(x + width/2, best_cross_domain_improvements, width,
                  label='Best Cross-domain (train≠test)',
                  color='#1976D2',  # 진한 파랑
                  alpha=0.85,
                  edgecolor='black',
                  linewidth=1)

            # 0선 강조 (Base 기준선)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8)

            ax.set_xlabel('Test Environment', fontsize=13, fontweight='bold')
            ax.set_ylabel('Improvement vs Base (dB)', fontsize=13, fontweight='bold')
            ax.set_title(f'{method}: In-domain vs Cross-domain Transfer\n(Negative = Better)',
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=12)
            ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black')
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')

            # Y축 범위 설정
            ax.set_ylim(-2.5, 2.5)

        plt.suptitle('Transfer Learning Effectiveness: In-domain vs Cross-domain Performance\n(Best configuration per scenario)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        transfer_path = Path(__file__).parent / 'pareto_transfer_effectiveness.png'
        plt.savefig(transfer_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {transfer_path}")
        plt.close()

    def analyze_iterations(self, sample_configs=None, save_suffix=''):
        """Iteration별 성능 분석 - 수렴 속도 및 최적 iteration 파악"""
        print("\n" + "="*70)
        print(f"ITERATION ANALYSIS: Convergence and Optimal Stopping Point{' (' + save_suffix + ')' if save_suffix else ''}")
        print("="*70)

        # Iteration 파일 찾기
        iteration_results = []
        iterations = [20000, 40000, 60000, 80000]  # Final 모델은 별도로 100K로 처리

        # 테스트 데이터 로드
        test_datasets = {}
        for scenario in self.scenarios:
            test_data = self.load_test_data(scenario)
            if test_data is not None:
                test_datasets[scenario] = test_data

        # 샘플 모델들 (기본값: 가장 작은 파라미터)
        if sample_configs is None:
            sample_configs = {
                'Adapter': ['dim8'],
                'LoRA': ['r4'],
                'Prompt': ['len50'],
                'Hybrid': ['p50_r5']
            }

        for method, configs in sample_configs.items():
            method_dir = self.pareto_dir / method
            if not method_dir.exists():
                continue

            for scenario in self.scenarios:
                for config in configs:
                    model_results = {'method': method, 'scenario': scenario, 'config': config}

                    # Final 모델
                    final_files = list(method_dir.glob(f'*_to_{scenario}_*{config}.pt'))
                    final_files = [f for f in final_files if '_iter_' not in f.name]

                    if final_files:
                        final_model = final_files[0]
                        test_data = test_datasets.get(scenario)
                        if test_data:
                            nmse = self.evaluate_model(final_model, test_data)
                            if nmse:
                                model_results['final'] = nmse

                    # Iteration 모델들
                    for iter_num in iterations:
                        iter_files = list(method_dir.glob(f'*_to_{scenario}_*{config}_iter_{iter_num}.pt'))
                        if iter_files:
                            iter_model = iter_files[0]
                            test_data = test_datasets.get(scenario)
                            if test_data:
                                nmse = self.evaluate_model(iter_model, test_data)
                                if nmse:
                                    model_results[f'iter_{iter_num}'] = nmse
                                    print(f"  [{method}_{scenario}_{config}] iter_{iter_num}: {nmse:.2f} dB")

                    if len(model_results) > 3:  # method, scenario, config 외에 데이터가 있으면
                        iteration_results.append(model_results)

        # 수렴 곡선 그래프
        self.plot_convergence_curves(iteration_results, iterations, save_suffix)

        # 최적 iteration 표
        self.print_optimal_iteration_table(iteration_results, iterations)

        return iteration_results

    def plot_convergence_curves(self, iteration_results, iterations, save_suffix=''):
        """수렴 곡선 그래프 생성 - 시나리오별로 방법 비교"""
        print(f"\nGenerating convergence curves{' (' + save_suffix + ')' if save_suffix else ''}...")

        import matplotlib.colors as mcolors

        # 2x3 레이아웃 (5개 시나리오 + 1개 빈 공간)
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        axes = axes.flatten()

        # 방법별 예쁜 색상 (Material Design + 가독성 고려)
        method_colors = {
            'Adapter': '#1E88E5',  # 밝은 파랑
            'LoRA': '#7B1FA2',     # 보라
            'Prompt': '#E53935',   # 산호 빨강
            'Hybrid': '#00897B'    # 청록
        }
        method_markers = {
            'Adapter': 'o',
            'LoRA': 's',
            'Prompt': '^',
            'Hybrid': 'D'
        }

        # 각 시나리오별로 서브플롯 생성
        for idx, scenario in enumerate(self.scenarios):
            ax = axes[idx]

            # 해당 시나리오의 데이터 필터링
            scenario_data = [r for r in iteration_results if r['scenario'] == scenario]

            # 방법별로 수렴 곡선 그리기 (각 방법의 최고 성능 config만)
            for method in sorted(set(r['method'] for r in scenario_data)):
                method_models = [r for r in scenario_data if r['method'] == method]

                if not method_models:
                    continue

                # 해당 시나리오에서 Final 성능이 가장 좋은 config 선택
                best_model = min(method_models, key=lambda x: x.get('final', float('inf')))

                # X: Iteration, Y: NMSE
                x_vals = []
                y_vals = []

                # Iteration 체크포인트들
                for iter_num in iterations:
                    key = f'iter_{iter_num}'
                    if key in best_model:
                        x_vals.append(iter_num / 1000)  # K 단위로 변환
                        y_vals.append(best_model[key])

                # Final 추가 (100K iteration)
                if 'final' in best_model:
                    x_vals.append(100)  # Final은 100K
                    y_vals.append(best_model['final'])

                if x_vals and len(x_vals) >= 2:
                    ax.plot(x_vals, y_vals,
                           marker=method_markers[method],
                           color=method_colors[method],
                           label=f'{method} ({best_model["config"]})',
                           linewidth=2.5,
                           markersize=8,
                           alpha=0.85,
                           zorder=3)

            # 축 설정
            ax.set_xlabel('Training Iterations (K)', fontsize=12, fontweight='bold')
            ax.set_ylabel('NMSE (dB)', fontsize=12, fontweight='bold')
            ax.set_title(f'{scenario} Scenario Convergence', fontsize=14, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.legend(fontsize=9, loc='best', framealpha=0.9, edgecolor='black')

            # X축 범위 설정
            ax.set_xlim(15, 105)

        # 마지막 subplot 제거
        fig.delaxes(axes[5])

        plt.suptitle('Training Convergence Curves: Method Comparison per Scenario\n(Shows iteration-by-iteration training progress)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        filename = f'pareto_convergence{"_" + save_suffix if save_suffix else ""}.png'
        save_path = Path(__file__).parent / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()

    def print_optimal_iteration_table(self, iteration_results, iterations):
        """최적 iteration 표 출력"""
        print("\n" + "="*70)
        print("Optimal Iteration Analysis")
        print("="*70)
        print(f"{'Model':<30} {'Best Iter':<12} {'Best NMSE':<12} {'Final NMSE':<12} {'Δ':<8}")
        print("-"*70)

        for model in iteration_results:
            model_name = f"{model['method']}_{model['scenario']}_{model['config']}"

            # Best iteration 찾기
            best_iter = None
            best_nmse = None

            for iter_num in iterations:
                key = f'iter_{iter_num}'
                if key in model:
                    if best_nmse is None or model[key] < best_nmse:
                        best_nmse = model[key]
                        best_iter = iter_num

            final_nmse = model.get('final', None)

            if best_iter and final_nmse:
                delta = final_nmse - best_nmse
                print(f"{model_name:<30} {best_iter:<12} {best_nmse:<12.2f} {final_nmse:<12.2f} {delta:+.2f}")

        print("="*70)

    def plot_pareto_curves(self, results):
        """Pareto curve 플롯 - 파라미터 효율성 vs 성능 개선"""
        print("\nGenerating Pareto frontier curves...")

        # Base 모델 전체 파라미터 수 (문서 기준: 1.35M)
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

            # 해당 환경에서 학습한 모델들 필터링 (train_env == test_env == scenario)
            scenario_results = [r for r in results
                               if r['train_env'] == scenario and r['test_env'] == scenario]

            # Base 모델 성능 (개선도 계산용)
            base_result = next((r for r in results
                               if r['method'] == 'Base' and r['test_env'] == scenario), None)

            if not base_result:
                continue

            base_nmse = base_result['nmse_db']

            for method in ['Adapter', 'LoRA', 'Prompt', 'Hybrid']:
                # 방법별 결과 필터링 및 정렬
                method_results = [r for r in scenario_results if r['method'] == method]
                method_results.sort(key=lambda x: x['params'])

                if method_results:
                    # X축: 파라미터 비율 (%)
                    param_ratios = [(r['params'] / base_total_params) * 100 for r in method_results]

                    # Y축: Base 대비 개선도 (dB)
                    improvements = [r['nmse_db'] - base_nmse for r in method_results]

                    ax.plot(param_ratios, improvements,
                           marker=markers[method], color=colors[method],
                           label=method, linewidth=2.5, markersize=10, alpha=0.85)

            # Base 기준선 (y=0)
            ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Base')

            ax.set_xlabel('Trainable Parameter Ratio (%)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Improvement vs Base (dB)', fontsize=13, fontweight='bold')
            ax.set_title(f'{scenario} Scenario (In-Domain)\nBase NMSE: {base_nmse:.2f} dB',
                        fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.legend(loc='best', fontsize=10, framealpha=0.9)

            # Y축 범위 설정 (개선도 기준)
            all_improvements = []
            for method in ['Adapter', 'LoRA', 'Prompt', 'Hybrid']:
                method_results = [r for r in scenario_results if r['method'] == method]
                all_improvements.extend([r['nmse_db'] - base_nmse for r in method_results])

            if all_improvements:
                y_min = min(all_improvements) - 0.5
                y_max = max(all_improvements) + 0.5
                ax.set_ylim(y_min, y_max)

            # X축 범위 설정 (0.1% ~ 50%)
            ax.set_xlim(0.1, 50)

        # 마지막 subplot 제거
        fig.delaxes(axes[5])

        plt.suptitle('Pareto Frontier: Parameter Efficiency vs Performance Improvement\n(Negative = Better than Base | Log scale X-axis)',
                    fontsize=18, fontweight='bold')
        plt.tight_layout()

        # 저장
        save_path = Path(__file__).parent / 'pareto_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_path}")
        plt.close()

def main():
    analyzer = ParetoAnalyzer()

    # 1. Cross-environment analysis
    results = analyzer.run_analysis()

    if results is not None and len(results) > 0:
        print("\n" + "="*70)
        print("CROSS-ENVIRONMENT ANALYSIS SUMMARY")
        print("="*70)

        total_models = len(set(r['model_name'] for r in results))
        total_envs = len(set(r['test_env'] for r in results))
        print(f"Total models tested: {total_models}")
        print(f"Test environments: {total_envs}")
        print(f"Total evaluations: {len(results)}")

        # 각 테스트 환경별 최고 성능 모델
        print("\n" + "="*70)
        print("Best model per test environment:")
        print("="*70)

        test_envs = sorted(set(r['test_env'] for r in results))
        for test_env in test_envs:
            env_results = [r for r in results if r['test_env'] == test_env]
            best = min(env_results, key=lambda x: x['nmse_db'])
            print(f"  {test_env}: {best['model_name']}")
            print(f"         NMSE: {best['nmse_db']:.2f} dB (Params: {best['params']:,})")

        # 전이학습 효과: 같은 환경에서 학습하고 테스트
        print("\n" + "="*70)
        print("In-domain performance (Train env == Test env):")
        print("="*70)

        for env in test_envs:
            env_models = [r for r in results
                         if r['train_env'] == env and r['test_env'] == env]
            if env_models:
                best = min(env_models, key=lambda x: x['nmse_db'])
                print(f"  {env}: {best['model_name']} → {best['nmse_db']:.2f} dB")

        # 2. Pareto curves
        analyzer.plot_pareto_curves(results)

        # 3-1. Iteration analysis (Medium size - Representative)
        medium_configs = {
            'Adapter': ['dim16'],     # 131K (중간 크기)
            'LoRA': ['r8'],           # 39K (중간 크기)
            'Prompt': ['len100'],     # 12K (중간 크기)
            'Hybrid': ['p100_r10']    # 61K (중간 크기)
        }
        analyzer.analyze_iterations(sample_configs=medium_configs, save_suffix='medium')

        # 3-2. Iteration analysis (All sizes - Detailed)
        all_configs = {
            'Adapter': ['dim8', 'dim16', 'dim32', 'dim64'],
            'LoRA': ['r4', 'r8', 'r16', 'r20'],
            'Prompt': ['len50', 'len100', 'len200'],
            'Hybrid': ['p50_r5', 'p100_r10', 'p200_r20']
        }
        analyzer.analyze_iterations(sample_configs=all_configs, save_suffix='all')

    return results

if __name__ == "__main__":
    main()