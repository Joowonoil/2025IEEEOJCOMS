"""
다중 테스트 셋 생성 (통계적 검증용)

각 시나리오당 5개의 서로 다른 seed로 테스트 셋 생성
- 총 25개 테스트 셋 (5 scenarios × 5 seeds)
- 파일명: {scenario}_input_seed{seed}.npy, {scenario}_true_seed{seed}.npy
"""

import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from dataset import get_dataset_and_dataloader


def generate_test_set_with_seed(scenario_name, scenario_params, seed):
    """특정 seed로 테스트 셋 생성"""
    print(f"  Generating {scenario_name} test set with seed={seed}...")

    # Seed 설정 (재현성 보장)
    params = scenario_params.copy()
    params['rnd_seed'] = seed

    # NumPy와 PyTorch seed도 설정
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 데이터셋 및 데이터로더 생성
    dataset, dataloader = get_dataset_and_dataloader(params)

    # 한 배치만 가져오기
    for batch_idx, data in enumerate(dataloader):
        if batch_idx == 0:  # 첫 번째 배치만
            # 딕셔너리에서 데이터 추출
            rx_signal = data['ref_comp_rx_signal']  # (batch, 14, 3072) - 복소수
            ch_freq = data['ch_freq']  # (batch, 3072) - 복소수

            # 복소수를 실수부/허수부로 분리
            rx_input = np.stack([rx_signal.real, rx_signal.imag], axis=-1)  # (batch, 14, 3072, 2)
            ch_true_np = ch_freq  # (batch, 3072) - 복소수 그대로

            # 저장 경로 설정
            test_data_dir = Path(__file__).parent / 'simple_test_data'
            test_data_dir.mkdir(exist_ok=True)

            # Seed 포함 파일명으로 저장
            input_file = test_data_dir / f'{scenario_name}_input_seed{seed}.npy'
            true_file = test_data_dir / f'{scenario_name}_true_seed{seed}.npy'

            np.save(input_file, rx_input)
            np.save(true_file, ch_true_np)

            print(f"    Saved: {input_file.name} ({rx_input.shape})")
            print(f"    Saved: {true_file.name} ({ch_true_np.shape})")
            break

    return True


def generate_multiple_test_sets(num_seeds=5):
    """모든 시나리오에 대해 다중 seed 테스트 셋 생성"""

    print("="*70)
    print("Generating Multiple Test Sets for Statistical Validation")
    print("="*70)
    print(f"Seeds per scenario: {num_seeds}")
    print(f"Total test sets: {5 * num_seeds} (5 scenarios × {num_seeds} seeds)")
    print("="*70)

    # Base 파라미터 설정
    base_params = {
        'batch_size': 100,
        'noise_spectral_density': -174.0,
        'subcarrier_spacing': 120.0,
        'transmit_power': 30.0,
        'carrier_freq': 28.0,
        'mod_order': 64,
        'ref_conf_dict': {'dmrs': [0, 3072, 6]},
        'fft_size': 4096,
        'num_guard_subcarriers': 1024,
        'num_symbol': 14,
        'cp_length': 590,
        'max_random_tap_delay_cp_proportion': 0.2,
        'num_workers': 0,
        'is_phase_noise': False,
        'is_channel': True,
        'is_noise': True
    }

    # 시나리오별 설정
    scenarios = {
        'InF': {
            'channel_type': ["InF_Los", "InF_Nlos"],
            'distance_ranges': {
                'InF_Los': [10.0, 100.0],
                'InF_Nlos': [10.0, 100.0]
            }
        },
        'InH': {
            'channel_type': ["InH_Los", "InH_Nlos"],
            'distance_ranges': {
                'InH_Los': [5.0, 100.0],
                'InH_Nlos': [5.0, 100.0]
            }
        },
        'UMi': {
            'channel_type': ["UMi_Los", "UMi_Nlos"],
            'distance_ranges': {
                'UMi_Los': [10.0, 500.0],
                'UMi_Nlos': [10.0, 500.0]
            }
        },
        'UMa': {
            'channel_type': ["UMa_Los", "UMa_Nlos"],
            'distance_ranges': {
                'UMa_Los': [10.0, 10000.0],
                'UMa_Nlos': [10.0, 10000.0]
            }
        },
        'RMa': {
            'channel_type': ["RMa_Los", "RMa_Nlos"],
            'distance_ranges': {
                'RMa_Los': [10.0, 10000.0],
                'RMa_Nlos': [10.0, 10000.0]
            }
        }
    }

    # 각 시나리오 × 각 seed로 생성
    total_generated = 0
    for scenario_name, scenario_config in scenarios.items():
        print(f"\n[{scenario_name}] Generating test sets...")

        # Base params + scenario specific params
        params = base_params.copy()
        params.update(scenario_config)

        # 5개 seed로 생성
        for seed in range(1, num_seeds + 1):
            success = generate_test_set_with_seed(scenario_name, params, seed)
            if success:
                total_generated += 1

    print("\n" + "="*70)
    print(f"Test set generation complete!")
    print(f"Total generated: {total_generated} / {5 * num_seeds}")
    print("="*70)

    # 생성된 파일 목록 출력
    test_data_dir = Path(__file__).parent / 'simple_test_data'
    print(f"\nFiles saved in: {test_data_dir}")
    print("File list:")

    for scenario_name in scenarios.keys():
        print(f"\n  {scenario_name}:")
        for seed in range(1, num_seeds + 1):
            input_file = test_data_dir / f'{scenario_name}_input_seed{seed}.npy'
            true_file = test_data_dir / f'{scenario_name}_true_seed{seed}.npy'

            input_exists = "✓" if input_file.exists() else "✗"
            true_exists = "✓" if true_file.exists() else "✗"

            print(f"    seed{seed}: input {input_exists}  true {true_exists}")


if __name__ == "__main__":
    # 5개 seed로 생성 (필요시 인자로 변경 가능)
    generate_multiple_test_sets(num_seeds=5)