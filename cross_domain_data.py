"""
Cross-Domain Test Data Generator

Cross-Domain 전이학습 분석을 위한 도메인별 테스트 데이터 생성 도구입니다.
학습 데이터와 겹치지 않도록 별도의 _10000.mat 파일들을 사용합니다.

생성되는 테스트 데이터:
- indoor: InF + InH 환경 통합
- outdoor: UMa + RMa 환경 통합  
- urban: UMa + UMi 환경 통합 (UMi는 UMa로 대체)
- rural: RMa 환경

각 도메인별로 여러 환경의 데이터를 결합하여 해당 도메인을 대표하는 테스트셋 생성
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from dataset import get_dataset_and_dataloader

# 테스트 데이터 설정 (유연하게 변경 가능)
TEST_DATA_CONFIG = {
    'indoor': {
        'environments': [
            {
                'name': 'InF',
                'channel_type': 'InF_Los_10000',
                'distance': 50.0,
                'samples_ratio': 0.7  # 전체 샘플 중 70%
            },
            {
                'name': 'InH', 
                'channel_type': 'InH_Los_10000',
                'distance': 30.0,
                'samples_ratio': 0.3  # 전체 샘플 중 30%
            }
        ],
        'description': 'Indoor environments (Factory + Hotspot)'
    },
    
    'outdoor': {
        'environments': [
            {
                'name': 'UMa',
                'channel_type': 'UMa_Los_10000', 
                'distance': 100.0,
                'samples_ratio': 0.5
            },
            {
                'name': 'RMa',
                'channel_type': 'RMa_Los_10000',
                'distance': 300.0, 
                'samples_ratio': 0.5
            }
        ],
        'description': 'Outdoor environments (Urban Macro + Rural Macro)'
    },
    
    'urban': {
        'environments': [
            {
                'name': 'UMa',
                'channel_type': 'UMa_Los_10000',
                'distance': 100.0,
                'samples_ratio': 0.7
            },
            {
                'name': 'UMi_substitute', 
                'channel_type': 'UMa_Los_10000',  # UMi 데이터가 없으므로 UMa로 대체
                'distance': 50.0,  # UMi는 더 짧은 거리
                'samples_ratio': 0.3
            }
        ],
        'description': 'Urban environments (UMa + UMi substitute)'
    },
    
    'rural': {
        'environments': [
            {
                'name': 'RMa',
                'channel_type': 'RMa_Los_10000',
                'distance': 300.0,
                'samples_ratio': 1.0  # RMa만 사용
            }
        ],
        'description': 'Rural environments (Rural Macro)'
    }
}

# 공통 파라미터
COMMON_PARAMS = {
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
    'rnd_seed': 0,
    'num_workers': 0,
    'is_phase_noise': False,
    'is_channel': True,
    'is_noise': True
}

def check_available_files():
    """사용 가능한 MAT 파일들 확인"""
    dataset_dir = Path(__file__).parent / 'dataset' / 'PDP_processed'
    available_files = []
    
    print("Checking available MAT files...")
    for domain, config in TEST_DATA_CONFIG.items():
        print(f"\n{domain.upper()} domain:")
        for env in config['environments']:
            mat_file = f"PDP_{env['channel_type']}.mat"
            file_path = dataset_dir / mat_file
            
            if file_path.exists():
                available_files.append(mat_file)
                print(f"  [OK] {mat_file} - Available")
            else:
                print(f"  [NOT FOUND] {mat_file} - Not found")
    
    return available_files

def generate_domain_data(domain_name, domain_config, total_samples=100):
    """특정 도메인의 테스트 데이터 생성"""
    print(f"\nGenerating {domain_name} domain data...")
    print(f"Description: {domain_config['description']}")
    
    all_inputs = []
    all_truths = []
    
    for env in domain_config['environments']:
        env_name = env['name']
        channel_type = env['channel_type']
        distance = env['distance']
        ratio = env['samples_ratio']
        
        # 현재 환경에서 생성할 샘플 수
        env_samples = int(total_samples * ratio)
        if env_samples == 0:
            continue
            
        print(f"  Generating {env_samples} samples from {env_name} ({channel_type}) at {distance}m...")
        
        # 데이터 생성 파라미터 설정
        params = COMMON_PARAMS.copy()
        params.update({
            'channel_type': [channel_type],
            'distance_range': [distance, distance],
            'batch_size': env_samples
        })
        
        try:
            # 데이터셋 및 데이터로더 생성
            dataset, dataloader = get_dataset_and_dataloader(params)
            
            # 한 배치만 가져오기
            for batch_idx, data in enumerate(dataloader):
                if batch_idx == 0:  # 첫 번째 배치만
                    # 데이터 형식 확인 및 처리
                    if isinstance(data, (list, tuple)):
                        rx_signal, ch_true = data[0], data[1]
                    else:
                        rx_signal = data['ref_comp_rx_signal']
                        ch_true = data['ch_freq']
                    
                    # NumPy 배열로 변환 (이미 numpy array인 경우 그대로 사용)
                    if hasattr(rx_signal, 'numpy'):
                        rx_input = rx_signal.numpy()  # (batch, 14, 3072, 2)
                        ch_true_np = ch_true.numpy()  # (batch, 3072) - 복소수
                    else:
                        rx_input = rx_signal  # 이미 numpy array
                        ch_true_np = ch_true  # 이미 numpy array
                    
                    # 입력 데이터가 3차원인 경우 4차원으로 변환
                    if rx_input.ndim == 3:  # (batch, 14, 3072) -> (batch, 14, 3072, 2)
                        # 복소수를 실수/허수 분리하여 마지막 차원 추가
                        rx_input_real = np.real(rx_input)
                        rx_input_imag = np.imag(rx_input)
                        rx_input = np.stack([rx_input_real, rx_input_imag], axis=-1)
                        print(f"    [INFO] Converted input shape from 3D to 4D: {rx_input.shape}")
                    
                    all_inputs.append(rx_input)
                    all_truths.append(ch_true_np)
                    
                    print(f"    [OK] Generated {rx_input.shape[0]} samples")
                    break
                    
        except Exception as e:
            print(f"    [ERROR] Failed to generate {env_name} data: {e}")
            continue
    
    if not all_inputs:
        print(f"  [ERROR] No data generated for {domain_name} domain")
        return None, None
    
    # 모든 환경의 데이터 결합
    combined_input = np.concatenate(all_inputs, axis=0)
    combined_truth = np.concatenate(all_truths, axis=0)
    
    print(f"  [OK] Combined {domain_name} data: {combined_input.shape} input, {combined_truth.shape} truth")
    
    return combined_input, combined_truth

def save_domain_data(domain_name, input_data, truth_data):
    """도메인 데이터 저장"""
    # 저장 경로 설정
    test_data_dir = Path(__file__).parent / 'cross_domain_test_data'
    test_data_dir.mkdir(exist_ok=True)
    
    # 파일 저장
    input_path = test_data_dir / f'{domain_name}_input.npy'
    truth_path = test_data_dir / f'{domain_name}_true.npy'
    
    np.save(input_path, input_data)
    np.save(truth_path, truth_data)
    
    print(f"  [SAVED] {domain_name} data:")
    print(f"    Input: {input_path}")
    print(f"    Truth: {truth_path}")

def generate_cross_domain_test_data():
    """모든 도메인의 Cross-Domain 테스트 데이터 생성"""
    print("="*80)
    print("Cross-Domain Test Data Generation")
    print("="*80)
    
    # 사용 가능한 파일 확인
    available_files = check_available_files()
    
    if not available_files:
        print("\n[ERROR] No MAT files available for test data generation!")
        return
    
    print(f"\n[INFO] Generating test data for {len(TEST_DATA_CONFIG)} domains...")
    
    # 각 도메인별 데이터 생성
    generated_domains = []
    
    for domain_name, domain_config in TEST_DATA_CONFIG.items():
        try:
            input_data, truth_data = generate_domain_data(domain_name, domain_config)
            
            if input_data is not None and truth_data is not None:
                save_domain_data(domain_name, input_data, truth_data)
                generated_domains.append(domain_name)
            else:
                print(f"  [WARNING] Skipping {domain_name} domain due to generation failure")
                
        except Exception as e:
            print(f"  [ERROR] Error generating {domain_name} domain: {e}")
    
    # 결과 요약
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    
    if generated_domains:
        print(f"[SUCCESS] Successfully generated {len(generated_domains)} domain test datasets:")
        for domain in generated_domains:
            print(f"  - {domain}")
        
        print(f"\n[INFO] Files saved in: cross_domain_test_data/")
        
        # Cross-Domain 시나리오 매핑 안내
        print(f"\n[INFO] Cross-Domain Test Mapping:")
        scenarios = {
            'Urban_to_Rural': 'rural',
            'Rural_to_Urban': 'urban', 
            'Indoor_to_Outdoor': 'outdoor',
            'Outdoor_to_Indoor': 'indoor'
        }
        
        for scenario, test_domain in scenarios.items():
            status = "[OK]" if test_domain in generated_domains else "[MISSING]"
            print(f"  {status} {scenario} -> {test_domain} test data")
            
    else:
        print("[ERROR] No domain test data generated!")
    
    print("="*80)

def update_config(new_config=None):
    """테스트 데이터 설정 업데이트 (필요 시 사용)"""
    global TEST_DATA_CONFIG
    
    if new_config:
        TEST_DATA_CONFIG.update(new_config)
        print("[OK] Test data configuration updated")
    else:
        print("Current configuration:")
        for domain, config in TEST_DATA_CONFIG.items():
            print(f"\n{domain}:")
            for env in config['environments']:
                print(f"  - {env['name']}: {env['channel_type']} @ {env['distance']}m ({env['samples_ratio']*100:.0f}%)")

if __name__ == "__main__":
    # 설정 확인 옵션
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        update_config()
    else:
        generate_cross_domain_test_data()