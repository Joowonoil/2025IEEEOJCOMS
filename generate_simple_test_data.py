import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from dataset import get_dataset_and_dataloader

def generate_simple_test_data():
    """간단한 테스트 데이터 생성 (TensorRT, 제어신호 등 제거)"""
    
    # InF 테스트 데이터 설정 (Los + Nlos 섞음)
    inf_params = {
        'channel_type': ["InF_Los", "InF_Nlos"],  # Los와 Nlos 섞어서 사용
        'batch_size': 100,
        'noise_spectral_density': -174.0,
        'subcarrier_spacing': 120.0,
        'transmit_power': 30.0,
        'distance_ranges': {  # dict 형식으로!
            'InF_Los': [10.0, 100.0],
            'InF_Nlos': [10.0, 100.0]
        },
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

    # RMa 테스트 데이터 설정 (Los + Nlos 섞음)
    rma_params = inf_params.copy()
    rma_params['channel_type'] = ["RMa_Los", "RMa_Nlos"]
    rma_params['distance_ranges'] = {
        'RMa_Los': [10.0, 10000.0],
        'RMa_Nlos': [10.0, 10000.0]
    }

    # InH 테스트 데이터 설정 (Los + Nlos 섞음)
    inh_params = inf_params.copy()
    inh_params['channel_type'] = ["InH_Los", "InH_Nlos"]
    inh_params['distance_ranges'] = {
        'InH_Los': [5.0, 100.0],
        'InH_Nlos': [5.0, 100.0]
    }

    # UMi 테스트 데이터 설정 (Los + Nlos 섞음)
    umi_params = inf_params.copy()
    umi_params['channel_type'] = ["UMi_Los", "UMi_Nlos"]
    umi_params['distance_ranges'] = {
        'UMi_Los': [10.0, 500.0],
        'UMi_Nlos': [10.0, 500.0]
    }

    # UMa 테스트 데이터 설정 (Los + Nlos 섞음)
    uma_params = inf_params.copy()
    uma_params['channel_type'] = ["UMa_Los", "UMa_Nlos"]
    uma_params['distance_ranges'] = {
        'UMa_Los': [10.0, 10000.0],
        'UMa_Nlos': [10.0, 10000.0]
    }

    # 데이터 생성 및 저장
    for dataset_name, params in [
        ('InF', inf_params),
        ('RMa', rma_params),
        ('InH', inh_params),
        ('UMi', umi_params),
        ('UMa', uma_params)
    ]:
        print(f"Generating {dataset_name} test data...")
        
        # 데이터셋 및 데이터로더 생성
        dataset, dataloader = get_dataset_and_dataloader(params)
        
        # 한 배치만 가져오기
        for batch_idx, data in enumerate(dataloader):
            if batch_idx == 0:  # 첫 번째 배치만
                # 딕셔너리에서 데이터 추출
                rx_signal = data['ref_comp_rx_signal']  # (batch, 14, 3072) - 복소수, reference compensated!
                ch_freq = data['ch_freq']  # (batch, 3072) - 복소수

                # 복소수를 실수부/허수부로 분리하여 NumPy 배열로 변환
                rx_input = np.stack([rx_signal.real, rx_signal.imag], axis=-1)  # (batch, 14, 3072, 2)
                ch_true_np = ch_freq  # (batch, 3072) - 복소수 그대로 저장

                # 저장 경로 설정
                test_data_dir = Path(__file__).parent / 'simple_test_data'
                test_data_dir.mkdir(exist_ok=True)

                # 파일 저장
                np.save(test_data_dir / f'{dataset_name}_input.npy', rx_input)
                np.save(test_data_dir / f'{dataset_name}_true.npy', ch_true_np)

                print(f"  Saved: {rx_input.shape} input samples")
                print(f"  Saved: {ch_true_np.shape} ground truth samples")
                break
    
    print("Test data generation complete!")

if __name__ == "__main__":
    generate_simple_test_data()