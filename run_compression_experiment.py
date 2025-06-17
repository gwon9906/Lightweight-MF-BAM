# run_compression_experiment.py

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 프로젝트 모듈 임포트
# (VSCode 등에서 실행 시 경로 문제를 피하기 위해)
sys.path.append(os.getcwd())
from bam_compressor.models.main_model import MFBAM
from bam_compressor.data.data_loader import NumpyDataLoader
from bam_compressor.training.trainer import Trainer


# 0. 실험 설정
# ============================================
# -- 데이터 관련 설정 --
# ⚠️ 이 경로는 preprocess_lora_log.py 실행 후 생성되는 파일 경로입니다.
DATA_PATH = 'data/original/clean_lora_data_combined.csv' 
# 학습된 가중치가 저장될 폴더 이름
WEIGHTS_PATH = 'data/compression_weights'

# -- 모델 아키텍처 설정 --
INPUT_DIM = 12
LATENT_DIM = 8
MF_LAYER_DIMS = [INPUT_DIM, 16, 14] # 기존 구조 유지

#-- 학습 하이퍼파라미터 --
LEARNING_RATE = 4e-5  # 학습률
MAX_EPOCHS_PER_LAYER = 15000 # 레이어당 최대 학습 에폭
CONVERGENCE_THRESHOLD = 1e-7 # 이 값보다 Loss 개선이 적으면 수렴으로 간주
PATIENCE = 100 # 개선 없이 100 에폭을 기다림
BATCH_SIZE = 32  # 배치 크기

# 실제 데이터 컬럼 이름 (그래프 레이블용)
FEATURE_NAMES = [
    'acc_ax', 'acc_ay', 'acc_az', 'gyro_gx', 'gyro_gy', 'gyro_gz',
    'ang_roll', 'ang_pitch', 'ang_yaw', 'gps_lat', 'gps_lon', 'gps_alt'
]


def main():
    """ 전체 실험 파이프라인을 실행하는 메인 함수 """

    # 1. 데이터 준비
    # ============================================
    print("--- 1. 데이터 준비 ---")
    try:
        data_loader = NumpyDataLoader(filepath=DATA_PATH, batch_size=BATCH_SIZE, shuffle=True, test_size=0.15)
    except FileNotFoundError:
        print(f"오류: 데이터 파일을 찾을 수 없습니다: '{DATA_PATH}'")
        print("솔루션: 'scripts/preprocess_lora_log.py'를 먼저 실행하여 데이터를 정제해주세요.")
        sys.exit(1)

    # 2. 모델 초기화
    # ============================================
    print("\n--- 2. 모델 초기화 ---")
    model = MFBAM(mf_layer_dims=MF_LAYER_DIMS, latent_dim=LATENT_DIM)
    print("MFBAM 모델 생성 완료:")
    print(f"  - 입력 차원: {INPUT_DIM}")
    print(f"  - MF 구조: {' -> '.join(map(str, MF_LAYER_DIMS))}")
    print(f"  - 잠재(압축) 차원: {LATENT_DIM}")

    # 3. 모델 학습
    # ============================================
    print("\n--- 3. 모델 학습 ---")
    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        learning_rate=LEARNING_RATE,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        patience=PATIENCE,
        max_epochs_per_layer=MAX_EPOCHS_PER_LAYER
    )
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    trainer.train()
    model.save_weights(WEIGHTS_PATH)

    # 4. 압축 및 복원 성능 테스트
    # ============================================
    print("\n\n--- 4. 압축/복원 성능 테스트 ---")
    X_test, _ = data_loader.get_test_data()
    
    if len(X_test) == 0:
        print("테스트 데이터가 없습니다. 성능 평가를 건너뜁니다.")
        return
        
    # 전체 테스트셋에 대한 인코딩 및 디코딩
    encoded_test = model.encode(X_test)
    reconstructed_test = model.decode(encoded_test)

    # 전체 테스트셋에 대한 복원 오류(MSE) 계산
    mse = np.mean((X_test - reconstructed_test)**2)
    print(f"✅ 전체 테스트셋 복원 오류 (MSE): {mse:.6f}")
    print(f"   (0에 가까울수록 복원이 잘 된 것입니다.)")

    # 5. 단일 샘플 결과 상세 비교 및 시각화
    # ============================================
    print("\n--- 5. 단일 샘플 상세 비교 ---")
    sample_original_scaled = X_test[0:1] # 테스트 데이터 첫 번째 샘플 (스케일링 된 상태)
    
    # 인코딩 (압축)
    latent_vector = model.encode(sample_original_scaled)
    # 디코딩 (복원)
    reconstructed_data_scaled = model.decode(latent_vector)

    # 결과를 사람이 읽을 수 있도록 원래 값의 범위로 되돌리기
    sample_original_unscaled = data_loader.inverse_transform(sample_original_scaled)
    reconstructed_data_unscaled = data_loader.inverse_transform(reconstructed_data_scaled)
    
    print(f"\n압축률: {INPUT_DIM}개 피처 -> {LATENT_DIM}개 피처 ({INPUT_DIM/LATENT_DIM:.1f} : 1)")
    print("-" * 50)
    print(f"압축된 잠재 벡터 (LoRa로 전송될 데이터):")
    print(np.round(latent_vector.flatten(), 4))
    print("-" * 50)
    print(f"{'Feature':<15} | {'Original':>12} | {'Reconstructed':>15} | {'Error':>10}")
    print("-" * 50)
    for i in range(INPUT_DIM):
        orig = sample_original_unscaled.flatten()[i]
        recon = reconstructed_data_unscaled.flatten()[i]
        error = orig - recon
        # ⚠️ data_loader에서 피처 이름을 가져와 사용
        print(f"{data_loader.feature_names[i]:<15} | {orig:12.4f} | {recon:15.4f} | {error:10.4f}")
    print("-" * 50)
    
    # 시각화
    plt.figure(figsize=(15, 7))
    plt.plot(sample_original_unscaled.flatten(), 'o-', color='blue', label='Original Data', markersize=7, linewidth=2)
    plt.plot(reconstructed_data_unscaled.flatten(), 'x--', color='red', label='Reconstructed Data', markersize=7)
    plt.title("Original vs. Reconstructed Sensor Data (Single Sample)", fontsize=16)
    plt.xlabel("Sensor Feature Index", fontsize=12)
    plt.ylabel("Original Sensor Value", fontsize=12)
    # ⚠️ data_loader에서 피처 이름을 가져와 사용
    plt.xticks(range(INPUT_DIM), labels=data_loader.feature_names, rotation=45, ha="right")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()