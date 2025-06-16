# run_experiment.py

import numpy as np
import sys
import matplotlib.pyplot as plt

# 프로젝트 모듈 임포트
from bam_compressor.models.main_model import MFBAM
from bam_compressor.data.data_loader import NumpyDataLoader
from bam_compressor.training.trainer import Trainer

# 0. 실험 설정
# ============================================
MF_LAYER_DIMS = [2, 16, 32]  # Moons(2) -> UL1(16) -> UL2(32)
TARGET_DIM = 1
LEARNING_RATE = 1e-4  # 학습률
MF_EPOCHS = 100
SL_EPOCHS = 200
BATCH_SIZE = 16
DATA_PATH = 'data/original/moons_dataset.csv'
WEIGHTS_PATH = 'data/weights'

# 1. 데이터 준비
# ============================================
print("--- 1. 데이터 준비 ---")
try:
    data_loader = NumpyDataLoader(filepath=DATA_PATH, batch_size=BATCH_SIZE, shuffle=True, test_size=0.2)
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# 2. 모델 초기화
# ============================================
print("\n--- 2. 모델 초기화 ---")
mf_bam_model = MFBAM(mf_layer_dims=MF_LAYER_DIMS, target_dim=TARGET_DIM)

# 3. 모델 학습
# ============================================
print("\n--- 3. 모델 학습 ---")
trainer = Trainer(
    model=mf_bam_model,
    data_loader=data_loader,
    learning_rate=LEARNING_RATE,
    mf_epochs=MF_EPOCHS,
    sl_epochs=SL_EPOCHS
)
trainer.train()

# 학습된 가중치 저장
mf_bam_model.save_weights(WEIGHTS_PATH)

# (선택) 저장된 가중치 다시 불러오기 테스트
# new_model = MFBAM(mf_layer_dims=MF_LAYER_DIMS, target_dim=TARGET_DIM)
# new_model.load_weights(WEIGHTS_PATH)

# 4. 순방향 테스트 (x -> y') 및 성능 평가
# ============================================
print("\n\n--- 4. 순방향 테스트 (x -> y') ---")
X_test, y_test = data_loader.get_test_data()

if len(X_test) > 0:
    predictions_raw = mf_bam_model.predict(X_test)
    predictions_class = np.sign(predictions_raw)
    
    accuracy = np.mean(predictions_class == y_test) * 100
    print(f"✅ 테스트 데이터셋 정확도: {accuracy:.2f}%")

    # 결정 경계(Decision Boundary) 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    
    # 그리드 생성
    h = .02  # step size in the mesh
    x_min, x_max = X_test[:, 0].min() - .5, X_test[:, 0].max() + .5
    y_min, y_max = X_test[:, 1].min() - .5, X_test[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 그리드 포인트에 대한 예측
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mf_bam_model.predict(grid_points)
    Z = np.sign(Z)
    Z = Z.reshape(xx.shape)

    # 시각화
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title("결정 경계 (Decision Boundary)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

# 5. 역방향 테스트 (y -> x')
# ============================================
print("\n--- 5. 역방향 테스트 (y -> x') ---")
target_class1 = np.array([[1.0]])   # 클래스 1
target_class2 = np.array([[-1.0]])  # 클래스 -1

reconstructed_pattern1 = mf_bam_model.reconstruct(target_class1)
reconstructed_pattern2 = mf_bam_model.reconstruct(target_class2)

print(f"클래스 {target_class1[0][0]}에 대한 전형적인(prototypical) 입력 패턴 x': {reconstructed_pattern1.flatten()}")
print(f"클래스 {target_class2[0][0]}에 대한 전형적인(prototypical) 입력 패턴 x': {reconstructed_pattern2.flatten()}")

# 시각화
plt.subplot(1, 2, 2)
# 원본 데이터 분포
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap=plt.cm.RdYlBu, alpha=0.3, label='Test Data')
# 복원된 패턴
plt.scatter(reconstructed_pattern1[:, 0], reconstructed_pattern1[:, 1], 
            color='red', marker='*', s=200, edgecolor='black', label=f'Reconstructed for Class {target_class1[0][0]}')
plt.scatter(reconstructed_pattern2[:, 0], reconstructed_pattern2[:, 1], 
            color='blue', marker='*', s=200, edgecolor='black', label=f'Reconstructed for Class {target_class2[0][0]}')
plt.title("역방향 복원 (y -> x')")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()