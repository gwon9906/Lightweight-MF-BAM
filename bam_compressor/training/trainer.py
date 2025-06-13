# bam_compressor/training/trainer.py (수정된 최종 버전)

import numpy as np
import os
import sys

sys.path.append(os.getcwd())

# 이제 main_model을 직접 사용합니다.
from bam_compressor.models.main_model import MFBAM
from bam_compressor.data.data_loader import NumpyDataLoader

class Trainer:
    """
    전체 MF-BAM 모델의 2단계 학습 파이프라인을 관리합니다.
    1. 비지도 학습: Feature Extractor (MF) 모듈을 계층별로 학습
    2. 지도 학습: Supervised Layer (SL) 모듈을 학습
    """
    def __init__(self, model: MFBAM, data_loader: NumpyDataLoader, learning_rate: float, mf_epochs: int, sl_epochs: int):
        self.model = model
        self.data_loader = data_loader
        self.lr = learning_rate
        self.mf_epochs = mf_epochs
        self.sl_epochs = sl_epochs

    def _calculate_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        return np.mean((original - reconstructed) ** 2)

    def _train_mf_module(self):
        """ 1단계: Feature Extractor (MF) 모듈을 계층별로 순차 학습 """
        print("\n********** [PHASE 1] Feature Extractor (MF) Training Start **********")
        
        num_layers = len(self.model.feature_extractor.layers)
        for i in range(num_layers):
            self._train_single_ul_layer(layer_idx=i)
        
        print("********** [PHASE 1] Feature Extractor (MF) Training Finished **********")

    def _train_single_ul_layer(self, layer_idx: int):
        """ MF 모듈의 단일 비지도 계층(UL)을 학습 (이전 로직과 동일) """
        target_layer = self.model.feature_extractor.layers[layer_idx]
        print(f"\n--- Training MF Layer {layer_idx + 1}/{len(self.model.feature_extractor.layers)} ---")

        for epoch in range(self.mf_epochs):
            epoch_loss = 0.0
            for batch_X, _ in self.data_loader:
                current_input = batch_X
                if layer_idx > 0:
                    for i in range(layer_idx):
                        _, _, _, current_input = self.model.feature_extractor.layers[i].forward(current_input)
                
                x0, _, xc, _ = target_layer.forward(current_input)
                target_layer.update_weights(x0, _, xc, _, self.lr) # y0, yc는 업데이트에 필요하나 반환값 혼동 방지
                
                loss = self._calculate_mse(x0, xc)
                epoch_loss += loss
            
            avg_epoch_loss = epoch_loss / self.data_loader.n_batches
            if (epoch + 1) % (self.mf_epochs // 5 or 1) == 0:
                print(f"  Epoch [{epoch + 1:>{len(str(self.mf_epochs))}}/{self.mf_epochs}] | Reconstruction Loss: {avg_epoch_loss:.6f}")
    
    def _train_sl_module(self):
        """ 2단계: Supervised Layer (SL) 모듈을 지도 학습 """
        print("\n********** [PHASE 2] Supervised Layer (SL) Training Start **********")
        
        target_layer = self.model.supervised_layer
        for epoch in range(self.sl_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in self.data_loader:
                # 1. MF 모듈로 특성 추출 (학습된 가중치 사용)
                extracted_features = self.model.feature_extractor.predict(batch_X)
                
                # 2. SL 모듈 학습
                x0, y0, xc, yc = target_layer.forward(extracted_features, batch_y)
                target_layer.update_weights(x0, y0, xc, yc, self.lr)
                
                # 지도 학습의 손실은 타겟에 대한 예측 오류
                loss = self._calculate_mse(y0, yc)
                epoch_loss += loss
                
            avg_epoch_loss = epoch_loss / self.data_loader.n_batches
            if (epoch + 1) % (self.sl_epochs // 5 or 1) == 0:
                print(f"  Epoch [{epoch + 1:>{len(str(self.sl_epochs))}}/{self.sl_epochs}] | Association Loss: {avg_epoch_loss:.6f}")
                
        print("********** [PHASE 2] Supervised Layer (SL) Training Finished **********")

    def train(self):
        """ 전체 MF-BAM 모델의 학습 파이프라인을 실행합니다. """
        self._train_mf_module()
        self._train_sl_module()
        print("\n✅ Total training process completed.")


# --- 사용 예시 (메인 실행 스크립트) ---
if __name__ == '__main__':
    # 1. 설정값 정의
    MF_LAYER_DIMS = [2, 10, 20]  # Moons(2) -> UL1(10) -> UL2(20)
    TARGET_DIM = 1
    LEARNING_RATE = 0.01
    MF_EPOCHS = 100       # MF 모듈의 각 레이어를 학습할 에폭 수
    SL_EPOCHS = 200       # SL 모듈을 학습할 에폭 수
    BATCH_SIZE = 16
    DATA_PATH = 'data/original/moons_dataset.csv'

    # 2. 모델 및 데이터 로더 준비
    try:
        data_loader = NumpyDataLoader(filepath=DATA_PATH, batch_size=BATCH_SIZE, shuffle=True)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    mf_bam_model = MFBAM(mf_layer_dims=MF_LAYER_DIMS, target_dim=TARGET_DIM)

    # 3. Trainer 인스턴스화 및 학습 시작
    trainer = Trainer(
        model=mf_bam_model,
        data_loader=data_loader,
        learning_rate=LEARNING_RATE,
        mf_epochs=MF_EPOCHS,
        sl_epochs=SL_EPOCHS
    )
    trainer.train()
    
    # 4. 학습된 가중치 저장
    mf_bam_model.save_weights()

    # 5. 학습된 모델로 테스트 데이터에 대한 성능 평가
    print("\n--- Evaluating on Test Data ---")
    X_test, y_test = data_loader.get_test_data()
    
    if len(X_test) > 0:
        predictions_raw = mf_bam_model.predict(X_test)
        predictions = np.sign(predictions_raw) # 클래스(-1 또는 1)로 변환
        
        accuracy = np.mean(predictions == y_test) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")