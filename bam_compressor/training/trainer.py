# bam_compressor/training/trainer.py (최종 완성본)

import numpy as np
import sys
import os
import time

sys.path.append(os.getcwd())
from bam_compressor.models.main_model import MFBAM
from bam_compressor.data.data_loader import NumpyDataLoader

class Trainer:
    """
    논문 기반의 2단계 학습 파이프라인을 관리합니다.
    - 각 계층은 Loss가 수렴할 때까지 학습합니다.
    """
    def __init__(self, model: MFBAM, data_loader: NumpyDataLoader, learning_rate: float, convergence_threshold: float = 1e-7, patience: int = 15, max_epochs_per_layer: int = 1000):
        self.model = model
        self.data_loader = data_loader
        self.lr = learning_rate
        self.threshold = convergence_threshold
        self.patience = patience
        self.max_epochs = max_epochs_per_layer

    def _calculate_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        return np.mean((original - reconstructed)**2)

    def train(self):
        """ 전체 MF-BAM 모델의 학습 파이프라인을 실행합니다. """
        print("********** MF-BAM Training Start (Convergence-based) **********")
        self._train_mf_module()
        self._train_sl_module()
        print("\n✅ Total training process completed.")

    def _train_until_convergence(self, layer_name, training_step_func):
        """
        특정 레이어의 Loss가 수렴할 때까지 학습을 진행하는 범용 함수.
        """
        print(f"\n--- Training {layer_name} (until convergence) ---")
        
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.max_epochs):
            # training_step_func()는 한 에폭 동안의 학습을 수행하고 평균 loss를 반환
            epoch_loss = training_step_func()
            
            # 수렴 조건 체크 (Early Stopping)
            if best_loss - epoch_loss > self.threshold:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # 10 에폭마다 로그 출력
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:<4} | Loss: {epoch_loss:.8f} | Best Loss: {best_loss:.8f} | Patience: {patience_counter}/{self.patience}")

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}. No improvement for {self.patience} epochs.")
                break
        
        if epoch == self.max_epochs - 1:
            print(f"Reached max epochs ({self.max_epochs}).")

        end_time = time.time()
        print(f"--- Training for {layer_name} finished in {end_time - start_time:.2f} seconds. Final Loss: {best_loss:.8f} ---")

    def _train_mf_module(self):
        """ Phase 1: Feature Extractor (MF) 모듈을 계층별로 학습 """
        print("\n********** [PHASE 1] Feature Extractor (MF) Training **********")
        num_layers = len(self.model.feature_extractor.layers)
        
        for i in range(num_layers):
            def training_step():
                target_layer = self.model.feature_extractor.layers[i]
                total_loss = 0.0
                with np.errstate(all='ignore'):
                    for batch_X, _ in self.data_loader:
                        current_input = batch_X
                        if i > 0:
                            h_temp = batch_X
                            for j in range(i):
                                h_temp = self.model.feature_extractor.layers[j].predict(h_temp)
                            current_input = h_temp
                        
                        x0, y0, xc, yc = target_layer.forward(current_input)
                        target_layer.update_weights(x0, y0, xc, yc, self.lr)
                        total_loss += self._calculate_mse(x0, xc)
                return total_loss / self.data_loader.n_batches
            
            self._train_until_convergence(f"MF Layer {i+1}", training_step)

    def _train_sl_module(self):
        """ Phase 2: 압축 모듈(SL)을 오토인코더 역할로 학습 """
        
        def training_step():
            sl_layer = self.model.supervised_layer
            total_loss = 0.0
            
            with np.errstate(all='ignore'):
                for batch_X, _ in self.data_loader:
                    # 1. 고정된 MF 모듈로 특성 추출 (이것이 SL의 입력이 됨)
                    features_original = self.model.feature_extractor.predict(batch_X)
                    
                    # 2. SL 모듈을 이용한 인코딩 및 디코딩 과정
                    # x0는 SL의 입력, 즉 원본 특성입니다.
                    x0 = features_original  # Shape: (N, 7)
                    
                    # 인코딩: x0 -> y0 (latent_vector)
                    a0 = sl_layer.W @ x0.T
                    y0 = sl_layer.transmission_function(a0).T # Shape: (N, 4)

                    # 디코딩: y0 -> xc (reconstructed_features)
                    b0 = sl_layer.V @ y0.T
                    xc = sl_layer.transmission_function(b0).T # Shape: (N, 7)
                    
                    # 재-인코딩: xc -> yc
                    ac = sl_layer.W @ xc.T
                    yc = sl_layer.transmission_function(ac).T # Shape: (N, 4)

                    # 3. 올바른 shape의 변수들로 가중치 업데이트
                    # x0, xc: (N, 7) / y0, yc: (N, 4)
                    sl_layer.update_weights(x0, y0, xc, yc, self.lr)

                    # 4. 손실은 원본 특성과 복원된 특성 간의 차이로 계산
                    loss = self._calculate_mse(x0, xc)
                    total_loss += loss
                    
            return total_loss / self.data_loader.n_batches
        
        # 정의된 학습 로직을 사용하여 수렴할 때까지 학습 실행
        self._train_until_convergence("SL as Autoencoder", training_step)