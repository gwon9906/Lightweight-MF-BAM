# bam_compressor/models/main_model.py

import numpy as np
import os
import sys

# 프로젝트의 다른 모듈을 가져오기 위해 경로 추가
sys.path.append(os.getcwd())

from bam_compressor.models.feature_extractor import FeatureExtractor
from bam_compressor.models.compression_model import SupervisedLayer

class MFBAM:
    """
    전체 MF-BAM 아키텍처.
    - Feature Extractor (MF) 모듈과 Supervised Layer (SL) 모듈을 통합합니다.
    """
    def __init__(self, mf_layer_dims: list[int], target_dim: int, delta: float = 0.2):
        """
        Args:
            mf_layer_dims (list[int]): MF 모듈의 계층별 뉴런 수. 
                                       [input_dim, hidden1, hidden2, ...]
            target_dim (int): 최종 타겟의 차원.
            delta (float): 모든 계층에서 사용할 transmission function 파라미터.
        """
        if len(mf_layer_dims) < 2:
            raise ValueError("mf_layer_dims는 최소 [input_dim, hidden_dim] 2개 이상이어야 합니다.")
            
        feature_dim = mf_layer_dims[-1] # MF 모듈의 최종 출력이 SL 모듈의 입력이 됨

        self.feature_extractor = FeatureExtractor(layer_dims=mf_layer_dims, delta=delta)
        self.supervised_layer = SupervisedLayer(feature_dim=feature_dim, target_dim=target_dim, delta=delta)

        

    def predict(self, p: np.ndarray) -> np.ndarray:
        """
        (추론) 원본 입력 데이터(p)를 받아 최종 타겟을 예측합니다.
        
        Args:
            p (np.ndarray): 원본 입력 데이터.

        Returns:
            np.ndarray: 예측된 최종 타겟.
        """
        # 1. Feature Extractor를 통과시켜 특성을 추출합니다.
        extracted_features = self.feature_extractor.predict(p)
        
        # 2. 추출된 특성을 Supervised Layer에 넣어 최종 타겟을 예측합니다.
        predicted_target = self.supervised_layer.predict(extracted_features)
        
        return predicted_target

    def save_weights(self, path='data/weights'):
        """ 학습된 모든 가중치를 파일로 저장합니다. """
        os.makedirs(path, exist_ok=True)
        # MF 모듈 가중치 저장
        for i, layer in enumerate(self.feature_extractor.layers):
            np.savez(os.path.join(path, f'mf_layer_{i}_weights.npz'), W=layer.W, V=layer.V)
        # SL 모듈 가중치 저장
        np.savez(os.path.join(path, 'sl_weights.npz'), W=self.supervised_layer.W, V=self.supervised_layer.V)
        print(f"모든 가중치가 '{path}' 디렉토리에 저장되었습니다.")

    def load_weights(self, path='data/weights'):
        """ 파일에서 가중치를 불러옵니다. """
        try:
            # MF 모듈 가중치 로드
            for i, layer in enumerate(self.feature_extractor.layers):
                data = np.load(os.path.join(path, f'mf_layer_{i}_weights.npz'))
                layer.W = data['W']
                layer.V = data['V']
            # SL 모듈 가중치 로드
            data = np.load(os.path.join(path, 'sl_weights.npz'))
            self.supervised_layer.W = data['W']
            self.supervised_layer.V = data['V']
            print(f"모든 가중치를 '{path}' 디렉토리에서 불러왔습니다.")
        except FileNotFoundError:
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: '{path}'. 모델을 먼저 학습시켜주세요.")
        
    def reconstruct(self, o: np.ndarray) -> np.ndarray:
        """
        (역방향 추론) 타겟(o)을 입력받아 연관된 원본 입력 패턴(p)을 복원/연상합니다.
        """
        # ... reconstruct 메서드 코드 ...
        # 1. SL 모듈에서 역방향으로 특성 연상 (V 행렬 사용)
        b_0 = self.supervised_layer.V @ o.T
        reconstructed_features = self.supervised_layer.transmission_function(b_0).T

        # 2. 연상된 특성을 MF 모듈에서 역방향으로 통과시켜 원본 입력 복원
        reconstructed_input = reconstructed_features
        for layer in reversed(self.feature_extractor.layers):
            b_c = layer.V @ reconstructed_input.T
            reconstructed_input = layer.transmission_function(b_c).T
            
        return reconstructed_input