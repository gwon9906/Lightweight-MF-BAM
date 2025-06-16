# bam_compressor/models/main_model.py

import numpy as np
import os
import sys

# 이 부분에 자기 자신을 import 하는 라인이 없어야 합니다.

from bam_compressor.models.feature_extractor import FeatureExtractor
from bam_compressor.models.compression_model import SupervisedLayer


class MFBAM:
    def __init__(self, mf_layer_dims: list[int], latent_dim: int, delta: float = 0.2):
        if len(mf_layer_dims) < 2:
            raise ValueError("mf_layer_dims는 최소 [input_dim, hidden_dim] 2개 이상이어야 합니다.")
            
        feature_dim = mf_layer_dims[-1]
        self.feature_extractor = FeatureExtractor(layer_dims=mf_layer_dims, delta=delta)
        self.supervised_layer = SupervisedLayer(feature_dim=feature_dim, target_dim=latent_dim, delta=delta)

    def encode(self, p: np.ndarray) -> np.ndarray:
        extracted_features = self.feature_extractor.predict(p)
        a = self.supervised_layer.W @ extracted_features.T
        latent_vector = self.supervised_layer.transmission_function(a).T
        return latent_vector

    def decode(self, latent_vector: np.ndarray) -> np.ndarray:
        b = self.supervised_layer.V @ latent_vector.T
        reconstructed_features = self.supervised_layer.transmission_function(b).T
        
        reconstructed_input = reconstructed_features
        for layer in reversed(self.feature_extractor.layers):
            b_c = layer.V @ reconstructed_input.T
            reconstructed_input = layer.transmission_function(b_c).T
        return reconstructed_input

    def save_weights(self, path='data/weights'):
        os.makedirs(path, exist_ok=True)
        for i, layer in enumerate(self.feature_extractor.layers):
            np.savez(os.path.join(path, f'mf_layer_{i}_weights.npz'), W=layer.W, V=layer.V)
        np.savez(os.path.join(path, 'sl_weights.npz'), W=self.supervised_layer.W, V=self.supervised_layer.V)
        print(f"모든 가중치가 '{path}' 디렉토리에 저장되었습니다.")

    def load_weights(self, path='data/weights'):
        try:
            for i, layer in enumerate(self.feature_extractor.layers):
                data = np.load(os.path.join(path, f'mf_layer_{i}_weights.npz'))
                layer.W = data['W']
                layer.V = data['V']
            data = np.load(os.path.join(path, 'sl_weights.npz'))
            self.supervised_layer.W = data['W']
            self.supervised_layer.V = data['V']
            print(f"모든 가중치를 '{path}' 디렉토리에서 불러왔습니다.")
        except FileNotFoundError:
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: '{path}'. 모델을 먼저 학습시켜주세요.")
