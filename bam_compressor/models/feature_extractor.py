# bam_compressor/models/lightweight_feature_extractor.py
import numpy as np

class UnsupervisedLayer:
    """
    NumPy 기반으로 구현된 Unsupervised Layer (UL/FEBAM).
    """
    def __init__(self, input_dim: int, hidden_dim: int, delta: float = 0.2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.delta = delta
        self.reset_parameters()

    def reset_parameters(self):
        """ -0.1과 0.1 사이의 균등 분포로 가중치를 초기화합니다. """
        self.W = np.random.uniform(-0.1, 0.1, size=(self.hidden_dim, self.input_dim))
        self.V = np.random.uniform(-0.1, 0.1, size=(self.input_dim, self.hidden_dim))

    def transmission_function(self, activation: np.ndarray) -> np.ndarray:
        """ NumPy를 사용한 입방(cubic) 전달 함수. """
        term1 = (self.delta + 1) * activation
        term2 = self.delta * np.power(activation, 3)
        output = term1 - term2
        return np.clip(output, a_min=-1.0, a_max=1.0)

    def forward(self, x_0: np.ndarray, num_cycles: int = 1) -> tuple:
        """ 정보 순환 과정을 수행하고 학습에 필요한 변수들을 반환합니다. """
        a_0 = self.W @ x_0.T
        y_0 = self.transmission_function(a_0).T
        
        y_c = y_0
        for _ in range(num_cycles):
            b_c = self.V @ y_c.T
            x_c = self.transmission_function(b_c).T
            
            a_c = self.W @ x_c.T
            y_c = self.transmission_function(a_c).T

        return x_0, y_0, x_c, y_c
    
    def update_weights(self, x_0, y_0, x_c, y_c, learning_rate):
        """ Hebbian/anti-Hebbian 학습 규칙 (Eq. 3)에 따라 가중치를 업데이트합니다. """
        # (y[0] - y[c])와 (x[0] + x[c])는 행 벡터. 외적(outer product)을 위해 형태 변환 필요.
        delta_W = (y_0 - y_c).T @ (x_0 + x_c)
        delta_V = (x_0 - x_c).T @ (y_0 + y_c)
        
        self.W += learning_rate * delta_W
        self.V += learning_rate * delta_V

class FeatureExtractor:
    """ NumPy 기반으로 구현된 Multi-Feature extraction (MF) 모듈. """
    def __init__(self, layer_dims: list[int], delta: float = 0.2):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                UnsupervisedLayer(layer_dims[i], layer_dims[i+1], delta)
            )

    def predict(self, p: np.ndarray, num_cycles: int = 1) -> np.ndarray:
        """ (추론) 입력 p를 받아 최종 특성 패턴 h_l을 생성합니다. """
        h = p
        for layer in self.layers:
            # 추론 시에는 학습과 달리 y_c만 다음 입력으로 넘겨주면 됨
            _, _, _, h = layer.forward(h, num_cycles)
        return h

# --- 사용 예시 ---
if __name__ == '__main__':
    # 설정
    layer_dims = [10, 30, 50, 20]  # 입력 10 -> UL1(30) -> UL2(50) -> UL3(20)
    
    print(f"경량화된 특성 추출 모듈(MF) 생성: {' -> '.join(map(str, layer_dims))}")
    feature_extractor = FeatureExtractor(layer_dims=layer_dims)
    
    # 임의의 입력 데이터 및 학습 설정
    input_data = np.random.randn(1, layer_dims[0])
    learning_rate = 1e-4  # 학습률 설정

    print(f"\n입력 데이터 (p) shape: {input_data.shape}")

    # --- 계층별 순차 학습 시뮬레이션 ---
    print("\n--- 계층별 학습 과정 (시뮬레이션) ---")
    current_input = input_data
    for i, layer in enumerate(feature_extractor.layers):
        print(f"\nLayer {i+1} 학습 중...")
        # 해당 계층에 대한 학습 변수 계산
        x0, y0, xc, yc = layer.forward(current_input)
        
        # 가중치 업데이트
        layer.update_weights(x0, y0, xc, yc, learning_rate)
        
        # 다음 계층의 입력은 현재 계층의 출력이 됨
        current_input = yc
        print(f"Layer {i+1}의 출력 shape (다음 Layer의 입력): {current_input.shape}")

    # --- 추론 시뮬레이션 ---
    print("\n\n--- 전체 모듈 추론 과정 ---")
    final_features = feature_extractor.predict(input_data)
    print(f"최종 특성맵 (h_l) shape: {final_features.shape}")