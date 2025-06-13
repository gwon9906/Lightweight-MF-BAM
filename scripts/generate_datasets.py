# scripts/generate_datasets.py

import numpy as np
import os
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def generate_and_save_dataset():
    """
    논문에 나온 비선형 데이터셋을 생성하고 CSV 파일로 저장합니다.
    - Moons, Eclipse(Circles) 등
    """
    # 데이터셋 저장 경로 생성
    data_path = 'data/original'
    os.makedirs(data_path, exist_ok=True)

    # 1. Moons 데이터셋 생성 (논문 Section 4.2 참조)
    # n_samples=40으로 되어 있지만, 더 풍부한 데이터셋을 위해 200개 생성
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    # 타겟 값을 논문처럼 -1과 1로 변환
    y_moons = np.where(y_moons == 0, -1.0, 1.0).reshape(-1, 1)
    
    # 입력 값을 -1과 1 사이로 스케일링
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_moons_scaled = scaler.fit_transform(X_moons)

    # DataFrame으로 변환 후 저장
    moons_df = pd.DataFrame(np.hstack([X_moons_scaled, y_moons]), columns=['feature1', 'feature2', 'target'])
    moons_filepath = os.path.join(data_path, 'moons_dataset.csv')
    moons_df.to_csv(moons_filepath, index=False)
    print(f"Moons 데이터셋 저장 완료: {moons_filepath}")

    # (추가) Eclipse(Circles) 데이터셋도 생성 가능
    # X_circles, y_circles = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
    # ... 동일한 저장 로직 ...

if __name__ == '__main__':
    generate_and_save_dataset()