# MF-BAM: 경량화된 비선형 과제 학습용 양방향 연상 메모리
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

이 프로젝트는 "A multilayered bidirectional associative memory model for learning nonlinear tasks" (Rolon-Mérette et al., 2023) 논문에서 제안된 **MF-BAM (Multi-Feature extraction Bidirectional Associative Memory)** 모델을 NumPy 기반으로 가볍게 구현한 것입니다.

PyTorch, TensorFlow와 같은 무거운 딥러닝 프레임워크 없이, 순수 Python과 NumPy만으로 구현하여 **저사양 환경으로의 이식성**을 염두에 두고 설계되었습니다.


## 🌟 주요 응용 사례: LoRa 통신 성능 개선 프로젝트

이 MF-BAM 모델은 **불안정한 저전력 통신 환경(LLN)에서 데이터 전송률을 높이는 프로젝트**에 성공적으로 적용되었습니다. 경량 압축기로서의 성능은 아래와 같습니다.
(압축률: 32B -> 20B 62.5%)
| 구분 | 값 | 비고 |
|---|---|---|
| **복원 오차 (MSE)** | **0.003619** | 원본 정보 손실 최소화 |
| **압축 비율** | **1.5 : 1** | 12 피처 → 8 잠재 벡터 |

<br>
<p align="center">
  <br>
</p>

> **➡️ [LoRa 통신 개선 프로젝트 전체 내용 보기 (GitHub: 4xvgal/ChirpChirp)](https://github.com/4xvgal/ChirpChirp)**
> 
> **➡️ [LoRa_통신개선_프로젝트_최종보고서.pdf](https://github.com/user-attachments/files/21096150/LoRa_._._.pdf)**  <!-- "https://..." 부분을 님의 PDF 파일 링크 URL로 교체하세요 -->
> <br>
> (실제 필드 테스트 사진, 하드웨어 구성 등은 위 GitHub 링크 또는 보고서에서 확인하실 수 있습니다.)

---


*실제 통신 데이터에 대한 압축 -> 복원 결과
![Final bset](https://github.com/user-attachments/assets/4cc89d83-8910-4856-9a12-08cc1973a969)


## 📈 데이터 압축·복원 성능 요약

### ✅ 전체 테스트셋 복원 오차  
- **Mean Squared Error (MSE)**: **0.003619**  


### 📦 압축 비율  
- **입력 피처**: 12 개  
- **잠재 벡터**: 8 개  
- **Compression ratio**: **1.5 : 1**

---

### 🔍 단일 샘플 상세 비교

#### 전송될 잠재 벡터  
`[0.0168, 0.1664, -0.6171, 0.1189, -0.4530, 0.8912, -0.1595, -0.2937]`

| Feature | Original | Reconstructed | |Absolute&nbsp;Error|
|---|---:|---:|---|---:|
| decoded_accel_ax | **0.9130** | **0.8961** | ✔️ | 0.0169 |
| decoded_accel_ay | -0.2010 | -0.1818 | ✔️ | 0.0192 |
| decoded_accel_az | 0.4000 | 0.3997 | ✔️ | 0.0003 |
| decoded_gyro_gx | 0.0000 | -0.6075 | ✔️ | 0.6075 |
| decoded_gyro_gy | 0.0000 | -1.0846 | ⚠️ | 1.0846 |
| decoded_gyro_gz | 0.0000 | 1.3382 | ⚠️ | 1.3382 |
| decoded_angle_roll | -26.7000 | -25.7422 | ✔️ | 0.9578 |
| decoded_angle_pitch | -63.8000 | -62.7246 | ⚠️ | 1.0754 |
| decoded_angle_yaw | -29.8000 | -31.1870 | ⚠️ | 1.3870 |
| decoded_gps_lat | 0.1477 | 0.1416 | ✔️ | 0.0061 |
| decoded_gps_lon | 0.0269 | 0.0296 | ✔️ | 0.0027 |
| decoded_gps_altitude | 84.7000 | 77.2924 | ⚠️ | 7.4076 |

> **참고**  
> - ✔️ : 절대 오차 ≤ 1.0 (허용 범위)  
> - ⚠️ : 절대 오차 > 1.0 (추가 개선 필요)

---

### ✨ 인사이트
- **평균 MSE 0.0036** 으로, 전체적으로 우수한 복원 성능을 유지하고 있습니다.    
- 가속도 및 GPS 위치 값은 거의 완벽하게 복원되어, **위치·가속 정보 전송 신뢰도**가 높음을 확인했습니다.


## ✨ 주요 특징

- **🧠 인지과학 기반 아키텍처**: 인간의 뇌가 정보를 처리하는 방식에 영감을 받은 2단계 모듈(비지도 특성 추출 + 지도 연관 학습) 구조를 가집니다.
- **⚙️ 프레임워크 독립**: 외부 딥러닝 라이브러리 의존성 없이 NumPy로만 핵심 로직이 구현되어 있어 매우 가볍고 이해하기 쉽습니다.
- **🚀 2단계 학습 파이프라인**:
  1. **비지도 특성 추출**: 입력 데이터의 본질적인 특성을 점진적으로 학습합니다.
  2. **지도 연관 학습**: 추출된 특성과 목표 레이블을 양방향으로 연결합니다.
- **↔️ 양방향 메모리**:
  - **순방향 (x → y')**: 입력에 대한 분류(Classification)를 수행합니다.
  - **역방향 (y → x')**: 특정 클래스의 전형적인(prototypical) 입력 패턴을 복원/연상합니다.
  - **오토인코더형 압축·복원**: BAM의 양방향 구조(순방향 압축 / 역방향 복원)를 이용해 오토인코더와 같은 압축·복원 기능을 수행합니다. 오토인코더를 그대로 사용한 것이 아니라, BAM 구조로 동일한 기능을 구현한 것입니다.

## 🏛️ 프로젝트 아키텍처

모델은 두 가지 주요 모듈이 순차적으로 연결된 구조입니다.

**`Original Data (p)`** ➡️ **`[Feature Extractor (MF) Module]`** ➡️ **`Features (h_l)`** ➡️ **`[Supervised Layer (SL) Module]`** ➡️ **`Target (o)`**

1. **Feature Extractor (MF) Module**: 여러 개의 `UnsupervisedLayer` (UL)가 쌓여있는 구조입니다. 각 UL은 입력의 재구성을 목표로 하는 비지도 학습을 통해 데이터의 점진적이고 깊은 특성을 추출합니다.
2. **Supervised Layer (SL) Module**: MF 모듈이 추출한 최종 특성(`h_l`)과 실제 정답 레이블(`o`)을 입력으로 받아, 둘 사이의 연관 관계를 학습하는 양방향 연상 메모리(BAM)입니다.

## 📁 폴더 구조

```
BAM_COMPRESSOR/
├── .venv/                  # 가상 환경
├── bam_compressor/         # 메인 소스 패키지
│   ├── data/
│   │   └── data_loader.py  # NumPy 기반 경량 데이터 로더
│   ├── models/
│   │   ├── feature_extractor.py # Feature Extractor (MF) 모듈
│   │   ├── compression_model.py # Supervised Layer (SL) 모듈
│   │   └── main_model.py        # 전체 MFBAM 아키텍처 통합
│   └── training/
│       └── trainer.py      # 2단계 학습 파이프라인 관리
├── data/
│   ├── original/           # 원본 데이터셋 (e.g., .csv)
│   └── weights/            # 학습된 모델 가중치 저장 위치
├── scripts/
│   └── generate_datasets.py # 예제 데이터셋 생성 스크립트
├── run_experiment.py       # 전체 학습 및 평가 실행 스크립트
├── requirements.txt        # 필요한 라이브러리 목록
└── README.md               # 프로젝트 설명 (이 파일)
```

## 🛠️ 설치 및 설정

1. **저장소 복제:**
   ```bash
   git clone https://your-repository-url/BAM_COMPRESSOR.git
   cd BAM_COMPRESSOR
   ```

2. **가상 환경 생성 및 활성화:**
   ```bash
   # 가상 환경 생성
   python -m venv .venv

   # 가상 환경 활성화 (macOS/Linux)
   source .venv/bin/activate

   # 가상 환경 활성화 (Windows)
   .\.venv\Scripts\activate
   ```

3. **필요한 라이브러리 설치:**
   `requirements.txt` 파일에 아래 내용을 작성하세요.
   ```
   numpy
   pandas
   scikit-learn
   matplotlib
   ```
   그리고 아래 명령어로 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 사용 방법

1. **(최초 1회) 예제 데이터셋 생성:**
   `run_experiment.py`를 실행하기 위해 필요한 `moons_dataset.csv` 파일을 생성합니다.
   ```bash
   python scripts/generate_datasets.py
   ```

2. **모델 학습 및 평가 실행:**
   전체 학습 파이프라인(MF 모듈 학습 → SL 모듈 학습)을 실행하고, 테스트 데이터에 대한 정확도 평가 및 결과 시각화를 수행합니다.
   ```bash
   python run_experiment.py
   ```
   학습이 완료되면, 테스트 정확도와 함께 결정 경계 및 역방향 복원 패턴을 보여주는 `matplotlib` 그래프 창이 나타납니다. 학습된 가중치는 `data/weights/` 폴더에 자동으로 저장됩니다.

## 🔮 향후 계획

- [ ] **하이퍼 파라미터 조정**: 경량화를 위한 적절한 하이퍼 파라미터 탐색 
- [ ] **C/C++ 포팅**: 학습된 가중치를 사용하여 임베디드 시스템용 추론 엔진 구현
- [ ] **모델 양자화(Quantization)**: `float32` 가중치를 `int8`로 변환하여 메모리 사용량 및 연산 속도 최적화
- [ ] **설정 파일 도입**: 하이퍼파라미터를 `.py` 파일이 아닌 `YAML` 파일로 관리하여 실험 용이성 증대

## 📜 참고 문헌

> Rolon-Mérette, D., Rolon-Mérette, T., & Chartier, S. (2023). A multilayered bidirectional associative memory model for learning nonlinear tasks. *Neural Networks, 167*, 244-265.
