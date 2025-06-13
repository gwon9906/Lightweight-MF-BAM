MF-BAM: 경량화된 비선형 과제 학습용 양방향 연상 메모리
이 프로젝트는 "A multilayered bidirectional associative memory model for learning nonlinear tasks" (Rolon-Mérette et al., 2023) 논문에서 제안된 MF-BAM (Multi-Feature extraction Bidirectional Associative Memory) 모델을 NumPy 기반으로 가볍게 구현한 것입니다.

PyTorch, TensorFlow와 같은 무거운 딥러닝 프레임워크 없이, 순수 Python과 NumPy만으로 구현하여 IoT 장치나 저사양 환경으로의 이식성을 염두에 두고 설계되었습니다.

Moons 데이터셋에 대한 학습 후 결정 경계 및 역방향 복원 결과

✨ 주요 특징
🧠 인지과학 기반 아키텍처: 인간의 뇌가 정보를 처리하는 방식에 영감을 받은 2단계 모듈(비지도 특성 추출 + 지도 연관 학습) 구조를 가집니다.
⚙️ 프레임워크 독립: 외부 딥러닝 라이브러리 의존성 없이 NumPy로만 핵심 로직이 구현되어 있어 매우 가볍고 이해하기 쉽습니다.
🚀 2단계 학습 파이프라인:
비지도 특성 추출: 입력 데이터의 본질적인 특성을 점진적으로 학습합니다.
지도 연관 학습: 추출된 특성과 목표 레이블을 양방향으로 연결합니다.
↔️ 양방향 메모리:
순방향 (x → y'): 입력에 대한 분류(Classification)를 수행합니다.
역방향 (y → x'): 특정 클래스의 전형적인(prototypical) 입력 패턴을 복원/연상합니다.
💡 IoT 확장성: 경량 설계 덕분에 향후 C/C++로 포팅하여 마이크로컨트롤러(MCU)와 같은 임베디드 시스템에 배포하기에 용이합니다.
🏛️ 프로젝트 아키텍처
모델은 두 가지 주요 모듈이 순차적으로 연결된 구조입니다.

Original Data (p) ➡️ [Feature Extractor (MF) Module] ➡️ Features (h_l) ➡️ [Supervised Layer (SL) Module] ➡️ Target (o)

Feature Extractor (MF) Module: 여러 개의 UnsupervisedLayer (UL)가 쌓여있는 구조입니다. 각 UL은 입력의 재구성을 목표로 하는 비지도 학습을 통해 데이터의 점진적이고 깊은 특성을 추출합니다.
Supervised Layer (SL) Module: MF 모듈이 추출한 최종 특성(h_l)과 실제 정답 레이블(o)을 입력으로 받아, 둘 사이의 연관 관계를 학습하는 양방향 연상 메모리(BAM)입니다.
📁 폴더 구조
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
Use code with caution.
🛠️ 설치 및 설정
저장소 복제:
git clone https://your-repository-url/BAM_COMPRESSOR.git
cd BAM_COMPRESSOR
Use code with caution.
Bash
가상 환경 생성 및 활성화:
# 가상 환경 생성
python -m venv .venv

# 가상 환경 활성화 (macOS/Linux)
source .venv/bin/activate

# 가상 환경 활성화 (Windows)
.\.venv\Scripts\activate
Use code with caution.
Bash
필요한 라이브러리 설치:
pip install -r requirements.txt
Use code with caution.
Bash
🚀 사용 방법
(최초 1회) 예제 데이터셋 생성:
run_experiment.py를 실행하기 위해 필요한 moons_dataset.csv 파일을 생성합니다.
python scripts/generate_datasets.py
Use code with caution.
Bash
모델 학습 및 평가 실행:
전체 학습 파이프라인(MF 모듈 학습 → SL 모듈 학습)을 실행하고, 테스트 데이터에 대한 정확도 평가 및 결과 시각화를 수행합니다.
python run_experiment.py
Use code with caution.
Bash
학습이 완료되면, 테스트 정확도와 함께 결정 경계 및 역방향 복원 패턴을 보여주는 matplotlib 그래프 창이 나타납니다. 학습된 가중치는 data/weights/ 폴더에 자동으로 저장됩니다.
🔮 향후 계획
C/C++ 포팅: 학습된 가중치를 사용하여 임베디드 시스템용 추론 엔진 구현
모델 양자화(Quantization): float32 가중치를 int8로 변환하여 메모리 사용량 및 연산 속도 최적화
데이터셋 확장: 논문에 나온 N-bit Parity, Suns, Eclipse 등 다른 비선형 과제 추가
설정 파일 도입: 하이퍼파라미터를 .py 파일이 아닌 YAML 파일로 관리하여 실험 용이성 증대
📜 참고 문헌
Rolon-Mérette, D., Rolon-Mérette, T., & Chartier, S. (2023). A multilayered bidirectional associative memory model for learning nonlinear tasks. Neural Networks, 167, 244-265.
📄 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.
