
# SMT_Machine_InspectionCNN

SMT(표면실장) 머신에서 촬영한 PCB 납땜부 이미지를 이용해 **정상/불량(2-class)** 를 분류하는 CNN 프로젝트입니다.  
Base 모델은 TorchVision의 **ResNet18**을 사용하고, Stratified K-Fold로 성능을 검증하며 AUC/Accuracy/Confusion Matrix/ROC 등을 저장합니다.

> 참고: 학습에 사용한 **이미지 데이터는 비공개**입니다. 

---

## Features
- ResNet18(Pretrained) 기반 이진 분류
- Stratified K-Fold Cross Validation (n_splits=5)
- GridSearch 결과값을 (`Dataset/ParameterSettings.txt`)로 export하고 학습에 활용
- Fold별 best 모델 ~.pt 파일로 저장(최고 AUC에 따라 갱신하여 저장, 동률 시 Accuracy 기반 우선반영)
- 전체 ROC/Confusion Matrix/Accuracy 결과 이미지 저장
- 학습 로그 visualization 개선 (tqdm library 활용)
- 이미지 원본 폴더(예: `BOARD4_0.124`)에서 cropcode.py를 통해 전처리 → `Dataset/Output/` 생성)
- checkpoint pt 모델을 통해 Inputdata 이미지에서 defect여부를 추론 (inference.py)

## Hardware Specification
- Render hardware : ASUS G14 (2021) R9 5900HS, RTX3050Ti(Cuda yes)

- SMT Machine Hardware : Opulo based SMT Machine
  - Mainboard : BTT Octopus V1.1 (STM32F446)
  - TOP Camera : SC200AI 2MP sensor with '6mm' lens USB Camera
  - BOTTOM Camera : SC200AI 2MP sensor with '3mm' lens USB Camera
  - Toolhead : SMT Mountor 28mm Nema 11 cp40 holder with N08 nozzle

---

## Directory
```text
Dataset/
 ├─ Output/                 # 전처리된 이미지(학습 입력)
 ├─ Output_Hue/             # 참고용 Hue 변환 이미지(선택)
 ├─ Inputdata/              # 추론용 입력 이미지 폴더
 ├─ DefectLabel.xlsx        # 라벨 파일 (ImageName, Defect)
 └─ ParameterSettings.txt   # JSON(배치/러닝레이트/옵티마이저)

CNN3.py                      # 학습(with 교차검증) 스크립트
inferencecode.py            # 추론 스크립트
cropcode.py                 # 원본 폴더 일괄 전처리(회전/센터크롭)
```

---

## Requirements
- Python 3.10.x 권장 (3.10.11 사용)


코드 실행 전 준비사항 ( venv 환경 추천 )
```bash
# CUDA 사용 시(예: CUDA 12.1 휠)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 공통 패키지
pip install opencv-python pandas matplotlib scikit-learn tqdm openpyxl
```

> CPU만 사용하는 경우에는 아래만 설치
```bash
pip install torch torchvision torchaudio
```

---

## 전처리(Preprocessing)
1) `Dataset/` 아래에 원본 폴더를 배치합니다. 예: `Dataset/BOARD4_0.124/` 내부에 PNG 이미지들을 위치합니다. format형식은 다음과 같습니다. BOARD4 -> 4번 PCB , _0.124 -> OpenPnP에서 fiducial mark offseting 이후 표기되는 Board Rotation 값 입력 (이미지가 회전되어 캡쳐되기 때문에 회전값을 정의해야함)
  
2) 전처리 실행:
```bash
python cropcode.py
```
3) 실행 결과
- `Dataset/Output/`, `Dataset/Output_Hue/` 생성  
- `Dataset/DefectLabel.xlsx` 자동 생성(기본 `Defect=0`)  

필요하면 `DefectLabel.xlsx`의 `Defect` 컬럼을 수동 수정하세요. (`0=정상`, `1=불량`)

---

## Hyperparameter Settings
학습 스크립트(`CNN.py`)는 **최적 파라미터를 `Dataset/ParameterSettings.txt`에서 로드**합니다.  
파일이 없으면 에러가 발생하므로 먼저 준비하세요.

예시(JSON):
```json
{
  "batch_size": 32,
  "learning_rate": 0.001,
  "optimizer": "adam"
}
```

경로: `Dataset/ParameterSettings.txt`  
> 추후 GridSearch 자동화 스크립트를 추가해 본 파일을 자동 생성/갱신할 예정입니다.

---

## Training
```bash
python CNN.py
# (파일명이 다르면 해당 파일명으로 실행)
```

생성물:
- 로그 루트: `./log_YYYYMMDD_HHMMSS/`
  - `fold_1/ ... fold_5/` : 각 폴드별 best 모델(`best_model.pt`)
  - `best_model.pt` : 전체 폴드 중 최고 AUC(동률 시 Acc) 모델
  - `overall_roc_curve.png`, `overall_confusion_matrix.png`

터미널에 fold/epoch별 Acc, AUC, 모델 저장 시점이 출력됩니다.

---

## Inference
1) 사용할 모델을 한 곳에 둡니다(예: `./Dataset/best_model.pt`).  
2) `Dataset/Inputdata/`에 추론할 PNG 이미지를 넣습니다.  
3) 실행:
```bash
python inferencecode.py
```

결과:
- `./Dataset/inference_output_YYYYMMDD_HHMMSS/PredictionResults.xlsx`  
  - 컬럼: `ImageName`, `DefectProb`(불량 확률), `PredictedLabel`(0/1)

결과에서 DefectProb Threshold에 따른 PredictedLabel이 정해집니다. 해당 값 변경이 필요한 경우, inferencecode.py 에서 # 이미지 추론 구문의  "pred = 1 if prob > 0.5 else 0" 값을 조절하세요.