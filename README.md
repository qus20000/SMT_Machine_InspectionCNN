
# SMT_Machine_InspectionCNN

SMT(표면실장) 머신에서 촬영한 PCB 납땜부 이미지를 이용해 **정상/불량(2-class)** 를 분류하는 CNN 프로젝트입니다.  
Base 모델은 TorchVision의 **ResNet18**을 사용하고, Stratified K-Fold로 성능을 검증하며 AUC/Accuracy/Confusion Matrix/ROC 등을 저장합니다.

> 참고: 학습에 사용한 **이미지 데이터는 비공개**입니다. (OpenPnP로 캡처한 PNG 이미지, USB 카메라 + 6mm 렌즈)

---

## Features
- ResNet18(Pretrained) 기반 이진 분류
- Stratified K-Fold Cross Validation (n_splits=5)
- GridSearch 결과(`Dataset/ParameterSettings.txt`) 재사용
- Fold별 best 모델 저장(최고 AUC, 동률 시 Accuracy 우선)
- 전체 ROC/Confusion Matrix/Accuracy 저장
- 학습 로그 및 진행바(tqdm)
- 전처리 스크립트로 원본 폴더(예: `BOARD4_0.124`)에서 자동 크롭 → `Dataset/Output/` 생성

---

## Directory
```text
Dataset/
 ├─ Output/                 # 전처리된 이미지(학습 입력)
 ├─ Output_Hue/             # 참고용 Hue 변환 이미지(선택)
 ├─ Inputdata/              # 추론용 입력 이미지 폴더
 ├─ DefectLabel.xlsx        # 라벨 파일 (ImageName, Defect)
 └─ ParameterSettings.txt   # JSON(배치/러닝레이트/옵티마이저)

CNN.py                      # 학습(교차검증) 스크립트
inferencecode.py            # 추론 스크립트
cropcode.py                 # 원본 폴더 일괄 전처리(회전/센터크롭)
```

---

## Requirements
- Python 3.10.x 권장

```bash
# CUDA 사용 시(예: CUDA 12.1 휠)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 공통 패키지
pip install opencv-python pandas matplotlib scikit-learn tqdm openpyxl
```

> CPU만 사용하는 경우에는 아래만 설치해도 됩니다.
```bash
pip install torch torchvision torchaudio


---

## 전처리(Preprocessing)
1) `Dataset/` 아래에 원본 폴더를 배치합니다. 예: `Dataset/BOARD4_0.124/` 내부에 PNG 이미지들  
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

