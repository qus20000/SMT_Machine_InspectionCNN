
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
- 이미지 전처리 및 데이터셋 생성 기능 개선:
  - 이미지 원본 폴더(예: `BOARD4_0.124`)에서 cropcode.py를 통해 전처리 → `Dataset/Output/` 생성
  - TEST 포함된 폴더는 자동으로 테스트셋으로 분류
  - 원본 폴더명에 따른 자동 라벨링:
    - `BOARD숫자_각도값` → 정상 데이터
    - `BOARD숫자_각도값_SHIFT/SLIDING 등` → 불량 데이터
    - `BOARD숫자_각도값_TEST` → 테스트 데이터
- Excel 파일 정리 기능 추가:
  - 소자 타입별 정렬 (C -> FID -> R -> 기타 알파벳순)
  - 보드 번호와 소자 번호 기준 정렬
  - 자동 덮어쓰기 확인
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
 ├─ Imageset/              # 추론용 입력 이미지 폴더 (해당 폴더에 BOARD1_-0.O24 와 같은 폴더명 포맷으로 추가)
 ├─ DefectLabel.xlsx        # 라벨 파일 (ImageName, Defect)
 └─ ParameterSettings.txt   # JSON(배치/러닝레이트/옵티마이저)

CNN3.py                      # 학습(with 교차검증) 스크립트
inferencecode.py            # 추론 스크립트
cropcode.py                 # 원본 폴더 일괄 전처리(회전/센터크롭)
ExcelSorter.py              # 각 스크립트에서 엑셀파일 생성시 호출되어 파일명 정렬처리
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
1) `Dataset/Imageset` 아래에 학습 및 추론을 진행할 원본 폴더를 배치합니다. 예: `Dataset/BOARD4_0.124/` 내부에 PNG 이미지들을 위치합니다. format형식은 다음과 같습니다. BOARD4 -> 4번 PCB , _0.124 -> OpenPnP에서 fiducial mark offseting 이후 표기되는 Board Rotation 값 입력 (이미지가 회전되어 캡쳐되기 때문에 회전값을 정의해야함)

 Defect만 존재하는 이미지를 전처리할 경우, 폴더명을 `BOARD4_0.124_NOCOMPONENTS` 와 같이 마지막에 언더바와 함께 DEFECT 타입을 지정해주세요.(DEFECT TYPE은 따로 정해진 포맷이 없으며, Cropcode에서 단순히 언더바가 한개 더 존재하는지만을 파악합니다. 이렇게 전처리 데이터를 준비하시면 DefectLabel.xlsx 파일에 Defect여부가 1로 자동할당 됩니다. 필요시 수동으로 수정해주시기 바랍니다.) 
  
2) 전처리 실행:
```bash
python cropcode.py
```
3) 실행 결과
- `Dataset/Imageset/Output/`, `Dataset/Imageset/Output_Hue/` 에 크롭된 이미지 생성  
- `Dataset/Imageset/DefectLabel.xlsx` 자동 생성(기본 `Defect=0`)  


테스트 데이터가 있는 경우 별도의 `DefectLabel_Test.xlsx`가 생성됩니다.
필요한 경우 `DefectLabel.xlsx`와 `DefectLabel_Test.xlsx`의 `Defect` 컬럼을 수동으로 수정할 수 있습니다. (`0=정상`, `1=불량`)

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

## Training 실행
```bash
python CNN3.py
```

생성물:
- 로그 루트: `./Dataset/CNNlog_YYYYMMDD_HHMMSS/`
  - `fold_1/ ... fold_5/` : 각 폴드별 best 모델(`best_model.pt`)
  - `best_model.pt` : 전체 폴드 중 최고 AUC(동률 시 Acc) 모델
  - `overall_roc_curve.png`, `overall_confusion_matrix.png`

터미널에 fold/epoch별 Acc, AUC, 모델 저장 시점이 출력됩니다.

---

## Inference
1) 사용할 모델을 정해진 곳에 둡니다(예: `./Dataset/best_model.pt`).  
2) Cropcode.py를 실행했을 때 Test 데이터를 포함한 경우, `./Dataset/Imageset/Output_Test` 폴더가 생성되며, 해당 폴더를 자동으로 인식합니다. 존재하지 않는 경우, 
3) 실행:
```bash
python inferencecode.py
```

결과:
- `./Dataset/inference_output_YYYYMMDD_HHMMSS/PredictionResults.xlsx`  
  - 컬럼: `ImageName`, `DefectProb`(불량 확률), `PredictedLabel`(0/1)

결과에서 DefectProb Threshold에 따른 PredictedLabel이 정해집니다. 해당 값 변경이 필요한 경우, inferencecode.py 에서 # 이미지 추론 구문의  "pred = 1 if prob > 0.5 else 0" 값을 조절하세요.