import os
import re
import cv2
import torch
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import datetime
from torch import nn

# =========================
# 설정
# =========================
# - input_dir: 추론할 이미지 폴더
# - model_path: 학습 시 저장한 best 모델 경로(예: .../CNNlog_YYYYMMDD_HHMMSS/best_model.pt)
# - output_dir: 추론 결과 저장 폴더
input_dir = "./Dataset/Imageset/Output_Test"
model_path = "./Dataset/model.pt"  # 필요시 실제 model.pt 경로로 교체
output_dir = f"./Dataset/inferenceResult_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 디바이스 설정
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# 이미지 전처리 정의 (학습/검증과 동일)
# =========================
# 학습/검증 코드의 transform:
#   ToPILImage() -> ToTensor() -> Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# 리사이즈는 학습 코드에 없었으므로 여기서도 하지 않음.
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# 모델 로드 (학습 시와 동일 아키텍처)
# =========================
# 학습 코드에서 resnet18(weights=ResNet18_Weights.DEFAULT) 후 fc만 교체했음.
# conv1는 원래와 동일 파라미터(7x7,stride=2,padding=3)로 재정의했지만 값상 동일 구조이므로
# 여기서는 기본 resnet18(weights=None) 생성 후 fc 교체 -> state_dict 로드로 충분.
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# state_dict 로드 (device 매핑 일치)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)

model.to(device)
model.eval()  # 추론 모드 고정(BN/Dropout 고정)

# =========================
# 정렬 함수 정의 (파일명 정렬 규칙 그대로 사용)
# =========================
def split_filename_parts(filename: str):
    """
    파일명을 (board_num, prefix, number, is_defect) 형태로 분리
    - 예: BOARD6_C118.png, BOARD6_R10_D.png 등
    """
    board_match = re.search(r"BOARD(\d+)", filename)
    board_num = int(board_match.group(1)) if board_match else 999

    is_defect = filename.endswith("_D.png")

    comp_match = re.search(r"_([A-Za-z]+)(\d+)(?:_D)?\.png", filename)
    if comp_match:
        prefix = comp_match.group(1)
        number = int(comp_match.group(2))
    else:
        prefix = ""
        number = 10**12

    return board_num, prefix, number, is_defect

def prefix_rank(prefix: str) -> int:
    """
    접두사 정렬 순위: C -> FID -> R -> 그 외 알파벳순
    """
    PREFIX_ORDER = {"C": 0, "FID": 1, "R": 2}
    if prefix in PREFIX_ORDER:
        return PREFIX_ORDER[prefix]
    return 100 + ord(prefix[0]) if prefix else 999

# =========================
# 이미지 추론
# =========================
predictions = []

# 파일명 정렬(선택): 기존 규칙을 적용하려면 아래처럼 키 함수를 사용
def sort_key(name: str):
    b, p, n, d = split_filename_parts(name)
    return (b, prefix_rank(p), p, n, d)

file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
file_list.sort(key=sort_key)

with torch.no_grad():  # 전체 추론 구간에서 gradient 비활성화
    for fname in file_list:
        img_path = os.path.join(input_dir, fname)

        # cv2는 BGR로 읽기 때문에 RGB로 변환 (학습/검증 파이프라인과 동일)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # 파일 누락/손상 방지 로그
            print(f"[경고] 이미지를 읽지 못했습니다: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 동일 transform 적용
        x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

        # 추론
        logits = model(x)
        prob_defect = torch.softmax(logits, dim=1)[0, 1].item()  # 클래스1(Defect) 확률
        pred = 1 if prob_defect >= 0.5 else 0

        predictions.append({
            "ImageName": fname,
            "DefectProb": prob_defect,
            "PredictedLabel": pred
        })

# =========================
# 결과 저장 (ExcelSorter 폴백 포함)
# =========================
result_df = pd.DataFrame(predictions)

# 정렬된 결과 엑셀 저장: ExcelSorter 모듈이 있으면 사용, 없으면 기본 저장
result_path = os.path.join(output_dir, "PredictionResults.xlsx")
try:
    import ExcelSorter
    # ExcelSorter.sort_excel(df, sort_column, save_path)
    ExcelSorter.sort_excel(result_df, "ImageName", result_path)
except Exception as e:
    # 폴백: 기본 정렬 후 저장
    print(f"[알림] ExcelSorter 사용 불가로 기본 저장 수행: {e}")
    result_df.sort_values(by="ImageName", inplace=True)
    # openpyxl 엔진 사용을 위해 의존성 필요(openpyxl은 학습 코드에서 이미 설치)
    result_df.to_excel(result_path, index=False)

print(f"추론 완료. 결과 저장됨: {result_path}")
