import os
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
input_dir = "./Dataset/Imageset/Output_Test"  # 입력 이미지 폴더
model_path = "./Dataset/best_model.pt"  # 모델 경로
output_dir = f"./Dataset/inferenceResult_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 디바이스 설정
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# 이미지 전처리 정의
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =========================
# 모델 로드
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# =========================
# 정렬 함수 정의
# =========================
import re

def split_filename_parts(filename: str):
    """
    파일명을 (board_num, prefix, number, is_defect) 형태로 분리
    """
    # BOARD 번호 추출
    board_match = re.search(r"BOARD(\d+)", filename)
    board_num = int(board_match.group(1)) if board_match else 999
    
    # Defect 여부 확인
    is_defect = filename.endswith("_D.png")
    
    # 소자 타입과 번호 추출
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
    접두사 정렬 순위 반환: C -> FID -> R -> 알파벳순
    """
    PREFIX_ORDER = {"C": 0, "FID": 1, "R": 2}
    if prefix in PREFIX_ORDER:
        return PREFIX_ORDER[prefix]
    return 100 + ord(prefix[0]) if prefix else 999

# =========================
# 이미지 추론
# =========================
predictions = []

for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_dir, fname)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)[0][1].item()  # Defect일 확률
        pred = 1 if prob > 0.5 else 0

    predictions.append({"ImageName": fname, "DefectProb": prob, "PredictedLabel": pred})

# =========================
# 결과 정렬 및 저장
# =========================
import ExcelSorter

result_df = pd.DataFrame(predictions)
result_path = os.path.join(output_dir, "PredictionResults.xlsx")
ExcelSorter.sort_excel(result_df, "ImageName", result_path)

print(f"추론 완료. 결과 저장됨: {result_path}")
