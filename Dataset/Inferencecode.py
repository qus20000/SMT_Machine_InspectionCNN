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
input_dir = "./Dataset/Inputdata"  # 입력 이미지 폴더
model_path = "./Dataset/best_model.pt"  # 모델 경로
output_dir = f"./Dataset/inference_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
# 결과 저장
# =========================
result_df = pd.DataFrame(predictions)
result_path = os.path.join(output_dir, "PredictionResults.xlsx")
result_df.to_excel(result_path, index=False)

print(f"✅ 추론 완료. 결과 저장됨: {result_path}")
