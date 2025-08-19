import os
import sys
import subprocess

# =========================
# 필요한 패키지 자동 설치
# =========================
def install_packages(packages):
    for package in packages:
        try:
            if package in ["torch", "torchvision"]:
                continue  # PyTorch는 별도로 설치
            __import__(package.split("-")[0])
        except ImportError:
            print(f"'{package}' 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_pytorch():
    try:
        import torch
        import torchvision
        if torch.cuda.is_available():
            print("CUDA 가능한 PyTorch가 이미 설치되어 있습니다.")
            return
    except ImportError:
        pass
    
    print("CUDA 가능한 PyTorch 설치 중...")
    try:
        # 기존 설치 제거
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        
        # PyTorch 설치
        print("torch 설치 중...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
        # torchvision 설치
        print("torchvision 설치 중...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
        # torchaudio 설치
        print("torchaudio 설치 중...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        
        print("PyTorch CUDA 설치 완료!")
        
    except subprocess.CalledProcessError as e:
        print(f"PyTorch 설치 중 오류 발생: {e}")
        print("수동으로 다음 명령어들을 순서대로 실행해보세요:")
        print(f"{sys.executable} -m pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print(f"{sys.executable} -m pip install torchvision --index-url https://download.pytorch.org/whl/cu118")
        print(f"{sys.executable} -m pip install torchaudio --index-url https://download.pytorch.org/whl/cu118")
        raise

# 필요한 패키지 목록 (PyTorch 제외)
required_packages = [
    "opencv-python",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "tqdm",
    "numpy",
    "openpyxl"
]

# PyTorch CUDA 버전 먼저 설치
install_pytorch()

# 패키지 설치 실행
install_packages(required_packages)

# 패키지 임포트
import cv2
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score

from datetime import datetime
from tqdm import tqdm


# =========================
# 기본 설정
# =========================
dataset_root = "./Dataset"
output_folder = os.path.join(dataset_root, "Output")
label_path = os.path.join(dataset_root, "DefectLabel.xlsx")
df = pd.read_excel(label_path)

# =========================
# 파라미터 설정 로딩 or 튜닝
# =========================
param_file = os.path.join(dataset_root, "ParameterSettings.txt")
param_grid = {
    "batch_size": [16, 32],
    "learning_rate": [1e-3, 1e-4],
    "optimizer": ["adam", "sgd"],
}
num_epochs = 5
num_folds = 5
num_classes = 2

def create_model(learning_rate, optimizer_name):
    # 커스텀 ResNet18 모델 생성
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # 첫 번째 컨볼루션 레이어를 650x270 입력 크기에 맞게 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Average Pooling 레이어 수정 (최종 피쳐맵 크기 조정)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # 마지막 fully connected 레이어 수정
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer")
    return model, optimizer

# =========================
# 커스텀 데이터셋 클래스
# =========================
class SolderDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["ImageName"]
        label = self.df.iloc[idx]["Defect"]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

# =========================
# 이미지 변환
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 표준화 값 사용
])

# =========================
# 디바이스 설정
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# 로그 디렉토리 생성
# =========================
global_log_root = os.path.join(".", f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(global_log_root, exist_ok=True)
print(f"로그 디렉토리 생성됨: {global_log_root}")

# =========================
# ParameterSettings 로드
# =========================
param_file = os.path.join(dataset_root, "ParameterSettings.txt")
if os.path.exists(param_file):
    print(f"📂 기존 ParameterSettings.txt 파일 로드됨: {param_file}")
    with open(param_file, "r") as f:
        best_param = json.load(f)
    batch_size = best_param["batch_size"]
    learning_rate = best_param["learning_rate"]
    optimizer_name = best_param["optimizer"]
    print(f"사용 파라미터: {best_param}")
else:
    raise FileNotFoundError("ParameterSettings.txt 파일이 존재하지 않습니다. 하이퍼파라미터 튜닝부터 먼저 진행하세요.")

# =========================
# 학습 시작
# =========================
print(" Stratified K-Fold 학습 시작")
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

all_true = []
all_probs = []
best_overall_auc = 0.0
final_best_model_path = os.path.join(global_log_root, "best_model.pt")

for fold, (train_idx, val_idx) in enumerate(skf.split(df["ImageName"], df["Defect"])):
    print(f"\n Fold {fold+1}/{num_folds} 시작")
    log_dir = os.path.join(global_log_root, f"fold_{fold+1}")
    os.makedirs(log_dir, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(SolderDataset(train_df, output_folder, transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SolderDataset(val_df, output_folder, transform), batch_size=batch_size)

    model, optimizer = create_model(learning_rate, optimizer_name)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    best_model_path = os.path.join(log_dir, "best_model.pt")

    for epoch in range(num_epochs):
        print(f"[Fold {fold+1}] Epoch {epoch+1}/{num_epochs} 시작")
        model.train()
        total_loss, correct = 0, 0

        for images, labels in tqdm(train_loader, desc=f"[Fold {fold+1}] Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        val_total_loss, val_correct = 0, 0
        fold_true, fold_probs = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                val_total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

                fold_true.extend(labels.cpu().numpy())
                fold_probs.extend(probs.cpu().numpy())

        val_loss = val_total_loss / len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        auc = roc_auc_score(fold_true, fold_probs)

        print(f"  Fold {fold+1} Epoch {epoch+1} ▶ Acc: {val_acc:.4f}, AUC: {auc:.4f}")

        # Fold별 모델 저장
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)
            print(f"    모델 저장됨 (Fold {fold+1}) ▶ {best_model_path}")

        # 전체 모델 저장
        if auc > best_overall_auc:
            best_overall_auc = auc
            torch.save(model.state_dict(), final_best_model_path)
            print(f"    최종 최고 AUC 갱신 ▶ {final_best_model_path}")

    all_true.extend(fold_true)
    all_probs.extend(fold_probs)

# =========================
# 전체 결과 저장
# =========================
print("\n 전체 결과 저장 중...")
global_auc = roc_auc_score(all_true, all_probs)
global_acc = accuracy_score(all_true, [1 if p > 0.5 else 0 for p in all_probs])
global_cm = confusion_matrix(all_true, [1 if p > 0.5 else 0 for p in all_probs])
fpr, tpr, _ = roc_curve(all_true, all_probs)

# 전체 ROC Curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {global_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Overall AUC-ROC")
plt.legend()
plt.savefig(os.path.join(global_log_root, "overall_roc_curve.png"))

# 전체 Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(global_cm, cmap="Blues")
plt.title(f"Overall Confusion Matrix\nResNet18, Acc: {global_acc:.3f}, AUC: {global_auc:.3f}")
plt.colorbar()
plt.xticks([0, 1], ["Normal", "Defect"])
plt.yticks([0, 1], ["Normal", "Defect"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(global_cm[i, j]), ha="center", va="center", color="black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(global_log_root, "overall_confusion_matrix.png"))

print(f"\n 전체 최종 결과 저장 완료: {global_log_root}")
print(f" 최종 모델 저장 위치: {final_best_model_path}")
