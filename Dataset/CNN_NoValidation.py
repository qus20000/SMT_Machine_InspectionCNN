import os
import sys
import subprocess
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

from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm

# =========================
# 패키지 자동 설치 함수
# =========================
def install_packages(packages):
    for package in packages:
        try:
            if package in ["torch", "torchvision"]:
                continue
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
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
        print("PyTorch CUDA 설치 완료!")
    except subprocess.CalledProcessError as e:
        print(f"PyTorch 설치 중 오류 발생: {e}")
        raise

# =========================
# 기본 설정
# =========================
dataset_root = "./Dataset"
output_folder = os.path.join(dataset_root, "Imageset/Output")
label_path = os.path.join(dataset_root, "Imageset/DefectLabel.xlsx")
df = pd.read_excel(label_path)

# =========================
# 파라미터 로딩
# =========================
param_file = os.path.join(dataset_root, "ParameterSettings.txt")
if os.path.exists(param_file):
    print(f"기존 ParameterSettings.txt 파일 로드됨: {param_file}") 
    with open(param_file, "r") as f:
        best_param = json.load(f)
    batch_size = best_param["batch_size"]
    learning_rate = best_param["learning_rate"]
    optimizer_name = best_param["optimizer"]
    print(f"사용 파라미터: {best_param}")
else:
    raise FileNotFoundError("ParameterSettings.txt 파일이 존재하지 않습니다. 하이퍼파라미터 튜닝부터 먼저 진행하세요.")

num_epochs = 5
num_classes = 2

# =========================
# 디바이스 설정
# =========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# 로그 디렉토리 생성
# =========================
global_log_root = os.path.join(dataset_root, f"CNNlog_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(global_log_root, exist_ok=True)
print(f"로그 디렉토리 생성됨: {global_log_root}")

# =========================
# 모델 생성 함수
# =========================
def create_model(learning_rate, optimizer_name):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
# 커스텀 데이터셋
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

# -------------------------
# 데이터 변환 (증강: train, 원본: val)
# -------------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=0.3, hue=0.1
    ),

    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =========================
# Train/Validation Split
# =========================
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["Defect"], random_state=42
)

train_loader = DataLoader(
    SolderDataset(train_df, output_folder, train_transform),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    SolderDataset(val_df, output_folder, val_transform),
    batch_size=batch_size, shuffle=False
)

# =========================
# 학습 시작
# =========================
print("\n전체 데이터 학습 시작")
model, optimizer = create_model(learning_rate, optimizer_name)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs} 시작")

    # ---------- Train ----------
    model.train()
    total_loss, correct = 0, 0
    for images, labels in tqdm(train_loader, desc=f"Train {epoch+1}"):
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

    # ---------- Validation ----------
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Val {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = val_correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1} -> "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 모델 저장
    epoch_folder = os.path.join(global_log_root, f"epoch_{epoch+1}")
    os.makedirs(epoch_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(epoch_folder, "model.pt"))
    print(f"모델 저장됨 -> {os.path.join(epoch_folder, 'model.pt')}")

print(f"\n학습 완료! 최종 모델 저장 위치: {global_log_root}")
