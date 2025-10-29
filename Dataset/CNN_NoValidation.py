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
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm
import random

# =========================
# [2025/10/31 추가] 패키지 및 시드 설정
# =========================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # [2025/10/31 수정] 성능 최적화 허용

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

# =========================
# 데이터 변환 정의
# =========================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
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
# Windows-safe main 시작
# =========================
if __name__ == "__main__":

    # =========================
    # 기본 설정
    # =========================
    dataset_root = "./Dataset"
    output_folder = os.path.join(dataset_root, "Imageset/Output")
    label_path = os.path.join(dataset_root, "Imageset/DefectLabel.xlsx")
    df = pd.read_excel(label_path)

    # =========================
    # 하이퍼파라미터 로드
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
    print(f"현재 사용 디바이스: {device}")

    # =========================
    # [2025/10/31 추가] 학습 모드 선택 (터미널 입력)
    # =========================
    print("\n학습 모드를 선택하세요:")
    print("1: 일반화 성능 측정용 (train/val split)")
    print("2: 모델 공개용 Full Train (100% 학습 + 원본 검증)")
    mode_input = input("선택 (1 또는 2): ").strip()

    if mode_input == "2":
        mode = "fulltrain"
    else:
        mode = "generalization"
    print(f"[모드 선택] 현재 학습 모드: {mode}")

    # =========================
    # 데이터 분할
    # =========================
    if mode == "generalization":
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df["Defect"], random_state=42
        )
    else:
        train_df = df.copy()
        val_df = df.sample(frac=0.2, random_state=42)
        print(f"[FullTrain 모드] 학습 데이터: {len(train_df)}개, 검증용 원본 샘플: {len(val_df)}개")

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
    # DataLoader 설정
    # =========================
    train_loader = DataLoader(
        SolderDataset(train_df, output_folder, train_transform),
        batch_size=batch_size, shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        SolderDataset(val_df, output_folder, val_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )

    # =========================
    # 학습 시작
    # =========================
    print("\n전체 데이터 학습 시작")
    model, optimizer = create_model(learning_rate, optimizer_name)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies, val_aucs = [], [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} 시작")

        # ---------- Train ----------
        model.train()
        total_loss, correct = 0, 0
        for images, labels in tqdm(train_loader, desc=f"Train {epoch+1}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ---------- Validation ----------
        model.eval()
        val_loss, val_correct = 0, 0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Val {epoch+1}"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        val_auc = roc_auc_score(all_labels, all_probs)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_aucs.append(val_auc)

        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ROC-AUC: {val_auc:.4f}")

        # 모델 저장
        epoch_folder = os.path.join(global_log_root, f"epoch_{epoch+1}")
        os.makedirs(epoch_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(epoch_folder, "model.pt"))
        print(f"모델 저장됨 -> {os.path.join(epoch_folder, 'model.pt')}")

    # =========================
    # 결과 그래프 저장
    # =========================
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(range(1, num_epochs+1), train_losses, '-o', label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, '-o', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss per Epoch')
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(range(1, num_epochs+1), train_accuracies, '-o', label='Train Acc')
    plt.plot(range(1, num_epochs+1), val_accuracies, '-o', label='Val Acc')
    plt.plot(range(1, num_epochs+1), val_aucs, '-o', label='Val ROC-AUC')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Accuracy & ROC-AUC per Epoch')
    plt.legend(); plt.grid(True)

    save_path = os.path.join(global_log_root, "performance_plot.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"성능 그래프 저장됨: {save_path}")

    print(f"\n학습 완료! 최종 모델 저장 위치: {global_log_root}")
