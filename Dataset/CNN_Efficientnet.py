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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights   # [2025/11/06 추가] ViT 모델 임포트

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
# [2025/11/06 추가] 색상비 + HSV + CLAHE 전처리 함수
# =========================
def preprocess_with_ratio_hsv(image):
    """RGB 이미지 입력 -> HSV CLAHE + R/G, B/G 채널 추가"""
    # HSV 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE 적용 (밝기 균일화)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    image_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 색상비 계산 (R/G, B/G)
    img_f = image_hsv.astype(np.float32) + 1e-6
    R, G, B = cv2.split(img_f)
    RG_ratio = (R / G).clip(0, 5)
    BG_ratio = (B / G).clip(0, 5)

    # 5채널로 병합
    merged = np.stack([R, G, B, RG_ratio, BG_ratio], axis=-1)
    merged = np.clip(merged / 255.0, 0, 1)
    return merged.astype(np.float32)


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

        # [2025/11/06 추가] CLAHE + 색상비 전처리 적용
        image = preprocess_with_ratio_hsv(image)

        if self.transform:
            image = self.transform(image)
        return image, label


# =========================
# [2025/11/06 추가] Tensor 변환 함수 (5채널 호환)
# =========================
class ToTensor5Ch:
    def __call__(self, image):
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        return image


# =========================
# 데이터 변환 정의
# =========================
train_transform = transforms.Compose([
    ToTensor5Ch(),  # [2025/11/06 수정] 5채널 입력으로 교체
])

val_transform = transforms.Compose([
    ToTensor5Ch(),
])

# =========================
# [2025/11/06 추가] FocalLoss 클래스 정의
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()


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
    def create_model(learning_rate, optimizer_name, model_type="efficientnet"):
        """[2025/11/06 수정] EfficientNet / ViT / ResNet 중 선택"""
        if model_type == "efficientnet":
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            model.features[0][0] = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_type == "vit":
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            model.conv_proj = nn.Conv2d(5, 768, kernel_size=16, stride=16)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        else:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
    model, optimizer = create_model(learning_rate, optimizer_name, model_type="efficientnet")  # [2025/11/06 수정] 모델타입 지정
    criterion = FocalLoss()  # [2025/11/06 수정] Focal Loss 사용

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
