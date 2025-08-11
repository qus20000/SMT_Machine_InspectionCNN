import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from datetime import datetime
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights

#=========================
# 2025-08-05 최종본
# 하이퍼파라미터 튜닝을 시도하기 이전의 코드이며, 혼동행렬 누적문제가 발생되고 있는 상태.
#=========================





# =========================
# 기본 설정
# =========================
dataset_root = "./Dataset"
output_folder = os.path.join(dataset_root, "Output")
label_path = os.path.join(dataset_root, "DefectLabel.xlsx")
df = pd.read_excel(label_path)

# =========================
# 하이퍼파라미터
# =========================
batch_size = 32
num_epochs = 10
learning_rate = 1e-4
num_classes = 2
num_folds = 5

# =========================
# 로그 디렉토리 생성
# =========================
global_log_root = os.path.join(".", f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(global_log_root, exist_ok=True)

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
# 이미지 변환 (해상도 유지)
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# =========================
# Stratified K-Fold 수행
# =========================
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

all_true = []
all_probs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df["ImageName"], df["Defect"])):
    print(f"\n🚀 Fold {fold+1}/{num_folds} 시작")

    log_dir = os.path.join(global_log_root, f"fold_{fold+1}")
    os.makedirs(log_dir, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = SolderDataset(train_df, output_folder, transform)
    val_dataset = SolderDataset(val_df, output_folder, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 저장용 변수
    best_auc = 0.0
    best_model_path = os.path.join(log_dir, "best_model.pt")

    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(num_epochs):
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
        train_acc = correct / len(train_dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        val_correct, val_total_loss = 0, 0
        fold_true, fold_probs = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

                fold_true.extend(labels.cpu().numpy())
                fold_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        val_loss = val_total_loss / len(val_loader)
        val_acc = val_correct / len(val_dataset)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # 모델 저장 (best AUC 기준)
        auc = roc_auc_score(fold_true, fold_probs)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_model_path)
            print(f"모델 저장됨 (Fold {fold+1}): {best_model_path}")

        all_true.extend(fold_true)
        all_probs.extend(fold_probs)

        print(f"  Fold {fold+1} - Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # === Fold별 그래프 저장 ===
    pred_labels = [1 if p > 0.5 else 0 for p in fold_probs]
    cm = confusion_matrix(fold_true, pred_labels)
    fpr, tpr, _ = roc_curve(fold_true, fold_probs)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "roc_curve.png"))

    # Confusion Matrix
    acc = accuracy_score(fold_true, pred_labels)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(
        f"Confusion Matrix (Fold {fold+1})\nResNet18, Epoch: {num_epochs}, Acc: {acc:.3f}, AUC: {auc:.3f}")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal", "Defect"])
    plt.yticks([0, 1], ["Normal", "Defect"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))

    # Accuracy & Loss
    plt.figure()
    plt.plot(train_acc_list, label="Train Acc")
    plt.plot(val_acc_list, label="Val Acc")
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.title(f"Fold {fold+1} - Accuracy and Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "accuracy_loss.png"))

# =========================
# 전체 통합 결과 그래프
# =========================
global_auc = roc_auc_score(all_true, all_probs)
global_acc = accuracy_score(all_true, [1 if p > 0.5 else 0 for p in all_probs])
global_cm = confusion_matrix(all_true, [1 if p > 0.5 else 0 for p in all_probs])
fpr, tpr, _ = roc_curve(all_true, all_probs)

# 전체 ROC
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
plt.title(f"Overall Confusion Matrix\nResNet18, Folds: {num_folds}, Acc: {global_acc:.3f}, AUC: {global_auc:.3f}")
plt.colorbar()
plt.xticks([0, 1], ["Normal", "Defect"])
plt.yticks([0, 1], ["Normal", "Defect"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(global_cm[i, j]), ha="center", va="center", color="black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(global_log_root, "overall_confusion_matrix.png"))

print(f"\n 전체 결과 저장됨: {global_log_root}")