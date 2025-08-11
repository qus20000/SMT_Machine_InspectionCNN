# 사용자가 제공한 최신 코드에 기반하여 멀티프로세싱을 적용한 버전 생성
# 단일 이미지 처리 함수 → 폴더 단위 병렬 처리로 최적화

import os
import subprocess
import sys

# =========================
# 필요한 패키지 자동 설치
# =========================
def install_packages(packages):
    for package in packages:
        try:
            __import__(package.split("-")[0])
        except ImportError:
            print(f"'{package}' 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# =========================
# 본 코드 시작
# =========================
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# =========================
# 사용자 설정
# =========================
dataset_root = "./Dataset"
output_root = os.path.join(dataset_root, "Output")
output_hue_root = os.path.join(dataset_root, "Output_Hue")
crop_width = 450
crop_height = 400
num_workers = 8

# === 처리할 소자 prefix 설정 ===
allowed_prefixes = ["C", "R", "LED"]

# =========================
# 회전 및 크롭 함수
# =========================
def rotate_image_keep_size(image, angle_deg):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def crop_center(image, crop_width, crop_height):
    (h, w) = image.shape[:2]
    cx, cy = w // 2, h // 2
    x1 = max(cx - crop_width // 2, 0)
    y1 = max(cy - crop_height // 2, 0)
    x2 = min(cx + crop_width // 2, w)
    y2 = min(cy + crop_height // 2, h)
    return image[y1:y2, x1:x2]

# =========================
# 개별 폴더 처리 함수
# =========================
def process_board_folder(folder_name):
    local_output_image_names = []

    board_path = os.path.join(dataset_root, folder_name)
    try:
        board_name, angle_str = folder_name.split("_")
        angle = float(angle_str)
    except ValueError:
        return []

    for filename in os.listdir(board_path):
        if not filename.lower().endswith(".png"):
            continue

        comp_name = os.path.splitext(filename)[0]

        if not any(comp_name.upper().startswith(p) for p in allowed_prefixes):
            continue

        image_path = os.path.join(board_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        rotated = rotate_image_keep_size(image, angle)
        cropped = crop_center(rotated, crop_width, crop_height)

        out_filename = f"{board_name}_{comp_name}.png"
        out_path = os.path.join(output_root, out_filename)
        cv2.imwrite(out_path, cropped)

        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        hue_only = np.zeros_like(hsv)
        hue_only[:, :, 0] = hsv[:, :, 0]
        hue_only[:, :, 1] = 255
        hue_only[:, :, 2] = 255
        hue_bgr = cv2.cvtColor(hue_only, cv2.COLOR_HSV2BGR)

        out_hue_path = os.path.join(output_hue_root, out_filename)
        cv2.imwrite(out_hue_path, hue_bgr)

        local_output_image_names.append(out_filename)

    return local_output_image_names

# =========================
# 멀티프로세싱 실행
# =========================

if __name__ == "__main__":
    # 패키지 설치는 메인에서만 실행
    required_packages = ["opencv-python", "tqdm", "numpy", "pandas", "openpyxl"]
    install_packages(required_packages)

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(output_hue_root, exist_ok=True)

    board_folders = [f for f in os.listdir(dataset_root)
                     if os.path.isdir(os.path.join(dataset_root, f)) and "_" in f]

    all_image_names = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_board_folder, board_folders), total=len(board_folders)):
            all_image_names.extend(result)

    # =========================
    # 라벨링 엑셀 생성
    # =========================
    label_df = pd.DataFrame({
        "ImageName": all_image_names,
        "Defect": [0] * len(all_image_names)
    })
    label_path = os.path.join(dataset_root, "DefectLabel.xlsx")
    label_df.to_excel(label_path, index=False)

    print(label_path)