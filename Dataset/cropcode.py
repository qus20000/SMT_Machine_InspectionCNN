# 사용자가 제공한 최신 코드에 기반하여 멀티프로세싱을 적용한 버전 생성
# 단일 이미지 처리 함수 -> 폴더 단위 병렬 처리로 최적화

import os
import subprocess
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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
# 출력 형식 설정
# =========================
def print_header(title):
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def print_info(msg):
    print(f"- {msg}")

# =========================
# 사용자 설정
# =========================
dataset_root = "./Dataset/Imageset"
output_root = os.path.join(dataset_root, "Output")
output_hue_root = os.path.join(dataset_root, "Output_Hue")
output_test_root = os.path.join(dataset_root, "Output_Test")
output_hue_test_root = os.path.join(dataset_root, "Output_Hue_Test")
crop_width = 650 # 크롭 너비
crop_height = 270 # 크롭 높이
num_workers = 16 # 멀티프로세싱 워커 수

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
    local_defect_labels = []

    board_path = os.path.join(dataset_root, folder_name)
    parts = folder_name.split("_")
    
    # 폴더명이 적절한 형식이 아니면 건너뜀
    if len(parts) < 2:
        return [], [], False
        
    try:
        board_name = parts[0]
        angle = float(parts[1])
        # TEST가 포함된 경우 테스트 데이터셋으로 처리
        is_test = any("TEST" in part.upper() for part in parts[2:]) if len(parts) > 2 else False
        # 언더바가 추가로 있으면서 TEST가 아닌 경우 (예: BOARD2_-0.015_SHIFT) Defect=1
        is_defect = len(parts) > 2 and not is_test
    except ValueError:
        return [], [], False

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

        # Defect 이미지의 경우 파일명 끝에 _D 추가
        suffix = "_D" if is_defect else ""
        out_filename = f"{board_name}_{comp_name}{suffix}.png"
        
        # 테스트 데이터셋인 경우 다른 출력 폴더 사용
        if is_test:
            out_path = os.path.join(output_test_root, out_filename)
            out_hue_path = os.path.join(output_hue_test_root, out_filename)
        else:
            out_path = os.path.join(output_root, out_filename)
            out_hue_path = os.path.join(output_hue_root, out_filename)
            
        cv2.imwrite(out_path, cropped)

        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        hue_only = np.zeros_like(hsv)
        hue_only[:, :, 0] = hsv[:, :, 0]
        hue_only[:, :, 1] = 255
        hue_only[:, :, 2] = 255
        hue_bgr = cv2.cvtColor(hue_only, cv2.COLOR_HSV2BGR)

        # Hue 이미지도 동일한 파일명으로 저장
        cv2.imwrite(out_hue_path, hue_bgr)

        local_output_image_names.append(out_filename)
        local_defect_labels.append(1 if is_defect else 0)

    return local_output_image_names, local_defect_labels, is_test

# =========================
# 멀티프로세싱 실행
# =========================

if __name__ == "__main__":
    # 패키지 설치는 메인에서만 실행
    required_packages = ["opencv-python", "tqdm", "numpy", "pandas", "openpyxl"]
    install_packages(required_packages)

    # 모든 출력 디렉토리 생성
    for dir_path in [output_root, output_hue_root, output_test_root, output_hue_test_root]:
        os.makedirs(dir_path, exist_ok=True)

    # BOARD로 시작하는 폴더만 선택
    board_folders = [f for f in os.listdir(dataset_root)
                     if os.path.isdir(os.path.join(dataset_root, f)) 
                     and f.startswith("BOARD")
                     and "_" in f]

    # 초기 정보 출력
    print_header("데이터셋 처리 시작")
    print_info(f"처리할 보드 폴더: {len(board_folders)}개")

    train_image_names = []
    train_defect_labels = []
    test_image_names = []
    test_defect_labels = []

    # 이미지 변환 진행
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(board_folders), desc="이미지 변환 진행률") as pbar:
            for images, labels, is_test in executor.map(process_board_folder, board_folders):
                if is_test:
                    test_image_names.extend(images)
                    test_defect_labels.extend(labels)
                else:
                    train_image_names.extend(images)
                    train_defect_labels.extend(labels)
                pbar.update(1)

    # 처리 결과 출력
    print_header("데이터 처리 결과")
    print_info(f"학습 이미지: {len(train_image_names)}개")
    print_info(f"테스트 이미지: {len(test_image_names)}개")

    # =========================
    # 엑셀 정렬 모듈 임포트
    # =========================
    sys.path.append(os.path.dirname(__file__))  # ExcelSorter.py 위치 추가
    import ExcelSorter

    # =========================
    # 학습용 라벨링 엑셀 생성
    # =========================
    train_df = pd.DataFrame({
        "ImageName": train_image_names,
        "Defect": train_defect_labels
    })
    train_label_path = os.path.join(dataset_root, "DefectLabel.xlsx")
    # 정렬 후 저장
    ExcelSorter.sort_excel(train_df, "ImageName", train_label_path)

    # =========================
    # 테스트용 라벨링 엑셀 생성
    # =========================
    if test_image_names:  # 테스트 데이터가 있는 경우에만 생성
        test_df = pd.DataFrame({
            "ImageName": test_image_names,
            "Defect": test_defect_labels
        })
        test_label_path = os.path.join(dataset_root, "DefectLabel_Test.xlsx")
        # 정렬 후 저장
        ExcelSorter.sort_excel(test_df, "ImageName", test_label_path)