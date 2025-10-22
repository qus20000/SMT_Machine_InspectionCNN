# inference_watcher_fixed.py
# -------------------------------------------------------------
# Folder watching inference (중복 방지 + 행(row)기반 결과 엑셀 저장)
# - ./Dataset/Inference 감시, 새 PNG 유입 즉시 추론
# - ./Dataset/inferenceResult_YYYYMMDD_HHMMSS 결과 폴더 생성
# - PredictionResults.xlsx에 행 방향으로 누적 저장(ExcelSorter 있으면 정렬)
# - DurationSec 컬럼을 맨 오른쪽으로 추가
# - Windows 환경에서는 torch.compile() 자동 비활성 (Triton 없음)
# - 디코딩 후 ROI(650x270) 센터 크롭. 이미 ROI 사이즈면 크롭 스킵
# - 콘솔에 raw score( threshold 적용 전 확률 ) 추가 출력
# -------------------------------------------------------------

import os
import re
import time
import queue
import threading
import platform
from datetime import datetime

import cv2
import numpy as np
import torch
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =========================
# 경로/폴더 설정
# =========================
INFER_DIR  = "./Dataset/Inference"
MODEL_PATH = "./Dataset/model.pt"

OUTPUT_DIR = f"./Dataset/inferenceResult_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULT_XLSX = os.path.join(OUTPUT_DIR, "PredictionResults.xlsx")

# =========================
# 환경/옵션
# =========================
CONF_THRESH      = 0.5
PRINT_OK         = True
IMG_EXTS         = {".png", ".PNG"}
STABILIZE_MS     = 120
DEBOUNCE_MS      = 250
BATCH_SAVE_EVERY = 1  # 1개마다 저장

# ROI 크기(사용자 크롭코드와 동일 규격)
ROI_W = 650
ROI_H = 270

# =========================
# CUDA 환경 설정
# =========================
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[INFO] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA unavailable. Running on CPU (slow).")
    return device

device = setup_device()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
USE_AMP = (device.type == "cuda")

# =========================
# 전처리 (학습 시와 동일)
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =========================
# 안전한 torch.compile 처리
# =========================
def _safe_compile(model: torch.nn.Module):
    if platform.system() == "Windows":
        print("[INFO] Windows detected: disable torch.compile (Inductor/Triton).")
        return model
    try:
        model = torch.compile(model, mode="reduce-overhead")  # Inductor(=Triton)
        print("[INFO] torch.compile enabled (Inductor).")
        return model
    except Exception as e:
        print(f"[WARN] torch.compile(Inductor) failed: {e} -> fallback to eager.")
        try:
            model = torch.compile(model, backend="eager")
            print("[INFO] torch.compile enabled (backend=eager).")
            return model
        except Exception as ee:
            print(f"[WARN] torch.compile(eager) failed: {ee} -> run without compile.")
            return model

# =========================
# 모델 로드
# =========================
def load_model_once(model_path: str) -> nn.Module:
    print(f"[INFO] Loading model: {model_path}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.to(memory_format=torch.channels_last)
    model = _safe_compile(model)
    return model

MODEL = load_model_once(MODEL_PATH)

# =========================
# ExcelSorter 감지
# =========================
try:
    import ExcelSorter
    HAS_EXCELSORTER = True
    print("[INFO] ExcelSorter detected.")
except ImportError:
    HAS_EXCELSORTER = False
    print("[WARN] ExcelSorter not found. Fallback to built-in sorting.")

# =========================
# 파일 안정화 검사
# =========================
def is_file_stable(path: str, wait_ms: int = STABILIZE_MS) -> bool:
    try:
        s1 = os.path.getsize(path)
        time.sleep(wait_ms / 1000.0)
        s2 = os.path.getsize(path)
        return (s1 == s2) and (s1 > 0)
    except FileNotFoundError:
        return False

# =========================
# 이름 파싱(정렬키 동일 유지)
# =========================
def split_filename_parts(filename: str):
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
    PRIORITY = {"C": 0, "FID": 1, "R": 2}
    return PRIORITY.get(prefix, 100 + ord(prefix[0]) if prefix else 999)

def sort_key(name: str):
    b, p, n, d = split_filename_parts(name)
    return (b, prefix_rank(p), p, n, d)

# =========================
# 크롭 유틸리티(센터 크롭)
# =========================
def crop_center(image_bgr, crop_w: int, crop_h: int):
    h, w = image_bgr.shape[:2]
    if w == crop_w and h == crop_h:
        return image_bgr  # 이미 ROI 사이즈면 스킵
    cx, cy = w // 2, h // 2
    x1 = max(cx - crop_w // 2, 0)
    y1 = max(cy - crop_h // 2, 0)
    x2 = min(x1 + crop_w, w)
    y2 = min(y1 + crop_h, h)
    # 경계 때문에 crop_w/crop_h와 다를 수 있음. 성능 위해 pad 없이 자르기만 함.
    return image_bgr[y1:y2, x1:x2]

# =========================
# 추론(단일 이미지)
# =========================
@torch.inference_mode()
def infer_one_image_bgr(img_bgr):
    # 크롭(센터, ROI 650x270). 이미 ROI면 스킵.
    img_bgr = crop_center(img_bgr, ROI_W, ROI_H)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = transform(img_rgb).unsqueeze(0).to(device, non_blocking=True)
    x = x.contiguous(memory_format=torch.channels_last)

    if USE_AMP:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = MODEL(x)
    else:
        logits = MODEL(x)

    prob_defect = torch.softmax(logits, dim=1)[0, 1].item()
    pred = 1 if prob_defect >= CONF_THRESH else 0
    return prob_defect, pred

# =========================
# 결과 누적/저장
# =========================
_predictions = []  # {"ImageName", "DefectProb", "PredictedLabel", "DurationSec"}

def save_results():
    if not _predictions:
        return
    df = pd.DataFrame(_predictions)
    base_cols = ["ImageName", "DefectProb", "PredictedLabel"]
    cols = base_cols + (["DurationSec"] if "DurationSec" in df.columns else [])
    df = df[cols]
    try:
        if HAS_EXCELSORTER:
            ExcelSorter.sort_excel(df, "ImageName", RESULT_XLSX)
        else:
            df.sort_values(by="ImageName", inplace=True, key=lambda s: s.map(str))
            df.to_excel(RESULT_XLSX, index=False)
        print(f"[SAVE] Results saved: {RESULT_XLSX}  (rows={len(df)})")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

# =========================
# watchdog 중복 방지
# =========================
class LatestHandler(FileSystemEventHandler):
    def __init__(self, q, pending, done, last_emit):
        self.q = q
        self.pending = pending
        self.done = done
        self.last_emit = last_emit

    def _should_enqueue(self, path: str) -> bool:
        if path in self.done or path in self.pending:
            return False
        now_ms = time.time() * 1000.0
        last = self.last_emit.get(path, 0.0)
        if (now_ms - last) < DEBOUNCE_MS:
            return False
        self.last_emit[path] = now_ms
        return True

    def on_created(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1]
        if ext not in IMG_EXTS:
            return
        if self._should_enqueue(event.src_path):
            self.q.put(event.src_path)

def start_watcher(watch_dir, q, pending, done, last_emit):
    obs = Observer()
    obs.schedule(LatestHandler(q, pending, done, last_emit), watch_dir, recursive=False)
    obs.start()
    print("=====================================================")
    print("Inference Watcher Started")
    print(f"  Watching folder : {watch_dir}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Model path      : {MODEL_PATH}")
    print(f"  Device          : {device}")
    print(f"  Threshold       : {CONF_THRESH}")
    print(f"  ROI (WxH)       : {ROI_W}x{ROI_H}")
    print("=====================================================")
    return obs

# =========================
# 메인 루프
# =========================
def main():
    global MODEL
    os.makedirs(INFER_DIR, exist_ok=True)

    # Warm-up
    try:
        dummy = torch.zeros((1, 3, 224, 224), device=device).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            _ = MODEL(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print("[INFO] Warm-up complete.")
    except Exception as e:
        print(f"[WARN] Warm-up failed: {e} -> reload without torch.compile.")
        def load_model_nocompile(model_path: str) -> nn.Module:
            m = models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 2)
            st = torch.load(model_path, map_location=device)
            m.load_state_dict(st)
            m.to(device)
            m.eval()
            m.to(memory_format=torch.channels_last)
            return m
        MODEL = load_model_nocompile(MODEL_PATH)

    q = queue.Queue(maxsize=32)
    pending, done, last_emit = set(), set(), {}
    obs = start_watcher(INFER_DIR, q, pending, done, last_emit)

    processed_since_save = 0

    try:
        while True:
            try:
                path = q.get(timeout=0.1)
            except queue.Empty:
                continue

            if path in done or path in pending:
                continue
            pending.add(path)

            ext = os.path.splitext(path)[1]
            if ext not in IMG_EXTS:
                pending.discard(path)
                continue

            if not is_file_stable(path):
                time.sleep(STABILIZE_MS / 1000.0)
                if not is_file_stable(path):
                    pending.discard(path)
                    continue

            fname = os.path.basename(path)
            print(f"image input detected : {fname}")

            try:
                with open(path, "rb") as f:
                    buf = f.read()
                arr = np.frombuffer(buf, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            except Exception as e:
                print(f"[WARN] Failed to read {path}: {e}")
                pending.discard(path)
                continue
            if img is None:
                print(f"[WARN] Invalid image: {path}")
                pending.discard(path)
                continue

            t0 = time.perf_counter()
            prob, pred = infer_one_image_bgr(img)
            t1 = time.perf_counter()
            duration_sec = round(t1 - t0, 6)

            # raw score(Threshold 이전 확률) 먼저 출력
            print(f"raw score            : {prob:.6f}")

            human = "Abnormal" if pred == 1 else "Normal"
            if PRINT_OK or pred == 1:
                print(f"inference result     : {pred} ({human})")
                print(f"inference duration   : {duration_sec:.2f}s")

            _predictions.append({
                "ImageName": fname,
                "DefectProb": float(prob),
                "PredictedLabel": int(pred),
                "DurationSec": duration_sec,
            })

            processed_since_save += 1
            if processed_since_save >= BATCH_SAVE_EVERY:
                save_results()
                processed_since_save = 0

            if PRINT_OK or pred == 1:
                print(f"result saved in      : {RESULT_XLSX}")
                print("-" * 50)

            done.add(path)
            pending.discard(path)

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        obs.stop()
        obs.join()
        save_results()
        print("[INFO] Exiting... Results saved at:")
        print(f"       {RESULT_XLSX}")

if __name__ == "__main__":
    main()
