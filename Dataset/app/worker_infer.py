import os, re, time
import numpy as np, cv2, torch
from torch import nn
from torchvision import transforms
from PySide6.QtCore import QThread, Signal
import torch.nn.functional as F
# =========================
# 정규식 / 유틸리티
# =========================
_DESIG_RE = re.compile(r"([A-Za-z]+)\s*0*([0-9]+)")

def canonical_designator(t):
    if not isinstance(t,str): return ""
    t=t.replace("\x00","").upper()
    m=_DESIG_RE.findall(t)
    if not m:
        m2=re.findall(r"[A-Z]+[0-9]+",t)
        if not m2: return ""
        t=m2[-1]; m=_DESIG_RE.findall(t)
        if not m: return t
    p,n=m[-1]
    try: n_i=int(n)
    except: n_i=int(re.sub(r"\D","",n) or "0")
    return f"{p}{n_i}"

def crop_center(img,w,h):
    H,W=img.shape[:2]
    if W==w and H==h: return img
    cx,cy=W//2,H//2
    x1=max(cx-w//2,0);y1=max(cy-h//2,0)
    return img[y1:y1+h,x1:x1+w]

def safe_read_image(path: str, retries: int = 5, delay: float = 0.15):
    """
    Windows에서 파일이 아직 저장 중일 때 PermissionError 나는 걸 막기 위해
    몇 번 재시도하면서 이미지를 읽는다.
    """
    for i in range(retries):
        try:
            # np.fromfile + cv2.imdecode 패턴 유지
            data = np.fromfile(path, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except PermissionError:
            time.sleep(delay)
        except FileNotFoundError:
            time.sleep(delay)
    return None


# =========================
# [2025/11/06 추가] CLAHE + 색상비 기반 전처리
# =========================
def preprocess_with_ratio_hsv(image):
    """RGB 이미지 -> HSV CLAHE + 5채널(R,G,B,R/G,B/G)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    img_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img_f = img_hsv.astype(np.float32) + 1e-6
    R, G, B = cv2.split(img_f)
    RG_ratio = (R / G).clip(0, 5)
    BG_ratio = (B / G).clip(0, 5)
    merged = np.stack([R, G, B, RG_ratio, BG_ratio], axis=-1)
    merged = np.clip(merged / 255.0, 0, 1)
    return merged.astype(np.float32)


# =========================
# InferenceWorker
# =========================
class InferenceWorker(QThread):
    image_ready=Signal(object,dict)
    log_ready=Signal(str)
    pred_ready=Signal(str,int,float)

    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg; self.stop_flag=False
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=None

        # [2025/11/06 수정] 5채널 입력을 위한 전처리 (cv2.resize 사용)
        self.preprocess = transforms.Compose([
            transforms.Lambda(
                lambda img: torch.from_numpy(
                    cv2.resize(img, (cfg["input_size"], cfg["input_size"]), interpolation=cv2.INTER_LINEAR)
                    .transpose(2, 0, 1)
                ).float()
            )
        ])

        self._seen={}

    def stop(self): self.stop_flag=True

    # =========================
    # [2025/11/06 수정] EfficientNet/ViT 기반 모델 로드
    # =========================
    def _load_model(self):
        mp=self.cfg["model_path"]
        model_type=self.cfg.get("model_type","efficientnet").lower()
        try:
            if model_type=="efficientnet":
                from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                m=efficientnet_b0(weights=None)
                m.features[0][0]=nn.Conv2d(5,32,kernel_size=3,stride=2,padding=1,bias=False)
                m.classifier[1]=nn.Linear(m.classifier[1].in_features,2)
            elif model_type=="vit":
                from torchvision.models import vit_b_16, ViT_B_16_Weights
                m=vit_b_16(weights=None)
                m.conv_proj=nn.Conv2d(5,768,kernel_size=16,stride=16)
                m.heads.head=nn.Linear(m.heads.head.in_features,2)
            else:
                from torchvision.models import resnet18
                m=resnet18(weights=None)
                m.conv1=nn.Conv2d(5,64,kernel_size=7,stride=2,padding=3,bias=False)
                m.fc=nn.Linear(m.fc.in_features,2)

            state=torch.load(mp,map_location=self.device)
            m.load_state_dict(state,strict=False)
            m.eval().to(self.device)
            self.model=m
            self.log_ready.emit(f"[worker] Model loaded ({model_type}): {mp}")
        except Exception as e:
            self.log_ready.emit(f"[worker] Model load failed: {e}")


    # =========================
    # 추론 스레드 루프
    # =========================
    def run(self):
        self._load_model()
        imgdir=self.cfg["watch_image_dir"]
        roi_w=self.cfg["roi_w"]; roi_h=self.cfg["roi_h"]
        th=self.cfg["threshold"]

        while not self.stop_flag:
            names=[f for f in os.listdir(imgdir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            for fn in names:
                p=os.path.join(imgdir,fn)
                try: mt=os.stat(p).st_mtime_ns
                except FileNotFoundError: continue
                if self._seen.get(p)==mt: continue

                img = safe_read_image(p)
                if img is None:
                    self.log_ready.emit(f"[worker] 이미지 읽기 실패(잠김): {p}")
                    continue

                patch=crop_center(img,roi_w,roi_h)
                des=canonical_designator(fn)
                self.image_ready.emit(patch,{"designator":des})

                # [2025/11/06 수정] CLAHE + 색상비 기반 5채널 전처리 적용
                rgb_img=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
                img_5ch=preprocess_with_ratio_hsv(rgb_img)
                x=self.preprocess(img_5ch).unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    prob=torch.softmax(self.model(x),1)[0,1].item()

                pred=int(prob>=th)
                self.log_ready.emit(f"[infer] {des} pred={pred} prob={prob:.4f}")
                self.pred_ready.emit(des,pred,prob)
                self._seen[p]=mt

            time.sleep(self.cfg["poll_sec"])
