import os, re, time
import numpy as np, cv2, torch
from torch import nn
from torchvision import transforms
from PySide6.QtCore import QThread, Signal

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
            # 아직 다른 프로세스가 쓰는 중 -> 잠깐 기다렸다가 다시
            time.sleep(delay)
        except FileNotFoundError:
            # 저장이 아주 느리면 바로 안 보일 수도 있음
            time.sleep(delay)
    return None

class InferenceWorker(QThread):
    image_ready=Signal(object,dict)
    log_ready=Signal(str)
    pred_ready=Signal(str,int,float)

    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg; self.stop_flag=False
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=None
        self.preprocess=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((cfg["input_size"],cfg["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        self._seen={}

    def stop(self): self.stop_flag=True

    def _load_model(self):
        mp=self.cfg["model_path"]
        try:
            from torchvision.models import resnet18
            m=resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,2)
            state=torch.load(mp,map_location=self.device)
            m.load_state_dict(state,strict=False); m.eval().to(self.device)
            self.model=m; self.log_ready.emit(f"[worker] Model loaded: {mp}")
        except Exception as e:
            self.log_ready.emit(f"[worker] Model load failed: {e}")

    def run(self):
        self._load_model()
        imgdir=self.cfg["watch_image_dir"]; roi_w=self.cfg["roi_w"]; roi_h=self.cfg["roi_h"]
        th=self.cfg["threshold"]
        while not self.stop_flag:
            names=[f for f in os.listdir(imgdir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            for fn in names:
                p=os.path.join(imgdir,fn); mt=os.stat(p).st_mtime_ns
                if self._seen.get(p)==mt: continue
                img = safe_read_image(p)
                if img is None:
                    self.log_ready.emit(f"[worker] 이미지 읽기 실패(잠김): {p}")
                    continue
                if img is None: continue
                patch=crop_center(img,roi_w,roi_h)
                des=canonical_designator(fn)
                self.image_ready.emit(patch,{"designator":des})
                x=self.preprocess(cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
                with torch.inference_mode(): prob=torch.softmax(self.model(x),1)[0,1].item()
                pred=int(prob>=th)
                self.log_ready.emit(f"[infer] {des} pred={pred} prob={prob:.4f}")
                self.pred_ready.emit(des,pred,prob)
                self._seen[p]=mt
            time.sleep(self.cfg["poll_sec"])
