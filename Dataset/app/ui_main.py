
import os, cv2
from PySide6.QtCore import Qt, QSize, Slot, QUrl
from PySide6.QtGui import QPixmap, QImage, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QTextEdit, QSplitter, QFrame, QSizePolicy
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from worker_infer import InferenceWorker
from pnp_html import build_boardmap_html

def rgbhex_to_bgr_tuple(hexcode: str):
    hexcode = hexcode.lstrip('#')
    r = int(hexcode[0:2], 16); g = int(hexcode[2:4], 16); b = int(hexcode[4:6], 16)
    return (b, g, r)

def cvimg_to_qpix(img_bgr) -> QPixmap:
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3*w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class Card(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("QCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8,8,8,8); lay.setSpacing(6)
        header = QLabel(title); header.setObjectName("QCardTitle")
        lay.addWidget(header)
        self.body = QVBoxLayout(); self.body.setContentsMargins(4,0,4,4)
        lay.addLayout(self.body)

class MainWindow(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("SMT Live Dashboard (v10)")
        self.resize(1280, 768)
        self._web_ready = False; self._pending_js = []
        self._last_pix = None; self._last_bgr = None

        self.COLOR_OK_BGR = rgbhex_to_bgr_tuple(self.cfg["color_ok"])
        self.COLOR_NG_BGR = rgbhex_to_bgr_tuple(self.cfg["color_ng"])

        root = QSplitter(Qt.Vertical)
        top  = QSplitter(Qt.Horizontal)
        root.addWidget(top)

        # Preview
        self.preview_card = Card("Live Preview")
        self.preview_label = QLabel("이미지 없음")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_card.body.addWidget(self.preview_label)
        top.addWidget(self.preview_card)

        # Board map
        self.html_card = Card("Board Map")
        self.web = QWebEngineView()
        self.html_card.body.addWidget(self.web)
        top.addWidget(self.html_card)

        # Logs
        self.log_card = Card("Logs")
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log_card.body.addWidget(self.log)
        root.addWidget(self.log_card)

        top.setStretchFactor(0, 1); top.setStretchFactor(1, 2)
        root.setStretchFactor(0, 3); root.setStretchFactor(1, 1)
        self.setCentralWidget(root)

        self._apply_qss()
        self._load_boardmap()

        self.worker = InferenceWorker(self.cfg)
        self.worker.image_ready.connect(self.on_image_ready)
        self.worker.log_ready.connect(self.on_log)
        self.worker.pred_ready.connect(self.on_pred)
        self.worker.start()

    def _apply_qss(self):
        qss_path = os.path.join(os.path.dirname(__file__), "theme.qss")
        if os.path.exists(qss_path):
            self.setStyleSheet(open(qss_path, "r", encoding="utf-8").read())

    def _load_boardmap(self):
        from PySide6.QtCore import QUrl
        try:
            path = build_boardmap_html(
                pnp_path=self.cfg["pnp_path"],
                out_html=self.cfg["html_out"],
                title="PnP Live Boardmap",
                color_ok=self.cfg["color_ok"],
                color_ng=self.cfg["color_ng"],
                color_neutral=self.cfg["color_neutral"],
            )
            self.web.load(QUrl.fromLocalFile(os.path.abspath(path)))
            self.web.loadFinished.connect(lambda ok: setattr(self, "_web_ready", ok))
            self.on_log(f"[ui] Board map loaded: {path}")
        except Exception as e:
            self.on_log(f"[ui] 보드맵 로드 실패: {e}")

    def _set_preview_pixmap(self, pix: QPixmap):
        self._last_pix = pix
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, e):
        if self._last_pix is not None:
            self._set_preview_pixmap(self._last_pix)
        return super().resizeEvent(e)

    @Slot(object, dict)
    def on_image_ready(self, img, meta):
        self._last_bgr = img.copy()
        pix = cvimg_to_qpix(img)
        self._set_preview_pixmap(pix)
        if meta.get("designator"): self.preview_label.setToolTip(meta["designator"])

    @Slot(str)
    def on_log(self, text: str):
        self.log.append(text); self.log.moveCursor(QTextCursor.End)

    @Slot(str, int, float)
    def on_pred(self, designator: str, pred: int, prob: float):
        js = f"PNP.setState('{designator}', {int(pred)});"
        if self._web_ready: self.web.page().runJavaScript(js)
        else: self._pending_js.append(js)

        if self._last_bgr is None: return
        bgr = self._last_bgr.copy()
        h, w = bgr.shape[:2]
        color = self.COLOR_NG_BGR if pred == 1 else self.COLOR_OK_BGR
        t = 5
        bgr[:t,:,:] = color; bgr[h-t:h,:,:] = color; bgr[:,:t,:] = color; bgr[:,w-t:w,:] = color
        cv2.putText(bgr, ("NG" if pred==1 else "PASS")+f" p={prob:.3f}", (16,36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        pix2 = cvimg_to_qpix(bgr); self._set_preview_pixmap(pix2)

    def closeEvent(self, e):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop(); self.worker.wait(1000)
        super().closeEvent(e)
