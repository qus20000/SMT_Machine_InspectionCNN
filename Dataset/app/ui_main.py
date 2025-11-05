import os, cv2
from PySide6.QtCore import Qt, QSize, Slot, QUrl, QObject, Signal
from PySide6.QtGui import QPixmap, QImage, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QTextEdit, QSplitter, QFrame,
    QSizePolicy, QPushButton, QMessageBox, QWidget, QHBoxLayout,QCheckBox
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel

from worker_infer import InferenceWorker
from pnp_html import build_boardmap_html

# ------------------ 작은 유틸 ------------------ #
def rgbhex_to_bgr_tuple(hexcode: str):
    hexcode = hexcode.lstrip('#')
    r = int(hexcode[0:2], 16)
    g = int(hexcode[2:4], 16)
    b = int(hexcode[4:6], 16)
    return (b, g, r)


def cvimg_to_qpix(img_bgr) -> QPixmap:
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ------------------ Web ↔ Qt 브리지 ------------------ #
class BoardClickBridge(QObject):
    clicked = Signal(str)

    @Slot(str)
    def onBoardClick(self, des: str):
        self.clicked.emit(des)


# ------------------ 카드 공통 위젯 ------------------ #
class Card(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("QCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        header = QLabel(title)
        header.setObjectName("QCardTitle")
        lay.addWidget(header)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(4, 0, 4, 4)
        lay.addLayout(self.body)


# =========================================================
#                       메인 윈도우
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.setWindowTitle("SMT Live Dashboard (v10)")
        self.resize(1280, 768)

        # 상태 변수들
        self._web_ready = False
        self._pending_js: list[str] = []
        self._last_pix = None
        self._last_bgr = None

        # 디자인별로 찍힌 이미지 임시 저장 (보드 클릭해서 다시 보기용)
        self._shot_cache: dict[str, any] = {}

        # 색상
        self.COLOR_OK_BGR = rgbhex_to_bgr_tuple(self.cfg["color_ok"])
        self.COLOR_NG_BGR = rgbhex_to_bgr_tuple(self.cfg["color_ng"])

        # -------------------------------------------------
        # 전체 레이아웃: 위/아래
        # -------------------------------------------------
        root = QSplitter(Qt.Vertical)
        self.setCentralWidget(root)

        # -------------------------------------------------
        # (1) 위쪽: 좌/우
        # -------------------------------------------------
        top = QSplitter(Qt.Horizontal)
        root.addWidget(top)

        # 1-1. 왼쪽: Live preview
        self.preview_card = Card("Live Preview")
        self.preview_label = QLabel("이미지 없음")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_card.body.addWidget(self.preview_label)
        top.addWidget(self.preview_card)

        # 1-2. 오른쪽: 보드맵 + 리셋 버튼
        right_wrap = QWidget()
        right_vbox = QVBoxLayout(right_wrap)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(6)

        self.html_card = Card("Board Map")
        # reset 버튼은 보드맵 카드 안에 넣음
        self.btn_reset = QPushButton("Reset board")
        self.btn_reset.clicked.connect(self.on_reset_board)
        self.html_card.body.addWidget(self.btn_reset)

        # 배경 on/off 체크박스
        self.chk_bg = QCheckBox("Show PCB background")
        self.chk_bg.setChecked(True)  # 기본은 켜진 상태
        self.chk_bg.toggled.connect(self.on_toggle_bg_background)
        self.html_card.body.addWidget(self.chk_bg)

        self.web = QWebEngineView()
        self.html_card.body.addWidget(self.web)

        right_vbox.addWidget(self.html_card)
        top.addWidget(right_wrap)

        # 위쪽 비율: Live : Board = 4 : 6 정도
        top.setStretchFactor(0, 4)
        top.setStretchFactor(1, 6)

        # -------------------------------------------------
        # (2) 아래쪽: Logs + Click board map (가로로 나란히)
        # -------------------------------------------------
        bottom_wrap = QWidget()
        bottom_hbox = QHBoxLayout(bottom_wrap)
        bottom_hbox.setContentsMargins(0, 0, 0, 0)
        bottom_hbox.setSpacing(6)

        # 2-1. Logs (왼쪽)
        self.log_card = Card("Logs")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log_card.body.addWidget(self.log)
        bottom_hbox.addWidget(self.log_card, 4)   # ← 여기 숫자 키우면 더 넓어짐

        # 2-2. Click board map (오른쪽)
        self.click_card = Card("Click Board map")

        # 제목/설명 라벨 (위)
        self.click_title_label = QLabel("Selected: -")
        self.click_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.click_card.body.addWidget(self.click_title_label)

        # 실제 이미지가 뜨는 영역 (아래)
        self.click_img_label = QLabel("보드에서 부품을 클릭하면 여기 표시됩니다.")
        self.click_img_label.setAlignment(Qt.AlignCenter)
        self.click_img_label.setMinimumHeight(140)
        self.click_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.click_card.body.addWidget(self.click_img_label)

        bottom_hbox.addWidget(self.click_card, 6)

        root.addWidget(bottom_wrap)

        # 위/아래 비율
        root.setStretchFactor(0, 9)   # 위쪽
        root.setStretchFactor(1, 2)   # 아래쪽

        # QSS 적용
        self._apply_qss()

        # -------------------------------------------------
        # 웹채널 준비 (JS → Python 클릭 신호 받기)
        # -------------------------------------------------
        self._board_bridge = BoardClickBridge()
        self._board_bridge.clicked.connect(self.on_board_clicked)

        self.channel = QWebChannel()
        self.channel.registerObject("qtBoard", self._board_bridge)
        self.web.page().setWebChannel(self.channel)

        # 보드맵 HTML 로드
        self._load_boardmap()

        # -------------------------------------------------
        # 백그라운드 추론 워커 시작
        # -------------------------------------------------
        self.worker = InferenceWorker(self.cfg)
        self.worker.image_ready.connect(self.on_image_ready)
        self.worker.log_ready.connect(self.on_log)
        self.worker.pred_ready.connect(self.on_pred)
        self.worker.start()

    # =========================================================
    #                    내부 함수들
    # =========================================================

    def _apply_qss(self):
        qss_path = os.path.join(os.path.dirname(__file__), "theme.qss")
        if os.path.exists(qss_path):
            self.setStyleSheet(open(qss_path, "r", encoding="utf-8").read())

    def _load_boardmap(self):
        try:
            path = build_boardmap_html(
                pnp_path=self.cfg["pnp_path"],
                out_html=self.cfg["html_out"],
                title="PnP Live Boardmap",
                color_ok=self.cfg["color_ok"],
                color_ng=self.cfg["color_ng"],
                color_neutral=self.cfg["color_neutral"],
                bg_image_path=self.cfg.get("board_bg"),
            )
            self.web.load(QUrl.fromLocalFile(os.path.abspath(path)))
            # 로드가 끝난 뒤에만 JS 를 보낼 수 있도록
            self.web.loadFinished.connect(self._on_web_loaded)
            self.on_log(f"[ui] Board map loaded: {path}")
        except Exception as e:
            self.on_log(f"[ui] 보드맵 로드 실패: {e}")

    def _on_web_loaded(self, ok: bool):
        self._web_ready = ok
        if ok and self._pending_js:
            for js in self._pending_js:
                self.web.page().runJavaScript(js)
            self._pending_js.clear()

    def _set_preview_pixmap(self, pix: QPixmap):
        self._last_pix = pix
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, e):
        if self._last_pix is not None:
            self._set_preview_pixmap(self._last_pix)
        return super().resizeEvent(e)

    # =========================================================
    #                    Reset board
    # =========================================================
    def on_reset_board(self):
        ret = QMessageBox.question(
            self,
            "Reset board",
            "보드맵 색상을 모두 초기화할까요?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return

        # 1) 보드맵 JS 리셋
        if self._web_ready:
            js = """
            if (window.PNP && window.PNP.resetAll) {
                window.PNP.resetAll();
            }
            """
            self.web.page().runJavaScript(js)
        else:
            self.on_log("[ui] board not ready yet, reset skipped")

        # 2) 파이썬 쪽 상태도 같이 리셋
        self._shot_cache.clear()
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("이미지 없음")

        self.click_img_label.setPixmap(QPixmap())
        self.click_img_label.setText("보드에서 부품을 클릭하면 여기 표시됩니다.")
        self.click_title_label.setText("Selected: -")

        self.on_log("[ui] board reset")
    # -------------------------------------------------
    # PCB 배경 on/off 토글
    # -------------------------------------------------
    @Slot(bool)
    def on_toggle_bg_background(self, checked: bool):
        js = (
            "if (window.PNP && window.PNP.setBgVisible) {"
            f"  window.PNP.setBgVisible({str(checked).lower()});"
            "}"
        )
        if self._web_ready:
            self.web.page().runJavaScript(js)
        else:
            self._pending_js.append(js)
    # =========================================================
    #        워커 → 이미지 들어왔을 때
    # =========================================================
    @Slot(object, dict)
    def on_image_ready(self, img, meta):
        # 원본 저장
        self._last_bgr = img.copy()

        # 왼쪽 미리보기 갱신
        pix = cvimg_to_qpix(img)
        self._set_preview_pixmap(pix)

        # 디자인레이터 캐시 (클릭해서 다시 보려고)
        des = meta.get("designator")
        if des:
            self._shot_cache[des.upper()] = img.copy()

        if des:
            self.preview_label.setToolTip(des)

    # =========================================================
    #        워커 → 로그 들어왔을 때
    # =========================================================
    @Slot(str)
    def on_log(self, text: str):
        self.log.append(text)
        self.log.moveCursor(QTextCursor.End)

    # =========================================================
    #        워커 → 예측 결과 들어왔을 때
    # =========================================================
    @Slot(str, int, float)
    def on_pred(self, designator: str, pred: int, prob: float):
        # 1) 보드맵 색 바꾸는 JS 보내기
        js = f"PNP.setState('{designator}', {int(pred)});"
        if self._web_ready:
            self.web.page().runJavaScript(js)
        else:
            self._pending_js.append(js)

        # 2) 왼쪽 프리뷰에도 테두리 + 텍스트
        if self._last_bgr is None:
            return
        bgr = self._last_bgr.copy()
        h, w = bgr.shape[:2]
        color = self.COLOR_NG_BGR if pred == 1 else self.COLOR_OK_BGR
        t = 5
        bgr[:t, :, :] = color
        bgr[h - t:h, :, :] = color
        bgr[:, :t, :] = color
        bgr[:, w - t:w, :] = color
        cv2.putText(
            bgr,
            ("NG" if pred == 1 else "PASS") + f" p={prob:.3f}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        pix2 = cvimg_to_qpix(bgr)
        self._set_preview_pixmap(pix2)

    # =========================================================
    #        보드 클릭(JS → Python)
    # =========================================================
    @Slot(str)
    def on_board_clicked(self, des: str):
        des_up = des.upper()
        self.click_title_label.setText(f"Selected: {des_up}")

        img = self._shot_cache.get(des_up)
        if img is None:
            # 아직 이 부품은 찍힌 이미지가 없음
            self.click_img_label.setPixmap(QPixmap())
            self.click_img_label.setText(f"{des_up} : image not captured yet.")
            return

        pix = cvimg_to_qpix(img)
        scaled = pix.scaled(self.click_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.click_img_label.setPixmap(scaled)
        self.click_img_label.setText("")  # 텍스트는 지움
        self.click_img_label.setToolTip(des_up)

    # =========================================================
    #                    종료 처리
    # =========================================================
    def closeEvent(self, e):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        super().closeEvent(e)
