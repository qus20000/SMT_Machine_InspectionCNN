import os, cv2, json, shutil 
import numpy as np
from PySide6.QtCore import Qt, QSize, Slot, QUrl, QObject, Signal
from PySide6.QtGui import QPixmap, QImage, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QTextEdit, QSplitter, QFrame,
    QSizePolicy, QPushButton, QMessageBox, QWidget, QHBoxLayout,QCheckBox, QComboBox
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel

from worker_infer import InferenceWorker
from pnp_html import build_boardmap_html
from worker_infer import crop_center, InferenceWorker

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
        ##self.board_dir = None 이거 지워야할까봐 일단 주석처리해둠

         # 완료된 보드들이 저장될 최상위 폴더 (기본값: ./Dataset/inference_output)
        self.outdir_base = os.path.abspath(
              cfg.get("watch_image_dir", "./Dataset/inference_output")
        )
        os.makedirs(self.outdir_base, exist_ok=True)

        # 현재 보고 있는 보드의 폴더 (기본은 실시간 폴더)
        self.board_dir = self.outdir_base
        
         # 윈도우 설정
        self.setWindowTitle("SMT Live Dashboard (v10)")
        self.resize(1280, 768)

        # 상태 변수들
        self._web_ready = False
        self._pending_js: list[str] = []
        self._last_pix = None
        self._last_bgr = None

        # 디자인별로 찍힌 이미지 임시 저장 (보드 클릭해서 다시 보기용)
        self._shot_cache: dict[str, any] = {}
        self._pred_cache: dict[str, tuple[int, float]] = {}
        self._finished_board_paths: list[str] = []

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
        header_row = QHBoxLayout()
        self.click_title_label = QLabel("Selected: -")
        self.click_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_row.addWidget(self.click_title_label)

        header_row.addStretch()

        header_row.addWidget(QLabel("Board:"))
        self.board_combo = QComboBox()
        self.board_combo.addItem("Current", userData=None)  # 0번: 현재 실시간 보드
        header_row.addWidget(self.board_combo)

        self.click_card.body.addLayout(header_row)

        # 실제 이미지가 뜨는 영역 (아래)
        self.click_img_label = QLabel("보드에서 부품을 클릭하면 여기 표시됩니다.")
        self.click_img_label.setAlignment(Qt.AlignCenter)
        self.click_img_label.setMinimumHeight(140)
        self.click_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.click_card.body.addWidget(self.click_img_label)

        bottom_hbox.addWidget(self.click_card, 6)

        root.addWidget(bottom_wrap)

        # 위/아래 비율
        root.setStretchFactor(0, 5)   # 위쪽
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

         # 완료된 보드 콤보박스 연결 + 초기 스캔
        self.board_combo.currentIndexChanged.connect(self.on_board_selected)
        self.refresh_board_list()

        # -------------------------------------------------
        # 백그라운드 추론 워커 시작
        # -------------------------------------------------
        self.worker = InferenceWorker(self.cfg)
        self.worker.image_ready.connect(self.on_image_ready)
        self.worker.log_ready.connect(self.on_log)
        self.worker.pred_ready.connect(self.on_pred)
        self.worker.start()

    def _scan_board_dirs(self, base: str):
        """
        상위 폴더(base) 밑에 있는 보드 폴더들(Board1, Board2...)을 스캔해서
        [(이름, 절대경로), ...] 리스트로 반환
        """
        boards: list[tuple[str, str]] = []

        if not os.path.isdir(base):
            return boards

        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            boards.append((name, path))

        return boards

    def _infer_dir(self) -> str:
        # config.json 에 "infer_dir" 키가 있으면 그걸 쓰고,
        # 없으면 기본값 ./Dataset/inference_output 사용
        return os.path.abspath(self.cfg.get("infer_dir", "./Dataset/inference_output"))
    
    # ========== 현재 보드를 파일로 정리 (저장 or 삭제) ==========
    def _finalize_current_board(self, save: bool):
        """
        save=True  : inference_output 에 있는 이미지를 새 BoardN 폴더로 이동 + result.json 저장
        save=False : inference_output 안의 현재 이미지들만 삭제
        """
        imgdir = self.outdir_base  # ./Dataset/inference_output 의 절대경로

        if not os.path.isdir(imgdir):
            self.on_log(f"[ui] watch dir not found: {imgdir}")
            return

        if save:
            # ---- 1) 다음 보드 번호 계산 (Board1, Board2, ...) ----
            boards = self._scan_board_dirs(self.outdir_base)
            next_idx = 1
            for name, _ in boards:
                if name.lower().startswith("board"):
                    # "Board12" → 12
                    num_part = "".join(ch for ch in name if ch.isdigit())
                    if num_part:
                        try:
                            n = int(num_part)
                            next_idx = max(next_idx, n + 1)
                        except ValueError:
                            pass

            board_name = f"Board{next_idx}"
            board_dir = os.path.join(self.outdir_base, board_name)
            os.makedirs(board_dir, exist_ok=True)

            # ---- 2) 이미지 파일들을 BoardN 폴더로 이동 ----
            moved = 0
            for fname in os.listdir(imgdir):
                src = os.path.join(imgdir, fname)
                if not os.path.isfile(src):
                    continue
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                dst = os.path.join(board_dir, fname)
                try:
                    shutil.move(src, dst)
                    moved += 1
                except Exception as e:
                    self.on_log(f"[ui] 파일 이동 실패: {src} -> {dst} ({e})")

            # ---- 3) 예측 결과를 result.json 으로 저장 ----
            if self._pred_cache:
                meta_path = os.path.join(board_dir, "result.json")
                try:
                    data = {
                        des: {"pred": int(p), "prob": float(prob)}
                        for des, (p, prob) in self._pred_cache.items()
                    }
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    self.on_log(f"[ui] saved board results: {meta_path}")
                except Exception as e:
                    self.on_log(f"[ui] 결과 JSON 저장 실패: {meta_path} ({e})")

            self.on_log(f"[ui] Board 저장: {board_name} ({moved} images)")
        else:
            # ---- 저장 안함: 현재 이미지들만 삭제 ----
            removed = 0
            for fname in os.listdir(imgdir):
                src = os.path.join(imgdir, fname)
                if not os.path.isfile(src):
                    continue
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                try:
                    os.remove(src)
                    removed += 1
                except Exception as e:
                    self.on_log(f"[ui] 파일 삭제 실패: {src} ({e})")

            self.on_log(f"[ui] 현재 보드 이미지 삭제 완료 (총 {removed}장)")

        # ---- 공통: 캐시 정리 + 보드 콤보 갱신 ----
        self._shot_cache.clear()
        self._pred_cache.clear()
        self._last_bgr = None
        self._last_pix = None

        self.refresh_board_list()

        self.preview_label.clear()
        self.preview_label.setText("이미지 없음")

    
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

     # ---------- 완료된 보드 목록 스캔 ----------
    def _scan_finished_boards(self):
        """
        inference_output 아래의 하위 폴더들을 '완료된 보드'로 간주해서
        (이름, 경로) 리스트로 리턴.
        """
        root_dir = self.cfg.get("inference_output", "./Dataset/inference_output")
        boards = []
        if not os.path.isdir(root_dir):
            return boards

        for name in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, name)
            if os.path.isdir(path):
                boards.append((name, path))
        return boards

     # ========== 저장된 보드 목록 갱신 ==========
    def refresh_board_list(self):
        """우측 하단 Board 콤보박스 리스트를 갱신"""
        if not hasattr(self, "board_combo"):
            return

        self.board_combo.blockSignals(True)
        self.board_combo.clear()

        # 0번: 현재 실시간 보드
        self.board_combo.addItem("Current", userData=None)

        # 나머지: 저장된 보드들
        boards = self._scan_board_dirs(self.outdir_base)
        for name, path in boards:
            self.board_combo.addItem(name, userData=path)

        # 기본 선택은 Current
        self.board_combo.setCurrentIndex(0)
        self.board_combo.blockSignals(False)

    @Slot(int)
    def on_board_selected(self, idx: int):
        """우측 Board 콤보박스에서 선택이 바뀌었을 때"""
        data = self.board_combo.itemData(idx)

        # 0번: Current → 실시간 모드로 복귀
        if data is None:
            self.board_dir = self.outdir_base
            self.on_log("[ui] switched to CURRENT board view")

            # 캐시/뷰/보드맵 초기화
            self._shot_cache.clear()
            self._pred_cache.clear()
            self._last_bgr = None
            self._last_pix = None

            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("이미지 없음")

            self.click_img_label.setPixmap(QPixmap())
            self.click_img_label.setText("보드에서 부품을 클릭하면 여기 표시됩니다.")
            self.click_title_label.setText("Selected: -")

            # 보드맵을 전부 회색으로
            js = "if (window.PNP && window.PNP.resetAll) { window.PNP.resetAll(); }"
            if self._web_ready:
                self.web.page().runJavaScript(js)
            else:
                self._pending_js.append(js)
            return

        # 그 외: 저장된 보드 폴더 로드
        folder = str(data)
        self.load_finished_board(folder)

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
    
    def _switch_to_current_board(self):
    #실시간(CURRENT) 보드 모드로 전환 + 상태 전부 리셋#
        self.board_dir = None
        self.on_log("[ui] switched to CURRENT board view")

    # 캐시/상태 초기화
        self._shot_cache.clear()
        self._pred_cache.clear()
        self._last_bgr = None
        self._last_pix = None

    # 프리뷰/클릭뷰 초기화
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("이미지 없음")

        self.click_img_label.setPixmap(QPixmap())
        self.click_img_label.setText("보드에서 부품을 클릭하면 여기 표시됩니다.")
        self.click_title_label.setText("Selected: -")

    # 보드맵 색 전부 회색으로
        reset_js = "if (window.PNP && window.PNP.resetAll) { window.PNP.resetAll(); }"
        if self._web_ready:
            self.web.page().runJavaScript(reset_js)
        else:
            self._pending_js.append(reset_js)
    # =========================================================
    #                    Reset board
    # =========================================================
    def on_reset_board(self):
        # 0) 어떻게 할지 먼저 물어보기
        msg = QMessageBox(self)
        msg.setWindowTitle("Reset board")
        msg.setText("보드맵을 초기화합니다.")
        msg.setInformativeText(
            "이 보드를 완료된 보드로 저장하시겠습니까?\n\n"
            "Yes    : 완료된 보드로 저장 (Board1, Board2 ...)\n"
            "No     : 이번에 찍힌 이미지만 삭제하고 다시 시작\n"
            "Cancel : 아무 작업도 하지 않음"
        )
        msg.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        msg.setDefaultButton(QMessageBox.No)
        ret = msg.exec()

        if ret == QMessageBox.Cancel:
            return

         # 0) infer worker 히스토리 리셋
        try:
        # self.worker 혹은 self.infer_worker 이름 확인해서 사용
            if hasattr(self, "worker") and hasattr(self.worker, "reset_history"):
                self.worker.clear_seen()
                self.on_log("[ui] infer worker history reset")
        except Exception as e:
            self.on_log(f"[ui] failed to reset worker history: {e}")

        # 1) 이미지 저장/삭제 처리
        try:
            if ret == QMessageBox.Yes:
                # 완료된 보드로 저장
                self._finalize_current_board(save=True)
            elif ret == QMessageBox.No:
                # 이번 이미지들만 삭제
                self._finalize_current_board(save=False)
        except Exception as e:
            self.on_log(f"[ui] board finalize error: {e}")

        # 워커의 seen 도 같이 리셋
        if self.worker is not None:
            self.worker.clear_seen()

        # 2) 파이썬 쪽 상태 리셋
        self._shot_cache.clear()
        self._pred_cache.clear()
        self._last_bgr = None
        self._last_pix = None

        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("이미지 없음")

        self.click_img_label.setPixmap(QPixmap())
        self.click_img_label.setText("보드에서 부품을 클릭하면 여기 표시됩니다.")
        self.click_title_label.setText("Selected: -")

        # 3) 보드맵 전체 초기화 (모두 회색)
        js = "if (window.PNP && window.PNP.resetAll) { window.PNP.resetAll(); }"
        if self._web_ready:
            self.web.page().runJavaScript(js)
        else:
            self._pending_js.append(js)

        self.on_log("[ui] board reset")

     # ========== 저장된 보드 불러오기 ==========
    def load_finished_board(self, folder: str):
        """
        완료된 보드 폴더를 기준으로
        - _shot_cache 를 폴더 이미지들로 채우고
        - result.json 이 있으면 보드맵 색상 복원
        """
        self.board_dir = folder
        self.on_log(f"[ui] loading finished board: {folder}")

        # 1) 보드맵 색상 전부 회색으로 초기화
        if self._web_ready:
            js = "if (window.PNP && window.PNP.resetAll) { window.PNP.resetAll(); }"
            self.web.page().runJavaScript(js)
        else:
            self._pending_js.append(
                "if (window.PNP && window.PNP.resetAll) { window.PNP.resetAll(); }"
            )

        # 2) 캐시/뷰 초기화
        self._shot_cache.clear()
        self._pred_cache.clear()
        self._last_bgr = None
        self._last_pix = None

        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("이미지 없음")

        self.click_img_label.setPixmap(QPixmap())
        self.click_img_label.setText("보드에서 부품을 클릭하면 여기 표시됩니다.")
        self.click_title_label.setText("Selected: -")

        # 3) 폴더에서 이미지 읽어서 _shot_cache 채우기
        if not os.path.isdir(folder):
            self.on_log(f"[ui] board folder not found: {folder}")
            return

        count = 0
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fpath = os.path.join(folder, fname)

            try:
                data = np.fromfile(fpath, np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception as e:
                self.on_log(f"[ui] failed to load {fpath}: {e}")
                continue
            if img is None:
                continue

            des = os.path.splitext(fname)[0].upper()  # "C14.png" -> "C14"
            self._shot_cache[des] = img
            count += 1

        self.on_log(
            f"[ui] finished board loaded: {os.path.basename(folder)} "
            f"({count} components in cache)"
        )

        # 4) result.json 이 있으면 보드맵 색상 복원
        result_path = os.path.join(folder, "result.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    data = json.load(f)  # { "C9": {"pred": 1, "prob": 0.99}, ... }

                self._pred_cache.clear()
                for des, info in data.items():
                    self._pred_cache[des.upper()] = (
                        int(info.get("pred", 0)),
                        float(info.get("prob", 0.0)),
                    )

                self._apply_pred_cache_to_boardmap()
                self.on_log(f"[ui] loaded board result: {result_path}")
            except Exception as e:
                self.on_log(f"[ui] failed to load board result: {e}")
        else:
            self.on_log(
                f"[ui] no result.json for {os.path.basename(folder)} "
                "(boardmap stays neutral)"
            )
    
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
        des_up = designator.upper()

    # 0) 현재 보드 상태 캐시에 저장 (나중에 result.json 저장·불러오기용)
        self._pred_cache[des_up] = (int(pred), float(prob))

    # 1) 보드맵 색 바꾸는 JS 보내기
        js = f"PNP.setState('{designator}', {int(pred)});"
        if self._web_ready:
            self.web.page().runJavaScript(js)
        else:
            self._pending_js.append(js)

    # 2) 왼쪽 Live Preview에도 테두리 + 텍스트 (공통 함수 사용)
        if self._last_bgr is None:
            return

        annotated = self._draw_result_overlay(self._last_bgr, pred, prob)
        if annotated is None:
            return

        pix2 = cvimg_to_qpix(annotated)
        self._set_preview_pixmap(pix2)
        # ================= 공통: PASS/NG 오버레이 그리기 =================
    def _draw_result_overlay(self, bgr, pred: int, prob: float):
        """
        bgr 이미지에 PASS/NG 테두리와 텍스트를 그려서 새로운 이미지를 리턴
        """
        if bgr is None:
            return None

        img = bgr.copy()
        h, w = img.shape[:2]

        color = self.COLOR_NG_BGR if pred == 1 else self.COLOR_OK_BGR
        t = 5  # 테두리 두께

        # 테두리
        img[:t, :, :] = color
        img[h - t:h, :, :] = color
        img[:, :t, :] = color
        img[:, w - t:w, :] = color

        # 텍스트
        text = ("NG" if pred == 1 else "PASS") + f" p={prob:.3f}"
        cv2.putText(
            img,
            text,
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        return img
 # ========== 캐시된 보드 상태를 보드맵에 적용 ==========
    def _apply_pred_cache_to_boardmap(self):
        """_pred_cache 에 저장된 예측 결과대로 보드맵 색상 다시 칠하기"""
        # 0) 전부 회색으로 리셋
        reset_js = "if (window.PNP && window.PNP.resetAll) { window.PNP.resetAll(); }"
        if self._web_ready:
            self.web.page().runJavaScript(reset_js)
        else:
            self._pending_js.append(reset_js)

        # 1) 캐시에 있는 결과대로 다시 색칠
        for des, (pred, prob) in self._pred_cache.items():
            js = f"PNP.setState('{des}', {int(pred)});"
            if self._web_ready:
                self.web.page().runJavaScript(js)
            else:
                self._pending_js.append(js)
    # =========================================================
    #        보드 클릭(JS → Python)
    # =========================================================
    @Slot(str)
    def on_board_clicked(self, designator: str):
    #"""
    #보드맵에서 부품을 클릭했을 때:
    #- _shot_cache 에 저장된 원본 이미지를 찾고
    #- worker_infer.py 의 crop_center 규칙으로 중앙 크롭
    #- 예측 결과가 있으면 테두리 + PASS/NG + prob 오버레이
    #- Click Board map 에 표시
    #"""
        des = (designator or "").strip().upper()
        base_title = f"Selected: {des}"
        self.on_log(f"[debug] board click: {des}")

        if not des:
        # 잘못된 클릭 혹은 공백
            self.click_title_label.setText("Selected: -")
            self.click_img_label.setPixmap(QPixmap())
            self.click_img_label.setText("보드에서 부품을 클릭하면 여기 표시됩니다.")
            return

        ase_title = f"Selected: {des}"

    # 1) 이 부품에 해당하는 이미지가 캐시에 있는지
        img = self._shot_cache.get(des)
        if img is None:
        # 아직 안 찍힌 부품
            self.click_img_label.setPixmap(QPixmap())
            self.click_img_label.setText(f"{des}: image not captured yet.")
            self.click_title_label.setText(base_title)
            return

    # 2) worker_infer.py 에서 쓰던 크롭 규칙 그대로 적용
        roi_w = int(self.cfg.get("roi_w", img.shape[1]))
        roi_h = int(self.cfg.get("roi_h", img.shape[0]))
        # 크롭 범위 살짝 확장 (10~20% 정도)
        roi_w = int(roi_w * 1.2)
        roi_h = int(roi_h * 1.2)

        patch = crop_center(img, roi_w, roi_h)

        

    # 3) 예측 결과가 있으면 테두리 + 텍스트 오버레이
        info = self._pred_cache.get(des)
        if info is not None:
            pred, prob = info
            patch = self._draw_result_overlay(patch, pred, prob)
            title = f"Selected: {des} ({'NG' if pred == 1 else 'PASS'} p={float(prob):.4f})"
        else:
            title = base_title

    # 4) QPixmap 으로 변환해서 라벨에 표시
        pix = cvimg_to_qpix(patch)
        self.click_img_label.setPixmap(pix)
        self.click_img_label.setText("")      # 텍스트는 비우고
        self.click_title_label.setText(title)

    # =========================================================
    #                    종료 처리
    # =========================================================
    def closeEvent(self, e):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        super().closeEvent(e)
