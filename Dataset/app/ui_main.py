import os, cv2, json, shutil, re # 25/11/08 03:04 수정사항 (숫자 정렬을 위해 re 모듈 임포트)
import time
import numpy as np
from PySide6.QtCore import Qt, QSize, Slot, QUrl, QObject, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QTextEdit, QSplitter, QFrame,
    QSizePolicy, QPushButton, QMessageBox, QWidget, QHBoxLayout,QCheckBox, QComboBox, QProgressBar
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


# ------------------ Web <-> Qt 브리지 ------------------ #
class BoardClickBridge(QObject):
    clicked = Signal(str)

    @Slot(str)
    def onBoardClick(self, des: str):
        self.clicked.emit(des)


# 25/11/07 23:03 수정사항 (QSplitter 드래그 시 이미지 크기 조절을 위한 커스텀 QLabel 클래스 추가)
class ScaledPixmapLabel(QLabel):
    """
    QSplitter 등으로 위젯 크기가 변경될 때, QPixmap을 자동으로
    비율에 맞게 리사이징하는 QLabel 서브클래스.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._full_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter) # 25/11/07 23:03 수정사항 (생성자에서 기본값 설정)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # 25/11/07 23:03 수정사항 (생성자에서 기본값 설정)
        # 25/11/07 23:14 수정사항 (LivePreview 축소 시 끊김 현상 해결을 위해 최소 크기를 1x1로 설정)
        # 25/11/07 23:14 수정사항 (이 코드는 QLabel이 QPixmap의 sizeHint를 무시하고 레이아웃을 따라 작아지도록 강제합니다.)
        self.setMinimumSize(1, 1)

    def setPixmap(self, pix: QPixmap | None):
        """
        (재정의) 원본 QPixmap을 저장하고, 리사이징된 버전을 표시합니다.
        None이나 빈 QPixmap이 들어오면 비웁니다.
        """
        if pix is None or pix.isNull():
            self._full_pixmap = None
            # 25/11/07 23:03 수정사항 (QLabel.setPixmap 호출을 명확히 하기 위해 super() 사용)
            super().setPixmap(QPixmap()) # 라벨을 비웁니다.
        else:
            self._full_pixmap = pix
            self._update_scaled_pixmap() # 현재 크기에 맞춰 즉시 업데이트합니다.

    def clear(self):
        """
        (재정의) QPixmap과 텍스트를 모두 비웁니다.
        """
        self._full_pixmap = None
        super().clear()

    def _update_scaled_pixmap(self):
        """
        내부 함수: 저장된 원본 QPixmap을 현재 위젯 크기에 맞게
        스케일링하여 `super().setPixmap`으로 표시합니다.
        """
        if self._full_pixmap:
            # 25/11/07 23:03 수정사항 (현재 라벨의 크기(self.size())에 맞춰 픽스맵을 스케일링)
            scaled_pix = self._full_pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pix) # 25/11/07 23:03 수정사항 (재귀 호출을 피하기 위해 super().setPixmap 사용)

    def resizeEvent(self, e):
        """
        (재정의) 위젯의 크기가 변경될 때마다(예: 스플리터 조작) 호출됩니다.
        """
        # 25/11/07 23:03 수정사항 (부모의 resizeEvent를 먼저 호출)
        super().resizeEvent(e) 
        if self._full_pixmap:
            # 25/11/07 23:03 수정사항 (크기가 변경되었으므로, 저장된 원본 픽스맵을 새 크기에 맞게 리스케일링)
            self._update_scaled_pixmap()


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
        self._last_pix = None # 25/11/07 23:03 수정사항 (MainWindow.resizeEvent는 제거했지만, _last_pix는 on_pred에서 마지막 이미지를 참조하기 위해 유지)
        self._last_bgr = None

        # 보드 진행률 관련
        self._all_des = []            # 전체 디자인레이터 리스트 (C1..R120)
        self._board_total = 0         # 현재 보드 전체 부품 수

        # 디자인별로 찍힌 이미지 임시 저장 (보드 클릭해서 다시 보기용)
        self._shot_cache: dict[str, any] = {}
        self._pred_cache: dict[str, tuple[int, float]] = {}
        self._finished_board_paths: list[str] = []

        self._all_designators: set[str] = set()   # 보드 전체 소자 집합
        self._seen_designators: set[str] = set()  # 이번 보드에서 이미 판정된 소자
        self._board_completed = False             # 이번 보드 완료 플래그

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
        
        # 25/11/07 23:03 수정사항 (QLabel을 ScaledPixmapLabel로 교체)
        self.preview_label = ScaledPixmapLabel("이미지 없음")
        # 25/11/07 23:03 수정사항 (setAlignment와 setSizePolicy는 ScaledPixmapLabel 생성자에서 처리되므로 제거)
        # self.preview_label.setAlignment(Qt.AlignCenter)
        # self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.preview_card.body.addWidget(self.preview_label)
        top.addWidget(self.preview_card)

        # 1-2. 오른쪽: 보드맵 + 리셋 버튼
        right_wrap = QWidget()
        right_vbox = QVBoxLayout(right_wrap)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(6)

        self.html_card = Card("Board Map")
        
        # 25/11/07 22:28 수정사항 (범례(legend)를 Reset 버튼과 같은 행에 넣기 위해, make_color_label 헬퍼 함수를 이 위치로 이동)
        def make_color_label(color, text):
            lbl_color = QLabel()
            lbl_color.setFixedSize(16, 16)
            lbl_color.setStyleSheet(f"background-color: {color}; border: 1px solid #999;")
            lbl_text = QLabel(text)
            hbox = QHBoxLayout()
            hbox.setSpacing(4)
            hbox.addWidget(lbl_color)
            hbox.addWidget(lbl_text)
            widget = QWidget()
            widget.setLayout(hbox)
            return widget
        
        # 25/11/07 22:28 수정사항 (Reset 버튼, 범례, BG 토글을 같은 행에 넣기 위한 QHBoxLayout 생성 및 재배치)
        button_toggle_row = QHBoxLayout()

        # 1. Reset 버튼 (왼쪽)
        self.btn_reset = QPushButton("Reset board")
        self.btn_reset.clicked.connect(self.on_reset_board)
        button_toggle_row.addWidget(self.btn_reset)

        # 2. 왼쪽 공백 (범례를 중앙으로 밀기)
        # 25/11/07 22:28 수정사항 (왼쪽 stretch 추가하여 범례를 중앙으로 밀어냄)
        button_toggle_row.addStretch(1)

        # 3. 범례 (중앙)
        # 25/11/07 22:28 수정사항 (범례 위젯들을 button_toggle_row에 직접 추가)
        button_toggle_row.addWidget(make_color_label("#00FF00", "PASS"))
        button_toggle_row.addWidget(make_color_label("#FF0000", "NG"))
        button_toggle_row.addWidget(make_color_label("#808080", "Not inspected"))

        # 4. 오른쪽 공백 (토글을 오른쪽으로 밀기)
        # 25/11/07 22:28 수정사항 (오른쪽 stretch 추가하여 토글 버튼을 맨 우측으로 밀어냄)
        button_toggle_row.addStretch(1)
    
        # 진행률 표시줄 (Logs 영역으로 이동됨)
        progress_row = QHBoxLayout()
        self.board_progress_label = QLabel("Board progress: 0 / 0")
        self.board_progress = QProgressBar()
        self.board_progress.setRange(0, 100)
        self.board_progress.setValue(0)
        self.board_progress.setTextVisible(True)

        progress_row.addWidget(self.board_progress_label)
        progress_row.addWidget(self.board_progress)

        # 5. 배경 on/off 체크박스 (오른쪽)
        self.chk_bg = QCheckBox("Show PCB background")
        self.chk_bg.setChecked(True)  # 기본은 켜진 상태
        self.chk_bg.toggled.connect(self.on_toggle_bg_background)
        # 25/11/07 22:28 수정사항 (BG 토글 체크박스를 레이아웃의 맨 오른쪽에 추가)
        button_toggle_row.addWidget(self.chk_bg)

        # 25/11/07 22:28 수정사항 (버튼/범례/토글 가로 레이아웃을 html_card body에 추가)
        self.html_card.body.addLayout(button_toggle_row)

        self.web = QWebEngineView()
        self.html_card.body.addWidget(self.web)

        # 25/11/07 22:28 수정사항 (범례(legend_row)가 button_toggle_row로 이동했으므로 이 섹션의 코드를 모두 제거)
        
        right_vbox.addWidget(self.html_card)
        top.addWidget(right_wrap)

        # 위쪽 비율: Live : Board = 4 : 6 정도
        top.setStretchFactor(0, 7)
        top.setStretchFactor(1, 3)

         # 2-1. Logs (왼쪽)
        self.log_card = Card("Logs")
        
        # 25/11/07 22:05 수정사항 (progress_row 레이아웃을 Logs 카드의 body 최상단으로 이동)
        self.log_card.body.addLayout(progress_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log_card.body.addWidget(self.log)
        
        # 2-2. Click board map (오른쪽)
        self.click_card = Card("Click Board map")

        # ----- Click Board map 상단 헤더 행 -----
        header_row = QHBoxLayout()

        # Selected 라벨
        self.click_title_label = QLabel("Selected: -")
        self.click_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_row.addWidget(self.click_title_label)

        header_row.addStretch(1)

        # Board 콤보박스
        header_row.addWidget(QLabel("Board:"))
        self.board_combo = QComboBox()
        self.board_combo.addItem("Current", userData=None)  # 0번: 현재 실시간 보드
        header_row.addWidget(self.board_combo)

        # Board result 
        self.btn_board_result = QPushButton("Board result")
        self.btn_board_result.clicked.connect(self.on_board_result_clicked)
        header_row.addWidget(self.btn_board_result)

        # All results 버튼
        self.btn_all_result = QPushButton("All results")
        self.btn_all_result.clicked.connect(self.on_all_result_clicked)
        header_row.addWidget(self.btn_all_result)

        # 실제 이미지가 뜨는 영역 (아래)
        # 25/11/07 23:03 수정사항 (QLabel을 ScaledPixmapLabel로 교체)
        self.click_img_label = ScaledPixmapLabel("보드에서 부품을 클릭하면 여기 표시됩니다.")
        # 25/11/07 23:03 수정사항 (setAlignment와 setSizePolicy는 ScaledPixmapLabel 생성자에서 처리되므로 제거)
        # self.click_img_label.setAlignment(Qt.AlignCenter)
        
        # 25/11/07 23:14 수정사항 (setMinimumSize(1, 1)이 생성자에서 호출되므로, 이 고정 높이 설정을 제거해야 함)
        # self.click_img_label.setMinimumHeight(140) 
        
        # self.click_img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # ----- Click Board map 카드 전체 레이아웃 -----
        click_layout = self.click_card.body  
        self.click_card.body.addLayout(header_row)
        self.click_card.body.addWidget(self.click_img_label)

        # -------------------------------------------------
        # (2) 아래쪽: Logs + Click board map (가로로 나란히)
        # -------------------------------------------------
        bottom_split = QSplitter(Qt.Horizontal)
        bottom_split.addWidget(self.log_card)    # 왼쪽: Logs
        bottom_split.addWidget(self.click_card)  # 오른쪽: Click Board map

# 초기 비율 (원래 4:6이었으니 비슷하게)
        bottom_split.setStretchFactor(0, 4)
        bottom_split.setStretchFactor(1, 6)

        root.addWidget(bottom_split)
        
        # 위/아래 비율
        root.setStretchFactor(0, 3)   # 위쪽
        root.setStretchFactor(1, 2)   # 아래쪽

        # QSS 적용
        self._apply_qss()

        # -------------------------------------------------
        # 웹채널 준비 (JS -> Python 클릭 신호 받기)
        # -------------------------------------------------
        self._board_bridge = BoardClickBridge()
        self._board_bridge.clicked.connect(self.on_board_clicked)

        self.channel = QWebChannel()
        self.channel.registerObject("qtBoard", self._board_bridge)
        self.web.page().setWebChannel(self.channel)

        # 보드맵 HTML 로드
        self._load_boardmap()
         # 보드맵 카드만 여백 0으로
        layout = self.html_card.layout()
        if layout is not None:
            layout.setContentsMargins(0, 0, 0, 0)
        self.html_card.body.setContentsMargins(0, 0, 0, 0)

        # boardmeta.json 읽기
        meta_path = os.path.join(os.path.dirname(self.cfg["html_out"]), "boardmeta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._all_designators = {str(d).upper() for d in meta.get("designators", [])}
            self.on_log(f"[ui] board meta loaded ({len(self._all_designators)} components)")
        except Exception as e:
            self._all_designators = set()
            self.on_log(f"[ui] board meta load failed: {e}")
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

    # 25/11/08 03:04 수정사항 (Board1, Board10, Board2... 문제를 해결하기 위해 숫자 기준 정렬로 변경)
    def _scan_board_dirs(self, base: str):
        """
        상위 폴더(base) 밑에 있는 보드 폴더들(Board1, Board2...)을 스캔해서
        [(이름, 절대경로), ...] 리스트로 반환 (숫자 오름차순 정렬)
        """
        # 25/11/08 03:04 수정사항 (Board1, Board10, Board2... 문제를 해결하기 위해 숫자 기준 정렬로 변경)
        board_dir_pattern = re.compile(r"Board(\d+)")
        
        boards_with_keys: list[tuple[int, str, str]] = []

        if not os.path.isdir(base):
            return []

        for name in os.listdir(base):
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            
            match = board_dir_pattern.match(name)
            if match:
                try:
                    # 25/11/08 03:04 수정사항 (숫자 부분을 key로 추출)
                    key = int(match.group(1))
                    boards_with_keys.append((key, name, path))
                except ValueError:
                    pass # 25/11/08 03:04 수정사항 (숫자 변환 실패 시 무시)
            
        # 25/11/08 03:04 수정사항 (숫자(key) 기준으로 오름차순 정렬)
        boards_with_keys.sort(key=lambda item: item[0])
        
        # 25/11/08 03:04 수정사항 (정렬된 BoardN 리스트를 (이름, 경로) 튜플 리스트로 변환하여 반환)
        sorted_boards = [(name, path) for key, name, path in boards_with_keys]
        
        return sorted_boards

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
            
            # 25/11/08 03:04 수정사항 (숫자 정렬된 _scan_board_dirs를 활용하여 다음 인덱스를 효율적으로 계산)
            if boards: # 25/11/08 03:04 수정사항 (보드 리스트가 비어있지 않다면)
                last_board_name, _ = boards[-1] # 25/11/08 03:04 수정사항 (가장 마지막 보드 이름을 가져옴)
                num_part = "".join(ch for ch in last_board_name if ch.isdigit())
                if num_part:
                    try:
                        n = int(num_part)
                        next_idx = n + 1 # 25/11/08 03:04 수정사항 (마지막 번호 + 1)
                    except ValueError:
                        pass # 25/11/08 03:04 수정사항 (혹시 모를 오류 시 next_idx=1 사용)
            
            # 25/11/08 03:04 수정사항 (기존 for 루프를 주석 처리)
            # for name, _ in boards:
            #     if name.lower().startswith("board"):
            #         # "Board12" -> 12
            #         num_part = "".join(ch for ch in name if ch.isdigit())
            #         if num_part:
            #             try:
            #                 n = int(num_part)
            #                 next_idx = max(next_idx, n + 1)
            #             except ValueError:
            #                 pass

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

             # ---- 4) 보드별 요약 정보 boards_summary.json 에 저장 ----
            try:
                summary_path = os.path.join(self.outdir_base, "boards_summary.json")

                # 기존 요약 불러오기 (없으면 빈 리스트)
                boards = []
                if os.path.exists(summary_path):
                    with open(summary_path, "r", encoding="utf-8") as f:
                        boards = json.load(f)

                # OK / NG 개수 집계
                total = len(self._pred_cache)
                ok_cnt = sum(1 for _, (p, _) in self._pred_cache.items() if int(p) == 0)
                ng_cnt = sum(1 for _, (p, _) in self._pred_cache.items() if int(p) == 1)

                boards.append(
                    {
                        "name": board_name,
                        "dir": board_dir,
                        "total": total,
                        "ok": ok_cnt,
                        "ng": ng_cnt,
                    }
                )

                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(boards, f, ensure_ascii=False, indent=2)

                self.on_log(
                    f"[ui] Board 완료: {board_name}  OK={ok_cnt}  NG={ng_cnt}  total={total}"
                )
                self.on_log(f"[ui] updated boards summary: {summary_path}")
            except Exception as e:
                self.on_log(f"[ui] boards_summary.json 저장 실패: {e}")
        # ---- 5) 세션용 전체 로그파일에도 한 줄 기록 ----
            try:
                log_path = os.path.join(self.outdir_base, "boards_log.txt")
                with open(log_path, "a", encoding="utf-8") as f:
                    now = time.strftime("%Y-%m-%d %H:%M:%S")
                    total = len(self._pred_cache)
                    ok_cnt = sum(1 for _, (p, _) in self._pred_cache.items() if int(p) == 0)
                    ng_cnt = sum(1 for _, (p, _) in self._pred_cache.items() if int(p) == 1)
                    f.write(f"[{now}] {board_name} | OK={ok_cnt} | NG={ng_cnt} | total={total}\n")
                self.on_log(f"[ui] session log updated: {log_path}")
            except Exception as e:
                self.on_log(f"[ui] session log write failed: {e}")

        # ---- 최종 로그 ----
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
            self._load_boardmeta()
        except Exception as e:
            self.on_log(f"[ui] 보드맵 로드 실패: {e}")

    def _load_boardmeta(self):
    #"""
    #pnp_html 에서 저장한 boardmeta.json 읽어서
    #self._all_des / self._board_total 초기화.
    #"""
        try:
            html_out = self.cfg["html_out"]  # 예: ./Dataset/app/boardmap.html
            meta_path = os.path.join(os.path.dirname(html_out), "boardmeta.json")
            if not os.path.exists(meta_path):
                self.on_log(f"[ui] boardmeta.json not found: {meta_path}")
                self._all_des = []
                self._board_total = 0
                self._update_board_progress()
                return

            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            des_list = data.get("designators", [])
            self._all_des = [str(d).upper() for d in des_list]
            self._board_total = len(self._all_des)
            self._seen_designators.clear()
            self._board_completed = False
            self._update_board_progress()
            self.on_log(f"[ui] board meta loaded: {self._board_total} components")
        except Exception as e:
            self.on_log(f"[ui] failed to load board meta: {e}")
            self._all_des = []
            self._board_total = 0
            self._update_board_progress()
    
    def _update_board_progress(self):
        done = len(self._seen_designators)
        total = self._board_total if self._board_total > 0 else 0

        if total > 0:
            ratio = int(done * 100 / total)
        else:
            ratio = 0

        self.board_progress.setValue(ratio)
        self.board_progress_label.setText(f"Board progress: {done} / {total}")

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
        
        self.board_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)  # 내용 길이에 맞게 폭 조절
        self.board_combo.setMinimumContentsLength(8)                       # 최소 글자 수 기준
        self.board_combo.setMinimumWidth(120)                             # 혹시 모자를 때를 대비한 최소 폭
        # 0번: 현재 실시간 보드
        self.board_combo.addItem("Current", userData=None)

        # 나머지: 저장된 보드들
        boards = self._scan_board_dirs(self.outdir_base) # 25/11/08 03:04 수정사항 (이제 숫자 정렬된 리스트를 반환함)
        for name, path in boards:
            self.board_combo.addItem(name, userData=path)

        # 기본 선택은 Current
        self.board_combo.setCurrentIndex(0)
        self.board_combo.blockSignals(False)

    @Slot(int)
    def on_board_selected(self, idx: int):
        """우측 Board 콤보박스에서 선택이 바뀌었을 때"""
        data = self.board_combo.itemData(idx)

        # 0번: Current -> 실시간 모드로 복귀
        if data is None:
            self._seen_designators.clear()
            self._board_completed = False
            self._update_board_progress()

            self.board_dir = self.outdir_base
            self.on_log("[ui] switched to CURRENT board view")

            # 캐시/뷰/보드맵 초기화
            self._shot_cache.clear()
            self._pred_cache.clear()
            self._last_bgr = None
            self._last_pix = None

            self.preview_label.setPixmap(QPixmap()) # 25/11/07 23:03 수정사항 (ScaledPixmapLabel.setPixmap(None)이 호출됨)
            self.preview_label.setText("이미지 없음")

            self.click_img_label.setPixmap(QPixmap()) # 25/11/07 23:03 수정사항 (ScaledPixmapLabel.setPixmap(None)이 호출됨)
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
        # 25/11/07 23:03 수정사항 (ScaledPixmapLabel이 자동 리사이징을 하므로 이 줄이 필요 없음)
        # scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # 25/11/07 23:03 수정사항 (원본 픽스맵을 ScaledPixmapLabel의 setPixmap으로 전달)
        self.preview_label.setPixmap(pix)

    # 25/11/07 23:03 수정사항 (ScaledPixmapLabel이 자체 resizeEvent를 가지므로 MainWindow의 resizeEvent는 더 이상 필요 없음)
    # def resizeEvent(self, e):
    #     if self._last_pix is not None:
    #         self._set_preview_pixmap(self._last_pix)
    #     return super().resizeEvent(e)
    
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


    @Slot()
    def on_board_result_clicked(self):
    #"""현재 선택된 보드(CURRENT 또는 Board1 등)의 OK/NG/총 개수 요약을 팝업으로 표시"""
        idx = self.board_combo.currentIndex()
        data = self.board_combo.itemData(idx)
        board_name = self.board_combo.currentText()

    # 1) CURRENT 보드인 경우: self._pred_cache 기준으로 바로 계산
        if data is None:
            if not self._pred_cache:
                QMessageBox.information(
                    self,
                    "Board result",
                    "현재 보드에 저장된 예측 결과가 없습니다.",
                )
                return

            total = len(self._pred_cache)
            ok_cnt = sum(1 for (p, _prob) in self._pred_cache.values() if p == 0)
            ng_cnt = sum(1 for (p, _prob) in self._pred_cache.values() if p == 1)

            msg = (
                "현재 보드 결과\n\n"
                f"OK  : {ok_cnt}\n"
                f"NG  : {ng_cnt}\n"
                f"Total: {total}"
            )
            QMessageBox.information(self, "Board result", msg)
            self.on_log(f"[ui] current board result -> OK={ok_cnt}, NG={ng_cnt}, total={total}")
            return

    # 2) 완료된 보드인 경우: boards_summary.json 우선 참고
        folder = str(data)

        summary_path = os.path.join(self.outdir_base, "boards_summary.json")
        stats = None

        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    all_summary = json.load(f)
            # boards_summary.json 이 {"Board1": {...}, "Board2": {...}} 형태라고 가정
                if isinstance(all_summary, dict):
                    stats = all_summary.get(board_name)
            except Exception as e:
                self.on_log(f"[ui] failed to read boards_summary.json: {e}")

    # 3) summary 에서 못 찾으면, 해당 보드 폴더의 result.json 으로부터 즉석 계산
        if stats is None:
            result_path = os.path.join(folder, "result.json")
            if os.path.exists(result_path):
                try:
                    with open(result_path, "r", encoding="utf-8") as f:
                        data = json.load(f)  # {"C9": {"pred": 1, "prob": 0.99}, ...}

                    ok_cnt = sum(
                        1 for info in data.values()
                        if int(info.get("pred", 0)) == 0
                    )
                    ng_cnt = sum(
                        1 for info in data.values()
                        if int(info.get("pred", 0)) == 1
                    )
                    total = ok_cnt + ng_cnt
                    stats = {"ok": ok_cnt, "ng": ng_cnt, "total": total}
                except Exception as e:
                    self.on_log(f"[ui] failed to read result.json for board result: {e}")

    # 4) 그래도 없으면 안내
        if stats is None:
            QMessageBox.information(
                self,
                "Board result",
                f"{board_name} 에 대한 요약 정보를 찾을 수 없습니다.",
            )
            return

    # 5) 팝업으로 표시
        ok_cnt = int(stats.get("ok", 0))
        ng_cnt = int(stats.get("ng", 0))
        total = int(stats.get("total", ok_cnt + ng_cnt))
        ts = stats.get("timestamp", "")

        msg_lines = [
            f"{board_name} 결과",
            "",
            f"OK   : {ok_cnt}",
            f"NG   : {ng_cnt}",
            f"Total: {total}",
        ]
        if ts:
            msg_lines.append("")
            msg_lines.append(f"Time: {ts}")

        msg = "\n".join(msg_lines)
        QMessageBox.information(self, "Board result", msg)
        self.on_log(f"[ui] {board_name} result -> OK={ok_cnt}, NG={ng_cnt}, total={total}")

    @Slot()
    def on_all_result_clicked(self):
    #"""
    #지금까지 저장된 모든 보드(Board1, Board2, ...)에 대한
    #OK/NG/Total/시간 요약을 한 번에 보여주는 팝업
    #"""
        log_path = os.path.join(self.outdir_base, "boards_log.txt")

        if not os.path.exists(log_path):
            QMessageBox.information(
                self,
                "All results",
                "현재까지 저장된 보드 결과 로그(boards_log.txt)가 없습니다.",
            )
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
        except Exception as e:
            self.on_log(f"[ui] failed to read boards_log.txt: {e}")
            QMessageBox.warning(
                self,
                "All results",
                f"boards_log.txt 읽기 실패:\n{e}",
            )
            return

        if not txt:
            QMessageBox.information(
                self,
                "All results",
                "boards_log.txt 에 기록된 내용이 없습니다.",
            )
            return
        
        # 그대로 보여주기
        QMessageBox.information(self, "All results", txt)
        self.on_log(f"[ui] all board results shown from {log_path}")
   
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
                # 이번 촬영 이미지만 삭제 (다시 같은 보드 촬영)
                if hasattr(self, "_finalize_current_board"):
                    self._finalize_current_board(save=False)

    # SMT 머신에게 "이번 보드 중단/재시작" 알리는 finished.txt 생성
                    try:
                        flag_dir = self.cfg.get("watch_image_dir", "./Dataset/inference_output")
                        flag_path = os.path.join(flag_dir, "finished.txt")
                        with open(flag_path, "w", encoding="utf-8") as f:
                            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
                        self.on_log(f"[ui] finished flag created: {flag_path}")

                        # 4초(4000ms) 뒤에 자동 삭제
                        QTimer.singleShot(4000, lambda p=flag_path: self._remove_finished_flag(p))
                    except Exception as e:
                        self.on_log(f"[ui] failed to create finished.txt: {e}")

        except Exception as e:
            self.on_log(f"[ui] board finalize error: {e}")

        # 워커의 seen 도 같이 리셋
        if self.worker is not None:
            self.worker.clear_seen()

        # 2) 파이썬 쪽 상태 리셋
        self._shot_cache.clear()
        if hasattr(self, "_pred_cache"):
            self._pred_cache.clear()
        self._last_bgr = None
        self._last_pix = None

        # 보드 완료 관련 상태도 리셋
        self._seen_designators.clear()
        self._board_completed = False
        self._update_board_progress()

        # Preview/Click 영역 초기화
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
    
    # 보드 진행률 및 완료 상태 갱신
        self._seen_designators = set(self._pred_cache.keys())
        self._board_completed = True   # 저장된 보드는 이미 완료된 상태
        self._update_board_progress()
        
    def _remove_finished_flag(self, path: str):
        try:
            if os.path.exists(path):
                os.remove(path)
                self.on_log(f"[ui] finished flag deleted: {path}")
            else:
                self.on_log(f"[ui] finished flag already gone: {path}")
        except Exception as e:
            self.on_log(f"[ui] failed to delete finished.txt: {e}")

    def _notify_board_completed(self):
   # """
    #현재 보드(실시간 Current)에 대해, 모든 소자에 대한 판정이 끝났을 때 한 번만 호출.
    #"""

    # 지금까지 저장된 보드 폴더 개수 조사
        boards = self._scan_board_dirs(self.outdir_base)
        next_idx = len(boards) + 1
        board_name = f"Board{next_idx}"

        QMessageBox.information(
            self,
            "Board 완료",
            f"{board_name} 검사 완료.\n\n"
            f"Reset Board 버튼을 눌러서 보드를 저장하거나 초기화해 주세요."
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
    #        워커 -> 이미지 들어왔을 때
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
    #        워커 -> 로그 들어왔을 때
    # =========================================================
    @Slot(str)
    def on_log(self, text: str):
        self.log.append(text)
        self.log.moveCursor(QTextCursor.End)

    # =========================================================
    #        워커 -> 예측 결과 들어왔을 때
    # =========================================================
    @Slot(str, int, float)
    def on_pred(self, designator: str, pred: int, prob: float):
        des_up = designator.upper()

    # 0) 현재 보드 상태 캐시에 저장 (나중에 result.json 저장·불러오기용)
        self._pred_cache[des_up] = (int(pred), float(prob))

         # 0-01) 이번 보드에서 처음 본 부품이면 set에 추가
        if des_up not in self._seen_designators:
            self._seen_designators.add(des_up)
            self._update_board_progress()
            self._check_board_completed()
         # 0-1) 이번 보드에서 판정된 소자 기록
        if self.board_combo.currentIndex() == 0:  # 0번은 항상 "Current" 라고 가정
            if self._all_designators:
                self._seen_designators.add(des_up)

            # 아직 완료 처리 안 했고, 전체 집합을 모두 포함하면 -> 완료
                if (not self._board_completed
                        and self._seen_designators.issuperset(self._all_designators)):
                    self._board_completed = True
                    self._notify_board_completed()

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
        self._set_preview_pixmap(pix2) # 25/11/07 23:03 수정사항 (원본 크기 픽스맵을 _set_preview_pixmap으로 전달)

    def _check_board_completed(self):
    #"""모든 부품이 한 번씩은 예측되었는지 검사하고, 끝났으면 팝업."""
        if self._board_completed:
            return  # 이미 한 번 완료 처리한 보드

        if self._board_total <= 0:
            return

        if len(self._seen_designators) < self._board_total:
            return

    # 여기까지 오면 전체 부품 검사 완료
        self._board_completed = True
        try:
            msg = QMessageBox(self)
            msg.setWindowTitle("Board completed")
            msg.setText("현재 보드의 모든 부품 검사가 완료되었습니다.")
            msg.setInformativeText(
                f"총 부품 수: {self._board_total}\n"
                f"검사된 부품 수: {len(self._seen_designators)}"
            )
            msg.setIcon(QMessageBox.Information)
            msg.exec()
        except Exception as e:
            self.on_log(f"[ui] board complete popup failed: {e}")
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
    #        보드 클릭(JS -> Python)
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
            self.click_img_label.setPixmap(QPixmap()) # 25/11/07 23:03 수정사항 (ScaledPixmapLabel.setPixmap(None) 호출)
            self.click_img_label.setText("보드에서 부품을 클릭하면 여기 표시됩니다.")
            return

        ase_title = f"Selected: {des}"

    # 1) 이 부품에 해당하는 이미지가 캐시에 있는지
        img = self._shot_cache.get(des)
        if img is None:
        # 아직 안 찍힌 부품
            self.click_img_label.setPixmap(QPixmap()) # 25/11/07 23:03 수정사항 (ScaledPixmapLabel.setPixmap(None) 호출)
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
        # 25/11/07 23:03 수정사항 (ScaledPixmapLabel의 커스텀 setPixmap 메서드를 호출)
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