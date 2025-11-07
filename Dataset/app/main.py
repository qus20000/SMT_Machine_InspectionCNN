# Dataset/app/main.py
import os, sys, json
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet
sys.path.append(os.path.dirname(__file__))
from ui_main import MainWindow


def main():
    app = QApplication(sys.argv)

    apply_stylesheet(app, theme='dark_cyan.xml')

    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("config.json 파일이 없습니다. 설정 파일을 확인하세요.")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    win = MainWindow(cfg)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
