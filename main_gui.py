import sys
from collections import Counter
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                               QHBoxLayout, QWidget, QLabel, QFileDialog, QListWidget,
                               QFrame, QTextEdit, QMessageBox, QSlider)
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "best_models" / "coin_v8s_768_best.pt",
    BASE_DIR / "runs" / "detect" / "coin_v8s_768" / "weights" / "best.pt",
]

COIN_KNOWLEDGE_BASE = {
    "PreQin_BuCoin": "【先秦布币】\n源于青铜农具‘镈’（bó），盛行于春秋战国时期。形状似铲。早期布币较大，后期逐渐小型化，反映了古代商品经济的发展与形制的演变。",
    "PreQin_DaoCoin": "【先秦刀币】\n起源于春秋战国齐、燕、赵等国，形制源于实用青铜刀具。齐国‘六字刀’和燕国‘明刀’最为著名。",
    "SongDynasty": "【宋代钱币】\n中国古代铸币技术的巅峰。最大的特点是书法艺术的完美结合，有‘御书钱’（真、草、隶、篆、行等）。铸造精美。",
    "MingQing": "【明清钱币】\n多为通宝、重宝。清代有著名的‘清十帝钱’。形制趋于统一，反映了中央集权下的货币管理制度。",
    "LiaoJinXiaXiYuan": "【辽金西夏元】\n北方少数民族政权铸币。深受中原汉文化影响，采用汉字铸面，同时也保留本民族特色，体现了文化交流。",
    "QinHanWeiJinNanBei": "【秦汉魏晋南北朝】\n从秦半两、汉五铢演变，确立了‘方孔圆钱’的标准形制，影响长达两千年。"
}


class CoinSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_pixmap = None
        self.current_detected_classes = []
        self.initUI()
        self.load_model()

    def initUI(self):
        self.setWindowTitle("古钱币自动化识别与智能科普演示系统")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #f5f5f5;")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- 左侧：控制与数据区 ---
        left_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_upload = QPushButton("📂 上传图片")
        self.btn_upload.setFixedHeight(50)
        self.btn_upload.setStyleSheet("background-color: #34495e; color: white; border-radius: 5px; font-weight:bold;")
        self.btn_upload.clicked.connect(self.run_detection)
        btn_layout.addWidget(self.btn_upload)

        self.btn_save = QPushButton("保存结果")
        self.btn_save.setFixedHeight(50)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("background-color: #16a085; color: white; border-radius: 5px; font-weight:bold;")
        self.btn_save.clicked.connect(self.save_current_result)
        btn_layout.addWidget(self.btn_save)
        left_layout.addLayout(btn_layout)

        threshold_layout = QHBoxLayout()
        self.conf_label = QLabel("置信度阈值: 0.50")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.update_conf_label)
        threshold_layout.addWidget(self.conf_label)
        threshold_layout.addWidget(self.conf_slider)
        left_layout.addLayout(threshold_layout)

        # 结果统计列表
        left_layout.addWidget(QLabel("1️⃣ 检测结果列表 (点击查看详情):"))
        self.result_list = QListWidget()
        self.result_list.setStyleSheet("background: white; border: 1px solid #ddd; font-size:13px;")
        self.result_list.itemClicked.connect(self.show_coin_details)  # 绑定点击事件
        left_layout.addWidget(self.result_list)

        # --- 钱币科普说明框 ---
        left_layout.addWidget(QLabel("2️⃣ 古钱币历史背景速览:"))
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)  # 只读
        self.detail_text.setStyleSheet("background: #fdfdfd; border: 1px dashed #7f8c8d; color: #2c3e50; padding:5px;")
        self.detail_text.setPlaceholderText("在这里显示识别到的钱币的历史背景...")
        left_layout.addWidget(self.detail_text)

        self.info_label = QLabel("状态：等待上传图片...")
        left_layout.addWidget(self.info_label)

        main_layout.addLayout(left_layout, 1)

        # --- 右侧：图片展示区 ---
        right_layout = QVBoxLayout()
        self.image_display = QLabel("预览区域")
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setFrameShape(QFrame.Panel)
        self.image_display.setStyleSheet("background-color: #ecf0f1; border: 2px solid #bdc3c7;")
        right_layout.addWidget(self.image_display)

        main_layout.addLayout(right_layout, 2)

    def load_model(self):
        model_path = next((path for path in MODEL_CANDIDATES if path.exists()), None)
        if model_path is None:
            self.btn_upload.setEnabled(False)
            expected_paths = "\n".join(str(path) for path in MODEL_CANDIDATES)
            self.info_label.setText("状态：模型文件不存在")
            QMessageBox.critical(self, "模型加载失败", f"找不到可用模型文件：\n{expected_paths}")
            return

        try:
            self.model = YOLO(str(model_path))
            self.info_label.setText(f"状态：模型加载完成：{model_path.name}")
        except Exception as exc:
            self.btn_upload.setEnabled(False)
            self.info_label.setText("状态：模型加载失败")
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.refresh_image_display)

    def refresh_image_display(self):
        if self.current_pixmap is None:
            return
        self.image_display.setPixmap(
            self.current_pixmap.scaled(
                self.image_display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def run_detection(self):
        if self.model is None:
            QMessageBox.warning(self, "模型未加载", "请确认模型文件存在后再运行检测。")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "图片文件 (*.jpg *.png *.jpeg)")

        if file_path:
            self.info_label.setText("状态：人工智能正在全力识别...")
            self.detail_text.clear()  # 清空旧科普文字
            self.result_list.clear()
            self.current_detected_classes = []  # 清空
            self.btn_save.setEnabled(False)
            QApplication.processEvents()

            try:
                results = self.model.predict(source=file_path, conf=self.confidence_threshold(), imgsz=768, verbose=False)
                res = results[0]
                self.show_annotated_image(res.plot())
                self.show_detection_results(res)
            except Exception as exc:
                self.info_label.setText("状态：检测失败")
                QMessageBox.critical(self, "检测失败", str(exc))

    def show_annotated_image(self, annotated_frame):
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.current_pixmap = QPixmap.fromImage(q_img)
        self.refresh_image_display()
        self.btn_save.setEnabled(True)

    def show_detection_results(self, result):
        class_names = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = self.model.names[cls_id]
            class_names.append(name)
            self.result_list.addItem(f"{name} | 置信度: {conf:.2f}")

        self.current_detected_classes = class_names
        total = len(class_names)

        if total == 0:
            self.detail_text.setText("未检测到古钱币，请尝试更清晰的图片或降低置信度阈值。")
            self.info_label.setText("状态：完成，未检测到钱币")
            return

        counts = Counter(class_names)
        summary = "；".join(f"{name} × {count}" for name, count in counts.items())
        self.detail_text.setText(f"检测汇总：{summary}\n\n点击左侧某条检测结果可查看对应历史背景。")
        self.info_label.setText(f"状态：完成，检测到 {total} 枚钱币")

    def show_coin_details(self, item):
        # 当左侧列表item被点击时触发
        row_index = self.result_list.row(item)
        if row_index >= 0 and row_index < len(self.current_detected_classes):
            cls_name = self.current_detected_classes[row_index]
            # 从知识库中匹配文字
            knowledge_blurb = COIN_KNOWLEDGE_BASE.get(cls_name, "抱歉，暂无该类钱币的详细科普资料。")
            self.detail_text.setHtml(f"<b style='color:#c0392b;'>{knowledge_blurb}</b>")

    def confidence_threshold(self):
        return self.conf_slider.value() / 100

    def update_conf_label(self, value):
        self.conf_label.setText(f"置信度阈值: {value / 100:.2f}")

    def save_current_result(self):
        if self.current_pixmap is None:
            QMessageBox.information(self, "暂无结果", "请先上传图片并完成检测。")
            return

        output_dir = BASE_DIR / "outputs"
        output_dir.mkdir(exist_ok=True)
        default_path = output_dir / "coin_detection_result.jpg"
        file_path, _ = QFileDialog.getSaveFileName(self, "保存检测结果", str(default_path), "图片文件 (*.jpg *.png)")
        if not file_path:
            return

        if self.current_pixmap.save(file_path):
            self.info_label.setText(f"状态：检测结果已保存到 {file_path}")
        else:
            QMessageBox.warning(self, "保存失败", "无法保存检测结果图片。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CoinSystem()
    window.show()
    sys.exit(app.exec())
