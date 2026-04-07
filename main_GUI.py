import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QStyledItemDelegate
from PyQt5.QtCore import Qt
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
import csv


class CustomImageFolder(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def data_standard(image_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None
    transform = transforms.Compose([
        transforms.Pad(50),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = CustomImageFolder(samples=[(image_path, 0)], transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key) / math.sqrt(C // 8)
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class OilMobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        ch = 960
        self.self_att = SelfAttention(ch)
        self.output = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.self_att(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)
        x = self.output(features)
        return x


def load_all_models(base_model_dir, DEVICE):
    global MODEL_CACHE
    MODEL_CACHE = {}

    cache_item = {}
    # -------- classifier --------
    cls_model = OilMobileNet(num_classes=3, pretrained=False)

    cls_path = os.path.join(base_model_dir, "OilMobileNet_classifier_best_model.pt")
    cls_model.load_state_dict(torch.load(cls_path, map_location=DEVICE))
    cls_model.to(DEVICE)
    cls_model.eval()
    cache_item["classifier"] = cls_model

    # -------- reg low --------
    reg_low = models.mobilenet_v3_large(weights=None)
    reg_low.classifier[3] = nn.Linear(reg_low.classifier[3].in_features, 1)

    reg_low_path = os.path.join(base_model_dir, "mobilenetv3L_regressor_low_best_model.pt")
    reg_low.load_state_dict(torch.load(reg_low_path, map_location=DEVICE))
    reg_low.to(DEVICE)
    reg_low.eval()
    cache_item["reg_low"] = reg_low

    # -------- reg high --------
    reg_high = models.efficientnet_v2_s(weights=None)
    reg_high.classifier[1] = nn.Linear(reg_high.classifier[1].in_features, 1)

    reg_high_path = os.path.join(base_model_dir, "efficientnetv2s_regressor_high_best_model.pt")
    reg_high.load_state_dict(torch.load(reg_high_path, map_location=DEVICE))
    reg_high.to(DEVICE)
    reg_high.eval()
    cache_item["reg_high"] = reg_high

    MODEL_CACHE['Green laser'] = cache_item


def validation_with_cache(model, loader, DEVICE, is_classifier):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            out = model(images)

            if is_classifier:
                out_probs = F.softmax(out, dim=1)
                pred = out_probs.argmax(dim=1).cpu().numpy().tolist()
            else:
                pred = out.squeeze().cpu().numpy().tolist()
                if isinstance(pred, float):
                    pred = [pred]

            predictions.extend(pred)
    return predictions


def run_pipeline_gui(image_path, base_model_dir, DEVICE, img_type):
    global MODEL_CACHE
    if not MODEL_CACHE:
        load_all_models(base_model_dir, DEVICE)
    loader = data_standard(image_path)

    cls_model = MODEL_CACHE[img_type]["classifier"]
    pred_class = validation_with_cache(cls_model, loader, DEVICE, is_classifier=True)[0]

    if pred_class == 0:
        plain_cls = "Pure oil"
        plain_pred = '<0.05%'
        cls_text = "<span style='color:black;'>Classification:</span><br>" \
                   "<span style='color:green; font-weight:bold;'>Pure oil</span>"

        pred_text = "<br><br><span style='color:black;'>Prediction:</span><br>" \
                    f"<span style='color:green; font-weight:bold;'>&lt;0.05%</span>"
        return cls_text + pred_text, plain_cls, plain_pred

    elif pred_class == 1:
        reg_model = MODEL_CACHE[img_type]["reg_low"]
        pred_val = validation_with_cache(reg_model, loader, DEVICE, is_classifier=False)[0]
        plain_cls = "Low-level contamination oil"
        plain_pred = f'{pred_val / 100:.2%}'
        cls_text = "<span style='color:black;'>Classification:</span><br>" \
                   "<span style='color:green; font-weight:bold;'>Low-level contamination oil</span>"

        pred_text = "<br><br><span style='color:black;'>Prediction:</span><br>" \
                    f"<span style='color:green; font-weight:bold;'>{pred_val / 100:.2%}</span>"
        return cls_text + pred_text, plain_cls, plain_pred

    elif pred_class == 2:
        reg_model = MODEL_CACHE[img_type]["reg_high"]
        pred_val = validation_with_cache(reg_model, loader, DEVICE, is_classifier=False)[0]
        plain_cls = "High-level contamination oil"
        plain_pred = f'{pred_val / 100:.1%}'
        cls_text = "<span style='color:black;'>Classification:</span><br>" \
                   "<span style='color:green; font-weight:bold;'>High-level contamination oil</span>"
        pred_text = "<br><br><span style='color:black;'>Prediction:</span><br>" \
                    f"<span style='color:green; font-weight:bold;'>{pred_val / 100:.2%}</span>"
        return cls_text + pred_text, plain_cls, plain_pred


class BatchResultWindow(QtWidgets.QDialog):
    def __init__(self, results):
        super().__init__()
        self.setWindowTitle("Batch Prediction Results")
        self.setGeometry(600, 200, 530, 500)
        self.results = results

        layout = QtWidgets.QVBoxLayout(self)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Image Name", "Classification", "Prediction"])
        self.table.setRowCount(len(results))

        for i, (img_path, cls, pred) in enumerate(results):
            name = os.path.basename(img_path)
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(cls))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(pred))

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_save = QtWidgets.QPushButton("Save Results")
        btn_save.clicked.connect(self.save_results)
        btn_layout.addWidget(btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

    def save_results(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Results",
            ".",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "Classification", "Prediction"])
            for img_path, cls, pred in self.results:
                writer.writerow([os.path.basename(img_path), cls, pred])
        QtWidgets.QMessageBox.information(self, "Saved", f"Results saved to:\n{path}")


class CenterDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignCenter


class BatchWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(list)

    def __init__(self, image_list, base_model_dir, device, img_type):
        super().__init__()
        self.image_list = image_list
        self.base_model_dir = base_model_dir
        self.device = device
        self.img_type = img_type

    def run(self):
        results = []
        for img_path in self.image_list:
            html_res, cls, pred = run_pipeline_gui(img_path, self.base_model_dir, self.device, self.img_type)
            results.append((img_path, cls, pred))
        self.result_ready.emit(results)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oil Contamination Detection")
        self.setGeometry(600, 200, 900, 450)

        self.image_path = None
        self.is_batch_mode = False
        self.batch_images = []

        self.init_ui()
        self.load_default_image()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        title_bar = QtWidgets.QLabel("Oil Contamination Detection")
        title_bar.setAlignment(Qt.AlignCenter)
        title_bar.setFixedHeight(50)
        title_bar.setStyleSheet(
            "background-color:#E6F2FF; font-size:24px; font-weight:bold; border: 1px solid #CCCCCC;"
        )

        main_layout.addWidget(title_bar)

        middle_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(middle_layout)

        left_panel = QtWidgets.QVBoxLayout()
        middle_layout.addLayout(left_panel, stretch=1)

        blue_btn_style = """
        QPushButton {
            background-color:#0078D7; 
            color:white; 
            padding:8px;
            font-size:16px;
            border-radius:4px;
        }
        QPushButton:hover {
            background-color:#3399FF;
        }
        """
        combo_style = """
                 QComboBox {
                    background-color:#FFFFFF; 
                    color:black; 
                    padding:8px;
                    font-size:16px;
                    border-radius:4px;
                }
                """

        lbl_laser_fix = QtWidgets.QLabel("Laser: ")
        lbl_laser_fix.setStyleSheet("font-size:18px; font-weight:bold; margin-top:8px;")
        left_panel.addWidget(lbl_laser_fix)

        lbl_laser_value = QtWidgets.QLabel("Green laser")
        lbl_laser_value.setAlignment(Qt.AlignCenter)
        lbl_laser_value.setStyleSheet(
            "background-color:#FFFFFF; "
            "color:black; "
            "padding:8px; "
            "font-size:16px; "
            "border-radius:4px;"
        )
        left_panel.addWidget(lbl_laser_value)
        left_panel.addSpacing(15)

        lbl_upload = QtWidgets.QLabel("File upload:")
        lbl_upload.setStyleSheet("font-size:18px; font-weight:bold; margin-top:18px;")
        left_panel.addWidget(lbl_upload)

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Image", "Folder"])
        self.combo_mode.setStyleSheet(combo_style)
        self.combo_mode.setEditable(True)
        self.combo_mode.lineEdit().setAlignment(Qt.AlignCenter)
        self.combo_mode.lineEdit().setReadOnly(True)
        self.combo_mode.setItemDelegate(CenterDelegate(self.combo_mode))
        self.combo_mode.currentIndexChanged.connect(self.update_load_buttons)
        left_panel.addWidget(self.combo_mode)

        self.btn_load_image = QtWidgets.QPushButton("Select Image")
        self.btn_load_image.setStyleSheet(blue_btn_style)
        self.btn_load_image.clicked.connect(self.load_image)
        left_panel.addWidget(self.btn_load_image)

        self.btn_load_folder = QtWidgets.QPushButton("Select Folder")
        self.btn_load_folder.setStyleSheet(blue_btn_style)
        self.btn_load_folder.clicked.connect(self.load_folder)
        self.btn_load_folder.hide()
        left_panel.addWidget(self.btn_load_folder)
        left_panel.addSpacing(45)

        self.btn_run = QtWidgets.QPushButton("Run Analysis")
        self.btn_run.setStyleSheet(blue_btn_style)
        self.btn_run.clicked.connect(self.predict)
        left_panel.addWidget(self.btn_run)
        left_panel.addSpacing(45)

        self.btn_clear = QtWidgets.QPushButton("Clear All")
        self.btn_clear.setStyleSheet("background-color:white; padding:8px; font-size:16px;")
        self.btn_clear.clicked.connect(self.clear_all)
        left_panel.addWidget(self.btn_clear)
        left_panel.addStretch()

        center_panel = QtWidgets.QVBoxLayout()
        right_panel = QtWidgets.QVBoxLayout()

        shared_area = QtWidgets.QWidget()
        shared_area.setStyleSheet("background-color:white;")
        shared_layout = QtWidgets.QHBoxLayout(shared_area)
        shared_layout.setContentsMargins(10, 10, 10, 10)

        shared_layout.addLayout(center_panel, stretch=3)
        shared_layout.addLayout(right_panel, stretch=2)

        middle_layout.addWidget(shared_area)

        self.image_panel = QtWidgets.QLabel("Image")
        self.image_panel.setStyleSheet("""
            background-color:#F0F0F0;
            padding:15px;
        """)
        self.image_panel.setAlignment(Qt.AlignCenter)
        self.image_panel.setStyleSheet("background-color:#F0F0F0;")
        self.image_panel.setFixedSize(470, 400)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.hide()

        self.top_area = QtWidgets.QStackedWidget()
        self.top_area.addWidget(self.image_panel)
        self.top_area.addWidget(self.scroll_area)

        center_panel.addWidget(self.top_area)

        self.batch_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.batch_container)
        self.scroll_area.setWidget(self.batch_container)

        self.scroll_area.setStyleSheet("background-color:#F0F0F0; border:none;")
        self.batch_container.setStyleSheet("background-color:#F0F0F0;")

        self.text_result = QtWidgets.QTextEdit()
        self.text_result.setReadOnly(True)
        self.text_result.setStyleSheet("background-color:white; font-size:18px;")
        self.text_result.setStyleSheet(
            "background-color:white; "
            "font-size:18px; "
            "border: 1px solid white;"
        )
        right_panel.addWidget(self.text_result)

    def update_load_buttons(self):
        mode = self.combo_mode.currentText()

        if mode == "Image":
            self.btn_load_image.show()
            self.btn_load_folder.hide()

        elif mode == "Folder":
            self.btn_load_image.hide()
            self.btn_load_folder.show()

    def load_default_image(self):
        if os.path.exists(DEFAULT_IMAGE):
            self.image_path = DEFAULT_IMAGE
            self.display_image(DEFAULT_IMAGE)
            self.text_result.setText(f"Display image {DEFAULT_IMAGE}")
            self.text_result.setFont(QtWidgets.QApplication.font())
        else:
            self.text_result.setText(f"Default image {DEFAULT_IMAGE} not found.")
            self.text_result.setFont(QtWidgets.QApplication.font())

    def display_image(self, path):
        pixmap = QtGui.QPixmap(path).scaled(
            self.image_panel.width(),
            self.image_panel.height(),
            Qt.KeepAspectRatio
        )
        self.image_panel.setPixmap(pixmap)

    def display_batch_images(self, paths):
        self.top_area.setCurrentIndex(1)

        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = 4
        img_w = 100
        img_h = 100

        for idx, p in enumerate(paths):
            r = idx // cols
            c = idx % cols

            lbl = QtWidgets.QLabel()
            lbl.setFixedSize(img_w, img_h)
            lbl.setAlignment(Qt.AlignCenter)

            pixmap = QtGui.QPixmap(p).scaled(
                img_w, img_h,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            lbl.setPixmap(pixmap)
            self.grid_layout.addWidget(lbl, r, c)

    def load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", ".", "Images (*.jpg *.png *.jpeg *.tiff)"
        )
        if path:
            self.is_batch_mode = False
            self.image_path = path
            self.top_area.setCurrentIndex(0)
            self.display_image(path)

            self.text_result.setHtml("")
            self.text_result.setFont(QtWidgets.QApplication.font())
            self.text_result.setText(f"Selected image: {path}\n")

    def load_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return

        imgs = []
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXT:
                imgs.append(os.path.join(folder, f))

        if not imgs:
            self.text_result.setHtml("")
            self.text_result.setFont(QtWidgets.QApplication.font())
            self.text_result.setText(f"No supported images found in {folder}")
            return

        self.is_batch_mode = True
        self.batch_images = imgs
        self.display_batch_images(imgs)

        self.text_result.setHtml("")
        self.text_result.setFont(QtWidgets.QApplication.font())
        self.text_result.setText(f"{folder}\n\nLoaded {len(imgs)} images\n")

    def clear_all(self):
        self.is_batch_mode = False
        self.batch_images = []
        self.text_result.clear()
        self.text_result.setHtml("")
        self.text_result.setFont(QtWidgets.QApplication.font())
        if hasattr(self, "combo_method"):
            self.combo_method.setCurrentIndex(0)
        self.combo_mode.setCurrentIndex(0)
        self.scroll_area.hide()
        self.image_panel.show()
        self.load_default_image()

    def show_batch_results(self, results):
        self.batch_window = BatchResultWindow(results)
        self.batch_window.exec_()
        self.text_result.setText("Prediction completed!\n")

    def predict(self):
        if not self.image_path:
            self.text_result.setText("No image loaded.")
            return

        img_type = "Green laser"

        if not self.is_batch_mode:
            self.text_result.setText(f"{self.image_path}\nRun analysis...\n")
            QtWidgets.QApplication.processEvents()
            html_res, cls, pred = run_pipeline_gui(self.image_path, BASE_MODEL_DIR, DEVICE, img_type)

            self.text_result.setHtml(
                f"<div style='text-align: justify;'>{self.image_path}</div>"
                "<br><br>"
                "<span style='font-size:22px; font-weight:bold; line-height:150%;'>Results:</span>"
            )

            self.text_result.append(html_res)
            self.text_result.setFont(QtWidgets.QApplication.font())
            return

        self.text_result.setText("Run analysis...\n")
        QtWidgets.QApplication.processEvents()

        self.worker = BatchWorker(self.batch_images, BASE_MODEL_DIR, DEVICE, img_type)
        self.worker.result_ready.connect(self.show_batch_results)
        self.worker.start()


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_MODEL_DIR = os.path.join(os.path.dirname(sys.argv[0]), "model")
    DEFAULT_IMAGE = "dataset/data_low/G_E1_Y4_0.9_1.jpg"
    SUPPORTED_EXT = [".jpg", ".jpeg", ".png", ".tiff"]

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Arial", 12))
    win = MainWindow()
    win.show()

    load_all_models(BASE_MODEL_DIR, DEVICE)
    sys.exit(app.exec_())