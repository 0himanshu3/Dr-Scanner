import sys
import os
import re
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from pytesseract import Output
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt

import scanner2
import file_manager

# Point pytesseract at your tesseract install
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

PDF_STORAGE_FOLDER = "scanned_documents"
os.makedirs(PDF_STORAGE_FOLDER, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def load_pdf_as_images(pdf_path):
    """Render each page of the PDF to a BGR OpenCV image via PyMuPDF."""
    doc = fitz.open(pdf_path)
    images = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        pix = page.get_pixmap()
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        images.append(cv_img)
    return images

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dr. Scanner")
        self.setGeometry(200, 100, 900, 600)
        self.setWindowIcon(QIcon("icons/app_icon.png"))

        self.image_labels = []
        self.images = []
        self.processed_images = []
        self.ocr_texts = []
        self.current_preview_index = 0

        self.pdf_filename_images = ""
        self.pdf_filename_text = ""

        self.initUI()

    def initUI(self):
        # Widgets
        self.image_label = QLabel("Preview Area")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.list_widget = QListWidget()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        # Buttons
        load_button = QPushButton("📂 Load Files")
        scan_button = QPushButton("🔍 Scan Documents")
        self.prev_button = QPushButton("⬅️ Previous")
        self.next_button = QPushButton("➡️ Next")
        save_text_pdf_button = QPushButton("💾 Save Text PDF")
        save_images_pdf_button = QPushButton("🖼️ Save Images PDF")
        search_button = QPushButton("📁 Open Output Folder")

        # Connect buttons
        load_button.clicked.connect(self.load_images)
        scan_button.clicked.connect(self.scan_documents)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        save_text_pdf_button.clicked.connect(self.save_all_output)
        save_images_pdf_button.clicked.connect(self.save_images_pdf)
        search_button.clicked.connect(self.open_saved_documents_folder)

        # Layouts
        image_controls_layout = QHBoxLayout()
        image_controls_layout.addWidget(self.prev_button)
        image_controls_layout.addWidget(self.next_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(load_button)
        left_layout.addWidget(scan_button)
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(save_text_pdf_button)
        left_layout.addWidget(save_images_pdf_button)
        left_layout.addWidget(search_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)
        right_layout.addLayout(image_controls_layout)
        right_layout.addWidget(QLabel("📝 Extracted Text:"))
        right_layout.addWidget(self.text_edit)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Files", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;PDF Files (*.pdf)",
            options=QFileDialog.Options()
        )
        if not files:
            return

        self.image_labels.clear()
        self.images.clear()
        self.processed_images.clear()
        self.ocr_texts.clear()
        self.list_widget.clear()
        self.current_preview_index = 0

        for path in files:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                pages = load_pdf_as_images(path)
                for i, img in enumerate(pages, start=1):
                    self.images.append(img)
                    label = f"{os.path.basename(path)} [page {i}]"
                    self.image_labels.append(label)
                    self.list_widget.addItem(label)
            else:
                img = cv2.imread(path)
                if img is not None:
                    self.images.append(img)
                    label = os.path.basename(path)
                    self.image_labels.append(label)
                    self.list_widget.addItem(label)
                else:
                    self.text_edit.append(f"⚠️ Failed to load: {path}")

        base_name, ok = QInputDialog.getText(
            self, "Save PDF", "Enter base filename (no extension):"
        )
        if not (ok and base_name.strip()):
            QMessageBox.warning(self, "Filename Missing", "No PDF filename provided.")
            return

        base = sanitize_filename(base_name.strip())
        self.pdf_filename_images = os.path.join(PDF_STORAGE_FOLDER, f"{base}_images.pdf")
        self.pdf_filename_text = os.path.join(PDF_STORAGE_FOLDER, f"{base}_text.pdf")

    def scan_documents(self):
        if not self.images:
            self.text_edit.setPlainText("❗ Please load images or a PDF first.")
            return

        self.processed_images.clear()
        self.ocr_texts.clear()
        self.text_edit.clear()

        for idx, img in enumerate(self.images):
            proc = scanner2.preprocess_image(img)
            txt = scanner2.extract_text(proc)
            self.processed_images.append(proc)
            self.ocr_texts.append(txt)

            snippet = txt.replace("\n", " ")[:200]
            if len(txt) > 200:
                snippet += "..."
            self.text_edit.append(
                f"✔️ {self.image_labels[idx]} | Len: {len(txt)}\n"
                f"📄 {snippet}\n"
            )

        if self.pdf_filename_images:
            out_img_pdf = file_manager.generate_pdf_scanned_document(
                self.processed_images, PDF_STORAGE_FOLDER, self.pdf_filename_images
            )
            QMessageBox.information(self, "Saved", f"Images PDF: {out_img_pdf}")
            os.startfile(out_img_pdf)

        if self.pdf_filename_text:
            out_txt_pdf = file_manager.generate_pdf_text_only(
                self.ocr_texts, PDF_STORAGE_FOLDER, self.pdf_filename_text
            )
            QMessageBox.information(self, "Saved", f"Text PDF: {out_txt_pdf}")
            os.startfile(out_txt_pdf)

        self.current_preview_index = 0
        self.update_preview()

    def update_preview(self):
        if not self.processed_images:
            return
        img = self.processed_images[self.current_preview_index]
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio
        )
        self.image_label.setPixmap(pix)
        self.prev_button.setEnabled(self.current_preview_index > 0)
        self.next_button.setEnabled(self.current_preview_index < len(self.images) - 1)

    def show_previous_image(self):
        if self.current_preview_index > 0:
            self.current_preview_index -= 1
            self.update_preview()

    def show_next_image(self):
        if self.current_preview_index < len(self.images) - 1:
            self.current_preview_index += 1
            self.update_preview()

    def save_all_output(self):
        QMessageBox.information(self, "Info", "This feature is already covered in scan_documents.")

    def save_images_pdf(self):
        if not self.processed_images:
            QMessageBox.warning(self, "Warning", "No processed images to save.")
            return
        out_img_pdf = file_manager.generate_pdf_scanned_document(
            self.processed_images, PDF_STORAGE_FOLDER, self.pdf_filename_images
        )
        QMessageBox.information(self, "Saved", f"Images PDF: {out_img_pdf}")
        os.startfile(out_img_pdf)

    def open_saved_documents_folder(self):
        os.startfile(PDF_STORAGE_FOLDER)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
