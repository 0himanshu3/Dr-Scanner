import sys
import os
import re
import cv2
import numpy as np
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QMessageBox, QFrame, QInputDialog
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt

import scanner2
import file_manager

# Ensure Tesseract is in the PATH
if "TESSDATA_PREFIX" not in os.environ:
    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_STORAGE_FOLDER = "scanned_documents"
os.makedirs(PDF_STORAGE_FOLDER, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def load_pdf_as_images(pdf_path):
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            mode = "RGB" if pix.alpha == 0 else "RGBA"
            pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR) \
                     if mode == "RGBA" else cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            images.append(cv_img)
        doc.close()
        return images
    except Exception as e:
        QMessageBox.critical(None, "PDF Load Error", f"Failed to load PDF {pdf_path}:\n{e}")
        return []

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dr. Scanner")
        self.setGeometry(200, 100, 1000, 700)
        if os.path.exists("icons/app_icon.png"):
            self.setWindowIcon(QIcon("icons/app_icon.png"))

        self.image_sources = []
        self.original_images = []
        self.processed_images = []
        self.ocr_texts = []
        self.current_preview_index = 0
        self.auto_images_pdf = ""
        self.auto_text_pdf = ""

        self.initUI()

    def initUI(self):
        # UI elements
        self.load_button = QPushButton("Load Files")
        self.scan_button = QPushButton("Scan & OCR")
        self.save_text_pdf_button = QPushButton("Save Text PDF")
        self.save_images_pdf_button = QPushButton("Save Image PDF")
        self.search_button = QPushButton("Open Saved Folder")

        self.list_widget = QListWidget()
        self.text_edit = QTextEdit()

        self.image_label = QLabel("Preview Area")
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)

        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.scan_button)
        button_layout.addWidget(self.save_text_pdf_button)
        button_layout.addWidget(self.save_images_pdf_button)
        button_layout.addWidget(self.search_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.list_widget)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect signals
        self.load_button.clicked.connect(self.load_files)
        self.scan_button.clicked.connect(self.scan_documents)
        self.save_text_pdf_button.clicked.connect(self.save_text_pdf_manual)
        self.save_images_pdf_button.clicked.connect(self.save_images_pdf_manual)
        self.search_button.clicked.connect(self.open_saved_documents_folder)

        self.update_button_states()

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Files", "", "PDF (*.pdf);;Images (*.png *.jpg *.jpeg *.bmp)", options=QFileDialog.Options()
        )
        if not files:
            return

        self.image_sources.clear()
        self.original_images.clear()
        self.processed_images.clear()
        self.ocr_texts.clear()
        self.list_widget.clear()
        self.current_preview_index = 0
        self.text_edit.clear()
        self.image_label.setText("Loading...")
        QApplication.processEvents()

        for path in files:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                pages = load_pdf_as_images(path)
                for i, img in enumerate(pages, 1):
                    self.original_images.append(img)
                    label = f"{os.path.basename(path)} [page {i}]"
                    self.image_sources.append(label)
                    self.list_widget.addItem(label)
            else:
                img = cv2.imread(path)
                if img is not None:
                    self.original_images.append(img)
                    label = os.path.basename(path)
                    self.image_sources.append(label)
                    self.list_widget.addItem(label)

        if not self.original_images:
            self.image_label.setText("No valid files loaded.")
        else:
            suggested = os.path.splitext(os.path.basename(files[0]))[0]
            base, ok = QInputDialog.getText(
                self, "Autosave Base Filename", "Enter base name for auto-saved PDFs:", text=suggested
            )
            if ok and base.strip():
                base = sanitize_filename(base.strip())
                self.auto_images_pdf = os.path.join(PDF_STORAGE_FOLDER, f"{base}_images.pdf")
                self.auto_text_pdf   = os.path.join(PDF_STORAGE_FOLDER, f"{base}_text.pdf")
            self.update_preview(use_original=True)

        self.update_button_states()

    def scan_documents(self):
        if not self.original_images:
            QMessageBox.warning(self, "No Files", "Load images or PDFs first.")
            return

        self.processed_images.clear()
        self.ocr_texts.clear()
        self.text_edit.clear()

        for idx, img in enumerate(self.original_images):
            self.text_edit.append(f"→ Processing {self.image_sources[idx]}...")
            QApplication.processEvents()

            processed = scanner2.preprocess_image(img)
            text = scanner2.extract_text(processed)

            self.processed_images.append(processed)
            self.ocr_texts.append(text)

        if self.auto_images_pdf:
            file_manager.generate_pdf_scanned_document(
                self.processed_images, os.path.dirname(self.auto_images_pdf), os.path.basename(self.auto_images_pdf)
            )
        if self.auto_text_pdf:
            file_manager.generate_pdf_text_only(
                self.ocr_texts, os.path.dirname(self.auto_text_pdf), os.path.basename(self.auto_text_pdf)
            )

        self.current_preview_index = 0
        self.update_preview()
        QMessageBox.information(self, "Done", "Scan & OCR complete.")

    def update_preview(self, use_original=False):
        if not self.original_images:
            return
        img = self.original_images[self.current_preview_index] if use_original else self.processed_images[self.current_preview_index]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio
        ))

        self.text_edit.setPlainText(self.ocr_texts[self.current_preview_index] if self.ocr_texts else "")

    def save_text_pdf_manual(self):
        if not self.ocr_texts:
            QMessageBox.warning(self, "Nothing to Save", "No OCR text found.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Text PDF", "", "PDF Files (*.pdf)")
        if filename:
            file_manager.generate_pdf_text_only(self.ocr_texts, os.path.dirname(filename), os.path.basename(filename))
            QMessageBox.information(self, "Saved", f"Text PDF saved to {filename}")

    def save_images_pdf_manual(self):
        if not self.processed_images:
            QMessageBox.warning(self, "Nothing to Save", "No processed images found.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image PDF", "", "PDF Files (*.pdf)")
        if filename:
            file_manager.generate_pdf_scanned_document(self.processed_images, os.path.dirname(filename), os.path.basename(filename))
            QMessageBox.information(self, "Saved", f"Image PDF saved to {filename}")

    def open_saved_documents_folder(self):
        os.startfile(os.path.abspath(PDF_STORAGE_FOLDER))

    def update_button_states(self):
        self.scan_button.setEnabled(bool(self.original_images))
        self.save_text_pdf_button.setEnabled(bool(self.ocr_texts))
        self.save_images_pdf_button.setEnabled(bool(self.processed_images))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
