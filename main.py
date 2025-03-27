import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QMessageBox, QFrame, QInputDialog
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt
import scanner2
import pytesseract
from pytesseract import Output
import file_manager 

# Folder for PDFs and text storage
PDF_STORAGE_FOLDER = "scanned_documents"
os.makedirs(PDF_STORAGE_FOLDER, exist_ok=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dr. Scanner")
        self.setGeometry(200, 100, 900, 600)
        self.setWindowIcon(QIcon("icons/app_icon.png"))
        self.image_paths = []
        self.images = []
        self.processed_images = []
        self.ocr_texts = []
        self.current_preview_index = 0
        # Two PDF filenames will be generated using the same base name.
        self.pdf_filename_images = ""
        self.pdf_filename_text = ""
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        title_label = QLabel("Dr. Scanner")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007BFF; margin-bottom: 15px;")
        
        content_layout = QHBoxLayout()
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(10, 10, 10, 10)
        sidebar.setSpacing(15)
        
        sidebar_heading = QLabel("📂 File Management")
        sidebar_heading.setStyleSheet("font-size: 16px; font-weight: bold; color: #343A40;")
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(220)
        self.list_widget.setStyleSheet("background-color: #F8F9FA; border-radius: 10px; padding: 5px;")
        
        load_button = QPushButton("📂 Load Images")
        scan_button = QPushButton("🔍 Scan Documents")
        search_button = QPushButton("🔎 View Saved Documents")
        button_style = """
            QPushButton {
                background-color: #007BFF; color: white; border-radius: 5px; padding: 8px;
                font-weight: bold; font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """
        for btn in [load_button, scan_button, search_button]:
            btn.setStyleSheet(button_style)
            btn.setFixedHeight(50)
        
        display_layout = QVBoxLayout()
        image_heading = QLabel("🖼 Document Preview")
        image_heading.setStyleSheet("font-size: 16px; font-weight: bold; color: #343A40;")
        self.image_label = QLabel("")
        self.image_label.setFixedSize(600, 400)
        self.image_label.setStyleSheet("border: 2px dashed #6C757D; background-color: #E9ECEF; border-radius: 10px;")
        
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("← Previous")
        self.next_button = QPushButton("Next →")
        for nav_btn in [self.prev_button, self.next_button]:
            nav_btn.setStyleSheet(button_style)
            nav_btn.setFixedHeight(40)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        
        text_heading = QLabel("📝 Extracted Text")
        text_heading.setStyleSheet("font-size: 16px; font-weight: bold; color: #343A40;")
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            background-color: #F8F9FA;
            padding: 10px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: bold;
        """)
        
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("background-color: white; border-radius: 10px; padding: 10px;")
        
        sidebar.addWidget(sidebar_heading)
        sidebar.addWidget(self.list_widget)
        sidebar.addWidget(load_button)
        sidebar.addWidget(scan_button)
        sidebar.addWidget(search_button)
        sidebar.addStretch()
        
        display_layout.addWidget(image_heading)
        display_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        display_layout.addLayout(nav_layout)
        display_layout.addWidget(text_heading)
        display_layout.addWidget(self.text_edit)
        frame.setLayout(display_layout)
        
        content_layout.addLayout(sidebar)
        content_layout.addWidget(frame)
        main_layout.addWidget(title_label)
        main_layout.addLayout(content_layout)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        load_button.clicked.connect(self.load_images)
        scan_button.clicked.connect(self.scan_documents)
        search_button.clicked.connect(self.open_saved_documents_folder)

    def load_images(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Open Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if files:
            self.image_paths = files
            self.images = []
            self.processed_images = []
            self.ocr_texts = []
            self.current_preview_index = 0
            self.list_widget.clear()
            for file in files:
                img = cv2.imread(file)
                if img is not None:
                    self.images.append(img)
                    self.list_widget.addItem(os.path.basename(file))
                else:
                    self.text_edit.append(f"Failed to load: {file}")
            # Ask user for a base filename and generate two PDFs (images & text)
            base_name, ok = QInputDialog.getText(self, "Save PDF", "Enter base filename for the PDFs (without extension):")
            if ok and base_name:
                self.pdf_filename_images = os.path.join(PDF_STORAGE_FOLDER, f"{base_name}_images.pdf")
                self.pdf_filename_text = os.path.join(PDF_STORAGE_FOLDER, f"{base_name}_text.pdf")
            else:
                QMessageBox.warning(self, "Filename Missing", "No PDF filename provided.")

    def scan_documents(self):
        if not self.images:
            self.text_edit.setText("Please load images first.")
            return

        self.processed_images = []
        self.ocr_texts = []
        self.text_edit.clear()

        for idx, img in enumerate(self.images):
            processed = scanner2.preprocess_image(img)
            text = scanner2.extract_text(processed)
            self.processed_images.append(processed)
            self.ocr_texts.append(text)
            self.text_edit.append(f"Processed {os.path.basename(self.image_paths[idx])}\nText length: {len(text)}\n")

        # Generate PDF for scanned images using file_manager
        if self.pdf_filename_images:
            pdf_images = file_manager.generate_pdf_scanned_document(self.processed_images, PDF_STORAGE_FOLDER, pdf_filename=self.pdf_filename_images)
            QMessageBox.information(self, "Images PDF Saved", f"Scanned images PDF saved as:\n{pdf_images}")
        
        # Generate PDF for extracted text using file_manager
        if self.pdf_filename_text:
            pdf_text = file_manager.generate_pdf_text_only(self.ocr_texts, PDF_STORAGE_FOLDER, pdf_filename=self.pdf_filename_text)
            QMessageBox.information(self, "Text PDF Saved", f"Extracted text PDF saved as:\n{pdf_text}")

        if self.processed_images:
            self.current_preview_index = 0
            self.update_preview()

        QMessageBox.information(self, "Scan Complete", "All images processed and both PDFs generated (if filenames provided).")

    def update_preview(self):
        if self.processed_images:
            img = self.processed_images[self.current_preview_index]
            height, width = img.shape[:2]
            bytes_per_line = width
            if len(img.shape) == 2:
                q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:
                q_image = QImage(img.data, width, height, bytes_per_line * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def show_previous_image(self):
        if self.processed_images and self.current_preview_index > 0:
            self.current_preview_index -= 1
            self.update_preview()

    def show_next_image(self):
        if self.processed_images and self.current_preview_index < len(self.processed_images) - 1:
            self.current_preview_index += 1
            self.update_preview()

    def open_saved_documents_folder(self):
        os.startfile(PDF_STORAGE_FOLDER)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
