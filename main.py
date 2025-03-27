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
import file_manager

# Define a folder 
PDF_STORAGE_FOLDER = "scanned_documents"
TEXT_STORAGE_FILE = os.path.join(PDF_STORAGE_FOLDER, "scanned_texts.txt")

# Ensure the storage folder exists
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
        self.current_preview_index = 0  # To keep track of which image is being shown

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Title Section
        title_label = QLabel("Dr. Scanner")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007BFF; margin-bottom: 15px;")

        # Main Content Layout
        content_layout = QHBoxLayout()

        # Sidebar Layout
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
        save_text_pdf_button = QPushButton("📝 Save Extracted Text PDF")
        save_images_pdf_button = QPushButton("📜 Save Scanned Images PDF")
        search_button = QPushButton("🔎 View Saved Documents")

        # Button Styling
        button_style = """
            QPushButton {
                background-color: #007BFF; color: white; border-radius: 5px; padding: 8px;
                font-weight: bold; font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """
        for btn in [load_button, scan_button, save_text_pdf_button, save_images_pdf_button, search_button]:
            btn.setStyleSheet(button_style)
            btn.setFixedHeight(50)

        # Main Display Layout
        display_layout = QVBoxLayout()

        image_heading = QLabel("🖼 Document Preview")
        image_heading.setStyleSheet("font-size: 16px; font-weight: bold; color: #343A40;")

        self.image_label = QLabel("")
        self.image_label.setFixedSize(600, 400)
        self.image_label.setStyleSheet("border: 2px dashed #6C757D; background-color: #E9ECEF; border-radius: 10px;")

        # Navigation Buttons Layout
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

        # Add sidebar widgets
        sidebar.addWidget(sidebar_heading)
        sidebar.addWidget(self.list_widget)
        sidebar.addWidget(load_button)
        sidebar.addWidget(scan_button)
        sidebar.addWidget(save_text_pdf_button)
        sidebar.addWidget(save_images_pdf_button)
        sidebar.addWidget(search_button)
        sidebar.addStretch()

        # Add display widgets
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

        # Connect Buttons
        load_button.clicked.connect(self.load_images)
        scan_button.clicked.connect(self.scan_documents)
        save_text_pdf_button.clicked.connect(self.save_all_output)
        save_images_pdf_button.clicked.connect(self.save_images_pdf)
        search_button.clicked.connect(self.open_saved_documents_folder)

    def load_images(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Open Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if files:
            self.image_paths = files
            self.images = []
            self.processed_images = []
            self.ocr_texts = []
            self.current_preview_index = 0  # Reset preview index
            self.list_widget.clear()
            for file in files:
                img = cv2.imread(file)
                if img is not None:
                    self.images.append(img)
                    self.list_widget.addItem(os.path.basename(file))
                else:
                    self.text_edit.append(f"Failed to load: {file}")

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
            self.text_edit.append(f"✅ Processed {os.path.basename(self.image_paths[idx])}\n📝 Extracted text length: {len(text)}\n")

        # Reset preview index and update preview to first processed image 
        if self.processed_images:
            self.current_preview_index = 0
            self.update_preview()

        # Append all extracted texts 
        with open(TEXT_STORAGE_FILE, "a", encoding="utf-8") as file:
            for text in self.ocr_texts:
                file.write(text + "\n" + "-" * 50 + "\n")

        QMessageBox.information(self, "Scan Complete", "All images have been processed.")

    def update_preview(self):
        if self.processed_images:
            img = self.processed_images[self.current_preview_index]
            height, width = img.shape
            bytes_per_line = width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
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

    def save_all_output(self):
        if not self.ocr_texts:
            self.text_edit.setText("No processed documents to save. Please scan documents first.")
            return

        filename, _ = QInputDialog.getText(self, "Save File", "Enter filename for the extracted text PDF:")
        if filename:
            pdf_filename = os.path.join(PDF_STORAGE_FOLDER, f"{filename}.pdf")
            file_manager.generate_pdf_text_only(self.ocr_texts, PDF_STORAGE_FOLDER, pdf_filename)
            QMessageBox.information(self, "Save Complete", f"Extracted text PDF saved as {pdf_filename}")

    def save_images_pdf(self):
        if not self.processed_images:
            self.text_edit.setText("No processed images to save. Please scan documents first.")
            return

        filename, _ = QInputDialog.getText(self, "Save File", "Enter filename for the scanned images PDF:")
        if filename:
            pdf_filename = os.path.join(PDF_STORAGE_FOLDER, f"{filename}.pdf")
            file_manager.generate_pdf_scanned_document(self.processed_images, PDF_STORAGE_FOLDER, pdf_filename)
            QMessageBox.information(self, "Save Complete", f"Scanned images PDF saved as {pdf_filename}")

    def open_saved_documents_folder(self):
        os.startfile(PDF_STORAGE_FOLDER)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
