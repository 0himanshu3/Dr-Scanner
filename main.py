import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QWidget, QHBoxLayout, QInputDialog, QListWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
import scanner
import file_manager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DocScanner")
        self.setStyleSheet("""
                QMainWindow {
                    background-color: #87CEEB; /* Sky Blue */
                }
                QLabel {
                    color: #333333;
                }
                QPushButton {
                    background-color: #ADD8E6;
                    border: 1px solid #1E90FF;
                    border-radius: 5px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #87CEFA;
                }
                QTextEdit {
                    background-color: #F0FFFF;
                    border: 1px solid #B0E0E6;
                }
                QListWidget {
                    background-color: #F0F8FF;
                    border: 1px solid #B0C4DE;
                }
            """)
        # Lists to store loaded image file paths, images, processed images, and OCR texts
        self.image_paths = []
        self.images = []
        self.processed_images = []
        self.ocr_texts = []

        # UI Elements
        self.image_label = QLabel("")
        self.image_label.setFixedSize(600, 400)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        # List widget to display file names of loaded images
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(200)
        self.list_widget.itemClicked.connect(self.display_selected_image)

        load_button = QPushButton("Load Images")
        load_button.clicked.connect(self.load_images)

        scan_button = QPushButton("Scan Documents")
        scan_button.clicked.connect(self.scan_documents)

        save_text_pdf_button = QPushButton("Save PDF with Extracted Text")
        save_text_pdf_button.clicked.connect(self.save_all_output)

        save_images_pdf_button = QPushButton("Save Scanned Images PDF")
        save_images_pdf_button.clicked.connect(self.save_images_pdf)

        search_button = QPushButton("Search Documents")
        search_button.clicked.connect(self.search_documents)

        # Layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(load_button)
        btn_layout.addWidget(scan_button)
        btn_layout.addWidget(save_text_pdf_button)
        btn_layout.addWidget(save_images_pdf_button)
        btn_layout.addWidget(search_button)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.list_widget)
        left_layout.addLayout(btn_layout)
        left_layout.addWidget(self.text_edit)
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_images(self):
        """
        Opens a file dialog to select multiple images and displays their file names.
        """
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Image Files", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options
        )
        if files:
            self.image_paths = files
            self.images = []
            self.processed_images = []
            self.ocr_texts = []
            self.list_widget.clear()
            for file in files:
                img = cv2.imread(file)
                if img is not None:
                    self.images.append(img)
                    self.list_widget.addItem(file)
                else:
                    self.text_edit.append(f"Failed to load: {file}")

    def scan_documents(self):
        """
        Processes all loaded images using document detection and OCR.
        """
        if not self.images:
            self.text_edit.setText("Please load images first.")
            return

        self.processed_images = []
        self.ocr_texts = []
        self.text_edit.clear()

        for idx, img in enumerate(self.images):
            processed = scanner.preprocess_image(img)
            text = scanner.extract_text(processed)
            self.processed_images.append(processed)
            self.ocr_texts.append(text)
            self.text_edit.append(f"Processed {self.image_paths[idx]}\nExtracted text length: {len(text)}\n")

        QMessageBox.information(self, "Scan Complete", "All images have been processed.")

    def save_all_output(self):
        """
        Generates a PDF that contains the extracted text of each image on separate pages.
        Prompts the user for a custom PDF name and saves the PDF in the output directory.
        """
        if not self.ocr_texts:
            self.text_edit.setText("No processed documents to save. Please scan documents first.")
            return

        output_dir = file_manager.create_output_directory()

        # Prompt the user for a custom PDF name.
        pdf_name, ok = QInputDialog.getText(self, "PDF File Name", "Enter PDF file name (without extension):")
        if ok and pdf_name:
            if not pdf_name.lower().endswith('.pdf'):
                pdf_name += ".pdf"
            pdf_filename = os.path.join(output_dir, pdf_name)
        else:
            pdf_filename = os.path.join(output_dir, "extracted_texts.pdf")

        pdf_file = file_manager.generate_pdf_text_only(self.ocr_texts, output_dir, pdf_filename)
        self.text_edit.append(f"\nGenerated PDF with extracted text: {pdf_file}")
        QMessageBox.information(self, "Save Complete", "Extracted text PDF generated successfully.")

    def save_images_pdf(self):
        """
        Generates a PDF containing the scanned images (one per page).
        Prompts the user for a custom PDF name and saves the PDF in the output directory.
        """
        if not self.processed_images:
            self.text_edit.setText("No processed images to save. Please scan documents first.")
            return

        output_dir = file_manager.create_output_directory()

        # Prompt the user for a custom PDF name.
        pdf_name, ok = QInputDialog.getText(self, "PDF File Name", "Enter PDF file name for scanned images (without extension):")
        if ok and pdf_name:
            if not pdf_name.lower().endswith('.pdf'):
                pdf_name += ".pdf"
            pdf_filename = os.path.join(output_dir, pdf_name)
        else:
            pdf_filename = os.path.join(output_dir, "scanned_documents.pdf")

        pdf_file = file_manager.generate_pdf_scanned_document(self.processed_images, output_dir, pdf_filename)
        self.text_edit.append(f"\nGenerated PDF with scanned images: {pdf_file}")
        QMessageBox.information(self, "Save Complete", "Scanned images PDF generated successfully.")

    def search_documents(self):
        """
        Prompts the user for a search query and displays the results from the stored documents.
        """
        query, ok = QInputDialog.getText(self, "Search Documents", "Enter search query:")
        if ok and query:
            results = file_manager.search_documents(query)
            if results:
                result_text = "Search Results:\n"
                for file_path, snippet in results:
                    result_text += f"\nFile: {file_path}\nContext: {snippet}\n"
            else:
                result_text = "No documents found containing the query."
            self.text_edit.setText(result_text)

    def display_selected_image(self, item):
        """
        Displays the processed image corresponding to the selected file in the list widget.
        """
        idx = self.list_widget.row(item)
        if idx < len(self.processed_images) and self.processed_images[idx] is not None:
            self.display_image(self.processed_images[idx])
        elif idx < len(self.images):
            self.display_image(self.images[idx])

    def display_image(self, cv_img):
        """
        Converts a cv2 image into a QImage and displays it in the QLabel.
        """
        if cv_img is None:
            return
        if len(cv_img.shape) == 2:
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            height, width, ch = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
