import sys
import cv2
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

        # Lists to store loaded image file paths, images, processed images, and OCR texts
        self.image_paths = []
        self.images = []
        self.processed_images = []
        self.ocr_texts = []

        # UI Elements
        self.image_label = QLabel("No Image Loaded")
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

        save_button = QPushButton("Save & Generate PDF with Selectable Text")
        save_button.clicked.connect(self.save_all_output)

        search_button = QPushButton("Search Documents")
        search_button.clicked.connect(self.search_documents)

        # Layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(load_button)
        btn_layout.addWidget(scan_button)
        btn_layout.addWidget(save_button)
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
        Saves all processed images and OCR texts, then generates a PDF with:
          - Each page showing the scanned image
          - An invisible text layer containing the OCR text (selectable/copyable)
        """
        if not self.processed_images or not self.ocr_texts:
            self.text_edit.setText("No processed documents to save. Please scan documents first.")
            return

        output_dir = file_manager.create_output_directory()
        saved_image_files = []
        for idx, processed_img in enumerate(self.processed_images):
            img_file, txt_file = file_manager.save_scanned_document(processed_img, self.ocr_texts[idx], output_dir)
            saved_image_files.append(img_file)
            self.text_edit.append(f"Saved image: {img_file}\nSaved text: {txt_file}\n")

        pdf_file = file_manager.generate_pdf_with_image_and_invisible_text(saved_image_files, self.ocr_texts,
                                                                           output_dir)
        self.text_edit.append(f"\nGenerated PDF with selectable text: {pdf_file}")
        QMessageBox.information(self, "Save Complete", "All documents saved and PDF generated.")

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
