import sys
import os
import re
import cv2
import numpy as np
import pytesseract
import fitz
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QMessageBox, QFrame, QInputDialog
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt, QDir

import scanner2
import file_manager

try:
    if "TESSDATA_PREFIX" not in os.environ:
         os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
         pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        pass

    pytesseract.get_tesseract_version()
except pytesseract.TesseractNotFoundError:
    QMessageBox.critical(None, "Tesseract Error",
                         "Tesseract OCR is not installed or not found in your system's PATH.\n"
                         "Please install it from https://github.com/tesseract-ocr/tesseract or update your PATH/config.")
except Exception as e:
     QMessageBox.critical(None, "Tesseract Configuration Error", f"An error occurred while configuring Tesseract: {e}")

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
            img_buffer = pix.samples
            pil_img = Image.frombytes(mode, (pix.width, pix.height), img_buffer)

            if mode == "RGBA":
                 cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
            else:
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            images.append(cv_img)
        doc.close()
        print(f"Successfully loaded {len(images)} pages from {os.path.basename(pdf_path)}")
        return images
    except fitz.FileDataError as e:
         print(f"Error loading PDF {pdf_path} (File Data Error - corrupted or unsupported PDF?): {e}")
         return []
    except Exception as e:
        print(f"An unexpected error occurred while loading PDF {pdf_path}: {e}")
        return []

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dr. Scanner")
        self.setGeometry(200, 100, 1000, 700)
        icon_path = "icons/app_icon.png"
        if os.path.exists(icon_path):
             self.setWindowIcon(QIcon(icon_path))

        self.image_sources = []
        self.original_images = []
        self.processed_images = []
        self.ocr_texts = []

        self.current_preview_index = 0

        self.auto_save_filename_images = ""
        self.auto_save_filename_text = ""

        self.load_button = None
        self.scan_button = None
        self.save_text_pdf_button = None
        self.save_images_pdf_button = None
        self.search_button = None
        self.prev_button = None
        self.next_button = None

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Dr. Scanner")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #0056b3; margin-bottom: 20px;")

        content_layout = QHBoxLayout()

        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(15)

        sidebar_frame = QFrame()
        sidebar_frame.setLayout(sidebar_layout)
        sidebar_frame.setFixedWidth(250)
        sidebar_frame.setStyleSheet("""
            QFrame {
                background-color: #E9ECEF;
                border: 1px solid #D3D3D3;
                border-radius: 8px;
                padding: 5px;
            }
        """)

        sidebar_heading = QLabel("📁 File Management")
        sidebar_heading.setStyleSheet("font-size: 18px; font-weight: bold; color: #343A40; margin-bottom: 5px;")

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("background-color: #F8F9FA; border: 1px solid #D3D3D3; border-radius: 8px; padding: 5px; font-size: 13px;")
        self.list_widget.setFixedHeight(150)
        self.list_widget.currentRowChanged.connect(self.show_selected_image)

        self.load_button = QPushButton("📂 Load Files (Images/PDF)")
        self.scan_button = QPushButton("🔍 Scan Documents & Auto-Save")
        self.save_text_pdf_button = QPushButton("📝 Save Text PDF (Manual)")
        self.save_images_pdf_button = QPushButton("📜 Save Scanned Images PDF (Manual)")
        self.search_button = QPushButton("🔎 View Saved Documents")

        button_style = """
            QPushButton {
                background-color: #007BFF; color: white; border: none; border-radius: 5px; padding: 10px;
                font-weight: bold; font-size: 14px;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """
        for btn in [self.load_button, self.scan_button, self.save_text_pdf_button, self.save_images_pdf_button, self.search_button]:
            btn.setStyleSheet(button_style)

        sidebar_layout.addWidget(sidebar_heading)
        sidebar_layout.addWidget(self.list_widget)
        sidebar_layout.addWidget(self.load_button)
        sidebar_layout.addWidget(self.scan_button)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(self.save_text_pdf_button)
        sidebar_layout.addWidget(self.save_images_pdf_button)
        sidebar_layout.addWidget(self.search_button)

        display_layout = QVBoxLayout()

        image_heading = QLabel("🖼 Document Preview")
        image_heading.setStyleSheet("font-size: 18px; font-weight: bold; color: #343A40; margin-bottom: 5px;")

        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.StyledPanel)
        image_frame.setStyleSheet("background-color: #E9ECEF; border: 2px dashed #6C757D; border-radius: 10px;")
        image_frame_layout = QVBoxLayout(image_frame)
        image_frame_layout.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel("Load files to see preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(600, 400)

        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(0, 10, 0, 0)

        self.prev_button = QPushButton("← Previous")
        self.next_button = QPushButton("Next →")

        for nav_btn in [self.prev_button, self.next_button]:
             nav_btn.setStyleSheet(button_style)
             nav_btn.setFixedWidth(120)
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addStretch()

        image_frame_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        image_frame_layout.addLayout(nav_layout)

        text_heading = QLabel("📝 Extracted Text")
        text_heading.setStyleSheet("font-size: 18px; font-weight: bold; color: #343A40; margin-top: 15px; margin-bottom: 5px;")

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            background-color: #F8F9FA;
            padding: 10px;
            border: 1px solid #D3D3D3;
            border-radius: 8px;
            font-size: 14px;
            font-weight: normal;
            font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
        """)

        display_layout.addWidget(image_heading)
        display_layout.addWidget(image_frame)
        display_layout.addWidget(text_heading)
        display_layout.addWidget(self.text_edit)

        content_layout.addWidget(sidebar_frame)
        content_layout.addLayout(display_layout)

        main_layout.addWidget(title_label)
        main_layout.addLayout(content_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.load_button.clicked.connect(self.load_files)
        self.scan_button.clicked.connect(self.scan_documents)
        self.save_text_pdf_button.clicked.connect(self.save_text_pdf_manual)
        self.save_images_pdf_button.clicked.connect(self.save_images_pdf_manual)
        self.search_button.clicked.connect(self.open_saved_documents_folder)

        self.update_button_states()

    def load_files(self):
        options = QFileDialog.Options()

        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Document Files", "",
            "Images (*.png *.jpg *.jpeg *.bmp);;PDF Files (*.pdf);;All Files (*)",
            options=options
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
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Loading...")
        self.update_button_states(scanning=True)

        loaded_count = 0
        for path in files:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                pages = load_pdf_as_images(path)
                if pages:
                    for i, img in enumerate(pages, start=1):
                        self.original_images.append(img)
                        label = f"{os.path.basename(path)} [page {i}]"
                        self.image_sources.append(label)
                        self.list_widget.addItem(label)
                    loaded_count += len(pages)
                else:
                    self.text_edit.append(f"⚠️ Failed to load PDF or no pages found: {path}\n")
            elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                img = cv2.imread(path)
                if img is not None:
                    self.original_images.append(img)
                    label = os.path.basename(path)
                    self.image_sources.append(label)
                    self.list_widget.addItem(label)
                    loaded_count += 1
                else:
                    self.text_edit.append(f"⚠️ Failed to load image: {path}\n")
            else:
                 self.text_edit.append(f"⚠️ Unsupported file type selected: {path}\n")
            QApplication.processEvents()

        if not self.original_images:
            self.image_label.setText("No files loaded.")
            self.update_button_states()
            return

        suggested_base_name = ""
        if files:
             first_file_name = os.path.basename(files[0])
             suggested_base_name = os.path.splitext(first_file_name)[0]

        base_name, ok = QInputDialog.getText(
            self, "Set Save Filename", "Enter base filename for auto-saved PDFs (e.g., 'MyDocument'). Two files will be saved:\n'[basename]_images.pdf'\n'[basename]_text.pdf'",
            text=suggested_base_name
        )

        if ok and base_name.strip():
            sanitized_base = sanitize_filename(base_name.strip())
            self.auto_save_filename_images = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_base}_images.pdf")
            self.auto_save_filename_text = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_base}_text.pdf")
            self.text_edit.append(f"\n✅ Auto-save filenames set:\n  Images: {os.path.basename(self.auto_save_filename_images)}\n  Text: {os.path.basename(self.auto_save_filename_text)}\n")
        else:
            self.text_edit.append("\nℹ️ No auto-save filename provided. PDFs will not be automatically saved after scanning. Use manual save buttons if needed.\n")
            self.auto_save_filename_images = ""
            self.auto_save_filename_text = ""

        self.image_label.setText("Files Loaded. Ready to scan.")
        self.update_preview(use_original=True)
        self.update_button_states()
        self.text_edit.append(f"\nLoaded {loaded_count} page(s)/image(s).\n")

    def scan_documents(self):
        if not self.original_images:
            QMessageBox.warning(self, "No Files", "Please load images or PDFs first.")
            return

        if len(self.processed_images) == len(self.original_images) and all(img is not None for img in self.processed_images):
             reply = QMessageBox.question(self, 'Already Scanned', 'Documents have already been scanned. Do you want to rescan?',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
             if reply == QMessageBox.No:
                  return

        self.update_button_states(scanning=True)
        self.text_edit.clear()
        self.text_edit.append("Scanning and processing documents...\n")
        QApplication.processEvents()

        self.processed_images.clear()
        self.ocr_texts.clear()

        for idx, img in enumerate(self.original_images):
            try:
                current_source_label = self.image_sources[idx] if idx < len(self.image_sources) else f"Document {idx+1}"
                self.text_edit.append(f"Processing {current_source_label}...")
                self.current_preview_index = idx
                self.update_preview(use_original=True)
                QApplication.processEvents()

                processed = scanner2.preprocess_image(img)
                text = scanner2.extract_text(processed)

                self.processed_images.append(processed)
                self.ocr_texts.append(text)

                snippet = text.replace("\n", " ")[:150]
                if len(text) > 150:
                    snippet += "..."
                self.text_edit.append(f"  ✅ OCR done (Length: {len(text)})")
                if snippet:
                    self.text_edit.append(f"    Snippet: \"{snippet}\"\n")
                else:
                     self.text_edit.append("    (No text found)\n")

            except Exception as e:
                error_msg = f"Error processing {current_source_label}: {e}"
                print(f"ERROR: {error_msg}")
                self.text_edit.append(f"  ❌ {error_msg}\n")
                self.processed_images.append(None)
                self.ocr_texts.append(f"[Error processing document {idx+1}: {e}]\n")
                QApplication.processEvents()

        self.text_edit.append("\n--- Processing Complete ---\n")
        QApplication.processEvents()

        saved_images_path = None
        saved_text_path = None

        if self.auto_save_filename_images or self.auto_save_filename_text:
            self.text_edit.append("\nAttempting automatic PDF saves...\n")
            QApplication.processEvents()

            if self.auto_save_filename_images:
                try:
                    self.text_edit.append(f"Saving processed images PDF to {os.path.basename(self.auto_save_filename_images)}...")
                    QApplication.processEvents()
                    valid_processed_images = [img for img in self.processed_images if img is not None]
                    if valid_processed_images:
                        saved_images_path = file_manager.generate_pdf_scanned_document(
                            valid_processed_images, PDF_STORAGE_FOLDER, self.auto_save_filename_images
                        )
                        self.text_edit.append(f"  ✅ Images PDF saved: {saved_images_path}\n")
                    else:
                        self.text_edit.append("  ℹ️ No valid processed images to save as PDF.\n")

                except Exception as e:
                    save_error = f"Error saving images PDF: {e}"
                    print(f"ERROR: {save_error}")
                    self.text_edit.append(f"  ❌ {save_error}\n")
                QApplication.processEvents()

            if self.auto_save_filename_text:
                try:
                    self.text_edit.append(f"Saving extracted text PDF to {os.path.basename(self.auto_save_filename_text)}...")
                    QApplication.processEvents()
                    if self.ocr_texts and any(self.ocr_texts):
                        saved_text_path = file_manager.generate_pdf_text_only(
                            self.ocr_texts, PDF_STORAGE_FOLDER, self.auto_save_filename_text
                        )
                        self.text_edit.append(f"  ✅ Text PDF saved: {saved_text_path}\n")
                    else:
                         self.text_edit.append("  ℹ️ No OCR text extracted to save as PDF.\n")

                except Exception as e:
                    save_error = f"Error saving text PDF: {e}"
                    print(f"ERROR: {save_error}")
                    self.text_edit.append(f"  ❌ {save_error}\n")
                QApplication.processEvents()

        if self.processed_images and any(img is not None for img in self.processed_images):
            first_valid_index = next((i for i, img in enumerate(self.processed_images) if img is not None), 0)
            self.current_preview_index = first_valid_index
            self.update_preview()
        elif self.original_images:
             self.image_label.setText("Processing failed for all documents.")
             self.image_label.setPixmap(QPixmap())
             self.text_edit.append("All documents failed processing.\n")
        else:
             self.image_label.setText("No files loaded or processed.")
             self.image_label.setPixmap(QPixmap())

        self.update_button_states()

        completion_messages = ["Scanning and Processing Complete."]
        if saved_images_path: completion_messages.append(f"Images PDF: {os.path.basename(saved_images_path)}")
        if saved_text_path: completion_messages.append(f"Text PDF: {os.path.basename(saved_text_path)}")
        if not saved_images_path and not saved_text_path and (self.auto_save_filename_images or self.auto_save_filename_text):
            completion_messages.append("Auto-save failed (check messages).")
        elif not self.auto_save_filename_images and not self.auto_save_filename_text:
             completion_messages.append("No auto-save filename was set.")

        QMessageBox.information(self, "Scan Complete", "\n".join(completion_messages))

    def update_preview(self, use_original=False):
        image_list = self.original_images if use_original else self.processed_images

        if not image_list or self.current_preview_index < 0 or self.current_preview_index >= len(image_list):
             self.image_label.setPixmap(QPixmap())
             if use_original:
                  self.image_label.setText("No original images loaded.")
             else:
                  self.image_label.setText("No processed images available.")
             self.text_edit.setPlainText("")
             self.update_button_states()
             return

        img = image_list[self.current_preview_index]

        if not use_original and img is None:
             self.image_label.setPixmap(QPixmap())
             source_label = self.image_sources[self.current_preview_index] if self.current_preview_index < len(self.image_sources) else f"Document {self.current_preview_index+1}"
             self.image_label.setText(f"Preview unavailable for\n{source_label}\n(Processing failed)")
             self.text_edit.setPlainText(self.ocr_texts[self.current_preview_index] if self.current_preview_index < len(self.ocr_texts) else "Error text unavailable.")
             self.update_button_states()
             return

        if len(img.shape) == 2 or img.shape[2] == 1:
            height, width = img.shape
            bytes_per_line = width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(img.shape) == 3 and img.shape[2] == 3:
             height, width, channel = img.shape
             bytes_per_line = 3 * width
             q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_BGR888)
        else:
             self.image_label.setPixmap(QPixmap())
             self.image_label.setText(f"Unsupported image format for preview: {img.shape}")
             self.text_edit.setPlainText("Cannot display preview due to unsupported image format.")
             self.update_button_states()
             return

        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

        if not use_original:
             if self.current_preview_index < self.list_widget.count():
                  self.list_widget.blockSignals(True)
                  self.list_widget.setCurrentRow(self.current_preview_index)
                  self.list_widget.blockSignals(False)

             if self.current_preview_index < len(self.ocr_texts):
                  self.text_edit.setPlainText(self.ocr_texts[self.current_preview_index])
             else:
                  self.text_edit.setPlainText("No OCR text available for this item.")
        else:
             if self.current_preview_index < self.list_widget.count():
                  self.list_widget.blockSignals(True)
                  self.list_widget.setCurrentRow(self.current_preview_index)
                  self.list_widget.blockSignals(False)
             self.text_edit.setPlainText("Scan documents to see extracted text.")

        self.update_button_states()

    def show_selected_image(self, row):
        if self.processed_images and 0 <= row < len(self.processed_images):
            if row != self.current_preview_index or self.image_label.pixmap() is None:
                 self.current_preview_index = row
                 self.update_preview(use_original=False)

    def show_previous_image(self):
        if self.processed_images and self.current_preview_index > 0:
            self.current_preview_index -= 1
            self.update_preview()

    def show_next_image(self):
        if self.processed_images and self.current_preview_index < len(self.processed_images) - 1:
            self.current_preview_index += 1
            self.update_preview()

    def save_text_pdf_manual(self):
        if not self.ocr_texts or all(not text or text.isspace() or text.strip().startswith("[Error processing") for text in self.ocr_texts):
            QMessageBox.warning(self, "No Text", "No significant text has been extracted yet. Please scan documents first or check for processing errors.")
            return

        suggested_name = "extracted_text"
        if self.auto_save_filename_text:
             suggested_name = os.path.splitext(os.path.basename(self.auto_save_filename_text))[0]
             if suggested_name.endswith("_text"):
                  suggested_name = suggested_name[:-len("_text")]

        filename, ok = QInputDialog.getText(
            self, "Save Text PDF", "Enter filename for the text PDF (without extension):",
            text=suggested_name
        )
        if ok and filename.strip():
            sanitized_filename = sanitize_filename(filename.strip())
            pdf_path = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_filename}_text.pdf")
            try:
                saved_path = file_manager.generate_pdf_text_only(self.ocr_texts, PDF_STORAGE_FOLDER, pdf_path)
                QMessageBox.information(self, "Save Complete", f"Extracted text PDF saved as:\n{saved_path}")
            except Exception as e:
                 QMessageBox.critical(self, "Save Error", f"Failed to save text PDF:\n{e}")
        elif ok:
             QMessageBox.warning(self, "Filename Missing", "Save cancelled. No filename provided.")

    def save_images_pdf_manual(self):
        valid_processed_images = [img for img in self.processed_images if img is not None]
        if not valid_processed_images:
            QMessageBox.warning(self, "No Images", "No valid processed images available. Please scan documents first.")
            return

        suggested_name = "scanned_images"
        if self.auto_save_filename_images:
             suggested_name = os.path.splitext(os.path.basename(self.auto_save_filename_images))[0]
             if suggested_name.endswith("_images"):
                  suggested_name = suggested_name[:-len("_images")]

        filename, ok = QInputDialog.getText(
            self, "Save Images PDF", "Enter filename for the images PDF (without extension):",
            text=suggested_name
        )
        if ok and filename.strip():
            sanitized_filename = sanitize_filename(filename.strip())
            pdf_path = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_filename}_images.pdf")
            try:
                saved_path = file_manager.generate_pdf_scanned_document(valid_processed_images, PDF_STORAGE_FOLDER, pdf_path)
                QMessageBox.information(self, "Save Complete", f"Scanned images PDF saved as:\n{saved_path}")
            except Exception as e:
                 QMessageBox.critical(self, "Save Error", f"Failed to save images PDF:\n{e}")
        elif ok:
             QMessageBox.warning(self, "Filename Missing", "Save cancelled. No filename provided.")

    def open_saved_documents_folder(self):
        target_folder = os.path.abspath(PDF_STORAGE_FOLDER)
        if not os.path.exists(target_folder):
            try:
                os.makedirs(target_folder, exist_ok=True)
                QMessageBox.information(self, "Folder Created", f"Created folder: {os.path.basename(target_folder)}\nIt is currently empty.")
            except Exception as e:
                 QMessageBox.critical(self, "Error Creating Folder", f"Could not create folder:\n{target_folder}\nError: {e}")
                 return

        try:
             from PyQt5.QtGui import QDesktopServices
             from PyQt5.QtCore import QUrl
             QDesktopServices.openUrl(QUrl.fromLocalFile(target_folder))
        except ImportError:
             try:
                if sys.platform == "win32":
                    os.startfile(target_folder)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.call(["open", target_folder])
                else:
                    import subprocess
                    subprocess.call(["xdg-open", target_folder])
             except Exception as e:
                QMessageBox.critical(self, "Error Opening Folder", f"Could not open folder:\n{target_folder}\nError: {e}")

    def update_button_states(self, scanning=False):
        has_original_images = len(self.original_images) > 0
        has_processed_images = len(self.processed_images) > 0
        has_valid_processed_images = has_processed_images and any(img is not None for img in self.processed_images)
        has_significant_ocr_text = len(self.ocr_texts) > 0 and any(text and not text.isspace() and not text.strip().startswith("[Error processing") for text in self.ocr_texts)

        if self.load_button:
             self.load_button.setEnabled(not scanning)
        if self.scan_button:
             self.scan_button.setEnabled(has_original_images and not scanning)
        if self.save_text_pdf_button:
             self.save_text_pdf_button.setEnabled(has_significant_ocr_text and not scanning)
        if self.save_images_pdf_button:
             self.save_images_pdf_button.setEnabled(has_valid_processed_images and not scanning)
        if self.search_button:
             self.search_button.setEnabled(True)

        if self.prev_button:
             self.prev_button.setEnabled(has_valid_processed_images and self.current_preview_index > 0 and not scanning)
        if self.next_button:
             self.next_button.setEnabled(has_valid_processed_images and self.current_preview_index < len(self.processed_images) - 1 and not scanning)

        if self.list_widget:
             self.list_widget.setEnabled(has_processed_images and not scanning)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())