# In main.py
import sys
import os
import re
import cv2
import numpy as np
# import pytesseract # Removed
import fitz
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QTextEdit,
    QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QMessageBox, QFrame,
    QInputDialog, QProgressDialog
)
from PyQt5.QtGui import QPixmap, QIcon, QImage, QDesktopServices
from PyQt5.QtCore import (
    Qt, QDir, QThread, pyqtSignal, QObject, QUrl
)

import scanner # Uses updated scanner.py
import file_manager2 # Uses updated file_manager2.py

# Tesseract config removed

PDF_STORAGE_FOLDER = "scanned_documents"
os.makedirs(PDF_STORAGE_FOLDER, exist_ok=True)

# sanitize_filename function remains the same
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)

# load_pdf_as_images function remains the same
def load_pdf_as_images(pdf_path):
    # ... (implementation from previous step) ...
    images = []
    print(f"Loading PDF: {os.path.basename(pdf_path)}...")
    try:
        doc = fitz.open(pdf_path)
        dpi = 300; zoom = dpi / 72.0; matrix = fitz.Matrix(zoom, zoom)
        num_pages = len(doc)
        for page_index in range(num_pages):
            page = doc[page_index]
            print(f"  Loading page {page_index + 1}/{num_pages}...")
            QApplication.processEvents()
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_buffer = pix.samples
            try:
                pil_img = Image.frombytes("RGB", (pix.width, pix.height), img_buffer)
            except ValueError as e:
                 print(f"  Error creating PIL Image from pixmap samples (page {page_index + 1}): {e}")
                 continue
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            images.append(cv_img)
        doc.close()
        print(f"Successfully loaded {len(images)} pages from {os.path.basename(pdf_path)}")
        return images
    except fitz.FileDataError as e:
         print(f"Error loading PDF {pdf_path} (File Data Error): {e}")
         QMessageBox.warning(None, "PDF Load Error", f"Failed to load {os.path.basename(pdf_path)}.\nLikely corrupted or unsupported.\nError: {e}")
         return []
    except Exception as e:
        print(f"An unexpected error occurred while loading PDF {pdf_path}: {e}")
        QMessageBox.critical(None, "PDF Load Error", f"Unexpected error loading {os.path.basename(pdf_path)}.\nError: {e}")
        return []


# ScanWorker class remains the same (already emits detailed OCR data)
class ScanWorker(QObject):
    progress_update = pyqtSignal(int, str)
    page_processed = pyqtSignal(int, object, list) # index, processed_image, ocr_data_list
    finished = pyqtSignal()
    error = pyqtSignal(str)
    # ... (run, stop methods remain the same) ...
    def __init__(self, original_images, image_sources):
        super().__init__()
        self.original_images = original_images
        self.image_sources = image_sources
        self._is_running = True
    def run(self):
        try:
            total_pages = len(self.original_images)
            for idx, img in enumerate(self.original_images):
                if not self._is_running:
                    self.progress_update.emit(idx, "Scan cancelled.")
                    break
                current_source_label = self.image_sources[idx] if idx < len(self.image_sources) else f"Document {idx+1}"
                self.progress_update.emit(idx, f"Processing {current_source_label} ({idx+1}/{total_pages})...")
                processed = None; ocr_data = []
                try:
                    if scanner.reader is None: raise RuntimeError("EasyOCR Reader not initialized.")
                    processed = scanner.preprocess_image(img)
                    if processed is None: raise ValueError("Preprocessing returned None.")
                    ocr_data = scanner.get_ocr_data(processed)
                    self.page_processed.emit(idx, processed, ocr_data)
                    page_text_for_snippet = scanner.extract_text_from_ocr_data(ocr_data)
                    snippet = page_text_for_snippet.replace("\n", " ")[:80] + ("..." if len(page_text_for_snippet) > 80 else "")
                    self.progress_update.emit(idx, f"  ✅ OCR done for {current_source_label}. Boxes: {len(ocr_data)}. Snippet: '{snippet}'")
                except Exception as e:
                    error_msg = f"Error processing {current_source_label}: {e}"
                    print(f"ERROR in worker for page {idx+1}: {error_msg}")
                    self.page_processed.emit(idx, None, [])
                    self.progress_update.emit(idx, f"  ❌ Error on {current_source_label}: {e}")
            if self._is_running: self.progress_update.emit(total_pages, "--- All pages processed ---")
        except Exception as e:
            critical_error_msg = f"Critical error during scanning thread: {e}"; print(f"CRITICAL ERROR in worker: {critical_error_msg}")
            self.error.emit(critical_error_msg)
        finally: self.finished.emit()
    def stop(self): self._is_running = False


# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... (Window setup) ...
        self.setWindowTitle("Dr. Scanner - OCR & PDF Tools")
        self.setGeometry(150, 80, 1150, 780) # Slightly wider

        # Data Storage
        self.image_sources = []
        self.original_images = []      # Keep original dimensions needed for structured text
        self.processed_images = []
        self.ocr_data_results = []     # Stores [(bbox, text, conf),...] per page
        self.ocr_texts_preview = []    # Stores plain text string per page for display

        self.current_preview_index = -1

        # === Updated Auto-save filenames ===
        self.auto_save_filename_searchable = "" # For image + invisible text
        self.auto_save_filename_structured_text = "" # For text-only, layout preserved

        # Threading objects
        self.scan_thread = None
        self.scan_worker = None
        self.progress_dialog = None

        # UI Elements
        self.load_button = None
        self.scan_button = None
        self.save_searchable_pdf_button = None
        self.save_structured_text_pdf_button = None # New button
        # self.save_images_pdf_button = None # Removed image-only button
        self.search_button = None # Renamed to "Open Folder"
        self.prev_button = None
        self.next_button = None
        self.list_widget = None
        self.image_label = None
        self.text_edit = None

        self.initUI()

        # Check OCR reader status after UI built
        if hasattr(scanner, 'reader') and scanner.reader is None:
             QMessageBox.critical(self, "OCR Initialization Error", "EasyOCR failed. Scanning disabled.")
             if self.scan_button: self.scan_button.setEnabled(False)


    def initUI(self):
        # ... (Main layout, title, content_layout setup) ...
        main_layout = QVBoxLayout(); main_layout.setContentsMargins(15, 15, 15, 15); main_layout.setSpacing(15)
        title_label = QLabel("Dr. Scanner"); title_label.setAlignment(Qt.AlignCenter); title_label.setStyleSheet("font-size: 26px; font-weight: bold; color: #2A5A8C; margin-bottom: 15px;")
        content_layout = QHBoxLayout()

        # --- Sidebar ---
        sidebar_layout = QVBoxLayout(); sidebar_layout.setContentsMargins(10, 10, 10, 10); sidebar_layout.setSpacing(12)
        sidebar_frame = QFrame(); sidebar_frame.setLayout(sidebar_layout); sidebar_frame.setFixedWidth(270); # Wider
        sidebar_frame.setStyleSheet("QFrame { background-color: #F0F0F0; border: 1px solid #C8C8C8; border-radius: 6px; padding: 8px; }")
        sidebar_heading = QLabel("📁 File Operations"); sidebar_heading.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-bottom: 8px;")
        self.list_widget = QListWidget(); self.list_widget.setStyleSheet("QListWidget { background-color: #FFFFFF; border: 1px solid #C8C8C8; border-radius: 4px; padding: 5px; font-size: 12px; } QListWidget::item:selected { background-color: #D0E0F0; color: #000000; }")
        self.list_widget.currentRowChanged.connect(self.show_selected_image)

        # --- Define Buttons ---
        self.load_button = QPushButton("📂 Load Files")
        self.scan_button = QPushButton("⚙️ Process Documents") # Changed icon/text
        self.save_searchable_pdf_button = QPushButton("💾 Save Searchable PDF") # Image + Text
        # === New Button ===
        self.save_structured_text_pdf_button = QPushButton("📄 Save Structured Text PDF") # Text only, layout
        # self.save_images_pdf_button = QPushButton("🖼️ Save Images PDF") # Removed/Commented
        self.search_button = QPushButton("↗️ Open Save Folder") # Changed text/icon

        button_style = """
            QPushButton { /* Base style */
                background-color: #4A7DBB; color: white; border: none; border-radius: 4px;
                padding: 9px 12px; font-weight: bold; font-size: 13px;
                min-height: 32px; text-align: left; padding-left: 15px;
            }
            QPushButton:hover { background-color: #3A6AA0; }
            QPushButton:disabled { background-color: #D0D0D0; color: #777777; }
        """
        # Apply style
        buttons_in_sidebar = [self.load_button, self.scan_button, self.save_searchable_pdf_button,
                              self.save_structured_text_pdf_button, self.search_button]
        for btn in buttons_in_sidebar:
             if btn: btn.setStyleSheet(button_style) # Check if button exists (if one was removed)

        # --- Add Widgets to Sidebar ---
        sidebar_layout.addWidget(sidebar_heading)
        sidebar_layout.addWidget(self.list_widget) # Allow list to take available space
        sidebar_layout.addWidget(self.load_button)
        sidebar_layout.addWidget(self.scan_button)
        sidebar_layout.addStretch(1) # Spacer
        sidebar_layout.addWidget(self.save_searchable_pdf_button)
        sidebar_layout.addWidget(self.save_structured_text_pdf_button) # Add new button
        # if self.save_images_pdf_button: sidebar_layout.addWidget(self.save_images_pdf_button) # Add if kept
        sidebar_layout.addWidget(self.search_button)


        # --- Display Area (Preview + Text) ---
        display_layout = QVBoxLayout(); display_layout.setSpacing(10)
        # Preview Area
        preview_area_layout = QVBoxLayout()
        image_heading = QLabel("🖼️ Document Preview"); image_heading.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-bottom: 5px;")
        image_frame = QFrame(); image_frame.setFrameShape(QFrame.StyledPanel); image_frame.setStyleSheet("background-color: #FFFFFF; border: 1px solid #C8C8C8; border-radius: 4px;")
        image_frame_layout = QVBoxLayout(image_frame); image_frame_layout.setContentsMargins(5, 5, 5, 5); image_frame_layout.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel("Load files and process to see preview"); self.image_label.setAlignment(Qt.AlignCenter); self.image_label.setMinimumSize(500, 350); self.image_label.setStyleSheet("color: #555;")
        self.image_label.setScaledContents(False)
        nav_layout = QHBoxLayout(); nav_layout.setContentsMargins(0, 5, 0, 0)
        self.prev_button = QPushButton("← Previous"); self.next_button = QPushButton("Next →")
        nav_button_style = "QPushButton { background-color: #6c757d; color: white; border-radius: 3px; padding: 5px 10px; font-size: 12px; max-width: 100px; } QPushButton:hover { background-color: #5a6268; } QPushButton:disabled { background-color: #E0E0E0; color: #999; }"
        self.prev_button.setStyleSheet(nav_button_style); self.next_button.setStyleSheet(nav_button_style)
        self.prev_button.clicked.connect(self.show_previous_image); self.next_button.clicked.connect(self.show_next_image)
        nav_layout.addStretch(); nav_layout.addWidget(self.prev_button); nav_layout.addWidget(self.next_button); nav_layout.addStretch()
        image_frame_layout.addWidget(self.image_label, 1); image_frame_layout.addLayout(nav_layout)
        preview_area_layout.addWidget(image_heading); preview_area_layout.addWidget(image_frame, 1)
        # Text Area
        text_area_layout = QVBoxLayout()
        text_heading = QLabel("📝 Extracted Text (Preview)"); text_heading.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-top: 10px; margin-bottom: 5px;")
        self.text_edit = QTextEdit(); self.text_edit.setReadOnly(True); self.text_edit.setStyleSheet("QTextEdit { background-color: #FDFDFD; border: 1px solid #C8C8C8; border-radius: 4px; padding: 8px; font-size: 12px; font-family: Consolas, Menlo, Monaco, 'Courier New', monospace; color: #222; }")
        self.text_edit.setMinimumHeight(120)
        text_area_layout.addWidget(text_heading); text_area_layout.addWidget(self.text_edit, 1)
        # Add preview and text areas to display layout
        display_layout.addLayout(preview_area_layout, 3); display_layout.addLayout(text_area_layout, 1)

        # --- Assemble Main Window ---
        content_layout.addWidget(sidebar_frame); content_layout.addLayout(display_layout, 1)
        main_layout.addWidget(title_label); main_layout.addLayout(content_layout)
        container = QWidget(); container.setLayout(main_layout); self.setCentralWidget(container)

        # --- Connect Signals ---
        self.load_button.clicked.connect(self.load_files)
        self.scan_button.clicked.connect(self.start_scan_thread)
        self.save_searchable_pdf_button.clicked.connect(self.save_searchable_pdf_manual)
        # === Connect NEW Button ===
        self.save_structured_text_pdf_button.clicked.connect(self.save_structured_text_pdf_manual)
        # if self.save_images_pdf_button: self.save_images_pdf_button.clicked.connect(self.save_images_pdf_manual) # Connect if kept
        self.search_button.clicked.connect(self.open_saved_documents_folder)

        self.update_button_states() # Set initial button states


    def load_files(self):
        # ... (Check for running scan - same as before) ...
        if self.scan_thread and self.scan_thread.isRunning():
             QMessageBox.warning(self, "Busy", "Processing active. Wait or cancel."); return

        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Document Files", QDir.homePath(), "All Supported (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pdf);;Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;PDF (*.pdf);;All (*)", options=options)
        if not files: return

        # --- Reset state ---
        self.image_sources.clear(); self.original_images.clear(); self.processed_images.clear()
        self.ocr_data_results.clear(); self.ocr_texts_preview.clear(); self.list_widget.clear()
        self.current_preview_index = -1; self.text_edit.clear()
        self.image_label.setPixmap(QPixmap()); self.image_label.setText("Loading files...")
        # === Reset BOTH auto-save paths ===
        self.auto_save_filename_searchable = ""
        self.auto_save_filename_structured_text = ""
        # self.auto_save_filename_images = "" # Remove if image-only save removed
        self.update_button_states(loading=True)
        QApplication.processEvents()

        # --- Load files with progress ---
        loaded_image_count = 0; total_files = len(files); progress_step = 100 / total_files if total_files > 0 else 0
        loading_progress = QProgressDialog("Loading files...", "Cancel", 0, 100, self); loading_progress.setWindowModality(Qt.WindowModal); loading_progress.setMinimumDuration(500); loading_progress.setValue(0)
        loading_cancelled = False

        for i, path in enumerate(files):
            if loading_progress.wasCanceled(): loading_cancelled = True; break
            loading_progress.setLabelText(f"Loading: {os.path.basename(path)}"); loading_progress.setValue(int(i * progress_step)); QApplication.processEvents()
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf": pages = load_pdf_as_images(path) # Handles errors internally
            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                try: img = cv2.imread(path); pages = [img] if img is not None else []
                except Exception as img_e: print(f"Error reading {path}: {img_e}"); pages = []
                if not pages: self.text_edit.append(f"⚠️ Failed to load image: {path}\n")
            else: self.text_edit.append(f"⚠️ Unsupported file skipped: {path}\n"); pages = []
            if pages:
                for page_num, img in enumerate(pages, start=1): self.original_images.append(img); label = f"{os.path.basename(path)}" + (f" [Page {page_num}]" if len(pages) > 1 else ""); self.image_sources.append(label)
                loaded_image_count += len(pages)

        loading_progress.close()
        if loading_cancelled: self.text_edit.append("\n--- Loading cancelled. ---"); self.image_label.setText("Loading cancelled."); self.original_images.clear(); self.image_sources.clear(); self.update_button_states(); return
        if not self.original_images: self.image_label.setText("No valid files loaded."); self.update_button_states(); QMessageBox.warning(self, "Loading Failed", "No valid images/PDFs loaded."); return

        # --- Init placeholders & list ---
        num_pages = len(self.original_images); self.processed_images = [None] * num_pages; self.ocr_data_results = [[] for _ in range(num_pages)]; self.ocr_texts_preview = [""] * num_pages
        self.list_widget.clear(); self.list_widget.addItems(self.image_sources)

        # --- Ask for Auto-Save Base Name (Updated Prompt) ---
        suggested_base_name = ""; first_file = files[0] if files else ""
        if first_file: suggested_base_name = re.sub(r'\s*\[Page\s*\d+\]$', '', os.path.splitext(os.path.basename(first_file))[0]).strip()
        base_name, ok = QInputDialog.getText(self, "Set Auto-Save Base Name",
            "Enter base name for auto-saved PDFs (e.g., 'Report_XYZ'). Blank=disable.\n\n"
            "If enabled, files created after processing:\n"
            "  • [basename]_searchable.pdf (Image + Text)\n"
            "  • [basename]_structured_text.pdf (Text Only, Layout)",
            text=suggested_base_name)

        if ok and base_name.strip():
            sanitized_base = sanitize_filename(base_name.strip())
            # === Set BOTH auto-save paths ===
            self.auto_save_filename_searchable = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_base}_searchable.pdf")
            self.auto_save_filename_structured_text = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_base}_structured_text.pdf")
            # self.auto_save_filename_images = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_base}_images.pdf") # Remove if image-only save removed
            self.text_edit.append(f"\n✅ Auto-save enabled ('{sanitized_base}').\n"
                                  f"   Searchable: ...{os.path.basename(self.auto_save_filename_searchable)}\n"
                                  f"   Structured Text: ...{os.path.basename(self.auto_save_filename_structured_text)}\n")
        else:
            self.text_edit.append("\nℹ️ Auto-save disabled.\n"); self.auto_save_filename_searchable = ""; self.auto_save_filename_structured_text = ""

        # --- Final UI Update ---
        self.image_label.setText(f"{loaded_image_count} page(s) loaded. Ready to process.")
        self.current_preview_index = 0; self.update_preview(use_original=True)
        self.update_button_states(); self.text_edit.append(f"\n--- Loaded {loaded_image_count} total page(s) ---\n")


    def start_scan_thread(self):
        # ... (Checks for no files, OCR reader status remain same) ...
        if not self.original_images: QMessageBox.warning(self, "No Files", "Load files first."); return
        if hasattr(scanner, 'reader') and scanner.reader is None: QMessageBox.critical(self, "OCR Error", "OCR engine unavailable."); return

        # ... (Ask to reprocess logic remains same) ...
        if any(img is not None for img in self.processed_images) or any(data for data in self.ocr_data_results):
             reply = QMessageBox.question(self, 'Reprocess?', 'Reprocess all documents?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
             if reply == QMessageBox.No: self.update_preview(use_original=False); return
             else: # Clear old results
                 num = len(self.original_images); self.processed_images=[None]*num; self.ocr_data_results=[[] for _ in range(num)]; self.ocr_texts_preview=[""]*num; self.text_edit.clear(); self.text_edit.append("Reprocessing...\n"); self.update_preview(use_original=True)

        # --- Setup Progress Dialog & Worker Thread ---
        num_items = len(self.original_images)
        self.progress_dialog = QProgressDialog("Processing...", "Cancel", 0, num_items, self); self.progress_dialog.setWindowModality(Qt.WindowModal); self.progress_dialog.setMinimumDuration(1000); self.progress_dialog.setValue(0); self.progress_dialog.setAutoReset(True); self.progress_dialog.setAutoClose(True)
        self.scan_worker = ScanWorker(self.original_images, self.image_sources); self.scan_thread = QThread(); self.scan_worker.moveToThread(self.scan_thread)
        self.scan_worker.progress_update.connect(self.update_scan_progress); self.scan_worker.page_processed.connect(self.handle_page_processed); self.scan_worker.finished.connect(self.scan_finished); self.scan_worker.error.connect(self.scan_error); self.scan_thread.started.connect(self.scan_worker.run); self.progress_dialog.canceled.connect(self.cancel_scan)
        self.update_button_states(scanning=True); self.text_edit.append("\n--- Starting Processing ---\n"); self.scan_thread.start(); self.progress_dialog.show()


    def update_scan_progress(self, index, message):
        # ... (remains the same) ...
        if self.progress_dialog and self.progress_dialog.isVisible(): self.progress_dialog.setLabelText(message); self.progress_dialog.setValue(index + 1)
        self.text_edit.append(message); self.text_edit.ensureCursorVisible(); QApplication.processEvents()

    def handle_page_processed(self, index, processed_image, ocr_data_list):
        # ... (remains the same - stores processed_image, ocr_data_results, ocr_texts_preview) ...
        if 0 <= index < len(self.processed_images):
            self.processed_images[index] = processed_image
            self.ocr_data_results[index] = ocr_data_list
            self.ocr_texts_preview[index] = scanner.extract_text_from_ocr_data(ocr_data_list)
            if index == self.current_preview_index: self.update_preview(use_original=False)

    def scan_finished(self):
        # ... (Thread cleanup remains same) ...
        self.text_edit.append("\n--- Processing Finished ---");
        if self.progress_dialog: self.progress_dialog.close()
        if self.scan_thread: self.scan_thread.quit(); self.scan_thread.wait(1000); self.scan_thread = None
        self.scan_worker = None

        # --- Attempt Auto-Save (Updated) ---
        saved_searchable_path = None
        saved_structured_path = None # New variable
        # saved_images_path = None # Removed
        auto_save_attempted = False

        # Check if *any* auto-save name is set
        if self.auto_save_filename_searchable or self.auto_save_filename_structured_text:
            auto_save_attempted = True
            self.text_edit.append("\nAttempting automatic PDF saves...\n"); QApplication.processEvents()

            # --- Auto-Save Searchable PDF (Type 1) ---
            if self.auto_save_filename_searchable:
                 try:
                     images_to_save = []; data_to_save = []
                     for img, data in zip(self.processed_images, self.ocr_data_results):
                          if img is not None: images_to_save.append(img); data_to_save.append(data if data is not None else [])
                     if images_to_save:
                         if not hasattr(file_manager2, 'generate_searchable_pdf'): self.text_edit.append("❌ Error: generate_searchable_pdf missing.\n")
                         else:
                             self.text_edit.append(f"Saving searchable PDF to {os.path.basename(self.auto_save_filename_searchable)}..."); QApplication.processEvents()
                             saved_searchable_path = file_manager2.generate_searchable_pdf(images_to_save, data_to_save, self.auto_save_filename_searchable)
                             self.text_edit.append(f"  {'✅ Saved' if saved_searchable_path else '⚠️ Failed'}\n")
                     else: self.text_edit.append("  ℹ️ No valid images for searchable PDF.\n")
                 except Exception as e: self.text_edit.append(f"  ❌ Error auto-saving searchable PDF: {e}\n"); print(f"ERROR auto-save searchable: {e}")

            # === Auto-Save Structured Text PDF (Type 2) ===
            if self.auto_save_filename_structured_text:
                 try:
                     # Needs OCR data and original dimensions
                     ocr_data_to_save = self.ocr_data_results # Full list
                     # Get original dimensions (important!)
                     original_dims = [(img.shape[1], img.shape[0]) for img in self.original_images] # WxH

                     # Check if there's any actual OCR data to process
                     if any(ocr_data_to_save):
                         if not hasattr(file_manager2, 'generate_structured_text_pdf'): self.text_edit.append("❌ Error: generate_structured_text_pdf missing.\n")
                         else:
                             self.text_edit.append(f"Saving structured text PDF to {os.path.basename(self.auto_save_filename_structured_text)}..."); QApplication.processEvents()
                             saved_structured_path = file_manager2.generate_structured_text_pdf(ocr_data_to_save, original_dims, self.auto_save_filename_structured_text)
                             self.text_edit.append(f"  {'✅ Saved' if saved_structured_path else '⚠️ Failed'}\n")
                     else: self.text_edit.append("  ℹ️ No OCR data to create structured text PDF.\n")
                 except Exception as e: self.text_edit.append(f"  ❌ Error auto-saving structured text PDF: {e}\n"); print(f"ERROR auto-save structured: {e}")

            # --- Auto-Save Images-Only PDF (Removed) ---
            # if self.auto_save_filename_images:
            #    ...

        # --- Final UI Update & Message ---
        # ... (Update preview logic remains same) ...
        first_valid_index = -1; # ... find first valid index ...
        if self.processed_images: first_valid_index = next((i for i, img in enumerate(self.processed_images) if img is not None), -1); # ... rest of update logic ...
        self.current_preview_index = first_valid_index if first_valid_index !=-1 else (0 if self.original_images else -1)
        if self.current_preview_index != -1: self.update_preview(use_original=False) # ... rest of UI update ...
        else: self.image_label.setText("Processing failed or no files."); self.image_label.setPixmap(QPixmap())
        self.update_button_states() # Re-enable buttons

        # --- Completion Message Box (Updated) ---
        completion_title = "Processing Complete"
        completion_messages = []
        if auto_save_attempted:
            completion_messages.append("Auto-save finished.")
            if saved_searchable_path: completion_messages.append(f"✓ Searchable PDF: {os.path.basename(saved_searchable_path)}")
            elif self.auto_save_filename_searchable: completion_messages.append("✗ Searchable PDF failed (see log).")
            # === Add message for structured text save ===
            if saved_structured_path: completion_messages.append(f"✓ Structured Text PDF: {os.path.basename(saved_structured_path)}")
            elif self.auto_save_filename_structured_text: completion_messages.append("✗ Structured Text PDF failed (see log).")
            # Removed image-only message
        else: completion_messages.append("Auto-save was disabled.")
        QMessageBox.information(self, completion_title, "\n".join(completion_messages))


    def scan_error(self, error_message):
        # ... (remains the same) ...
        if self.progress_dialog: self.progress_dialog.close()
        QMessageBox.critical(self, "Processing Error", f"Critical error:\n{error_message}"); self.text_edit.append(f"\n--- CRITICAL ERROR: {error_message} ---\n")
        if self.scan_thread and self.scan_thread.isRunning(): self.scan_thread.quit(); self.scan_thread.wait(1000)
        self.scan_thread = None; self.scan_worker = None; self.update_button_states()

    def cancel_scan(self):
        # ... (remains the same) ...
        self.text_edit.append("\n--- Requesting Cancellation ---");
        if self.scan_worker: self.scan_worker.stop()
        if self.progress_dialog: self.progress_dialog.setLabelText("Cancelling..."); self.progress_dialog.setEnabled(False)

    def update_preview(self, use_original=False):
        # ... (image display logic remains same, ensure it uses self.ocr_texts_preview for text) ...
        if not (0 <= self.current_preview_index < len(self.original_images)): # ... handle invalid index ...
             self.image_label.setPixmap(QPixmap()); self.image_label.setText("No item selected."); self.text_edit.setPlainText(""); self.list_widget.setCurrentRow(-1); self.update_button_states(); return
        # Update List Widget Selection
        if self.list_widget.count() > self.current_preview_index and self.list_widget.currentRow() != self.current_preview_index: self.list_widget.blockSignals(True); self.list_widget.setCurrentRow(self.current_preview_index); self.list_widget.blockSignals(False)
        # Determine Image
        img_to_display = self.original_images[self.current_preview_index] if use_original else self.processed_images[self.current_preview_index]; source_label = self.image_sources[self.current_preview_index]
        # Update Image Label (handle None, convert BGR->RGB for QImage)
        if img_to_display is None: self.image_label.setPixmap(QPixmap()); self.image_label.setText(f"Preview unavailable:\n{source_label}\n(Processing failed/skipped)")
        else: # ... (try/except block for QImage conversion and pixmap scaling) ...
             try: # ... (QImage creation from numpy array) ...
                if len(img_to_display.shape)==2: height,width=img_to_display.shape; bytes_per_line=width; q_image=QImage(img_to_display.data,width,height,bytes_per_line,QImage.Format_Grayscale8)
                elif len(img_to_display.shape)==3 and img_to_display.shape[2]==3: rgb_image=cv2.cvtColor(img_to_display,cv2.COLOR_BGR2RGB); height,width,channel=rgb_image.shape; bytes_per_line=3*width; q_image=QImage(rgb_image.data,width,height,bytes_per_line,QImage.Format_RGB888)
                else: q_image = None; self.image_label.setText(f"Unsupported format:\n{source_label}")
                if q_image: pixmap=QPixmap.fromImage(q_image); scaled_pixmap=pixmap.scaled(self.image_label.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation); self.image_label.setPixmap(scaled_pixmap)
             except Exception as e: print(f"Err converting image: {e}"); self.image_label.setPixmap(QPixmap()); self.image_label.setText(f"Err display preview:\n{source_label}\n{e}")
        # Update Text Edit
        if use_original: self.text_edit.setPlainText(f"(Original: {source_label})\n\nProcess to see text.")
        else: preview_text=self.ocr_texts_preview[self.current_preview_index] if self.current_preview_index < len(self.ocr_texts_preview) else "N/A"; self.text_edit.setPlainText(f"(Processed: {source_label})\n\n{preview_text if preview_text else '(No text detected or processing failed)'}")
        # Update Buttons
        self.update_button_states()

    def resizeEvent(self, event):
        # ... (remains the same - calling update_preview is simplest) ...
        super().resizeEvent(event)
        if self.image_label and self.image_label.pixmap() and not self.image_label.pixmap().isNull():
             if 0 <= self.current_preview_index < len(self.original_images): showing_original = self.text_edit.toPlainText().startswith("(Original"); self.update_preview(use_original=showing_original)

    def show_selected_image(self, row):
        # ... (remains the same - update index, show processed view) ...
        if 0 <= row < len(self.original_images):
            if row != self.current_preview_index: self.current_preview_index = row; self.update_preview(use_original=False)

    def show_previous_image(self):
        # ... (remains the same - update index, show processed view) ...
         if self.current_preview_index > 0: self.current_preview_index -= 1; self.update_preview(use_original=False)

    def show_next_image(self):
        # ... (remains the same - update index, show processed view) ...
         if self.current_preview_index < len(self.original_images) - 1: self.current_preview_index += 1; self.update_preview(use_original=False)

    # --- Manual Save Functions (Updated) ---

    def save_searchable_pdf_manual(self):
        """Manually saves the Searchable PDF (Image + Invisible Text)."""
        has_valid_images = any(img is not None for img in self.processed_images)
        has_valid_data = any(data for data in self.ocr_data_results)
        if not has_valid_images or not has_valid_data: QMessageBox.warning(self, "Missing Data", "Need both processed images and OCR results to save searchable PDF."); return

        suggested_name = "searchable_doc"
        if self.auto_save_filename_searchable: suggested_name = re.sub(r'_searchable$', '', os.path.splitext(os.path.basename(self.auto_save_filename_searchable))[0])
        filename, ok = QInputDialog.getText(self, "Save Searchable PDF", "Filename (no ext):", text=suggested_name)
        if not (ok and filename.strip()): QMessageBox.warning(self, "Cancelled", "Save cancelled."); return

        sanitized_filename = sanitize_filename(filename.strip())
        pdf_path = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_filename}.pdf") # Simple name

        try: # ... (Prepare image/data lists, check file_manager2.generate_searchable_pdf exists) ...
            images_to_save = []; data_to_save = []
            for i, (img, data) in enumerate(zip(self.processed_images, self.ocr_data_results)):
                 if img is not None: images_to_save.append(img); data_to_save.append(data if data is not None else [])
            if not images_to_save: QMessageBox.warning(self, "Save Error", "No valid pages found."); return
            if not hasattr(file_manager2, 'generate_searchable_pdf'): QMessageBox.critical(self, "Error", "'generate_searchable_pdf' missing."); return

            saving_msg = QMessageBox(QMessageBox.Information, "Saving...", f"Generating searchable PDF:\n{os.path.basename(pdf_path)}", QMessageBox.NoButton, self); saving_msg.setWindowModality(Qt.WindowModal); saving_msg.show(); QApplication.processEvents()
            saved_path = file_manager2.generate_searchable_pdf(images_to_save, data_to_save, pdf_path)
            saving_msg.close()
            if saved_path: QMessageBox.information(self, "Success", f"Searchable PDF saved:\n{saved_path}")
            else: QMessageBox.warning(self, "Failed", "PDF generation failed (see logs).")
        except Exception as e: # ... (Handle exceptions, close saving_msg) ...
             if 'saving_msg' in locals() and saving_msg.isVisible(): saving_msg.close()
             QMessageBox.critical(self, "Save Error", f"Error saving searchable PDF:\n{e}"); print(f"ERROR manual save searchable: {e}")


    # === NEW Manual Save Function for Structured Text PDF ===
    def save_structured_text_pdf_manual(self):
        """Manually saves the Structured Text PDF (Text Only, Layout Preserved)."""
        has_valid_data = any(data for data in self.ocr_data_results)
        if not has_valid_data: QMessageBox.warning(self, "No OCR Data", "No text data extracted. Cannot create structured text PDF."); return

        suggested_name = "structured_text_doc"
        if self.auto_save_filename_structured_text: suggested_name = re.sub(r'_structured_text$', '', os.path.splitext(os.path.basename(self.auto_save_filename_structured_text))[0])
        filename, ok = QInputDialog.getText(self, "Save Structured Text PDF", "Filename (no ext):", text=suggested_name)
        if not (ok and filename.strip()): QMessageBox.warning(self, "Cancelled", "Save cancelled."); return

        sanitized_filename = sanitize_filename(filename.strip())
        pdf_path = os.path.join(PDF_STORAGE_FOLDER, f"{sanitized_filename}.pdf")

        try:
            # Needs OCR data and original image dimensions
            ocr_data_to_save = self.ocr_data_results
            # Get original dimensions (crucial for layout scaling)
            if len(self.original_images) != len(ocr_data_to_save):
                 QMessageBox.warning(self,"Data Mismatch", "Mismatch between original images and OCR results. Layout may be incorrect.")
                 # Attempt to proceed? Or stop? Let's try to proceed with available pairs.
                 min_len = min(len(self.original_images), len(ocr_data_to_save))
                 if min_len == 0: raise ValueError("No matching image/OCR data.")
                 original_dims = [(img.shape[1], img.shape[0]) for img in self.original_images[:min_len]] # WxH
                 ocr_data_to_save = ocr_data_to_save[:min_len]
            else:
                original_dims = [(img.shape[1], img.shape[0]) for img in self.original_images] # WxH


            if not hasattr(file_manager2, 'generate_structured_text_pdf'): QMessageBox.critical(self, "Error", "'generate_structured_text_pdf' missing."); return

            saving_msg = QMessageBox(QMessageBox.Information, "Saving...", f"Generating structured text PDF:\n{os.path.basename(pdf_path)}", QMessageBox.NoButton, self); saving_msg.setWindowModality(Qt.WindowModal); saving_msg.show(); QApplication.processEvents()
            saved_path = file_manager2.generate_structured_text_pdf(ocr_data_to_save, original_dims, pdf_path)
            saving_msg.close()
            if saved_path: QMessageBox.information(self, "Success", f"Structured Text PDF saved:\n{saved_path}")
            else: QMessageBox.warning(self, "Failed", "PDF generation failed (see logs).")
        except Exception as e:
             if 'saving_msg' in locals() and saving_msg.isVisible(): saving_msg.close()
             QMessageBox.critical(self, "Save Error", f"Error saving structured text PDF:\n{e}"); print(f"ERROR manual save structured: {e}")


    # Removed save_images_pdf_manual function

    def open_saved_documents_folder(self):
        # ... (remains the same) ...
        target_folder = os.path.abspath(PDF_STORAGE_FOLDER); # ... (create if not exists) ...;
        if not os.path.exists(target_folder): # ... create folder ...
             try: os.makedirs(target_folder, exist_ok=True); QMessageBox.information(self, "Folder Created", f"Created: {target_folder}\nIt is empty.")
             except OSError as e: QMessageBox.critical(self, "Error", f"Could not create folder:\n{target_folder}\n{e}"); return
        print(f"Opening folder: {target_folder}")
        try: success = QDesktopServices.openUrl(QUrl.fromLocalFile(target_folder)); # ... (fallback if !success) ...
        except Exception as e: QMessageBox.critical(self, "Error", f"Could not open folder:\n{target_folder}\n{e}"); print(f"Error opening folder: {e}")

    def update_button_states(self, scanning=False, loading=False):
        """Updates the enabled state of UI elements."""
        is_busy = scanning or loading
        has_original = len(self.original_images) > 0
        has_any_processed_image = any(img is not None for img in self.processed_images)
        has_any_valid_ocr_data = any(data for data in self.ocr_data_results)
        ocr_ready = hasattr(scanner, 'reader') and scanner.reader is not None

        if self.load_button: self.load_button.setEnabled(not is_busy)
        if self.scan_button: self.scan_button.setEnabled(has_original and not is_busy and ocr_ready)

        # === Updated Save Button Enables ===
        if self.save_searchable_pdf_button: self.save_searchable_pdf_button.setEnabled(has_any_processed_image and has_any_valid_ocr_data and not is_busy)
        if self.save_structured_text_pdf_button: self.save_structured_text_pdf_button.setEnabled(has_any_valid_ocr_data and not is_busy) # Needs only OCR data

        # Removed image-only button check
        # if self.save_images_pdf_button: self.save_images_pdf_button.setEnabled(has_any_processed_image and not is_busy)

        if self.search_button: self.search_button.setEnabled(True) # Always enabled

        can_navigate = has_original and not is_busy
        if self.prev_button: self.prev_button.setEnabled(can_navigate and self.current_preview_index > 0)
        if self.next_button: self.next_button.setEnabled(can_navigate and self.current_preview_index < len(self.original_images) - 1)
        if self.list_widget: self.list_widget.setEnabled(can_navigate)

    def closeEvent(self, event):
        # ... (remains the same) ...
        if self.scan_thread and self.scan_thread.isRunning():
             reply = QMessageBox.question(self,'Quit?', 'Processing active. Quit anyway?', QMessageBox.Yes|QMessageBox.No, QMessageBox.No)
             if reply == QMessageBox.Yes: self.cancel_scan(); self.scan_thread.wait(2000); event.accept()
             else: event.ignore()
        else: event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # ... (High DPI settings remain same) ...
    try: # High DPI
        if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception as e: print(f"Note: DPI setting error: {e}")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())