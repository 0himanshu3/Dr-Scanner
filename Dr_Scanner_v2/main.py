import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QListWidget,
    QLabel, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout,
    QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt
import fitz  # PyMuPDF
import scanner

PDF_FOLDER = "scanned_documents"
os.makedirs(PDF_FOLDER, exist_ok=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dr. Scanner")
        self.setGeometry(200, 100, 900, 600)
        self.setWindowIcon(QIcon("icons/app_icon.png"))
        self.paths = []
        self._build_ui()

    def _build_ui(self):
        title = QLabel("Dr. Scanner", alignment=Qt.AlignCenter)
        title.setStyleSheet("font-size:24px; font-weight:bold; color:#007BFF;")

        self.listwidget = QListWidget()
        load_btn = QPushButton("📂 Load Images"); load_btn.clicked.connect(self.load_images)
        scan_btn = QPushButton("📄 Scan & Save PDF"); scan_btn.clicked.connect(self.scan_and_save)
        open_btn = QPushButton("🔎 Open Folder"); open_btn.clicked.connect(self.open_folder)

        sidebar = QVBoxLayout()
        sidebar.addWidget(self.listwidget)
        sidebar.addWidget(load_btn)
        sidebar.addWidget(scan_btn)
        sidebar.addWidget(open_btn)
        sidebar.addStretch()

        self.preview = QLabel(); self.preview.setFixedSize(600, 400)
        self.preview.setStyleSheet("border:2px dashed #6C757D;")
        self.log = QTextEdit(); self.log.setReadOnly(True)

        right = QVBoxLayout()
        right.addWidget(QLabel("🖼 Preview")); right.addWidget(self.preview, alignment=Qt.AlignCenter)
        right.addWidget(QLabel("📝 Log")); right.addWidget(self.log)

        mainlay = QVBoxLayout()
        mainlay.addWidget(title)
        hl = QHBoxLayout(); hl.addLayout(sidebar); hl.addLayout(right)
        mainlay.addLayout(hl)

        container = QWidget(); container.setLayout(mainlay)
        self.setCentralWidget(container)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", "Images (*.png *.jpg *.jpeg *.bmp);;PDF Files (*.pdf)")
        if not files: return
        self.paths = files
        self.listwidget.clear()
        for f in files: self.listwidget.addItem(os.path.basename(f))

        file = files[0]
        ext = os.path.splitext(file)[1].lower()

        if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            pix = QPixmap(file).scaled(self.preview.size(), Qt.KeepAspectRatio)
            self.preview.setPixmap(pix)
        elif ext == '.pdf':
            doc = fitz.open(file)
            page = doc.load_page(0)
            pix = page.get_pixmap()
            qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(self.preview.size(), Qt.KeepAspectRatio)
            self.preview.setPixmap(pixmap)

        self.log.append(f"Loaded {len(files)} file(s).")

    def scan_and_save(self):
        if not self.paths:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return

        for p in self.paths:
            base = os.path.splitext(os.path.basename(p))[0]
            self.log.append(f"Processing {base}…")
            try:
                scanner.process_file(p)
                self.log.append(f"✔️ Done: {base}\n")
            except Exception as e:
                self.log.append(f"❌ Error: {e}\n")

    def open_folder(self):
        os.startfile(PDF_FOLDER)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
