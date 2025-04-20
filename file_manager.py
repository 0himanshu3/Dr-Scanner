import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import cv2
from PIL import Image

def generate_pdf_scanned_document(processed_images, output_dir, pdf_filename):
    """
    Each processed image fills one PDF page.
    processed_images: list of OpenCV BGR or gray arrays
    output_dir: folder where to save
    pdf_filename: filename (with .pdf) under output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, pdf_filename)
    c = canvas.Canvas(full_path, pagesize=A4)
    w, h = A4

    for img in processed_images:
        # Convert OpenCV→RGB PIL
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        c.drawImage(ImageReader(pil_img), 0, 0, width=w, height=h)
        c.showPage()

    c.save()
    return full_path

def generate_pdf_text_only(ocr_texts, output_dir, pdf_filename):
    """
    Each text block becomes one PDF page.
    ocr_texts: list of strings
    """
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, pdf_filename)
    c = canvas.Canvas(full_path, pagesize=A4)
    w, h = A4
    margin = 40

    for idx, text in enumerate(ocr_texts, start=1):
        c.setFont("Helvetica", 10)
        text_obj = c.beginText(margin, h - margin)
        text_obj.textLine(f"Page {idx}")
        text_obj.textLine("")
        for line in text.splitlines():
            text_obj.textLine(line)
        c.drawText(text_obj)
        c.showPage()

    c.save()
    return full_path
