import os
import datetime
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from PIL import Image
import pytesseract
from pytesseract import Output

def create_output_directory(base_dir="ScannedDocuments"):
    # Create folder with today's date.
    today = datetime.date.today().strftime("%Y-%m-%d")
    output_dir = os.path.join(base_dir, today)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def generate_scanned_images_pdf(processed_images, pdf_filename):
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4
    for img in processed_images:
        # Convert image to RGB if needed
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_reader = ImageReader(pil_img)
        c.drawImage(img_reader, 0, 0, width=page_width, height=page_height)
        c.showPage()
    c.save()
    print("Scanned images PDF saved as:", pdf_filename)
    
def generate_pdf_scanned_document(processed_images, output_dir, pdf_filename=None):
    # Generate a PDF with each scanned image filling a page.
    if pdf_filename is None:
        pdf_filename = os.path.join(output_dir, "scanned_documents.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4

    for img in processed_images:
        if len(img.shape) == 2:  # grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_reader = ImageReader(pil_img)
        c.drawImage(img_reader, 0, 0, width=page_width, height=page_height)
        c.showPage()
    c.save()
    return pdf_filename

def generate_pdf_text_only(ocr_texts, output_dir, pdf_filename=None):
    # Create a PDF with OCR text, one document per page.
    if pdf_filename is None:
        pdf_filename = os.path.join(output_dir, "extracted_texts.pdf")
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4
    margin = 50

    for i, text in enumerate(ocr_texts, start=1):
        c.setFont("Helvetica", 10)
        text_obj = c.beginText()
        text_obj.setTextOrigin(margin, page_height - margin)
        text_obj.textLine(f"Document {i}")
        text_obj.textLine("")
        for line in text.splitlines():
            text_obj.textLine(line)
        c.drawText(text_obj)
        c.showPage()

    c.save()
    return pdf_filename

def search_documents(query, base_dir="ScannedDocuments"):
    # Look for the query in all .txt files.
    results = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if query.lower() in content.lower():
                            index = content.lower().find(query.lower())
                            start = max(index - 30, 0)
                            end = min(index + len(query) + 30, len(content))
                            snippet = content[start:end].replace("\n", " ")
                            results.append((file_path, snippet))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return results

def generate_pdf_with_layout(cv_image, output_pdf="layout_ocr.pdf"):
    # Create a PDF with text drawn at OCR-determined positions.
    if len(cv_image.shape) == 2:
        pil_img = Image.fromarray(cv_image)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    img_width, img_height = pil_img.size
    data = pytesseract.image_to_data(pil_img, output_type=Output.DICT)
    c = canvas.Canvas(output_pdf, pagesize=A4)
    page_width, page_height = A4

    c.drawImage(ImageReader(pil_img), 0, 0, width=page_width, height=page_height)
    scale_x = page_width / float(img_width)
    scale_y = page_height / float(img_height)

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        conf = int(data['conf'][i])
        text = data['text'][i].strip()
        if conf > 50 and text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            pdf_x = x * scale_x
            pdf_y = page_height - (y * scale_y)
            font_size = max(6, int(h * scale_y * 0.8))
            c.setFont("Helvetica", font_size)
            c.drawString(pdf_x, pdf_y, text)
    c.showPage()
    c.save()
    return output_pdf