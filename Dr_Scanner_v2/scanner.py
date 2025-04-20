import cv2
import pytesseract
import os
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
from pytesseract import Output
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUTPUT_DIR = "scanned_documents"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_searchable_pdf(image, output_pdf, lang='eng'):
    pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, lang=lang, extension='pdf')
    with open(output_pdf, 'wb') as f:
        f.write(pdf_bytes)
    print(f"✅ Saved searchable PDF: {output_pdf}")

def generate_extracted_text_pdf(image, output_pdf, font_name="Helvetica", font_size=10):
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    lines = {}
    n_boxes = len(ocr_data['text'])

    for i in range(n_boxes):
        conf = int(ocr_data['conf'][i])
        word = ocr_data['text'][i].strip()
        if conf > 50 and word:
            key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
            if key not in lines:
                lines[key] = {
                    "words": [],
                    "min_left": ocr_data['left'][i],
                    "top": ocr_data['top'][i]
                }
            else:
                if ocr_data['left'][i] < lines[key]["min_left"]:
                    lines[key]["min_left"] = ocr_data['left'][i]
            lines[key]["words"].append(word)

    sorted_lines = sorted(lines.items(), key=lambda item: item[1]["top"])
    c = canvas.Canvas(output_pdf, pagesize=A4)
    page_width, page_height = A4
    img_height, img_width = image.shape[:2]
    scale_x = page_width / float(img_width)
    scale_y = page_height / float(img_height)

    for _, line_data in sorted_lines:
        line_text = " ".join(line_data["words"])
        x_pdf = line_data["min_left"] * scale_x
        y_pdf = page_height - (line_data["top"] * scale_y)
        c.setFont(font_name, font_size)
        c.drawString(x_pdf, y_pdf, line_text)

    c.showPage()
    c.save()
    print(f"📝 Saved extracted text PDF: {output_pdf}")

def process_file(input_path, lang='eng'):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    searchable_pdf_path = os.path.join(OUTPUT_DIR, f"{base_name}_searchable.pdf")
    extracted_pdf_path = os.path.join(OUTPUT_DIR, f"{base_name}_extracted.pdf")

    if input_path.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")):
        print("📷 Processing image...")
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        generate_searchable_pdf(image, searchable_pdf_path, lang=lang)
        generate_extracted_text_pdf(image, extracted_pdf_path)

    elif input_path.lower().endswith(".pdf"):
        print("📄 Processing PDF...")
        doc = fitz.open(input_path)
        searchable_output = fitz.open()
        extracted_output = canvas.Canvas(extracted_pdf_path, pagesize=A4)

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

            if img_array is not None:
                # Searchable
                pdf_bytes = pytesseract.image_to_pdf_or_hocr(img_array, lang=lang, extension='pdf')
                ocr_page = fitz.open("pdf", pdf_bytes)
                searchable_output.insert_pdf(ocr_page)

                # Extracted
                ocr_data = pytesseract.image_to_data(img_array, output_type=Output.DICT)
                lines = {}
                n_boxes = len(ocr_data['text'])

                for j in range(n_boxes):
                    conf = int(ocr_data['conf'][j])
                    word = ocr_data['text'][j].strip()
                    if conf > 50 and word:
                        key = (ocr_data['block_num'][j], ocr_data['par_num'][j], ocr_data['line_num'][j])
                        if key not in lines:
                            lines[key] = {
                                "words": [],
                                "min_left": ocr_data['left'][j],
                                "top": ocr_data['top'][j]
                            }
                        else:
                            if ocr_data['left'][j] < lines[key]["min_left"]:
                                lines[key]["min_left"] = ocr_data['left'][j]
                        lines[key]["words"].append(word)

                sorted_lines = sorted(lines.items(), key=lambda item: item[1]["top"])
                page_width, page_height = A4
                img_height, img_width = img_array.shape[:2]
                scale_x = page_width / float(img_width)
                scale_y = page_height / float(img_height)

                for _, line_data in sorted_lines:
                    line_text = " ".join(line_data["words"])
                    x_pdf = line_data["min_left"] * scale_x
                    y_pdf = page_height - (line_data["top"] * scale_y)
                    extracted_output.setFont("Helvetica", 10)
                    extracted_output.drawString(x_pdf, y_pdf, line_text)

                extracted_output.showPage()
            else:
                print(f"⚠️ Could not decode page {i + 1} of PDF.")

        searchable_output.save(searchable_pdf_path)
        extracted_output.save()
        print(f"✅ Saved searchable PDF: {searchable_pdf_path}")
        print(f"📝 Saved extracted text PDF: {extracted_pdf_path}")

    else:
        raise ValueError("Unsupported file format.")
