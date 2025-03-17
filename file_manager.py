import os
import datetime
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


def create_output_directory(base_dir="ScannedDocuments"):
    """
    Creates an output directory based on today's date (e.g., ScannedDocuments/2025-03-17).
    Returns the directory path.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    output_dir = os.path.join(base_dir, today)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def generate_pdf_scanned_document(processed_images, output_dir, pdf_filename=None):
    """
    Generates a PDF where each page shows the scanned document image.
    This PDF is created directly from in-memory images without saving them individually.
    """
    from PIL import Image
    from reportlab.lib.utils import ImageReader

    if pdf_filename is None:
        pdf_filename = os.path.join(output_dir, "scanned_documents.pdf")

    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4

    for img in processed_images:
        # Convert the cv2 image to a PIL image in RGB format.
        if len(img.shape) == 2:  # grayscale image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_reader = ImageReader(pil_img)

        # Draw the image to fill the entire page.
        c.drawImage(img_reader, 0, 0, width=page_width, height=page_height)
        c.showPage()
    c.save()
    return pdf_filename


def generate_pdf_text_only(ocr_texts, output_dir, pdf_filename=None):
    """
    Generates a PDF that contains the extracted OCR text for each image on a separate page.
    Each page of the PDF includes a header indicating the document number followed by the text.
    """
    if pdf_filename is None:
        pdf_filename = os.path.join(output_dir, "extracted_texts.pdf")

    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4
    margin = 50

    for i, text in enumerate(ocr_texts, start=1):
        c.setFont("Helvetica", 10)
        text_obj = c.beginText()
        text_obj.setTextOrigin(margin, page_height - margin)

        # Add a header for each document.
        text_obj.textLine(f"Document {i}")
        text_obj.textLine("")  # Empty line for spacing

        # Add the OCR text line by line.
        for line in text.splitlines():
            text_obj.textLine(line)

        c.drawText(text_obj)
        c.showPage()  # Complete the page.

    c.save()
    return pdf_filename


def search_documents(query, base_dir="ScannedDocuments"):
    """
    Searches for the query string in all extracted text files under the base directory.
    Returns a list of tuples (file_path, snippet) for files where the query was found.
    """
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
