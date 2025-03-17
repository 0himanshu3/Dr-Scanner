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


def save_scanned_document(processed_image, ocr_text, output_dir):
    """
    Saves the processed image and OCR text into the output directory.
    Returns the filenames for the saved image and text file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = os.path.join(output_dir, f"processed_{timestamp}.png")
    text_filename = os.path.join(output_dir, f"extracted_{timestamp}.txt")

    cv2.imwrite(image_filename, processed_image)
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    return image_filename, text_filename


def generate_pdf_with_image_and_invisible_text(image_files, ocr_texts, output_dir, pdf_filename=None):
    """
    Generates a PDF where each page shows the scanned image (filling the page) and
    overlays an invisible text layer with the OCR output. This makes the text selectable.
    """
    if pdf_filename is None:
        pdf_filename = os.path.join(output_dir, "scanned_documents_with_text.pdf")

    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4

    for image_file, text in zip(image_files, ocr_texts):
        # Draw the image so it fills the page.
        c.drawImage(image_file, 0, 0, width=page_width, height=page_height)

        # Create a text object for the OCR text.
        text_obj = c.beginText()
        # Set the starting position (from the bottom left; adjust margins as needed)
        text_obj.setTextOrigin(50, page_height - 50)
        text_obj.setFont("Helvetica", 10)
        try:
            # Try to set invisible text rendering mode (3 means invisible text in PDF specs)
            text_obj.setTextRenderMode(3)
        except AttributeError:
            # If not available, inject the PDF operator manually.
            text_obj._code.append("3 Tr")

        # Add the OCR text line by line.
        for line in text.splitlines():
            text_obj.textLine(line)
        c.drawText(text_obj)

        # Finish the page.
        c.showPage()
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
