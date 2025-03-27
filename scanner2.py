import cv2
import numpy as np
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"E:\Tesseract\tesseract.exe"

def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left.
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # Get a bird's-eye view of the image.
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_document(image):
    # Find a 4-point contour likely to be the document.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = image.shape[0] * image.shape[1]
    candidate = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            contour_area = cv2.contourArea(approx)
            area_ratio = contour_area / image_area
            if 0.2 < area_ratio < 0.7:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 0.5 < aspect_ratio < 2.0:
                    candidate = approx
                    break

    if candidate is not None:
        return four_point_transform(image, candidate.reshape(4, 2))
    return image

def preprocess_image(image):
    # Correct perspective, enhance contrast, and threshold.
    detected = detect_document(image)
    gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def extract_text(image):
    # Extract text using Tesseract.
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    try:
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except Exception as e:
        print("OCR error:", e)
        return ""

def scan_document(image_path):
    # Read and process the image.
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load the image. Check the path!")
        return None, None
    processed = preprocess_image(image)
    return processed, image

def generate_pdf_line_by_line(image_path, pdf_filename):
    # Read image and extract text with OCR details.
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4

    image = cv2.imread(image_path)
    if image is None:
        print("Could not load the image. Check the path!")
        return None

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
    c = canvas.Canvas(pdf_filename, pagesize=A4)
    page_width, page_height = A4
    img_height, img_width = image.shape[:2]
    scale_x = page_width / float(img_width)
    scale_y = page_height / float(img_height)

    for key, line_data in sorted_lines:
        line_text = " ".join(line_data["words"])
        x_pdf = line_data["min_left"] * scale_x
        y_pdf = page_height - (line_data["top"] * scale_y)
        c.setFont("Helvetica", 10)
        c.drawString(x_pdf, y_pdf, line_text)

    c.showPage()
    c.save()
    print("PDF saved as:", pdf_filename)
    return pdf_filename

if __name__ == "__main__":
    image_path = "sample_document.jpg" 
    pdf_file_name = "extracted_layout.pdf"
    generate_pdf_line_by_line(image_path, pdf_file_name)
