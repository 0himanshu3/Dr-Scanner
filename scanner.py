import cv2
import numpy as np
import easyocr
import os

reader = None
try:
    print("Initializing EasyOCR Reader (attempting GPU)...")
    reader = easyocr.Reader(['en'], gpu=True)
    print("EasyOCR initialized successfully using GPU.")
except Exception as e_gpu:
    print(f"EasyOCR GPU initialization failed: {e_gpu}.")
    print("Attempting EasyOCR initialization using CPU...")
    try:
         reader = easyocr.Reader(['en'], gpu=False)
         print("EasyOCR initialized successfully using CPU.")
    except Exception as e_cpu:
         print(f"FATAL: EasyOCR CPU initialization failed: {e_cpu}")
         print("OCR functionality will be unavailable.")

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
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
    if image is None:
        print("Error: Input image to detect_document is None.")
        return None
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image_resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            if cv2.contourArea(approx) > (image_resized.shape[0] * image_resized.shape[1] * 0.1):
                 screenCnt = approx
                 break
    if screenCnt is not None:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        print("Document contour detected and warped.")
        return warped
    else:
        print("Warning: No document contour detected. Processing original image without warping.")
        return orig

def preprocess_image(image):
    detected = detect_document(image)
    if detected is None:
        print("Error: Document detection returned None in preprocess_image.")
        return None
    gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def get_ocr_data(image, min_confidence=0.3):
    global reader
    if reader is None:
        print("Error: EasyOCR reader not initialized.")
        return []
    if image is None:
        print("Error: Cannot perform OCR on None image input.")
        return []
    try:
        print(f"Running EasyOCR on image with shape: {image.shape}...")
        results = reader.readtext(image, detail=1)
        print(f"EasyOCR found {len(results)} text boxes.")
        filtered_results = [res for res in results if res[2] >= min_confidence]
        print(f"Filtered to {len(filtered_results)} boxes with confidence >= {min_confidence}.")
        final_results = []
        for bbox, text, conf in filtered_results:
             float_bbox = [[float(p[0]), float(p[1])] for p in bbox]
             final_results.append((float_bbox, text, conf))
        return final_results
    except Exception as e:
        print(f"Error during EasyOCR data extraction: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_text_from_ocr_data(ocr_data):
    if not ocr_data:
        return ""
    text_lines = [result[1] for result in ocr_data]
    return "\n".join(text_lines).strip()

def scan_document(image_path):
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load the image: {image_path}. Check path/permissions.")
        return None, None
    print("Preprocessing image...")
    processed = preprocess_image(image)
    if processed is None:
        print("Warning: Image preprocessing failed. Returning None for processed image.")
        return None, image
    print("Preprocessing complete.")
    return processed, image

if __name__ == "__main__":
    print("Scanner module direct execution test.")
    test_image_path = 'test_document.jpg'
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please update the 'test_image_path' variable in scanner.py for testing.")
    elif reader is None:
         print("EasyOCR Reader failed to initialize. Cannot run test.")
    else:
        print(f"\n--- Testing scan_document for: {test_image_path} ---")
        processed_img, original_img = scan_document(test_image_path)
        if processed_img is not None:
            print("scan_document returned a processed image.")
            print(f"\n--- Testing get_ocr_data ---")
            ocr_results = get_ocr_data(processed_img, min_confidence=0.4)
            if ocr_results:
                print(f"Found {len(ocr_results)} text items with confidence >= 0.4.")
                for i, (bbox, text, conf) in enumerate(ocr_results[:5]):
                     print(f"  Item {i+1}: BBox={bbox}, Text='{text}', Conf={conf:.2f}")
                print(f"\n--- Testing extract_text_from_ocr_data ---")
                plain_text = extract_text_from_ocr_data(ocr_results)
                print("Extracted Plain Text (Preview):\n---")
                print(plain_text)
                print("---")
            else:
                print("get_ocr_data did not return any results.")
        else:
            print("scan_document failed to return a processed image.")
