import cv2
import numpy as np
import pytesseract

# Specify the path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract\tesseract.exe'


def order_points(pts):
    """
    Orders points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """
    Performs a perspective transform of the image using the given 4 points.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the new image dimensions
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
    """
    Detects the document (page) within the image by finding a 4-point contour that
    likely corresponds to the document. It uses edge detection with Canny, finds contours,
    and filters them based on area and aspect ratio to avoid extraneous elements.
    Returns a warped (top-down) view of the document if found; otherwise, returns the original image.
    """
    # Convert image to grayscale and apply Gaussian blur to reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge Detection using Canny
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours and sort by area (largest first)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = image.shape[0] * image.shape[1]
    candidate = None

    # Loop through contours to find a 4-point contour that meets our criteria
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            contour_area = cv2.contourArea(approx)
            area_ratio = contour_area / image_area
            # Assume the document occupies between 20% and 70% of the image.
            if 0.2 < area_ratio < 0.7:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                # Check if the aspect ratio is reasonable for a document.
                if 0.5 < aspect_ratio < 2.0:
                    candidate = approx
                    break

    if candidate is not None:
        return four_point_transform(image, candidate.reshape(4, 2))
    return image


def preprocess_image(image):
    """
    Preprocesses the image for OCR:
      - Performs document detection and perspective correction.
      - Converts the detected document to grayscale.
      - Enhances contrast using CLAHE.
      - Applies Otsu's thresholding.
      - Uses morphological opening and closing to refine edges and remove minor artifacts.
    Returns the cleaned binary image.
    """
    # Detect and warp the document
    detected = detect_document(image)

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define a small kernel for morphological operations
    kernel = np.ones((1, 1), np.uint8)

    # Apply morphological opening to remove small noise
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Apply morphological closing to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed


def extract_text(image):
    """
    Uses Tesseract OCR with a custom configuration to extract text.
    Returns the extracted text stripped of extra whitespace.
    """
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    try:
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except Exception as e:
        print("Error during OCR:", e)
        return ""


def scan_document(image_path):
    """
    Reads an image from the given path, applies document detection, perspective correction,
    and preprocessing. Returns a tuple (processed_image, original_image).
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load the image. Check the path!")
        return None, None
    processed = preprocess_image(image)
    return processed, image
