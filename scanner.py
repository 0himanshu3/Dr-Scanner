import cv2
import numpy as np
import easyocr
import os # Keep os for potential path operations if needed later

# --- Global EasyOCR Reader Initialization ---
reader = None
try:
    print("Initializing EasyOCR Reader (attempting GPU)...")
    # Try GPU first. Add other languages if needed: ['en', 'fr']
    reader = easyocr.Reader(['en'], gpu=True)
    print("EasyOCR initialized successfully using GPU.")
except Exception as e_gpu:
    print(f"EasyOCR GPU initialization failed: {e_gpu}.")
    print("Attempting EasyOCR initialization using CPU...")
    try:
         # Fallback to CPU
         reader = easyocr.Reader(['en'], gpu=False)
         print("EasyOCR initialized successfully using CPU.")
    except Exception as e_cpu:
         print(f"FATAL: EasyOCR CPU initialization failed: {e_cpu}")
         print("OCR functionality will be unavailable.")
         # reader remains None


# --- Image Processing Functions ---

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
    # Compute widths and heights at extremities
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Define destination points for the warped image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_document(image):
    """
    Finds the largest 4-sided contour likely representing a document.
    Returns a perspective-warped version of the detected document area.
    """
    if image is None:
        print("Error: Input image to detect_document is None.")
        return None # Return None if input is invalid

    # Keep a copy of the original for warping later
    orig = image.copy()
    # Resize for faster edge detection (optional, adjust height 500 as needed)
    ratio = image.shape[0] / 500.0
    image_resized = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # Preprocessing for edge detection
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Blur to reduce noise
    edged = cv2.Canny(blurred, 75, 200) # Canny edge detection

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None

    # Loop over contours to find a 4-sided one
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # Approximate contour shape

        if len(approx) == 4:
            # Basic check: ensure the contour is reasonably large
            # (e.g., at least 10% of the resized image area)
            if cv2.contourArea(approx) > (image_resized.shape[0] * image_resized.shape[1] * 0.1):
                 screenCnt = approx
                 break # Found a suitable candidate

    if screenCnt is not None:
        # Found contour, apply perspective transform using original image coordinates
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        print("Document contour detected and warped.")
        return warped
    else:
        # If no suitable contour found, return the original image unprocessed
        # Or return a standard processed version (e.g., grayscale)
        print("Warning: No document contour detected. Processing original image without warping.")
        # Decide fallback: return original, or original preprocessed? Let's return original.
        return orig

def preprocess_image(image):
    """
    Applies document detection, perspective correction (if detected),
    contrast enhancement, and binarization.

    Returns:
        np.array: The processed image (binary), or None if detection fails badly.
    """
    # 1. Detect document and warp perspective
    detected = detect_document(image)
    if detected is None:
        print("Error: Document detection returned None in preprocess_image.")
        # Handle failure: maybe try processing the original image without warp?
        # For now, return None to signal failure upstream.
        return None

    # 2. Convert to grayscale
    gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)

    # 3. Enhance contrast (CLAHE) - often helps OCR
    # clipLimit controls contrast factor, tileGridSize controls locality
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Binarization (Thresholding)
    # OTSU's method automatically finds a threshold value
    # Using the enhanced image for thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Further noise reduction or morphological operations could be added here
    # kernel = np.ones((1, 1), np.uint8)
    # cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # Opening removes small noise
    # return cleaned

    # Return the binary image - EasyOCR generally works well with this
    return binary
    # Alternative: return 'enhanced' (grayscale CLAHE) - might be better for some EasyOCR cases
    # return enhanced


# --- NEW OCR Data Extraction Function ---
def get_ocr_data(image, min_confidence=0.3):
    """
    Performs OCR using EasyOCR and returns the raw results including bounding boxes.

    Args:
        image (np.array): The processed image (BGR, Grayscale, or Binary numpy array).
        min_confidence (float): Minimum confidence threshold for results (0.0 to 1.0).

    Returns:
        list: A list of tuples, where each tuple is
              ([bbox], text, confidence). bbox is a list of 4 [x,y] points (float).
              Returns empty list on error or if reader is not ready.
    """
    global reader # Access the global reader instance
    if reader is None:
        print("Error: EasyOCR reader not initialized.")
        return []
    if image is None:
        print("Error: Cannot perform OCR on None image input.")
        return []

    try:
        # EasyOCR works with BGR, Grayscale, or Binary numpy arrays.
        # The output of preprocess_image (binary) should be suitable.
        print(f"Running EasyOCR on image with shape: {image.shape}...")
        results = reader.readtext(image, detail=1) # detail=1 ensures bbox format
        print(f"EasyOCR found {len(results)} text boxes.")

        # Filter results by confidence
        filtered_results = [res for res in results if res[2] >= min_confidence]
        print(f"Filtered to {len(filtered_results)} boxes with confidence >= {min_confidence}.")

        # Convert bbox coordinates to float if they aren't already (safer for later use)
        final_results = []
        for bbox, text, conf in filtered_results:
             float_bbox = [[float(p[0]), float(p[1])] for p in bbox]
             final_results.append((float_bbox, text, conf))

        return final_results

    except Exception as e:
        print(f"Error during EasyOCR data extraction: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return [] # Return empty list on error


# --- NEW Helper Function for Preview Text ---
def extract_text_from_ocr_data(ocr_data):
    """
    Extracts and joins text strings from the detailed OCR data list.
    Intended for generating plain text for display/preview panes.

    Args:
        ocr_data (list): The list of [(bbox, text, conf), ...] from get_ocr_data.

    Returns:
        str: A single string containing joined text lines.
    """
    if not ocr_data:
        return "" # Return empty string if no data

    # Simply extract the text part (index 1) from each result
    text_lines = [result[1] for result in ocr_data]

    # Join the lines with newline characters
    return "\n".join(text_lines).strip()


# --- REMOVED Old extract_text Function ---
# def extract_text(image, min_confidence=0.3): # This is now redundant
#    ...


# --- Main Scan Function (Reads image, calls preprocess) ---
# This function's role is primarily loading and preprocessing, not OCR itself.
def scan_document(image_path):
    """
    Reads an image file, applies preprocessing steps.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (processed_image, original_image)
               processed_image is the result of preprocess_image (e.g., binary np.array) or None on failure.
               original_image is the initially loaded color image (np.array) or None on failure.
    """
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load the image: {image_path}. Check path/permissions.")
        return None, None

    print("Preprocessing image...")
    processed = preprocess_image(image) # Calls detect_document internally
    if processed is None:
        print("Warning: Image preprocessing failed. Returning None for processed image.")
        # Return None for processed, but still return original for potential display
        return None, image

    print("Preprocessing complete.")
    return processed, image # Return processed (e.g., binary) and original color image


# --- REMOVED Old PDF Generation Function ---
# def generate_pdf_with_easyocr(...) : # PDF generation now in file_manager.py


# --- Example Usage (Optional: For testing this module directly) ---
if __name__ == "__main__":
    print("Scanner module direct execution test.")
    # Replace with a valid image path on your system for testing
    test_image_path = 'test_document.jpg' # <<<=== CHANGE THIS

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
            # Optional: Display processed image (requires GUI environment)
            # cv2.imshow("Processed Test Image", processed_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            print(f"\n--- Testing get_ocr_data ---")
            ocr_results = get_ocr_data(processed_img, min_confidence=0.4)

            if ocr_results:
                print(f"Found {len(ocr_results)} text items with confidence >= 0.4.")
                # Print first few results for inspection
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