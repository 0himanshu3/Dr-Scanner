import cv2
import numpy as np
import pytesseract
import os
from pytesseract import Output

# Tesseract config
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.hypot(br[0]-bl[0], br[1]-bl[1])
    widthB = np.hypot(tr[0]-tl[0], tr[1]-tl[1])
    maxW = max(int(widthA), int(widthB))
    heightA = np.hypot(tr[0]-br[0], tr[1]-br[1])
    heightB = np.hypot(tl[0]-bl[0], tl[1]-bl[1])
    maxH = max(int(heightA), int(heightB))

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

def detect_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bl = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(bl, 75, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    h, w = image.shape[:2]
    area = h*w

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            A = cv2.contourArea(approx)
            if 0.2*area < A < 0.7*area:
                x,y,ww,hh = cv2.boundingRect(approx)
                ar = ww/float(hh)
                if 0.5 < ar < 2.0:
                    return four_point_transform(image, approx.reshape(4,2))
    return image

def preprocess_image(image):
    # 1) Deskew / crop to document
    doc = detect_document(image)
    # 2) Grayscale
    gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)
    # 3) Contrast limited AHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # 4) Binarize
    _, bin_img = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # 5) Clean up
    kernel = np.ones((1,1), np.uint8)
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

def extract_text(image):
    config = r'--oem 3 --psm 6 preserve_interword_spaces=1'
    return pytesseract.image_to_string(image, config=config).strip()
