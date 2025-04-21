import os
import datetime
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from PIL import Image as PILImage
from io import BytesIO
import statistics

PDF_STORAGE_FOLDER = "scanned_documents"

def create_output_directory(base_dir=PDF_STORAGE_FOLDER):
    if not os.path.exists(base_dir):
        print(f"Base directory '{base_dir}' not found. Creating it.")
        os.makedirs(base_dir, exist_ok=True)
    today = datetime.date.today().strftime("%Y-%m-%d")
    output_dir = os.path.join(base_dir, today)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created dated output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return base_dir
    return output_dir

def generate_searchable_pdf(image_list, ocr_data_per_page, output_filename):
    if not image_list:
        raise ValueError("No images provided to generate searchable PDF.")
    if len(image_list) != len(ocr_data_per_page):
        print(f"Warning: Mismatch between images ({len(image_list)}) and OCR data ({len(ocr_data_per_page)}). Processing common pages.")
        min_len = min(len(image_list), len(ocr_data_per_page))
        if min_len == 0: raise ValueError("No matching image/OCR data pairs.")
        image_list = image_list[:min_len]
        ocr_data_per_page = ocr_data_per_page[:min_len]

    try:
        first_img_h, first_img_w = image_list[0].shape[:2]
    except IndexError:
        raise ValueError("Image list is empty after filtering.")

    page_width_pt = float(first_img_w)
    page_height_pt = float(first_img_h)
    print(f"Generating searchable PDF with page size: {page_width_pt:.2f}x{page_height_pt:.2f} points")

    c = canvas.Canvas(output_filename, pagesize=(page_width_pt, page_height_pt))

    for i, img_cv in enumerate(image_list):
        img_h, img_w = img_cv.shape[:2]
        ocr_data = ocr_data_per_page[i]

        try:
            if len(img_cv.shape) == 2:
                img_pil = PILImage.fromarray(img_cv).convert("RGB")
            elif img_cv.shape[2] == 3:
                img_pil = PILImage.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            elif img_cv.shape[2] == 4:
                img_pil = PILImage.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGB))
            else:
                raise ValueError(f"Unsupported image shape {img_cv.shape}")

            img_buffer = BytesIO()
            img_pil.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_reader = ImageReader(img_buffer)
            c.drawImage(img_reader, 0, 0, width=page_width_pt, height=page_height_pt)
        except Exception as e:
            print(f"Error drawing background image for page {i+1}: {e}")
            c.setFont("Helvetica-Bold", 12)
            c.setFillColorRGB(1, 0, 0)
            c.drawCentredString(page_width_pt / 2, page_height_pt / 2, f"Error displaying image {i+1}")
            c.setFillColorRGB(0, 0, 0)
            c.showPage()
            continue

        textobject = c.beginText()
        textobject.setTextRenderMode(3)

        if not ocr_data:
            continue

        processed_boxes = 0
        for item_index, ocr_item in enumerate(ocr_data):
            try:
                bbox, text, conf = ocr_item
                if not text or text.isspace(): continue

                tl_x, tl_y = bbox[0]
                bl_x, bl_y = bbox[3]
                tr_x, _ = bbox[1]
                pdf_x = float(tl_x)
                pdf_y = page_height_pt - float(bl_y)
                box_width = float(tr_x) - float(tl_x)
                box_height = float(bl_y) - float(tl_y)
                if box_height <= 1 or box_width <= 1: continue

                font_size = max(box_height * 0.80, 4)
                textobject.setFont('Helvetica', font_size)
                textobject.setTextOrigin(pdf_x, pdf_y)
                textobject.textLine(text)
                processed_boxes += 1
            except Exception as e_text:
                print(f"  Error processing text box {item_index+1} ('{text}') on page {i+1} for searchable PDF: {e_text}")

        c.drawText(textobject)
        c.showPage()

    c.save()
    print(f"Searchable PDF saved successfully as: {output_filename}")
    return output_filename

def generate_structured_text_pdf(ocr_data_per_page, original_page_dims, output_filename):
    if not ocr_data_per_page:
        print("Warning: No OCR data provided for generate_structured_text_pdf.")
        return None
    if len(ocr_data_per_page) != len(original_page_dims):
        print(f"Warning: Mismatch OCR data pages ({len(ocr_data_per_page)}) and page dimensions ({len(original_page_dims)}).")
        min_len = min(len(ocr_data_per_page), len(original_page_dims))
        if min_len == 0: return None
        ocr_data_per_page = ocr_data_per_page[:min_len]
        original_page_dims = original_page_dims[:min_len]

    print(f"Generating structured text PDF: {output_filename}")
    c = canvas.Canvas(output_filename, pagesize=A4)
    page_width_a4, page_height_a4 = A4
    margin = 0.75 * inch
    drawable_width = page_width_a4 - 2 * margin
    drawable_height = page_height_a4 - 2 * margin
    fixed_font_size = 9

    for page_index, ocr_data in enumerate(ocr_data_per_page):
        if not ocr_data:
            c.showPage()
            continue

        orig_w, orig_h = original_page_dims[page_index]
        if orig_w == 0 or orig_h == 0:
            c.showPage()
            continue

        scale_y = drawable_height / float(orig_h)
        scale_x = drawable_width / float(orig_w)

        lines = []
        box_heights = [float(res[0][3][1]) - float(res[0][0][1]) for res in ocr_data if res[0] and len(res[0])==4]
        median_height = statistics.median(box_heights) if box_heights else 10
        y_tolerance = median_height * 0.5
        sorted_boxes = sorted(ocr_data, key=lambda item: (item[0][0][1], item[0][0][0]))

        current_line = []
        last_y_center = -1

        for bbox, text, conf in sorted_boxes:
            if not text or text.isspace(): continue
            try:
                tl_x, tl_y = bbox[0]
                bl_x, bl_y = bbox[3]
                y_center = (float(tl_y) + float(bl_y)) / 2.0

                if not current_line or abs(y_center - last_y_center) <= y_tolerance:
                    current_line.append((float(tl_x), text))
                    last_y_center = y_center
                else:
                    current_line.sort(key=lambda item: item[0])
                    lines.append((last_y_center, " ".join([item[1] for item in current_line])))
                    current_line = [(float(tl_x), text)]
                    last_y_center = y_center
            except (IndexError, TypeError, ValueError):
                continue

        if current_line:
            current_line.sort(key=lambda item: item[0])
            lines.append((last_y_center, " ".join([item[1] for item in current_line])))

        c.setFont("Helvetica", fixed_font_size)
        line_height_pt = fixed_font_size * 1.2
        drawable_lines = []
        for original_y_center, line_text in lines:
            pdf_y = (page_height_a4 - margin) - (original_y_center * scale_y)
            drawable_lines.append((pdf_y, line_text))

        drawable_lines.sort(key=lambda item: item[0], reverse=True)

        text_obj = c.beginText()
        current_y = page_height_a4 - margin
        text_obj.setTextOrigin(margin, current_y)

        first_line_on_page = True
        for pdf_y, line_text in drawable_lines:
            target_y = pdf_y
            if not first_line_on_page and (current_y - target_y) < (line_height_pt * 0.5):
                current_y -= line_height_pt
            else:
                current_y = target_y

            if current_y < margin:
                c.drawText(text_obj)
                c.showPage()
                c.setFont("Helvetica", fixed_font_size)
                text_obj = c.beginText()
                current_y = page_height_a4 - margin
                text_obj.setTextOrigin(margin, current_y)
                first_line_on_page = True

            text_obj.setTextOrigin(margin, current_y)
            current_line_segment = ""
            words = line_text.split()
            for word in words:
                test_segment = f"{current_line_segment} {word}".strip()
                if c.stringWidth(test_segment, "Helvetica", fixed_font_size) < drawable_width:
                    current_line_segment = test_segment
                else:
                    text_obj.textLine(current_line_segment)
                    current_y -= line_height_pt
                    if current_y < margin:
                        c.drawText(text_obj)
                        c.showPage()
                        c.setFont("Helvetica", fixed_font_size)
                        text_obj = c.beginText()
                        current_y = page_height_a4 - margin
                        text_obj.setTextOrigin(margin, current_y)
                        first_line_on_page = True
                    current_line_segment = word
                    text_obj.setTextOrigin(margin, current_y)

            text_obj.textLine(current_line_segment)
            first_line_on_page = False

        c.drawText(text_obj)
        c.showPage()

    c.save()
    print(f"Structured text PDF saved as: {os.path.basename(output_filename)}")
    return output_filename

def search_documents(query, base_dir=PDF_STORAGE_FOLDER):
    results = []
    print(f"Searching for '{query}' in text files under '{base_dir}'...")
    if not os.path.isdir(base_dir):
        print(f"Error: Search directory '{base_dir}' not found.")
        return results
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                        content = f.read()
                        if query.lower() in content.lower():
                            index = content.lower().find(query.lower())
                            start = max(index - 50, 0)
                            end = min(index + len(query) + 50, len(content))
                            snippet = f"...{content[start:end].replace(chr(10), ' ').replace(chr(13), ' ')}..."
                            results.append((file_path, snippet))
                            print(f"  Found in: {file_path}")
                except Exception as e:
                    print(f"Error reading or processing {file_path}: {e}")
    print(f"Search complete. Found {len(results)} occurrences.")
    return results

if __name__ == '__main__':
    print("File Manager module loaded.")
