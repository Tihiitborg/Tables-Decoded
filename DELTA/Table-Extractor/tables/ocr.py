
from PIL import Image
import pytesseract
import easyocr
import torch
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

# Load EasyOCR once
reader = easyocr.Reader(['en'])

# Load Surya OCR v0.6.0 models once
langs = ["en"]  # you can pass multiple, e.g., ["en", "hi"]
det_processor, det_model = load_det_processor(), load_det_model("datalab-to/surya-detection")
rec_model, rec_processor = load_rec_model("datalab-to/surya-recognition"), load_rec_processor()

def get_cell_ocr(img, bbox, lang, ocr_engine):
    """
    img: numpy array (full image)
    bbox: [x1, y1, x2, y2]
    lang: language code (for tesseract/easyocr)
    ocr_engine: 'tess', 'easy', or 'surya'
    """
    cell_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cell_pil_img = Image.fromarray(cell_img)

    if ocr_engine == 'tess':
        ocr_result = pytesseract.image_to_string(cell_pil_img, config='--psm 6', lang=lang)
        ocr_result = ocr_result.replace("\n", " ").strip()

    elif ocr_engine == 'easy':
        ocr_result = get_easy_ocr(cell_img)

    elif ocr_engine == 'surya':
        predictions = run_ocr([cell_pil_img], [langs], det_model, det_processor, rec_model, rec_processor)
        # Join all recognized lines into one string
        ocr_result = " ".join([line['text'] for line in predictions[0]['text_lines']]).strip()

    else:
        raise ValueError(f"Unknown OCR engine: {ocr_engine}")

    return ocr_result.replace("|", "")

def get_table_ocr_all_at_once(cropped_img, soup, lang, x1, y1, ocr_engine):
    """
    cropped_img: numpy array (cropped table image)
    soup: BeautifulSoup object
    lang: OCR language
    x1, y1: table offset in original image
    ocr_engine: 'tess', 'easy', or 'surya'
    """
    for bbox in soup.find_all('td'):
        ocr_bbox = list(map(int, bbox['title'].split(' ')[1:]))
        bbox.string = get_cell_ocr(cropped_img, ocr_bbox, lang, ocr_engine)

        # Adjust to original image coordinates
        ocr_bbox[0] += x1
        ocr_bbox[1] += y1
        ocr_bbox[2] += x1
        ocr_bbox[3] += y1
        bbox['bbox'] = f'{ocr_bbox[0]} {ocr_bbox[1]} {ocr_bbox[2]} {ocr_bbox[3]}'

    return soup

def get_easy_ocr(image):
    result = reader.readtext(image, detail=0)
    return ' '.join(result)
