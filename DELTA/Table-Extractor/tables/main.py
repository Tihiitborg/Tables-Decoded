import matplotlib.pyplot as plt
from .td import TableDetector
from .physical_tsr import get_cols_from_tatr, get_cells_from_rows_cols, get_rows_from_tatr
from .utils import *
from .logical_tsr import get_logical_structure, align_otsl_from_rows_cols, convert_to_html
from .ocr import *
from bs4 import BeautifulSoup
import torch

def perform_td(image_path):
    image = cv2.imread(image_path)
    table_det = TableDetector()
    return table_det.predict(image = image)

def perform_tsr(img_file, x1, y1, struct_only, lang, ocr_engine='easy'):
    rows = get_rows_from_tatr(img_file)
    cols = get_cols_from_tatr(img_file)
    print('Physical TSR')
    print(str(len(rows)) + ' rows detected')
    print(str(len(cols)) + ' cols detected')
    rows, cols = order_rows_cols(rows, cols)
    ## Extracting Grid Cells
    cells = get_cells_from_rows_cols(rows, cols)
    ## Corner case if no cells detected
    if len(cells) == 0 or len(cols) == 0 or len(rows) == 0:
        table_img = cv2.imread(img_file)
        h, w, c = table_img.shape
        bbox = [0, 0, w, h]
        text = get_cell_ocr(table_img, bbox, lang)
        html_string = '<p>' + text + '</p>'
        return html_string, []
    print('Logical TSR')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    otsl_string = get_logical_structure(img_file, device)
    corrected_otsl = align_otsl_from_rows_cols(otsl_string, len(rows), len(cols))
    # Correction
    corrected_otsl = corrected_otsl.replace("E", "C")
    corrected_otsl = corrected_otsl.replace("F", "C")
    print('OTSL => ' + otsl_string)
    print("Corrected OTSL => " + corrected_otsl)
    html_string, struc_cells = convert_to_html(corrected_otsl, len(rows), len(cols), cells)
    # Parse the HTML
    soup = BeautifulSoup('<html>' + html_string + '</html>', 'html.parser')

    # Do this if struct_only flag is FALSE
    if struct_only == False:
        cropped_img = cv2.imread(img_file)
        # Perform OCR
        soup = get_table_ocr_all_at_once(cropped_img, soup, lang, x1, y1, ocr_engine)

    return soup, struc_cells


def get_full_page_hocr(img_file, lang):
    tabledata = get_table_hocrs(img_file, lang)
    finalimgtoocr = img_file
    # Hide all tables from images before perfroming recognizing text
    if len(tabledata) > 0:
        img = cv2.imread(img_file)
        for entry in tabledata:
            bbox = entry[1]
            tab_x = bbox[0]
            tab_y = bbox[1]
            tab_x2 = bbox[2]
            tab_y2 = bbox[3]
            img_x = int(tab_x)
            img_y = int(tab_y)
            img_x2 = int(tab_x2)
            img_y2 = int(tab_y2)
            cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (255, 0, 255), -1)
        finalimgfile = img_file[:-4] + '_filtered.jpg'
        cv2.imwrite(finalimgfile, img)
        finalimgtoocr = finalimgfile

    # Now we detect text using Tesseract
    hocr = pytesseract.image_to_pdf_or_hocr(finalimgtoocr, lang=lang, extension='hocr')
    soup = BeautifulSoup(hocr, 'html.parser')

    # Adding table hocr in final hocr at proper position
    if len(tabledata) > 0:
        for entry in tabledata:
            tab_element = entry[0]
            tab_bbox = entry[1]
            tab_position = tab_bbox[1]
            tab_inserted = False
            for elem in soup.find_all('span', class_="ocr_line"):
                find_all_ele = elem.attrs["title"].split(" ")
                line_position = int(find_all_ele[2])
                if tab_position < line_position:
                    elem.insert_before(tab_element)
                    tab_inserted = True
                    break
            if not tab_inserted:
                elem.insert_after(tab_element)

    return soup


def get_table_hocrs(image_file, lang):
    final_hocrs = []
    image = cv2.imread(image_file)
    dets = perform_td(image_file)
    print(str(len(dets)) + ' tables detected')
    for det in dets:
        x1, y1, x2, y2 = map(int, det)  # Convert coordinates to integers
        tab_box = [x1, y1, x2, y2]
        cropped_img = image[y1:y2, x1:x2]  # Crop the image using the bounding box
        plt.imsave("temp.jpg", cropped_img)
        img_path = "temp.jpg"
        hocr_string, struct_cells = perform_tsr(img_path, x1, y1, False, lang)
        final_hocrs.append([hocr_string, tab_box])

    return final_hocrs

if __name__=="__main__":
    print()
    # Try TD Call
    # tabs = perform_td('samples/page.png')
    # print(tabs)

    # Try Apps call
    # hocr = get_full_page_hocr('samples/page.png', 'eng')
    # print(hocr)

    # Try TSR Call
    img_path = "samples/cropped_img_2.jpg"
    hocr_string = perform_tsr(img_path, 0, 0, struct_only = True)
    print(hocr_string)

