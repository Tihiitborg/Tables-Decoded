import pdf2image
import cv2


def pdf_to_images(input_pdf):
    # Open the input PDF
    images = pdf2image.convert_from_path(input_pdf)
    return images

def draw_bboxes(img_file, bboxes, color = (255, 0, 255), thickness= 2):
    image = cv2.imread(img_file)
    for b in bboxes:
        start_point = (int(b[0]), int(b[1]))
        end_point = (int(b[2]), int(b[3]))
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

def order_rows_cols(rows, cols):
    # Order rows from top to bottom based on y1 (second value in the bounding box)
    rows = sorted(rows, key=lambda x: x[1])
    # Order columns from left to right based on x1 (first value in the bounding box)
    cols = sorted(cols, key=lambda x: x[0])
    return rows, cols
