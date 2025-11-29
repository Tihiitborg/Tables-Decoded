import os
import sys
import json
import cv2
from tqdm import tqdm
from tables.main import perform_td, perform_tsr, get_full_page_hocr

#change device setting according to availability
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# -------------------------------
# Resize helper
# -------------------------------
def resize_image_keep_aspect(image_path, target_width=2048):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]
    if width >= target_width:
        return image_path  # Skip resizing

    scale_ratio = target_width / width
    new_size = (int(width * scale_ratio), int(height * scale_ratio))

    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    # Save resized image to ./tmp instead of /tmp
    os.makedirs("./tmp", exist_ok=True)
    resized_path = os.path.join("./tmp", os.path.basename(image_path))
    cv2.imwrite(resized_path, resized_img)
    return resized_path

# -------------------------------
# Input arguments
# -------------------------------
img_dir = sys.argv[1]     # Path to image directory
mode = sys.argv[2]        # 'td', 'tsr', or 'hocr'
struc_only = sys.argv[3]  # 'True' or 'False'

struct_flag = struc_only != 'False'

# -------------------------------
# Supported image extensions
# -------------------------------
valid_exts = {'.png', '.jpg', '.jpeg'}

# -------------------------------
# Gather image files
# -------------------------------
img_files = sorted([
    f for f in os.listdir(img_dir)
    if any(f.lower().endswith(ext) for ext in valid_exts)
])

# -------------------------------
# Process images with progress bar
# -------------------------------
results = []
for filename in tqdm(img_files, desc=f"Processing in '{mode}' mode"):
    try:
        original_path = os.path.join(img_dir, filename)
        resized_path = resize_image_keep_aspect(original_path)

        if mode == 'td':
            result = perform_td(resized_path)
            results.append({
                'filename': filename,
                'result': str(result)
            })

        elif mode == 'tsr':
          #For parameter ocr_engine ='easy' corresponding to easy ocr the parameter "utk" can be anything, but for ocr_engine = 'tess' change 'utk' with 'hin+eng' for hindi and 'eng' for English ocr.
            result, struc_cells = perform_tsr(resized_path, 0, 0, struct_flag, 'utk', ocr_engine='easy')
            results.append({
                'filename': filename,
                'result': str(result),
                'struc_cells': str(struc_cells)
            })

        elif mode == 'hocr':
            result = get_full_page_hocr(resized_path, 'eng')
            results.append({
                'filename': filename,
                'result': str(result)
            })

        else:
            print(f"Unknown mode: {mode}, skipping {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# -------------------------------
# Save results to JSON
# -------------------------------
os.makedirs("./output", exist_ok=True)
#change filename according to yourself
output_path = os.path.join("./output", f"{mode}_easy_output.json")                   

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved results to: {output_path}")
