## scripts for inference DELTA + TARQA for TabVQA Task

import os
import sys
import json
import cv2
from tqdm import tqdm
from tables.main import perform_td, perform_tsr, get_full_page_hocr
import json
import re
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm
import torch
import json
import os
from tqdm import tqdm
import Levenshtein
from transformers import AutoTokenizer, LlamaForCausalLM


def extract_otsl_with_content(html_string):
    """
    Converts an HTML table into an OTSL matrix with content at the correct positions,
    handling complex structures with rowspan and colspan.
    """
    soup = BeautifulSoup(html_string, 'html.parser')
    table = soup.find('table')

    if not table:
        return "<otsl> </otsl>"  # Return empty OTSL if no table exists

    rows = table.find_all('tr')

    # Step 1: Compute Actual Row Count (`R`) and Column Count (`C`)
    row_spans = []  # Track ongoing rowspan usage
    R = len(rows)  # Base row count
    C = max(sum(int(cell.get('colspan', 1)) for cell in row.find_all(['td', 'th'])) for row in rows)

    # Adjust R based on `rowspan`
    for row in rows:
        row_span_count = [int(cell.get('rowspan', 1)) for cell in row.find_all(['td', 'th'])]
        if row_span_count:
            max_rowspan = max(row_span_count)
            if max_rowspan > 1:
                R += (max_rowspan - 1)

    # Step 2: Initialize OTSL Matrix and Cell Map
    otsl_matrix = [['<ecel>' for _ in range(C)] for _ in range(R)]
    cell_map = np.zeros((R, C), dtype=int)  # Tracks occupied cells

    row_idx = 0  # Tracks the actual row index
    for row in rows:
        col_idx = 0
        while row_idx < R and np.any(cell_map[row_idx]):  # Skip already occupied rows
            row_idx += 1

        for cell in row.find_all(['td', 'th']):
            while col_idx < C and cell_map[row_idx][col_idx] == 1:
                col_idx += 1

            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))

            if row_idx >= R or col_idx >= C:
                continue  # Skip if indices go out of bounds

            cell_text = cell.get_text(strip=True).replace(" ", "_")
            otsl_matrix[row_idx][col_idx] = f'<fcel> {cell_text}' if cell_text else '<ecel>'

            # Fill merged cells
            for c in range(1, colspan):
                if col_idx + c < C:
                    otsl_matrix[row_idx][col_idx + c] = '<lcel>'

            for r in range(1, rowspan):
                if row_idx + r < R:
                    otsl_matrix[row_idx + r][col_idx] = '<ucel>'
                    for c in range(1, colspan):
                        if col_idx + c < C:
                            otsl_matrix[row_idx + r][col_idx + c] = '<xcel>'

            # Mark occupied positions
            for r in range(rowspan):
                for c in range(colspan):
                    if row_idx + r < R and col_idx + c < C:
                        cell_map[row_idx + r][col_idx + c] = 1

            col_idx += colspan  # Move to next column after colspan width

        row_idx += 1  # Move to the next row

    # Convert matrix to OTSL string
    otsl_string = " ".join([" ".join(row) + " <nl>" for row in otsl_matrix]).strip()
    return otsl_string



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
input_question = sys.argv[2]
mode = 'tsr'        # 'td', 'tsr', or 'hocr'
struc_only = 'False'  # 'True' or 'False'

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
            otsl_string = extract_otsl_with_content(result)
            results.append({
                'filename': filename,
                'html': str(result),
                'otsl': otsl_string,
                'input_ques': input_question
            })

        elif mode == 'tsr':
          #For parameter ocr_engine ='easy' corresponding to easy ocr the parameter "utk" can be anything, but for ocr_engine = 'tess' change 'utk' with 'hin+eng' for hindi and 'eng' for English ocr.
            result, struc_cells = perform_tsr(resized_path, 0, 0, struct_flag, 'utk', ocr_engine='easy')
            otsl_string = extract_otsl_with_content(result)
            results.append({
                'filename': filename,
                'html': str(result),
                'struc_cells': str(struc_cells),
                'otsl': otsl_string,
                'input_ques': input_question
            })

        elif mode == 'hocr':
            result = get_full_page_hocr(resized_path, 'eng')
            otsl_string = extract_otsl_with_content(result)
            results.append({
                'filename': filename,
                'html': str(result),
                'otsl': otsl_string,
                'input_ques': input_question
            })

        else:
            print(f"Unknown mode: {mode}, skipping {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")



# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Model Wrapper ===
class TableVQAModel(torch.nn.Module):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__()
        print(f"Loading model: {model_name}")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Use automatic GPU placement
        )
        self.model.eval()

    def forward(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.loss, outputs.logits

def extract_answer(decoded_output, input_text):
    decoded_output = decoded_output.lower()
    if "### answer:" in decoded_output:
        start = decoded_output.find("### answer:") + len("### answer:")
        end = decoded_output.find("###", start)
        return decoded_output[start:end].strip() if end != -1 else decoded_output[start:].strip()
    else:
        return decoded_output.replace(input_text.lower(), "").strip()

# === Main Evaluation Logic ===
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
checkpoint_path = "/../tablevqa_epoch4.pth"  # update epoch number as needed

# === Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = TableVQAModel(model_name=model_name)

# Load trained weights
state_dict = torch.load(checkpoint_path, map_location="cpu")
new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
base_model.load_state_dict(new_state_dict)
base_model = base_model.to(device)
model = base_model.model  # unwrap inner model for `.generate`

print(f"Loaded model from: {checkpoint_path}")

predictions = []


input_text = f"""### Instruction:
Given the following table, answer the question in one word or short phrase. Do not provide an explanation.

### Table:
{results["otsl"]}

### Question:
{input_question}

### Answer:"""

inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=4096
).to(device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=False,
        num_beams=5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
predicted_answer = extract_answer(decoded_output, input_text)
predicted_answer = predicted_answer.strip().lower()

results.append({"pred_ans": predicted_answer})

# -------------------------------
# Save results to JSON
# -------------------------------
os.makedirs("./output", exist_ok=True)
#change filename according to yourself
output_path = os.path.join("./output", f"{mode}_easy_output.json")                   

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved results to: {output_path}")
