from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import csv
import json
import os
import re
from tqdm import tqdm
import Levenshtein

# === Configuration ===
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
CSV_PATH = "./torque-qa.csv"
IMAGE_DIR = "../HindiTableImages/"
OUTPUT_FILE = "qwen_results.jsonl"

# === Device Setup ===
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Use GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Normalization function for FinTabNet-style relieved accuracy ===
def fintabnet_normalize(text):
    def _normalize(s):
        if not isinstance(s, str):
            s = str(s)
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[,\.]", "", s)  # remove commas/periods
        s = s.replace(" ", "")
        return s

    gt = _normalize(text)
    return gt, [gt]

# === Load model and processor ===
print("Loading model and processor...")
with tqdm(total=2, desc="Loading components") as pbar:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    pbar.update(1)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    pbar.update(1)

# === Load test data from CSV ===
print("Loading test data from CSV...")
test_data = []

with open(CSV_PATH, 'r', encoding='utf-8') as csvfile:
    # Skip the empty first row
    next(csvfile)
    
    # Read the actual header from the second row
    reader = csv.DictReader(csvfile)
    rows = list(reader)
    
    for row in tqdm(rows, desc="Processing CSV rows"):
        # Get image name
        image_name = row.get('Image_name', '').strip()
        if not image_name:
            continue
            
        # Check for questions and answers
        for i in range(1, 4):
            question_key = f'Question {i}'
            answer_key = f'Answer {i}'
            
            if (question_key in row and answer_key in row and 
                row[question_key].strip() and row[answer_key].strip()):
                test_data.append({
                    'image_name': image_name,
                    'question': row[question_key].strip(),
                    'ground_truth': row[answer_key].strip()
                })

print(f"Loaded {len(test_data)} test samples.")

if len(test_data) == 0:
    print("No test data found! Checking CSV structure...")
    with open(CSV_PATH, 'r', encoding='utf-8') as csvfile:
        lines = csvfile.readlines()
        print(f"First few lines of CSV:")
        for i, line in enumerate(lines[:5]):
            print(f"Line {i}: {line.strip()}")
    exit(1)

# === Process each sample ===
exact_match = 0
similar_match = 0
relieved_match = 0
total = 0

# Create output file and write header
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("# Qwen2.5-VL Evaluation Results\n")

# Create progress bar
pbar = tqdm(test_data, desc="Processing samples", unit="sample")

for sample in pbar:
    image_path = os.path.join(IMAGE_DIR, sample['image_name'])
    question = f"संक्षेप में, केवल प्रत्यक्ष उत्तर दें। {sample['question']}"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found. Skipping.")
        continue
    
    try:
        # Prepare the input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        predicted_answer = output_text[0].strip()
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        predicted_answer = ""
    
    # Normalize and calculate metrics
    ground_truth = sample['ground_truth']
    
    # Normalize for comparison
    predicted_norm = predicted_answer.strip().lower()
    ground_truth_norm = ground_truth.strip().lower()
    
    # Calculate Levenshtein ratio
    lev_score = Levenshtein.ratio(predicted_norm, ground_truth_norm)
    is_exact = predicted_norm == ground_truth_norm
    is_similar = lev_score >= 0.8
    
    # Relieved Accuracy (FinTabNet style)
    norm_pred, norm_preds = fintabnet_normalize(predicted_answer)
    norm_gt, norm_gts = fintabnet_normalize(ground_truth)
    relieved_loose = int(any(_p == _g for _p in norm_preds for _g in norm_gts))
    
    # Update metrics
    exact_match += int(is_exact)
    similar_match += int(is_similar)
    relieved_match += relieved_loose
    total += 1
    
    # Create prediction data
    prediction_data = {
        "image_name": sample['image_name'],
        "question": sample['question'],
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "levenshtein_score": lev_score,
        "exact_match": is_exact,
        "lenient_match": is_similar,
        "relieved_match": relieved_loose
    }
    
    # Write to JSONL file immediately
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(prediction_data, ensure_ascii=False) + '\n')
    
    # Update progress bar description
    pbar.set_postfix({
        'EM': f'{exact_match/total:.2%}',
        'Lev≥0.8': f'{similar_match/total:.2%}',
        'Relieved': f'{relieved_match/total:.2%}',
        'Processed': f'{total}/{len(test_data)}'
    })

# === Save final metrics ===
final_metrics = {
    "total_samples": total,
    "exact_match_accuracy": exact_match / total * 100 if total > 0 else 0,
    "levenshtein_accuracy": similar_match / total * 100 if total > 0 else 0,
    "relieved_accuracy": relieved_match / total * 100 if total > 0 else 0
}

with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
    f.write("# FINAL METRICS\n")
    f.write(json.dumps(final_metrics, ensure_ascii=False) + '\n')

# === Print final metrics ===
print("\n=== Final Evaluation ===")
print(f"Total Samples                 : {total}")
print(f"Exact Match Accuracy          : {final_metrics['exact_match_accuracy']:.2f}%")
print(f"Levenshtein ≥ 0.8 Accuracy    : {final_metrics['levenshtein_accuracy']:.2f}%")
print(f"Relieved Accuracy (FinTabNet) : {final_metrics['relieved_accuracy']:.2f}%")

# === Print first 10 examples ===
print("\n=== First 10 Predictions ===")
# Read first 10 results from the JSONL file
first_10 = []
with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if line.startswith('#') or not line.strip():
            continue
        if len(first_10) >= 10:
            break
        try:
            data = json.loads(line)
            first_10.append(data)
        except:
            continue

for i, pred in enumerate(first_10):
    print(f"\nExample {i+1}")
    print(f"Image           : {pred['image_name']}")
    print(f"Question        : {pred['question']}")
    print(f"Ground Truth    : {pred['ground_truth']}")
    print(f"Predicted Answer: {pred['predicted_answer']}")
    print(f"Levenshtein     : {pred['levenshtein_score']:.2f}")
    print(f"Exact Match     : {pred['exact_match']}")
    print(f"Lenient Match   : {pred['lenient_match']}")
    print(f"Relieved Match  : {pred['relieved_match']}")

print(f"\nResults saved to {OUTPUT_FILE}")