import os
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
import csv
import json
import re
from tqdm import tqdm
import Levenshtein
import shutil

# === Configuration ===
MODEL_PATH = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LOCAL_SAVE_PATH = "./local_llama_model"
CSV_PATH = "./torque-qa.csv"
IMAGE_DIR = "./HindiTableImages/"
OUTPUT_FILE = "llama_results.jsonl"

# === Device Setup ===
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === COMPLETELY BYPASS CACHE - Create temporary environment ===
TEMP_HF_HOME = "./temp_hf_home"
os.makedirs(TEMP_HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = TEMP_HF_HOME
os.environ["TRANSFORMERS_CACHE"] = TEMP_HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = TEMP_HF_HOME

print(f"Using temporary HF home: {TEMP_HF_HOME}")

# === Check if model exists locally, else download and save ===
print("Checking for local model...")
os.makedirs(LOCAL_SAVE_PATH, exist_ok=True)

# Check if model files exist
required_files = ["config.json", "preprocessor_config.json", "model.safetensors", "model-00001-of-00005.safetensors"]
model_files_exist = all(os.path.exists(os.path.join(LOCAL_SAVE_PATH, f)) for f in required_files)

if model_files_exist:
    print(f"Loading model and processor from {LOCAL_SAVE_PATH}...")
    with tqdm(total=2, desc="Loading components") as pbar:
        model = MllamaForConditionalGeneration.from_pretrained(
            LOCAL_SAVE_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        pbar.update(1)
        processor = AutoProcessor.from_pretrained(
            LOCAL_SAVE_PATH,
            local_files_only=True
        )
        pbar.update(1)
else:
    print(f"Model not found locally. Downloading from {MODEL_PATH}...")
    
    download_dir = "./temp_download"
    os.makedirs(download_dir, exist_ok=True)
    
    try:
        with tqdm(total=2, desc="Downloading components") as pbar:
            model = MllamaForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=download_dir,
                force_download=True,
                local_files_only=False
            )
            pbar.update(1)
            
            processor = AutoProcessor.from_pretrained(
                MODEL_PATH,
                cache_dir=download_dir,
                force_download=True,
                local_files_only=False
            )
            pbar.update(1)
        
        print(f"Saving model and processor to {LOCAL_SAVE_PATH}...")
        model.save_pretrained(LOCAL_SAVE_PATH)
        processor.save_pretrained(LOCAL_SAVE_PATH)
        shutil.rmtree(download_dir, ignore_errors=True)
        print(f"Model and processor saved to {LOCAL_SAVE_PATH}")
        
    except Exception as e:
        print(f"Error during download: {e}")
        shutil.rmtree(download_dir, ignore_errors=True)
        raise

# === Clean up temporary HF home ===
shutil.rmtree(TEMP_HF_HOME, ignore_errors=True)

# === Function to clean up predicted answer ===
def clean_predicted_answer(answer):
    """Extract only the assistant's response from the generated text"""
    # Remove user prompt and assistant tag
    cleaned = re.sub(r'^.*?assistant\s*[\n\\]*', '', answer, flags=re.IGNORECASE)
    
    # Remove any remaining user text
    cleaned = re.sub(r'user.*?assistant', '', cleaned, flags=re.IGNORECASE)
    
    # Remove special characters and extra whitespace
    cleaned = re.sub(r'[\*\\"]', '', cleaned)
    cleaned = cleaned.strip()
    
    # If we still have the full conversation, try to extract just the answer
    if 'user' in cleaned.lower() and 'assistant' in cleaned.lower():
        parts = re.split(r'assistant\s*', cleaned, flags=re.IGNORECASE)
        if len(parts) > 1:
            cleaned = parts[-1].strip()
    
    return cleaned

# === Normalization function for FinTabNet-style relieved accuracy ===
def fintabnet_normalize(text):
    def _normalize(s):
        if not isinstance(s, str):
            s = str(s)
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[,\.]", "", s)
        s = s.replace(" ", "")
        return s

    gt = _normalize(text)
    return gt, [gt]

# === Load test data from CSV ===
print("Loading test data from CSV...")
test_data = []

with open(CSV_PATH, 'r', encoding='utf-8') as csvfile:
    next(csvfile)
    reader = csv.DictReader(csvfile)
    rows = list(reader)
    
    for row in tqdm(rows, desc="Processing CSV rows"):
        image_name = row.get('Image_name', '').strip()
        if not image_name:
            continue
            
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

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("# Llama-3.2-Vision Evaluation Results\n")

pbar = tqdm(test_data, desc="Processing samples", unit="sample")

for sample in pbar:
    image_path = os.path.join(IMAGE_DIR, sample['image_name'])
    question = f"संक्षेप में, केवल प्रत्यक्ष उत्तर दें। {sample['question']}"
    
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found. Skipping.")
        continue
    
    try:
        image = Image.open(image_path)
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]}
        ]
        
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128)
            predicted_answer = processor.decode(output[0], skip_special_tokens=True)
        
        # Clean up the predicted answer to extract only the assistant's response
        predicted_answer = clean_predicted_answer(predicted_answer)
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        predicted_answer = ""
    
    ground_truth = sample['ground_truth']
    predicted_norm = predicted_answer.strip().lower()
    ground_truth_norm = ground_truth.strip().lower()
    
    lev_score = Levenshtein.ratio(predicted_norm, ground_truth_norm)
    is_exact = predicted_norm == ground_truth_norm
    is_similar = lev_score >= 0.8
    
    norm_pred, norm_preds = fintabnet_normalize(predicted_answer)
    norm_gt, norm_gts = fintabnet_normalize(ground_truth)
    relieved_loose = int(any(_p == _g for _p in norm_preds for _g in norm_gts))
    
    exact_match += int(is_exact)
    similar_match += int(is_similar)
    relieved_match += relieved_loose
    total += 1
    
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
    
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(prediction_data, ensure_ascii=False) + '\n')
    
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

print("\n=== Final Evaluation ===")
print(f"Total Samples                 : {total}")
print(f"Exact Match Accuracy          : {final_metrics['exact_match_accuracy']:.2f}%")
print(f"Levenshtein ≥ 0.8 Accuracy    : {final_metrics['levenshtein_accuracy']:.2f}%")
print(f"Relieved Accuracy (FinTabNet) : {final_metrics['relieved_accuracy']:.2f}%")

print("\n=== First 10 Predictions ===")
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