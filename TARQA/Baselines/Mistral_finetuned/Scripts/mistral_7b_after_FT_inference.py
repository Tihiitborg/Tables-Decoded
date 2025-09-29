import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MistralForCausalLM
import json
from tqdm import tqdm
import Levenshtein
import logging
from datetime import datetime
import re

# === Setup Logging ===
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./mistralFTresults/inference_log_{timestamp}.log"
    os.makedirs("./mistralFTresults", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# === Device Setup ===
GPU_INDEX = 0  # Change this to 0, 1, 2, etc. to select the desired GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() and torch.cuda.device_count() > GPU_INDEX else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available() and torch.cuda.device_count() > GPU_INDEX:
    logger.info(f"GPU: {torch.cuda.get_device_name(GPU_INDEX)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(GPU_INDEX).total_memory / 1024**3:.1f} GB")
else:
    logger.warning(f"GPU {GPU_INDEX} not available (detected {torch.cuda.device_count()} GPUs). Falling back to {device}.")

# === Answer Cleaning Function ===
import re
import logging

def clean_predicted_answer(raw_answer):
    """
    Clean the predicted answer by removing explanations, formatting, and extra text.
    Extract only the core answer.
    
    Args:
        raw_answer (str): The raw predicted answer from the model.
    
    Returns:
        str: The cleaned answer, or "Generation failed" if cleaning fails.
    """
    if not raw_answer or raw_answer == "Generation failed":
        logger.warning("Empty or failed raw answer received.")
        return raw_answer

    # Define prefixes to remove
    prefixes_to_remove = [
        r"1 word or short phrase\s*:",
        r"1 word\s*:",
        r"1-word response\s*:",
        r"Answer\s*:",
        r"Short answer\s*:",
        r"Response\s*:",
        r"The answer is\s*:",
        r"^\d+\.\s*",  # Handles "1. " or similar numbering
    ]

    # Remove leading/trailing whitespace
    cleaned = raw_answer.strip()

    # Remove prefixes (case-insensitive)
    for prefix in prefixes_to_remove:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE).strip()

    # Remove common delimiters for explanations
    delimiters = [
        r"\n\n### Explanation\s*:",
        r"\n### Explanation\s*:",
        r"\nExplanation\s*:",
        r"\n\n",
    ]
    for delimiter in delimiters:
        if re.search(delimiter, cleaned):
            cleaned = cleaned.split(delimiter)[0].strip()

    # Handle multi-line answers by selecting the first non-empty, non-metadata line
    if '\n' in cleaned:
        lines = cleaned.split('\n')
        for line in lines:
            line = line.strip()
            # Skip lines that are empty, metadata, or instructions
            if line and not line.startswith('#') and not line.lower().startswith(('to ', 'the ', 'answer:', 'response:')):
                cleaned = line
                break
        else:
            # If no valid line is found, use the first non-empty line
            cleaned = next((line.strip() for line in lines if line.strip()), cleaned)

    # Remove trailing punctuation (except for special cases like Ph.D)
    cleaned = re.sub(r'[.\n\r\t]+$', '', cleaned).strip()

    # Remove extra internal whitespace while preserving case
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Final check: if cleaned answer is empty or looks like a prefix, log a warning
    if not cleaned or re.match(r"^(1 word|Answer|Response|1-word response|Short answer|The answer is)\s*:?$", cleaned, re.IGNORECASE):
        logger.warning(f"Cleaning failed to extract valid answer from: {raw_answer[:100]}...")
        return "Generation failed"

    logger.debug(f"Cleaned answer: {cleaned} from raw: {raw_answer[:100]}...")
    return cleaned

# === Dataset Class ===
class TableVQADataset:
    def __init__(self, json_path, tokenizer, max_seq_len=4096):
        logger.info(f"Loading dataset from {json_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        for sample in self.data:
            sample['question'] = re.sub(r',\s*Q:\s*Q:\s*Q:\s*Q:.*$', '', sample['question']).strip()
        
        logger.info(f"Loaded {len(self.data)} samples.")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self._log_sample_data()

    def _log_sample_data(self):
        logger.info("=== DATASET SAMPLE INSPECTION ===")
        for i in range(min(2, len(self.data))):
            sample = self.data[i]
            logger.info(f"Sample {i+1} keys: {list(sample.keys())}")
            logger.info(f"Question: {sample.get('question', 'N/A')[:100]}...")
            logger.info(f"Answer: {sample.get('answer_text', 'N/A')}")
            logger.info(f"OTSL: {sample.get('otsl', 'N/A')[:200]}...")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        question = entry["question"]
        answer = entry["answer_text"]
        table_context = entry["otsl"]

        prompt = f"""### Instruction:
Given the following table, answer the question in one word or short phrase. Do not provide an explanation.

### Table:
{table_context}

### Question:
{question}

### Answer: """
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors='pt'
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompt,
            "raw_question": question,
            "raw_answer": answer
        }

# === Model Wrapper ===
class TableVQAModel(torch.nn.Module):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        super().__init__()
        logger.info(f"Loading model: {model_name}")
        self.model = MistralForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            use_cache=False
        )
        self.model.to(device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

# === Inference Function ===
def run_inference(model, batch, tokenizer, device, max_new_tokens=50):
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(batch["input_ids"].size(0)):
            prompt = batch["prompt"][i]
            input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
            try:
                outputs = model.model.generate(
                    **input_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.1,
                    repetition_penalty=1.1
                )
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                raw_pred_answer = full_output[len(prompt):].strip()
                
                # Clean the predicted answer
                cleaned_pred_answer = clean_predicted_answer(raw_pred_answer)
                
            except Exception as e:
                logger.warning(f"Generation failed: {str(e)}")
                raw_pred_answer = "Generation failed"
                cleaned_pred_answer = "Generation failed"
            
            gt_answer = batch["raw_answer"][i]
            question = batch["raw_question"][i]
            results.append({
                "question": question,
                "gt_answer": gt_answer,
                "raw_pred_answer": raw_pred_answer,
                "cleaned_pred_answer": cleaned_pred_answer
            })
    return results

# === Normalization Function for FinTabNet-style Relieved Accuracy ===
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

# === Inference with Checkpoint ===
def infer_with_checkpoint(checkpoint_path, json_path, tokenizer, device, max_seq_len=3300, batch_size=1):
    # Load model
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    with tqdm(total=2, desc="Loading model components") as pbar:
        model = TableVQAModel(model_name="mistralai/Mistral-7B-Instruct-v0.3")
        pbar.update(1)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded checkpoint: {checkpoint_path}")
        pbar.update(1)

    # Load dataset
    logger.info(f"Loading test dataset: {json_path}")
    dataset = TableVQADataset(json_path, tokenizer, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    logger.info(f"Test dataset loaded. Batches: {len(dataloader)}")

    # Initialize metrics
    exact_match = 0
    similar_match = 0
    relieved_match = 0
    total = 0
    results = []

    # Output file setup
    os.makedirs("./mistralFTresults", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./mistral7bresults/cleaned_inference_results_{timestamp}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Mistral-7B TableVQA Cleaned Inference Results\n")

    # Inference loop
    for batch in tqdm(dataloader, desc="Inferencing", unit="batch"):
        batch_results = run_inference(model, batch, tokenizer, device, max_new_tokens=50)
        for result in batch_results:
            raw_predicted_answer = result["raw_pred_answer"]
            predicted_answer = result["cleaned_pred_answer"]
            ground_truth = result["gt_answer"]
            question = result["question"]

            # Normalize for comparison (use cleaned answer)
            predicted_norm = predicted_answer.strip().lower()
            ground_truth_norm = ground_truth.strip().lower()

            # Calculate metrics
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

            # Save individual result
            prediction_data = {
                "question": question,
                "ground_truth": ground_truth,
                "raw_predicted_answer": raw_predicted_answer,
                "cleaned_predicted_answer": predicted_answer,
                "levenshtein_score": lev_score,
                "exact_match": is_exact,
                "lenient_match": is_similar,
                "relieved_match": relieved_loose
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(prediction_data, ensure_ascii=False) + '\n')

            results.append(prediction_data)

        # Update progress bar
        if total > 0:
            tqdm.write(f"Processed {total}/{len(dataset)} samples | EM: {exact_match/total:.2%} | Lev≥0.8: {similar_match/total:.2%} | Relieved: {relieved_match/total:.2%}")

    # Save final metrics
    final_metrics = {
        "total_samples": total,
        "exact_match_accuracy": exact_match / total * 100 if total > 0 else 0,
        "levenshtein_accuracy": similar_match / total * 100 if total > 0 else 0,
        "relieved_accuracy": relieved_match / total * 100 if total > 0 else 0
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("# FINAL METRICS\n")
        f.write(json.dumps(final_metrics, ensure_ascii=False) + '\n')

    # Log final metrics
    logger.info("\n=== Final Evaluation ===")
    logger.info(f"Total Samples                 : {total}")
    logger.info(f"Exact Match Accuracy          : {final_metrics['exact_match_accuracy']:.2f}%")
    logger.info(f"Levenshtein ≥ 0.8 Accuracy    : {final_metrics['levenshtein_accuracy']:.2f}%")
    logger.info(f"Relieved Accuracy (FinTabNet) : {final_metrics['relieved_accuracy']:.2f}%")

    # Display first few predictions with cleaning comparison
    logger.info("\n=== First 10 Predictions (Before/After Cleaning) ===")
    for i, pred in enumerate(results[:10]):
        logger.info(f"\nExample {i+1}")
        logger.info(f"Question         : {pred['question']}")
        logger.info(f"Ground Truth     : {pred['ground_truth']}")
        logger.info(f"Raw Prediction   : {pred['raw_predicted_answer'][:100]}...")
        logger.info(f"Cleaned Prediction: {pred['cleaned_predicted_answer']}")
        logger.info(f"Levenshtein      : {pred['levenshtein_score']:.3f}")
        logger.info(f"Exact Match      : {pred['exact_match']}")
        logger.info(f"Lenient Match    : {pred['lenient_match']}")
        logger.info(f"Relieved Match   : {pred['relieved_match']}")

    logger.info(f"\nResults saved to {output_file}")
    return results

# === Main Function ===
def main():
    checkpoint_path = "./mistral7bresults/tablevqa_epoch4.pth"
    json_path = "./mistral7bresults/wtq_test.json"
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    max_seq_len = 3300
    batch_size = 1

    logger.info("=== STARTING MISTRAL INFERENCE WITH ANSWER CLEANING ===")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Test Dataset: {json_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Max sequence length: {max_seq_len}")
    logger.info(f"Batch size: {batch_size}")

    # Initialize tokenizer
    with tqdm(total=1, desc="Loading tokenizer") as pbar:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        pbar.update(1)

    # Run inference
    try:
        infer_with_checkpoint(checkpoint_path, json_path, tokenizer, device, max_seq_len=max_seq_len, batch_size=batch_size)
        logger.info("Inference completed successfully")
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

    logger.info("=== INFERENCE COMPLETED ===")

if __name__ == "__main__":
    main()