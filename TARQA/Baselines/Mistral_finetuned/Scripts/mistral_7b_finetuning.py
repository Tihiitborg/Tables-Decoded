import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, MistralForCausalLM, get_linear_schedule_with_warmup
import json
from tqdm import tqdm
import Levenshtein
import logging
from datetime import datetime
import re


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# === Setup Logging ===
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.log"
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Ensure only GPU0 is visible to prevent usage of other GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Dataset Class ===
class TableVQADataset(Dataset):
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

### Answer: {answer} </s>"""

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors='pt'
        )

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        labels = input_ids.clone()

        prompt_only = f"""### Instruction:
Given the following table, answer the question in one word or short phrase. Do not provide an explanation.

### Table:
{table_context}

### Question:
{question}

### Answer: """
        prompt_encoding = self.tokenizer(
            prompt_only,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors='pt'
        )
        prompt_len = prompt_encoding.input_ids.size(1)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_question": question,
            "raw_answer": answer,
            "prompt": prompt_only
        }

# === Model Wrapper ===
class TableVQAModel(nn.Module):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        super().__init__()
        logger.info(f"Loading model: {model_name}")
        self.model = MistralForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            use_cache=False
        )
        # Explicitly move model to cuda:0
        self.model.to(device)
        self.model.gradient_checkpointing_enable()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

# === Inference Function ===
def run_inference(model, batch, tokenizer, device, max_new_tokens=100):
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
                    eos_token_id=tokenizer.eos_token_id
                )
                full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred_answer = full_output[len(prompt):].strip()
            except Exception as e:
                logger.warning(f"Generation failed: {str(e)}")
                pred_answer = "Generation failed"
            
            gt_answer = batch["raw_answer"][i]
            question = batch["raw_question"][i]
            exact_match = pred_answer.lower().strip() == gt_answer.lower().strip()
            similarity = Levenshtein.ratio(pred_answer.lower().strip(), gt_answer.lower().strip())
            
            results.append({
                "question": question,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "exact_match": exact_match,
                "similarity": similarity
            })
    return results

# === Training Function ===
def train(model, dataloader, tokenizer, epochs=4, gradient_accumulation_steps=4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Total training steps: {total_steps}")
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch+1}/{epochs}")
        logger.info(f"{'='*50}")
        
        log_path = f"../Results/tablevqa_epoch{epoch+1}.log"
        with open(log_path, "a") as f:
            f.write(f"Starting this epoch {epoch+1}\n")
        
        total_loss = 0
        exact_match = 0
        similar_match = 0
        total_samples = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (batch_idx + 1) % 200 == 0:
                inference_results = run_inference(model, batch, tokenizer, device, max_new_tokens=100)
                for i, result in enumerate(inference_results):
                    pred_text = result["pred_answer"].lower().strip()
                    label_text = result["gt_answer"].lower().strip()
                    if pred_text == label_text:
                        exact_match += 1
                        logger.info(f"\n[Exact Match | Epoch {epoch+1} | Batch {batch_idx+1} | Sample {i+1}]")
                        logger.info(f"  Question: {result['question'][:100]}...")
                        logger.info(f"  Ground Truth: '{result['gt_answer']}'")
                        logger.info(f"  Predicted: '{result['pred_answer']}'")
                        logger.info(f"  Similarity: {result['similarity']:.3f}")
                    if Levenshtein.ratio(pred_text, label_text) >= 0.8:
                        similar_match += 1
                    total_samples += 1
            
            if total_samples > 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                    'Exact': f'{exact_match/total_samples*100:.1f}%',
                    'Similar': f'{similar_match/total_samples*100:.1f}%'
                })
        
        avg_loss = total_loss / len(dataloader)
        exact_acc = exact_match / total_samples * 100
        sim_acc = similar_match / total_samples * 100
        
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Avg Loss: {avg_loss:.4f} | Exact Match: {exact_acc:.2f}% | Levenshtein ≥ 0.8: {sim_acc:.2f}%")
        
        os.makedirs("../Results", exist_ok=True)
        
        checkpoint_path = f"../Results/tablevqa_epoch{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'exact_acc': exact_acc,
            'sim_acc': sim_acc
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

# === Main Function ===
def main():
    
    # CONFIGURATION
    json_path = "../Data/combined_wtq_html_otsl_sequential.json"
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    max_seq_len = 3300
    batch_size = 1
    epochs = 7
    
    logger.info("=== STARTING MISTRAL FINE-TUNING ===")
    logger.info(f"Dataset: {json_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Max sequence length: {max_seq_len}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    try:
        dataset = TableVQADataset(json_path, tokenizer, max_seq_len=max_seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        logger.info(f"Dataset loaded successfully. Batches: {len(dataloader)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return
    
    try:
        model = TableVQAModel(model_name)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
    try:
        train(model, dataloader, tokenizer, epochs=epochs, gradient_accumulation_steps=8)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    logger.info("=== TRAINING COMPLETED ===")

if __name__ == "__main__":
    # Set environment variable to use expandable segments for CUDA memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()