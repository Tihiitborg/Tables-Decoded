import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
from tqdm import tqdm
import Levenshtein

# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Dataset Class ===
class TableVQADataset(Dataset):
    def __init__(self, json_path, tokenizer, max_seq_len=4096, end_token="<END>"):
        print(f"Loading dataset from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples.")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.end_token = end_token

        # ensure there is a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        question = entry.get("question", "").strip()
        answer = entry.get("answer_text", "").strip()
        # prefer the otsl field if present, fallback to html or other fields
        table_context = entry.get("otsl", entry.get("html", "")).strip()

        # Clear, explicit prompt; end with newline so answer is clearly after the marker
        prompt_text = (
            "### Instruction:\n"
            "Given the following table, answer the question in one word or short phrase. Do not provide an explanation.\n\n"
            "### Table:\n"
            f"{table_context}\n\n"
            "### Question:\n"
            f"{question}\n\n"
            "### Answer:\n"
        )

        # append the end token to the answer so model learns when to stop
        answer_with_end = answer + " " + self.end_token

        # Tokenize prompt and answer separately (no added special tokens)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_with_end, add_special_tokens=False)

        # If combined length exceeds max_seq_len, truncate prompt from the LEFT (keep the answer)
        if len(prompt_ids) + len(answer_ids) > self.max_seq_len:
            available_for_prompt = self.max_seq_len - len(answer_ids)
            if available_for_prompt <= 0:
                # extreme case: answer itself too long -> truncate answer to max_seq_len
                answer_ids = answer_ids[: self.max_seq_len]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-available_for_prompt:]

        input_ids = prompt_ids + answer_ids
        input_len = len(input_ids)
        pad_len = self.max_seq_len - input_len

        # pad with eos_token_id (we set pad_token to eos earlier, but explicit is fine)
        input_ids = input_ids + [self.eos_id] * pad_len
        attention_mask = [1] * input_len + [0] * pad_len

        # labels: ignore prompt and padding (-100), compute loss only on answer token positions (including <END>)
        labels = [-100] * len(prompt_ids) + answer_ids + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": len(prompt_ids),     # integer
            "answer_text": answer              # raw string (without <END>), useful for metric comparisons
        }

# --- Model wrapper: will load model and keep it at .model ---
class TableVQAModel(nn.Module):
    def __init__(self, model_name="upstage/SOLAR-10.7B-Instruct-v1.0"):
        super().__init__()
        print(f"Loading model: {model_name}")
        # load model (device_map="" kept as originally used)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )
        # enable gradient checkpointing for memory saving
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

# --- Training loop (modified to use <END> token for generation stopping + save) ---
def train(model, dataloader, tokenizer, optimizer, epochs=4, grad_accum_steps=8, save_dir="solar10b_results", end_token="<END>"):
    model.train()
    # get id for our custom end token
    end_token_id = tokenizer.convert_tokens_to_ids(end_token)
    if end_token_id is None:
        raise ValueError(f"End token {end_token} not found in tokenizer vocab")

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0.0
        exact_match, similar_match, total = 0, 0, 0

        for i, batch in enumerate(tqdm(dataloader)):
            # move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # NaN check
            if torch.isnan(loss):
                print(f"⚠️ NaN loss at batch {i}, skipping")
                optimizer.zero_grad()
                continue

            # scale for grad accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps

            # ---- Evaluation / decoding using generate on only the prompt ----
            with torch.no_grad():
                answer_texts = batch["answer_text"]  # list of raw strings
                prompt_lens = batch["prompt_len"]     # list/tensor of integers

                for j in range(input_ids.size(0)):
                    prompt_len_j = int(prompt_lens[j])
                    # slice only the prompt tokens as generation input
                    if prompt_len_j == 0:
                        # nothing to generate from; skip
                        pred_text = ""
                        gold_text = answer_texts[j].strip().lower()
                    else:
                        prompt_input_ids = input_ids[j, :prompt_len_j].unsqueeze(0)
                        prompt_attention = attention_mask[j, :prompt_len_j].unsqueeze(0)

                        gen_ids = model.model.generate(
                            input_ids=prompt_input_ids,
                            attention_mask=prompt_attention,
                            max_new_tokens=16,   # small for short answers
                            do_sample=False,
                            eos_token_id=end_token_id,   # stop at <END>
                            pad_token_id=tokenizer.eos_token_id,
                            use_cache=True
                        )

                        # decode continuation (everything after prompt_len_j in gen_ids)
                        # but genotype may include prompt, so slice accordingly
                        generated_part = gen_ids[0][prompt_len_j:].cpu().tolist()
                        # decode and clean
                        pred_text = tokenizer.decode(generated_part, skip_special_tokens=True).strip()
                        pred_text = pred_text.replace(end_token, "").split("\n")[0].strip().lower()
                        gold_text = answer_texts[j].strip().lower()

                    # metrics
                    if pred_text == gold_text:
                        exact_match += 1
                    if Levenshtein.ratio(pred_text, gold_text) >= 0.8:
                        similar_match += 1
                    total += 1

                    # --- debug print for every sample (optional; can be noisy) ---
                    print(f"\nBatch {i} Sample {j}")
                    print(f"👉 Pred: {pred_text}")
                    print(f"✅ Label: {gold_text}")
                    print("-" * 40)

            # occasional summary print (every 10 batches)
            if i % 10 == 0 and total > 0:
                exact_acc = exact_match / total * 100
                sim_acc = similar_match / total * 100
                print(
                    f"Batch {i}, Loss: {loss.item() * grad_accum_steps:.4f}, "
                    f"Exact Acc: {exact_acc:.2f}%, Sim Acc: {sim_acc:.2f}%"
                )

        # epoch summary...
        avg_loss = total_loss / len(dataloader)
        exact_acc = exact_match / total * 100 if total else 0.0
        sim_acc = similar_match / total * 100 if total else 0.0

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Exact Match Accuracy: {exact_acc:.2f}%")
        print(f"Levenshtein ≥ 0.8 Accuracy: {sim_acc:.2f}%")

        # === SAVE: both .pth and HuggingFace style
        # 1) .pth (weights only)
        os.makedirs("solar10b_results", exist_ok=True)
        torch.save(model.state_dict(), f"/solar10b_results/tablevqa_epoch{epoch+1}.pth")
        print(f"Model checkpoint saved: solar10b_results/tablevqa_epoch{epoch+1}.pth")

        # 2) HuggingFace style save (model + tokenizer) — saves full weights and config
        hf_save_dir = os.path.join(save_dir, f"epoch{epoch+1}_hf")
        os.makedirs(hf_save_dir, exist_ok=True)
        try:
            model.model.save_pretrained(hf_save_dir)
            tokenizer.save_pretrained(hf_save_dir)
            print(f"HuggingFace checkpoint saved: {hf_save_dir}")
        except Exception as e:
            print(f"Warning: failed to save HF-style checkpoint: {e}")

# === Main ===
def main():
    json_path = "src/model/combined_wtq_html_otsl_sequential.json"
    model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
    end_token = "<END>"

    # Load tokenizer first, add end token if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if end_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([end_token])
        print(f"Added {end_token} to tokenizer vocab")

    # instantiate dataset (dataset uses tokenizer to encode)
    dataset = TableVQADataset(json_path, tokenizer, max_seq_len=4096, end_token=end_token)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # load model and resize embeddings if tokenizer grew
    model = TableVQAModel(model_name)
    # resize embeddings to match tokenizer length after adding tokens
    try:
        model.model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print("Warning: resize_token_embeddings failed:", e)

    # move model parameters to device is handled by device_map used earlier; ensure model on device if needed
    # model.model.to(device)

    # optimizer (bitsandbytes 8bit)
    global optimizer
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)

    # run training
    train(model, dataloader, tokenizer, optimizer, epochs=4, grad_accum_steps=8, save_dir="solar10b_results", end_token=end_token)

if __name__ == "__main__":
    main()

