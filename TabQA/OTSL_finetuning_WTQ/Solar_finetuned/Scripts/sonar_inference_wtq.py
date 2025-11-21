import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
import Levenshtein
from tqdm import tqdm

# === Device ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Dataset Class ===
class TableVQADataset(Dataset):
    def __init__(self, json_path, tokenizer, max_seq_len=4096, end_token="<END>"):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.end_token = end_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        question = entry.get("question", "").strip()
        answer = entry.get("answer_text", "").strip()
        table_context = entry.get("otsl", entry.get("html", "")).strip()

        prompt_text = (
            "### Instruction:\n"
            "Given the following table, answer the question in one word or short phrase. Do not provide an explanation.\n\n"
            "### Table:\n"
            f"{table_context}\n\n"
            "### Question:\n"
            f"{question}\n\n"
            "### Answer:\n"
        )

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > self.max_seq_len:
            prompt_ids = prompt_ids[-self.max_seq_len:]  # truncate from left

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "answer_text": answer,
            "question": question,
            "otsl": table_context
        }

# === Collate Function ===
def collate_fn(batch):
    prompt_ids = [item["prompt_ids"] for item in batch]
    answers = [item["answer_text"] for item in batch]
    questions = [item["question"] for item in batch]
    otsls = [item["otsl"] for item in batch]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Pad sequences
    prompt_ids_padded = pad_sequence(prompt_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = (prompt_ids_padded != pad_token_id).long()

    return {
        "prompt_ids": prompt_ids_padded,
        "attention_mask": attention_mask,
        "answer_text": answers,
        "question": questions,
        "otsl": otsls
    }

# === Evaluation Function ===
def evaluate(model, dataloader, tokenizer, save_path, end_token="<END>", threshold=0.8):
    model.eval()
    end_token_id = tokenizer.convert_tokens_to_ids(end_token)

    exact, lev_sim, relieved, total = 0, 0, 0, 0
    all_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            prompt_ids = batch["prompt_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            answers = batch["answer_text"]
            questions = batch["question"]
            otsls = batch["otsl"]

            for i in range(prompt_ids.size(0)):
                input_ids = prompt_ids[i].unsqueeze(0)
                input_mask = attn_mask[i].unsqueeze(0)

                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=end_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                generated = gen_ids[0][len(input_ids[0]):].cpu().tolist()
                pred = tokenizer.decode(generated, skip_special_tokens=True).replace(end_token, "").strip().lower()
                gold = answers[i].strip().lower()

                # --- Metrics ---
                if pred == gold:
                    exact += 1
                if Levenshtein.ratio(pred, gold) >= threshold:
                    lev_sim += 1
                if pred == gold or gold in pred or Levenshtein.ratio(pred, gold) >= threshold:
                    relieved += 1

                total += 1

                all_results.append({
                    "question": questions[i],
                    "otsl": otsls[i],
                    "predicted_answer": pred,
                    "ground_truth": gold
                })

    # save predictions to JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return {
        "Exact Match": exact / total * 100,
        "Levenshtein Acc": lev_sim / total * 100,
        "Relieved Acc": relieved / total * 100
    }

# === Main ===
def main():
    test_json = "combined_wtq_html_otsl_test.json"   # <-- your test set
    model_name = "upstage/SOLAR-10.7B-Instruct-v1.0"
    checkpoint_path = "/solar10b_results/tablevqa_epoch4.pth"   # <-- your saved .pth
    predictions_save = "/solar10b_results/test_predictions_wtq.json"

    end_token = "<END>"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if end_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([end_token])

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Dataset + Dataloader
    dataset = TableVQADataset(test_json, tokenizer, end_token=end_token)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Evaluate + save predictions
    results = evaluate(model, dataloader, tokenizer, predictions_save, end_token=end_token)
    print("\n=== Final Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.2f}%")
    print(f"\nPredictions saved to: {predictions_save}")

if __name__ == "__main__":
    main()


