import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
import Levenshtein

# === Device Setup ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Dataset Class using HTML format ===
class TableVQADataset(Dataset):
    def __init__(self, json_path, tokenizer, max_seq_len=4096):
        print(f"Loading dataset from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples.")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        question = entry["question"]
        answer = entry["answer_text"]
        table_context = entry["html"]  # Use HTML instead of OTSL

        input_text = f"""### Instruction:
        Given the following HTML table, answer the question in one word or short phrase. Do not provide an explanation.

        ### Table:
        {table_context}

        ### Question:
        {question}

        ### Answer:"""

        full_text = input_text + " " + answer

        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors='pt'
        )

        input_ids = encoded.input_ids.squeeze(0)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels
        }

# === Model Wrapper ===
class TableVQAModel(nn.Module):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__()
        print(f"Loading model: {model_name}")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            use_cache=False
        ).to(device)
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.loss, outputs.logits

# === Training Function ===
def train(model, dataloader, tokenizer, epochs=6):
    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0
        exact_match = 0
        similar_match = 0
        total = 0

        for i, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids=input_ids, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            for j in range(input_ids.size(0)):
                output_ids = torch.argmax(logits[j], dim=-1)
                pred = tokenizer.decode(output_ids, skip_special_tokens=True)
                label = tokenizer.decode(labels[j], skip_special_tokens=True)

                pred = pred.strip().lower().split("### answer:")[-1].strip()
                label = label.strip().lower().split("### answer:")[-1].strip()

                if pred == label:
                    exact_match += 1
                if Levenshtein.ratio(pred, label) >= 0.8:
                    similar_match += 1
                total += 1

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        exact_acc = exact_match / total * 100
        sim_acc = similar_match / total * 100

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Exact Match Accuracy: {exact_acc:.2f}%")
        print(f"Levenshtein ≥ 0.8 Accuracy: {sim_acc:.2f}%")

        os.makedirs("llama8bhtmlresults", exist_ok=True)
        torch.save(model.state_dict(), f"/llama8bhtmlresults/tablevqa_epoch{epoch+1}.pth")
        print(f"Model checkpoint saved: llama8bhtmlresults/tablevqa_epoch{epoch+1}.pth")

# === Main Function ===
def main():
    json_path = "src/model/combined_wtq_html_otsl_sequential.json"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = TableVQADataset(json_path, tokenizer, max_seq_len=4096)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = TableVQAModel(model_name)
    global optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train(model, dataloader, tokenizer, epochs=4)

    # === Sample Inference on 10 Examples ===
    model.eval()
    print("\nSample Predictions after Training:")
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            for i in range(input_ids.size(0)):
                input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                label = tokenizer.decode(labels[i], skip_special_tokens=True)
                pred = tokenizer.decode(outputs[i], skip_special_tokens=True)

                question = input_text.split("### Question:")[-1].split("### Answer:")[0].strip()
                gt_answer = label.strip().lower().split("### answer:")[-1].strip()
                pred_answer = pred.strip().lower().split("### answer:")[-1].strip()

                print(f"\nExample {count+1}")
                print(f"Question        : {question}")
                print(f"Ground Truth    : {gt_answer}")
                print(f"Predicted Answer: {pred_answer}")

                count += 1
                if count >= 10:
                    break
            if count >= 10:
                break

if __name__ == "__main__":
    main()



