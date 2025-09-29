# 📑 TableVQA Fine-Tuning with Meta-LLaMA-3-8B-Instruct

We fine-tuned a large language model (**Meta-LLaMA-3-8B-Instruct**) for the **Table Visual Question Answering (TableVQA)** task using a custom instruction-tuned setup.  
The training objective is to generate **short, accurate answers** conditioned on a table and a natural language question.

---

## 🔧 Model Architecture
- **Base Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
- **Type:** Causal Language Model (Auto-regressive)
- **Parameters:** ~8 Billion  
- **Modifications:**
  - `gradient_checkpointing_enable()` → reduced memory footprint  
  - `torch_dtype=torch.bfloat16` → mixed-precision training  
  - `use_cache=False` → disabled for gradient checkpointing  

---

## 📥 Input Format
Training follows an **instruction-based generation** setup.

**Prompt Template:**
Instruction:

Given the following table, answer the question in one word or short phrase. Do not provide an explanation.

Table:

{table_text}

Question:

{question}

Answer: {answer}


---

## 🗂️ Data Preprocessing
- **Dataset Format:** JSON  
- **Key Fields:**
  - `question` → Natural language question  
  - `answer_text` → Ground truth answer  
  - `otsl` → Serialized table format  
- **Tokenization:**
  - HuggingFace `AutoTokenizer` (LLaMA-3)  
  - `padding="max_length"`, `truncation=True`  
  - Max sequence length: **4096 tokens**

---

## ⚙️ Training Configuration

| Parameter               | Value                           |
|--------------------------|---------------------------------|
| **Model**               | Meta-LLaMA-3-8B-Instruct        |
| **Optimizer**           | AdamW                          |
| **Learning Rate**       | 2e-5                           |
| **Batch Size**          | 1 (per GPU)                    |
| **Epochs**              | 4                              |
| **Sequence Length**     | 4096 tokens                    |
| **Tokenizer Pad Token** | `eos_token`                    |
| **Gradient Checkpointing** | Enabled                     |
| **Precision**           | bfloat16                       |
| **Device**              | NVIDIA A100 80GB (CUDA)        |
| **Framework**           | PyTorch + HuggingFace          |

---

## 📊 Evaluation Metrics
- **Exact Match Accuracy (EM):** strict string match (case + whitespace normalized)  
- **Levenshtein Similarity ≥ 0.8:** edit distance–based soft match  
- **Relieved Accuracy:** relaxed EM, ignores minor formatting differences  

> Metrics computed **on-the-fly** during training after each batch.

---

## 💾 Checkpointing
- Checkpoints saved **after each epoch**  
- Naming format: `tablevqa_epochX.pth`

---

## 🚀 Inference
Inference was performed using HuggingFace `model.generate()`:

- **Beam size:** 1 (greedy decoding)  
- **max_new_tokens:** 100  
- **Sampling:** disabled (`do_sample=False`)  

Example:
```python
outputs = model.generate(
    inputs,
    max_new_tokens=100,
    num_beams=1,
    do_sample=False
)
