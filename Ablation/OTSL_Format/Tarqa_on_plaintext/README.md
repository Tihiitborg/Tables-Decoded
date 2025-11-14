# 📑 Supervised Fine-Tuning (SFT) on TARQA with Plaintext Tables

This project applies **Supervised Fine-Tuning (SFT)** to adapt a pretrained LLaMA model (**Meta-LLaMA-3-8B-Instruct**) for the **Table Visual Question Answering (TableVQA)** task on the **TARQA dataset** using **Plaintext tables**.  
The goal is to train the model to generate **short, accurate answers** given **Plaintext tables** and **natural language questions**.

---

## 🎯 Goal
- Adapt the pretrained model to perform **structured reasoning over Plaintext tables**.  
- **Training objective:** predict the correct answer text given a serialized Plaintext table + natural language question.  

---

## 📝 1) Input Format
We use an **instruction-based generation** setup where each example includes:
- An instruction  
- A table in **serialized Plaintext** format  
- A natural language question  
- The answer (as the label)  

**Prompt Template:**

Instruction:

Given the following table, answer the question in one word or short phrase. Do not provide an explanation.

Table:

{Plain_text}

Question:

{question}

Answer: {answer}


---

## 🗂️ 2) Data Preprocessing
- **Dataset Format:** JSON  
- **Key Fields Used:**
  - `question`: natural language question  
  - `answer_text`: ground truth answer  
  - `Plain_text`: serialized table format  
- **Tokenization:**
  - HuggingFace `AutoTokenizer` (LLaMA-3)  
  - `padding="max_length"`, `truncation=True`  
  - Max sequence length = **4096 tokens**  

---

## ⚙️ 3) Model Setup
- Wrapped `LlamaForCausalLM` inside a **TableVQAModel**.  
- **Efficiency features:**
  - `torch_dtype=bfloat16` → faster & memory-efficient on A100/H100 GPUs  
  - `gradient_checkpointing_enable()` → reduces memory footprint  

---

## 🔄 4) Training Pipeline
### Steps:
1. **Forward Pass**  
   - Input = tokenized Plaintext + question + answer  
   - Labels = same sequence (causal LM objective)  
   - Model computes **cross-entropy loss**  

2. **Backward & Optimization**  
   - `loss.backward()`  
   - Optimizer = **AdamW (LR = 2e-5)**  

3. **Metrics**  
   - **Exact Match (EM):** prediction == label  
   - **Levenshtein ≥ 0.8:** similarity-based accuracy (robust to typos)  
   - **Relieved Accuracy:** ignores minor formatting differences  

4. **Checkpointing**  
   - Saves weights each epoch as:  
     ```
     tablevqa_epochX.pth
     ```

---

## ⚙️ Training Configuration

| Category       | Parameter             | Our Setting                          | Description |
|----------------|-----------------------|--------------------------------------|-------------|
| **Model**      | `model_name`         | meta-llama/Meta-Llama-3-8B-Instruct  | Base model |
|                | `torch_dtype`        | bfloat16                             | Mixed precision |
|                | `gradient_checkpointing` | Enabled                          | Saves VRAM |
| **Tokenizer**  | `max_seq_len`        | 4096                                 | Max tokens |
|                | `pad_token`          | eos_token                            | Padding |
| **DataLoader** | `batch_size`         | 1                                    | Samples per pass |
|                | `shuffle`            | True                                 | Random order |
| **Optimization** | `optimizer`        | AdamW                                | Transformer optimizer |
|                | `learning_rate`      | 2e-5                                 | LR |
|                | `weight_decay`       | Default (0.01)                       | Regularization |
| **Training**   | `epochs`             | 4 (planned 6 future)                 | Dataset passes |
|                | `loss`               | CrossEntropy (causal LM)             | Next-token prediction |
|                | `metrics`            | EM, Levenshtein ≥ 0.8, Relieved Acc. | Evaluation |
| **Checkpointing** | `save_path`       | tablevqa_epochX.pth                  | Saved weights |

---

## 🤖 5) Inference
Inference is done with HuggingFace `model.generate()`:

- **Beam size:** 1 (greedy decoding)  
- **max_new_tokens:** 100  
- **do_sample:** False (no sampling)  

**Example:**
```python
outputs = model.generate(
    inputs,
    max_new_tokens=100,
    num_beams=1,
    do_sample=False
)
