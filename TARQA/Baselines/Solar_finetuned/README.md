# Supervised Finetuning (SFT) on TaRQA with Plaintext Format

## What is SFT?
Supervised Fine-tuning (SFT) means you take a pretrained language model (e.g., **LLaMA 3-8B-Instruct**) and fine-tune it on your task-specific labeled dataset — here, **Plaintext + questions + answers**.

### Goal
- Adapt the pretrained model so it learns **structured table reasoning** from your data.  
- **Training objective:** predict the correct answer text given a Plaintext + question.

---

## 1) Input Format
We formulated the task as **instruction-based generation**, where each training example was prepended with a structured prompt.

The prompt includes:
- An instruction  
- A table in serialized format (from the **Plaintext** field)  
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

## 2) Data Preprocessing
- **Dataset Format:** JSON  
- **Key Fields Used:**
  - `question`: Natural language question  
  - `answer_text`: Ground truth answer  
  - `Plain_text`: Serialized table format  

- **Tokenization:**
  - HuggingFace `AutoTokenizer` (corresponding to LLaMA-3)  
  - `padding="max_length"` and `truncation=True` enabled  
  - **Max sequence length:** 4096 tokens  

---

## 3) Model Setup
We wrap `LlamaForCausalLM` inside a custom **TableVQAModel**.

**Key points:**
- Uses **bfloat16** for efficiency (best on A100/H100 GPUs).  
- `gradient_checkpointing_enable()` reduces memory usage by recomputing activations during backpropagation.  

---

## 4) Training Pipeline

### Steps
1. **Forward pass**
   - Input = tokenized Plaintext + Question + Answer  
   - Labels = same sequence (causal LM objective)  
   - Model computes **Cross-Entropy loss**  

2. **Backward pass & Optimization**
   - `loss.backward()`  
   - `optimizer.step()` → Optimizer = **AdamW**, LR = `2e-5`  

3. **Metrics**
   - **Exact Match (EM):** `pred == label`  
   - **Levenshtein ≥ 0.8:** similarity-based accuracy (robust to typos)  
   - **Relieved Accuracy:** relaxed EM (ignores minor formatting issues)  

4. **Checkpointing**
   - Saves weights per epoch → `tablevqa_epochX.pth`

---

## Training Configuration

| Category      | Parameter              | Our Setting                             | Description |
|---------------|------------------------|------------------------------------------|-------------|
| **Model**     | `model_name`           | meta-llama/Meta-Llama-3-8B-Instruct      | Pretrained base model |
|               | `torch_dtype`          | bfloat16                                 | Mixed precision to save memory |
|               | `gradient_checkpointing` | Enabled                                | Saves VRAM by recomputing activations |
| **Tokenizer** | `max_seq_len`          | 4096                                     | Max tokens per input |
|               | `pad_token`            | eos_token                                | Ensures consistent padding |
| **DataLoader**| `batch_size`           | 1                                        | Samples per forward pass |
|               | `shuffle`              | True                                     | Randomizes samples each epoch |
| **Optimization** | `optimizer`         | AdamW                                    | Optimizer for transformers |
|               | `learning_rate`        | 2e-5                                     | Step size for updates |
|               | `weight_decay`         | Default (0.01)                           | Regularization |
| **Training Loop** | `epochs`           | 4 (planned 6 in final run)               | Number of dataset passes |
|               | `loss`                 | CrossEntropy (causal LM)                 | Predict next token |
|               | `metrics`              | EM, Levenshtein ≥ 0.8, Relieved accuracy | Evaluation |
| **Checkpointing** | `save_path`        | tablevqa_epochX.pth                      | Saves model weights |

---

## 5) Inference

After training, inference was performed using HuggingFace `generate()`.

**Settings:**
- `beam_size = 1` (**greedy decoding**)  
- `max_new_tokens = 100`  
- `do_sample = False` (deterministic output)  

**Example Inference Code:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("tablevqa_epoch4.pth")

# Example input
prompt = """### Instruction:
Given the following table, answer the question in one word or short phrase. Do not provide an explanation.

### Table:
{Plain_text}

### Question:
{question}

### Answer:"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
    num_beams=1
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
