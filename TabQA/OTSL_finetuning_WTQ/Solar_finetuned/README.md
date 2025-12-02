# TableVQA Training with SOLAR-10B

This project provides a training pipeline for **Table-based Question
Answering (TableVQA)** using the
[SOLAR-10.7B-Instruct](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)
model.\
It fine-tunes the model to answer concise questions given tabular data
(HTML or OTSL format).

------------------------------------------------------------------------

## 📌 Key Features

-   **Custom Dataset** loader for JSON-based TableVQA data.
-   **Instruction-style prompting** for table-question-answer pairs.
-   **Long sequence support** (up to 4096 tokens).
-   **Training loop** with:
    -   Gradient accumulation\
    -   Gradient clipping\
    -   NaN loss checks\
    -   Metrics: Exact Match & Levenshtein similarity
-   **Checkpoint saving** in both:
    -   PyTorch `.pth` weights\
    -   HuggingFace-compatible format (model + tokenizer)

------------------------------------------------------------------------

## 📂 Dataset Format

Your dataset JSON file should follow this structure:

``` json
[
  {
    "question": "Which country has the highest GDP?",
    "answer_text": "USA",
    "html": "<table>...</table>",
    "otsl": "Optional serialized table format"
  }
]
```

------------------------------------------------------------------------

**Default dataset path:**

    ../Data/combined_wtq_html_otsl_sequential.json

------------------------------------------------------------------------

## ⚙️ Setup

### Install Dependencies

    pip install torch torchvision torchaudio
    pip install transformers bitsandbytes datasets tqdm python-Levenshtein

------------------------------------------------------------------------

## 🚀 Training

Run the main script:

    python sonar10b_finetune_wtq.py

This will:

-   Load the tokenizer & add `<END>` as a stopping token\
-   Initialize dataset & dataloader\
-   Load the SOLAR-10B model with gradient checkpointing\
-   Train for 4 epochs

------------------------------------------------------------------------

## 📊 Evaluation Metrics

During training, two metrics are tracked:

-   **Exact Match Accuracy** → Prediction matches gold answer exactly.\
-   **Levenshtein ≥ 0.8 Accuracy** → Prediction has ≥80% similarity to
    gold answer.

------------------------------------------------------------------------

## 💾 Saving Checkpoints

At the end of each epoch, the model is saved in two formats:

### **PyTorch weights**

    solar10b_results/tablevqa_epoch.pth #See in releases for pth files

### **HuggingFace model + tokenizer**

    solar10b_results/epoch_hf/

------------------------------------------------------------------------

## 🖥️ Hardware Requirements

-   GPU ≥ 40GB memory recommended (A100, H100, or similar).\
-   Training uses **bfloat16** precision with **gradient checkpointing**
    for memory efficiency.\
-   Optimizer: **8-bit AdamW** (bitsandbytes).
