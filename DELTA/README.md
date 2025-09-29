# Eval Sub-Dir

The `./Eval/` directory contains resources for evaluating various baseline models for Hindi question answering (QA) tasks, focusing on extractive QA from HTML and OTSL tables. It includes scripts, input data, and output results for assessing model performance.

## Directory Structure
- **`Scripts/`**: Contains evaluation scripts for different models.
- **`Data/`**: Stores input data, including:
  - `torque_hindi_qa.json`: A JSON file containing questions, HTML table content, and ground truth answers.
- **`Results/`**: Stores output files, including (e.g.) :
  - `bert-base-multilingual-cased.jsonl`: JSONL file with individual predictions for the BERT model.
  - `bert-base-multilingual-cased.json`: JSON file with summary statistics for the BERT model.
  - Other JSONL/JSON files for different models (e.g., Qwen-based results).

## Evaluation Script Overview
The script in `./Eval/Scripts/` evaluates various multilingual models for extractive QA on Hindi questions and HTML table content. Key functionalities include:
- **Model Configuration**:
  - Model: `bert-base-multilingual-cased` (default) or other models like `Qwen-2.5-14B-Hindi` (configurable via `model_path="../Qwen-2.5-14B-Hindi"`).
  - Device: Uses CUDA if available, otherwise CPU.
- **Input Data**:
  - JSON file: `../torque_hindi_qa.json` (contains questions, HTML content, and ground truth answers).
- **Output**:
  - Predictions saved to `*.jsonl` in `./Eval/Results/`, with each line containing a prediction (index, question, ground truth, predicted answer, confidence, Levenshtein score, and match indicators).
  - Summary statistics saved to `*.json`.
  - Metrics include:
    - Total samples processed.
    - Exact Match Accuracy (% of predictions matching ground truth exactly).
    - Levenshtein ≥ 0.8 Accuracy (% of predictions with Levenshtein similarity ≥ 0.8).
    - Relieved Accuracy (FinTabNet-style, based on normalized string comparison).
- **Process**:
  - Initializes the `HindiQAInference` class, which downloads and loads the model if not already present.
  - Parses HTML tables into structured text for context.
  - Generates answers using a QA pipeline with multiple prompt strategies (direct, Hindi-instructed, and focused prompts).
  - Cleans answers to remove noise and improve relevance (e.g., removing Hindi particles, extracting numbers/percentages).
  - Computes metrics and saves results.
  - Prints the first 100 samples' progress and final metrics.
