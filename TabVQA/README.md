

# Eval

The `./Eval/` directory contains resources for evaluating baseline models on a Table visual question answering (TabVQA) task involving Hindi table images. It includes scripts, data, and results for assessing model performance.

## Directory Structure
- **`Scripts/`**: Contains evaluation scripts  (e.g., `blip2-opt-2.7b.py`) for different models.
- **`Data/`**: Stores input data, including:
  - `torque-qa.csv`: A CSV file with image names, questions, and ground truth answers.
  - `Hindi Table Images.zip/`: A directory containing Hindi table images referenced in the CSV.
- **`Results/`**: Stores output files in JSONL format (e.g., `blip2-opt-2.7b.jsonl`), containing individual predictions and final evaluation metrics.

## Evaluation Script Overview
The script in `./Eval/Scripts/` evaluates various baseline models for TabVQA on Hindi table images. Key functionalities include:
- **Model Configuration**:
  - Model: `Salesforce/blip2-opt-2.7b`
  - Device: Uses GPU 7 (set via `CUDA_VISIBLE_DEVICES="7"`) or CPU if CUDA is unavailable.
- **Input Data**:
  - CSV file: `../Data/torque-qa.csv` (contains image names, questions, and answers).
  - Image directory: `../Data/HindiTableImages/` (relative to the script, after unzipping directory).
- **Output**:
  - Results are saved in `./Eval/Results/` as JSONL files.
  - Each line in the JSONL file contains a prediction (image name, question, ground truth, predicted answer, and metrics).
  - Final metrics include:
    - Total samples processed.
    - Exact Match Accuracy (% of predictions matching ground truth exactly).
    - Levenshtein ≥ 0.8 Accuracy (% of predictions with Levenshtein similarity ≥ 0.8).
    - Relieved Accuracy (FinTabNet-style, based on normalized string comparison).
- **Process**:
  - Loads the BLIP-2 model and processor.
  - Reads questions and ground truth answers from the CSV.
  - Processes each image-question pair to generate answers.
  - Computes metrics (Exact Match, Levenshtein, Relieved Accuracy).
  - Saves results and metrics to the JSONL file.
  - Prints the first 10 predictions and final metrics for review.

