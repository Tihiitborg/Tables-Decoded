## TabVQA task on FinTabNetQA 

We evaluate the performance of our integrated pipeline, **DELTA + TARQA**, on the **FinTabNetQA** dataset. Our approach is compared against a wide range of open-source and closed-source Vision–Language Models. As shown below, **our OTSL-based pipeline achieves the highest score among all non-skyline methods**, demonstrating the effectiveness of structured table reconstruction in boosting downstream TabVQA performance.

### FinTabNetQA Relieved Accuracy Results


| **Type**        | **Model**                     | **FinTabNetQA** |
|-----------------|-------------------------------|:---------------:|
| **Open-source** | BLIP-2                        | 0.4             |
|                 | CogVLM-1k                     | 4.8             |
|                 | CogAgent-VQA                  | 22.8            |
|                 | SPHINX-v1-1k                  | 3.2             |
|                 | LLaVA-1.5                     | 0.8             |
|                 | QWEN-VL-Chat                  | 29.6            |
|                 | QWEN-VL                       | 34.0            |
| **Closed-source** | SPHINX-MoE-1k               | *36.0*          |
|                 | SPHINX-v2-1k                  | 31.2            |
|                 | SPHINX-MoE                    | 2.8             |
| **Ours**        | **DELTA + TARQA-HTML**     | 29.2            |
|                 | **DELTA + TARQA-OTSL**     | **45.2**        |
| **Skyline**     | GT-HTML + TARQA-HTML          | **51.2**        |
|                 | GT-OTSL + TARQA-OTSL          | **69.2**        |


---

## Interpretation

OTSL provides a more faithful, structured representation of tables compared to HTML.
This leads to:

* stronger grounding of cell relationships
* fewer structural ambiguities
* better alignment between question and table content

Thus, when plugged into **TARQA**, OTSL consistently yields superior performance across metrics and datasets.



























































# Eval

The `./Eval/` directory contains resources for evaluating baseline models on a Table visual question answering (TabVQA) task involving Hindi table images. It includes scripts, data, and results for assessing model performance.

## Directory Structure
- **`Scripts/`**: Contains evaluation scripts  (e.g., `blip2-opt-2.7b.py`) for different models.
- **`Data/`**: Stores input data, including:
  - `torque-qa.csv`: A CSV file with image names, questions, and ground truth answers.
- **`Results/`**: Stores output files in JSONL format (e.g., `blip2-opt-2.7b.jsonl`), containing individual predictions and final evaluation metrics.

## Evaluation Script Overview
The script in `./Eval/Scripts/` evaluates various baseline models for TabVQA on Hindi table images. Key functionalities include:
- **Model Configuration**:
  - Model: `Salesforce/blip2-opt-2.7b`
  - Device: Uses GPU 7 (set via `CUDA_VISIBLE_DEVICES="7"`) or CPU if CUDA is unavailable.
- **Input Data**:
  - CSV file: `../Data/torque-qa.csv` (contains image names, questions, and answers).
  - Image directory:
    - proposed TORQUE dataset directory path to be put here  
      `IMAGE_DIR = "../../HindiTableImages/"`

- **Output**:
  - Results are saved in `./Eval/Results/` as JSONL files.
  - Each line in the JSONL file contains a prediction (image name, question, ground truth, predicted answer, and metrics).
  - Final metrics include:
    - Total samples processed.
    - Exact Match Accuracy (% of predictions matching ground truth exactly).
    - Levenshtein ≥ 0.8 Accuracy (% of predictions with Levenshtein similarity ≥ 0.8).
    - Relieved Accuracy (FinTabNet-style, based on normalized string comparison).
- **Process**:
  - Loads the respective model and processor.
  - Reads questions and ground truth answers from the CSV.
  - Processes each image-question pair to generate answers.
  - Computes metrics (Exact Match, Levenshtein, Relieved Accuracy).
  - Saves results and metrics to the JSONL file.
  - Prints the first 10 predictions and final metrics for review.

