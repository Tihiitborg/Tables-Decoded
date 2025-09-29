# TARQA Project Directory Structure

This README provides an overview of the TARQA project directory, detailing the purpose of each folder and its contents.

---

## 📁 Root Directory: `TARQA`

This is the main project folder for Table Retrieval and Question Answering (TARQA) experiments.

### Subfolders:

---

### 1. `Baselines`

Contains pretrained or baseline models used for comparisons.

- **`Mistral_finetuned/`**
  - Fine-tuned Mistral model on TARQA dataset.
  - Typically includes model checkpoints and configuration files.
  - Search for `# CONFIGURATION` in all scripts od `Scripts` sub-dir to change the config
  - All checkpoints , models , output json , etc. saved in `Results` sub-dir

- **`Solar_finetuned/`**
  - Fine-tuned SOLAR model on TARQA dataset.
  - Includes model weights and associated configurations.

---

### 2. `Evaluations`

Contains all evaluation scripts, results, and datasets for running TARQA experiments.

#### Subfolders:

1. **`Data/`**
   - Stores evaluation datasets (HTML, OTSL, or plaintext formats) used for testing the models.

2. **`Tarqa_on_html/`**
   - **Purpose:** Evaluate TARQA models on HTML table inputs.
   - **Contents:**
     - `Results/` → Stores evaluation outputs, metrics, and predictions on HTML tables.
     - `Scripts/` → Contains Python or shell scripts for running evaluation.
     - `README.md` → Specific instructions for HTML-based evaluation.

3. **`Tarqa_on_otsl/`**
   - **Purpose:** Evaluate TARQA models on OTSL (serialized table) inputs.
   - **Contents:**
     - `Results/` → Stores evaluation outputs, metrics, and predictions on OTSL tables.
     - `Scripts/` → Evaluation scripts specific to OTSL input format.
     - `README.md` → Instructions and details for OTSL-based evaluation.

4. **`Tarqa_on_plaintext/`**
   - **Purpose:** Evaluate TARQA models on plain-text table inputs.
   - **Contents:**
     - `Results/` → Stores evaluation outputs, metrics, and predictions on plaintext tables.
     - `Scripts/` → Scripts for running plain-text evaluations.
     - `README.md` → Instructions and details for plain-text evaluation.

---

## 🔹 Notes

- Each evaluation folder (`Tarqa_on_html`, `Tarqa_on_otsl`, `Tarqa_on_plaintext`) follows the **same structure**: `Results`, `Scripts`, and `README.md`.
- `Baselines` contains pretrained/fine-tuned models for reference and comparison.
- `Evaluations/Data` contains the datasets necessary for running any evaluation script.

---

## 📌 Summary

- `TARQA/`
  - `Baselines/`
    - `Mistral_finetuned/` → Fine-tuned Mistral model
    - `Solar_finetuned/` → Fine-tuned SOLAR model
  - `Evaluations/`
    - `Data/` → Evaluation datasets
    - `Tarqa_on_html/` → Scripts + results + README for HTML tables
    - `Tarqa_on_otsl/` → Scripts + results + README for OTSL tables
    - `Tarqa_on_plaintext/` → Scripts + results + README for plaintext tables

---

This structure ensures clarity and separation of **training baselines** and **evaluation experiments** for HTML, OTSL, and plaintext table formats.
