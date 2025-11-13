***Tables Decoded: DELTA for Structure, TARQA for Understanding***
---

```
# 📘 Tables Decoded: DELTA for Structure, TARQA for Understanding

> **Accepted at WACV 2025**  
> Official implementation and resources for  
> **“Tables Decoded: DELTA for Structure, TARQA for Understanding.”**

---

## 🧩 Abstract

Table understanding is a core task in document intelligence, encompassing two key subtasks: **table reconstruction** and **table visual question answering (TabVQA)**.  
While recent approaches predominantly rely on **vision-language models (VLMs)** operating on table images, we propose a more scalable and effective alternative based on **structured textual representations**.  

These representations are easier to process, align naturally with **large language models (LLMs)**, and remove the need for language-specific visual encoders — making them ideal for **multilingual documents**.  
We present **DELTA**, a *decoupled table reconstruction framework* that separates structure recognition from OCR to extract both layout and content accurately. DELTA outputs tables in **Optimized Table Structure Language (OTSL)**, a compact and unified format that encodes cell arrangements and textual content. DELTA achieves high-fidelity table-to-text conversion, outperforming prior methods on structure metrics with superior **TEDS-Structure** scores across **FinTabNet**, **PubTabNet**, and **PubTables**.  
Built on DELTA, we introduce **TARQA (Table structure-Aware Representation for Question Answering)** — an LLM fine-tuned on OTSL-formatted tables for accurate and structure-aware **TabVQA**. TARQA outperforms baselines fine-tuned on HTML representations by **14.2%**, and improves answer accuracy on **WTQ** by **6.8%** and **FinTabNetQA** by **9.2%**.  
---

## 📁 Repository Structure

```

├── Ablation/        # Ablation experiments and analysis
├── DELTA/           # DELTA: Table reconstruction module
├── TARQA/           # TARQA: Structure-aware table QA module
├── TORQUE/          # TORQUE dataset sample images and scripts
├── TabVQA/          # Tabular question answering datasets and configs

````

---

## ⚙️ Framework Overview

| Component | Description |
|------------|-------------|
| **DELTA** | Decoupled table reconstruction framework separating OCR and layout recognition. Outputs tables in **OTSL** format. |
| **TARQA** | LLM-based reasoning model fine-tuned on OTSL tables for **TabVQA**. Enables structure-aware and multilingual question answering. |
| **Ablation** | Comprehensive studies on OTSL impact, OCR decoupling, and multilingual adaptation. |

---

## 🧠 Key Highlights

- **Text-based table representation** → more scalable than VLM-based approaches.  
- **Multilingual compatibility** → no dependence on visual encoders.  
- **Superior structure reconstruction** → highest TEDS-Structure scores on FinTabNet, PubTabNet, and PubTables.  
- **Structure-aware QA** → TARQA improves TabVQA accuracy by **+14.2%** over HTML baselines.  

---

## 🧪 Datasets

| Dataset | Description | Source |
|----------|--------------|---------|
| **TORQUE** | Temporal and causal reasoning dataset for structured QA | [Hugging Face: `(https://huggingface.co/datasets/jahanvirajput/TORQUE)`] |
| **FinTabNet** | Financial tables for structure extraction | [IBM Developer](https://developer.ibm.com/exchanges/data/all/fintabnet/) |
| **PubTabNet** | Scientific tables for reconstruction | [GitHub: PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet) |
| **WTQ** | WikiTableQuestions benchmark for logical QA | [GitHub: WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions) |
| **FinTabNetQA** | QA dataset over financial domain tables | *https://huggingface.co/datasets/terryoo/TableVQA-Bench/viewer/default/fintabnetqa* |

---

## 🚀 Getting Started

### 1️⃣ Installation
```bash
git clone https://github.com/JahanviRajputTIH/Tables-Understanding.git
cd Tables-Understanding
pip install -r requirements.txt
````

---

## 📈 Results

---

## 🔬 Ablation Studies

Located under `Ablation/`, covering:

* OCR–layout decoupling effect
* Impact of OTSL vs HTML representations
* Multilingual fine-tuning analysis
* Structure fidelity vs reasoning accuracy trade-off

---

## 🧾 Citation

If you use this work, please cite our **WACV 2025** paper:

```bibtex
@inproceedings{tablesdecoded2025,
  title={Tables Decoded: DELTA for Structure, TARQA for Understanding},
  author={},
  booktitle={},
  year={2025}
}
```

---


## 🏁 License

This project is licensed under the **MIT License**.
See [LICENSE](LICENSE) for details.

```

---

Would you like me to add a **“Model Architecture” diagram section** (DELTA → OTSL → TARQA flow) in Markdown using Mermaid or image reference? It would make it more visually appealing for GitHub and citations.
```
