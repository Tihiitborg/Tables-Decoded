# Ablation Study: Effectiveness of OTSL for TabVQA

We evaluate various table-to-text representation formats—**OTSL**, **HTML**, and **Plain Text**—to determine their impact on TabVQA performance. Each format is tested using:

* **Off-the-shelf LLM**
* **Finetuned LLM (TabQA-<format>)**

Across all settings, **OTSL achieves the best results**, outperforming other formats by a large margin.
Finetuning with OTSL leads to a **+14 p.p. improvement** in both **ANLS** and **Exact Match (EM)** over HTML representations.

## Ablation Results (Representation Formats)

| Inferred On    | LLM Model      | ANLS     | EM       |
| -------------- | -------------- | -------- | -------- |
| **OTSL**       | Off-the-Shelf  | 25.4     | 16.9     |
|                | **TabQA-OTSL** | **56.5** | **54.0** |
| **HTML**       | Off-the-Shelf  | 11.0     | 10.3     |
|                | TabQA-HTML     | 42.3     | 39.9     |
| **Plain Text** | Off-the-Shelf  | 29.8     | 27.8     |
|                | TabQA-Plain    | 52.8     | 50.2     |

---
