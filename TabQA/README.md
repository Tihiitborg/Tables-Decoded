## ANLS Comparison on WTQ Test Set (HTML vs OTSL)

The following table summarizes ANLS scores on the WTQ test set across multiple baselines, including OTSL- and HTML-based variants. 

### **ANLS Scores on WTQ Test Set**

| **Baseline**     | **HTML** | **OTSL** |
|------------------|:--------:|:--------:|
| UDOP             | **47.2** |    –     |
| Pix2Struct       |  39.8    |    –     |
| DocOwl           |  26.9    |    –     |
| Kosmos           |  32.4    |    –     |
| Donut            |  18.8    |    –     |
| TAPAS            | *46.4*   |    –     |
| Mistral          |    –     | *29.9*   |
| SOLAR            |    –     |  12.3    |
| **TabQA (Ours)** |  42.3    | **56.5** |
---
This evaluation demonstrates:

* **OTSL-based inputs significantly outperform HTML-based baselines**
  → OTSL TabQA achieves **56.5 ANLS**, beating the strongest HTML baseline by **+9.3** points.

* All OTSL finetuned models are trained:

  * with **unimodal text-only data**,
  * using the **same splits**,
  * for **4 epochs**.

* This folder (`Tarqa_on_plaintext/Scripts`) specifically contains:

  * inference pipelines
  * evaluation scripts
  * score aggregation logic
    for plaintext WTQ testing.

---

### Repository Flow — TabQA (WTQ Evaluation)


```
TabQA/
│
├── OTSL_finetuning_WTQ/
│   ├── Mistral_finetuned/
│   │   ├── Data/        # Processed WTQ training data 
│   │   ├── Results/     # Fine-tuned model outputs & metrics
│   │   └── Scripts/     # Training scripts for OTSL finetuning
│   └── Solar_finetuned/ 
│       ├── Data/        # Processed WTQ training data 
│       ├── Results/     # Fine-tuned model outputs & metrics
│       └── Scripts/     # Training scripts for OTSL finetuning
│
└── Results_test_WTQ/
    ├── Data/                 # WTQ test set (HTML, OTSL, Plaintext)
    ├── Tarqa_on_html/
    │   ├── Scripts/          # Eval scripts for HTML tables
    │   └── Results/          # Evaluation outputs, predictions & metrics
    ├── Tarqa_on_otsl/
    │   ├── Scripts/          # Eval scripts for OTSL tables
    │   └── Results/          # Evaluation outputs, predictions & metrics
    └── Tarqa_on_plaintext/
        ├── Scripts/          # Eval scripts for Plaintext tables
        └── Results/          # Evaluation outputs, predictions & metrics

```

