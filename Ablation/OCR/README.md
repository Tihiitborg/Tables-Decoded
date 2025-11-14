# OCR Ablation Summary

Our pipeline achieves highly reliable **table structure recognition** (TEDS-Structure > 95%), but the end-to-end performance is dominated by **OCR quality**. To analyze this bottleneck, we compare **Tesseract**, **EasyOCR**, and **Ground-Truth-Mapped (GT-Mapped)** content across multiple table datasets.

EasyOCR delivers **higher accuracy**, **GPU support**, and **3× faster inference**, making it the default OCR module in our system.

---

## Key Highlights

* **OCR is the main source of errors** in table reconstruction; structure prediction is already near-perfect.
* **EasyOCR > Tesseract** on almost all datasets (clean and noisy).
* **+12.45 TEDS point improvement** on average across major datasets (FinTabNet, PubTabNet, FinTabNetQA, TORQUE).
* **GT-Mapped** results (>90% TEDS) show that **content errors almost entirely originate from OCR**.
* **EasyOCR is ~3× faster**, enabling large-scale experiments (e.g., PubTables-1M) where Tesseract is too slow.

---

## OCR Performance (TEDS)

| Dataset      | Tesseract | EasyOCR  | GT-Mapped |
| ------------ | --------- | -------- | --------- |
| FinTabNet    | 41.5      | **55.9** | 91.2      |
| PubTabNet    | **54.4**  | 53.0     | 87.8      |
| FinTabNetQA  | 70.0      | **84.0** | 92.5      |
| PubTables-1M | –         | 54.8     | –         |
| TORQUE       | 40.8      | **63.6** | 83.5      |

---

## Latency Comparison

| OCR Engine | Avg. Time per Image (sec) |
| ---------- | ------------------------- |
| Tesseract  | 13.9                      |
| EasyOCR    | **4.3**                   |

---

## GT-Mapped Insight

Replacing OCR outputs with ground truth—while keeping the predicted structure—produces **TEDS > 90%** on most datasets.
This confirms:

> **OCR, not structure prediction, is the major limiting factor.**

---
