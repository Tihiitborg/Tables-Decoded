# TORQUE: Hindi Table Reconstruction & TabVQA Benchmark

TORQUE is a Hindi benchmark designed for evaluating both **table reconstruction** and **Tabular Visual Question Answering (TabVQA)** tasks. It is aimed at facilitating research on table understanding and QA in Hindi, including scanned and digital-born documents.

## Dataset Overview

- **Total tables:** 210  
  - **Scanned:** 109  
  - **Digital-born & cropped:** 101  
- **Source:**  
  - Government circulars   
  - Spiritual books from MUSTARD dataset 
- **Table complexity:**  
  - Simple: 149  
  - Complex: 61  

- **QA pairs:** 422 manually verified question-answer pairs  
  - Generated using **GPT-oss-20B** 
- **Table reconstruction:**  
  - Outputs from **ChatGPT-4o**  were post-corrected manually to retrieve exact HTML sequences  

## Benchmark Use

- Evaluates open-source models and our proposed **pipeline**  
- Demonstrates that our approach is **language-agnostic**  
- Outperforms conventional decoupled pipelines and large VLMs in **zero-shot Hindi settings**  

## Directory Structure

- `TORQUE/`
  - `Evaluation/` – Contains Htmlwork,Otslwork and Torque dataset
  - `sample_images/` – Contains sample table images from the dataset  
  - `qa_pairs.json` – JSON file containing a few QA pairs  
  - `README.md` – This README file  


