#!/usr/bin/env python3
"""
Hindi HTML Table Question Answering Inference Script for Entire Dataset
Using Qwen-2.5-14B-Hindi Model for Direct Answer Extraction
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
from tqdm import tqdm
import Levenshtein
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class HindiHTMLQAInference:
    def __init__(self, model_path: str = "../Qwen-2.5-14B-Hindi", use_gpu: bool = True):
        """
        Initialize the Hindi HTML QA Inference system
        
        Args:
            model_path: Local path where model will be stored
            use_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Model and tokenizer placeholders
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        logger.info(f"Initializing on device: {self.device}")
        
    def download_and_load_model(self):
        """Download model locally and load for inference"""
        try:
            model_name = "large-traversaal/Qwen-2.5-14B-Hindi"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model...")
            
            # Model loading configuration for optimal performance
            model_kwargs = {
                "cache_dir": self.model_path,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.use_gpu else torch.float32,
                "device_map": "auto" if self.use_gpu else None,
                "low_cpu_mem_usage": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Create text generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if self.use_gpu else None,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def parse_html_table(self, html_content: str) -> str:
        """
        Parse HTML table and convert to structured text format
        
        Args:
            html_content: Raw HTML table content
            
        Returns:
            Formatted table text for model input
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return html_content
            
            # Extract table data
            rows = table.find_all('tr')
            table_data = []
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:  # Skip empty rows
                    table_data.append(row_data)
            
            # Format as structured text
            formatted_text = "तालिका की जानकारी:\n"
            for i, row in enumerate(table_data):
                if len(row) >= 2:
                    formatted_text += f"{row[0]}: {row[1]}\n"
                else:
                    formatted_text += f"पंक्ति {i+1}: {' | '.join(row)}\n"
            
            return formatted_text.strip()
            
        except Exception as e:
            logger.warning(f"Error parsing HTML: {str(e)}")
            return html_content
    
    def create_prompt(self, question: str, html_content: str) -> str:
        """
        Create optimized prompt for the model using recommended format
        
        Args:
            question: Question in Hindi
            html_content: HTML table content
            
        Returns:
            Formatted prompt string
        """
        # Parse HTML to structured format
        table_text = self.parse_html_table(html_content)
        
        # Create prompt using recommended format for direct responses
        prompt = f"""संदर्भ तालिका:
{table_text}

संक्षेप में, केवल प्रत्यक्ष उत्तर दें। {question} ### DIRECT RESPONSE ###

उत्तर:"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using the loaded model
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Generated answer text
        """
        try:
            # Generation parameters optimized for direct answers
            generation_config = {
                "max_new_tokens": 50,  # Short, direct answers
                "temperature": 0.1,    # Low temperature for consistency
                "top_p": 0.9,         # Nucleus sampling
                "top_k": 40,          # Top-k sampling
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False  # Only return generated part
            }
            
            # Generate response
            outputs = self.pipe(prompt, **generation_config)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
                # Clean up the response
                answer = self.clean_response(generated_text)
                return answer
            else:
                return "उत्तर नहीं मिला"
                
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "त्रुटि हुई"
    
    def clean_response(self, response: str) -> str:
        """
        Clean and extract the direct answer from model response
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned direct answer
        """
        # Remove common prefixes and suffixes
        response = response.strip()
        
        # Remove any remaining prompt text
        if "उत्तर:" in response:
            response = response.split("उत्तर:")[-1].strip()
        
        # Remove extra whitespace and newlines
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove any trailing punctuation except essential ones
        response = re.sub(r'[।\s]*$', '', response)
        
        # Extract only the first sentence/phrase for direct answer
        sentences = re.split(r'[।\n]', response)
        if sentences:
            answer = sentences[0].strip()
            return answer if answer else response
        
        return response
    
    def infer(self, question: str, html_content: str) -> str:
        """
        Main inference method
        
        Args:
            question: Question in Hindi
            html_content: HTML table content
            
        Returns:
            Generated answer text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call download_and_load_model() first.")
        
        try:
            # Create prompt
            prompt = self.create_prompt(question, html_content)
            
            # Generate answer
            answer = self.generate_answer(prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return "त्रुटि हुई"

def fintabnet_normalize(text):
    """
    Normalization function for FinTabNet-style relieved accuracy
    """
    def _normalize(s):
        if not isinstance(s, str):
            s = str(s)
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[,\.]", "", s)  # remove commas/periods
        s = s.replace(" ", "")
        return s

    gt = _normalize(text)
    return gt, [gt]

def main():
    """Main function to run the inference on the entire dataset"""
    
    # Initialize inference pipeline
    print("Initializing Qwen-2.5-14B-Hindi Inference Pipeline...")
    qa_inference = HindiHTMLQAInference(model_path="../Qwen-2.5-14B-Hindi")
    
    # Download and load model
    print("Downloading and loading model...")
    qa_inference.download_and_load_model()
    
    # Load the dataset
    data_path = "../Data/torque_hindi_qa.json"
    print(f"Loading dataset from {data_path}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize metrics
    exact_match = 0
    similar_match = 0
    relieved_match = 0
    total = 0
    predictions = []
    
    # Process each sample
    for entry in tqdm(dataset, desc="Processing samples"):
        question = entry["question"]
        ground_truth = entry["answer"].strip().lower()
        html_content = entry["html"]
        
        # Get prediction
        answer = qa_inference.infer(question, html_content)
        predicted_answer = answer.strip().lower()
        
        # Calculate metrics
        lev_score = Levenshtein.ratio(predicted_answer, ground_truth)
        is_exact = predicted_answer == ground_truth
        is_similar = lev_score >= 0.8
        
        # Relieved Accuracy (FinTabNet style)
        norm_pred, norm_preds = fintabnet_normalize(predicted_answer)
        norm_gt, norm_gts = fintabnet_normalize(ground_truth)
        relieved_loose = int(any(_p == _g for _p in norm_preds for _g in norm_gts))
        
        # Update metrics
        exact_match += int(is_exact)
        similar_match += int(is_similar)
        relieved_match += relieved_loose
        total += 1
        
        # Save prediction
        prediction_data = {
            "index": entry["index"],
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "levenshtein_score": lev_score,
            "exact_match": is_exact,
            "lenient_match": is_similar,
            "relieved_match": relieved_loose
        }
        predictions.append(prediction_data)
        
        # Append to JSONL file after each prediction
        with open("../Results/qwen-2.5-14B-Hindi.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction_data, ensure_ascii=False) + "\n")
    
    # Save predictions to JSONL file (complete)
    output_file = "../Results/qwen-2.5-14B-Hindi.jsonl"
    print(f"Saving predictions to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for prediction in tqdm(predictions, desc="Writing predictions"):
            f.write(json.dumps(prediction, ensure_ascii=False) + '\n')
    
    # Print final statistics
    print("\n=== Final Evaluation ===")
    print(f"Total Samples             : {total}")
    print(f"Exact Match Accuracy      : {exact_match / total * 100:.2f}%")
    print(f"Levenshtein ≥ 0.8 Accuracy: {similar_match / total * 100:.2f}%")
    print(f"Relieved Accuracy         : {relieved_match / total * 100:.2f}%")
    
    # Save summary statistics
    stats = {
        "total_samples": total,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match / total * 100,
        "lenient_match": similar_match,
        "lenient_match_accuracy": similar_match / total * 100,
        "relieved_match": relieved_match,
        "relieved_accuracy": relieved_match / total * 100
    }
    
    with open("../Results/qwen-2.5-14B-Hindi.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Statistics saved to ../Results/qwen-2.5-14B-Hindi.json")

if __name__ == "__main__":
    main()