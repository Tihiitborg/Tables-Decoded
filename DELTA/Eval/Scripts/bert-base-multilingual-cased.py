#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hindi Question Answering Inference Script with Evaluation
Uses BERT multilingual model for extractive QA from HTML tables
"""

import os
import torch
import json
import re
from tqdm import tqdm
import Levenshtein
from bs4 import BeautifulSoup
from transformers import (
    BertTokenizer, 
    BertForQuestionAnswering,
    pipeline
)
import warnings
warnings.filterwarnings('ignore')

class HindiQAInference:
    def __init__(self, model_path="../bert-base-multilingual-cased"):
        """
        Initialize the QA inference pipeline
        
        Args:
            model_path (str): Path to the locally downloaded model
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        self._setup_model()
    
    def _download_model(self):
        """Download model locally if not exists"""
        if not os.path.exists(self.model_path):
            print("Downloading BERT multilingual model...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Download tokenizer and model
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            model = BertForQuestionAnswering.from_pretrained('bert-base-multilingual-cased')
            
            # Save locally
            tokenizer.save_pretrained(self.model_path)
            model.save_pretrained(self.model_path)
            print(f"Model saved to {self.model_path}")
        else:
            print(f"Model found at {self.model_path}")
    
    def _setup_model(self):
        """Setup model and tokenizer from local path"""
        try:
            # Download if not exists
            self._download_model()
            
            # Load from local path
            print("Loading model and tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForQuestionAnswering.from_pretrained(self.model_path)
            
            # Create QA pipeline
            self.qa_pipeline = pipeline(
                'question-answering',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=False
            )
            
            print("Model loaded successfully!")
            print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def parse_html_table(self, html_content):
        """
        Parse HTML table and convert to clean, structured text
        
        Args:
            html_content (str): HTML table string
            
        Returns:
            str: Clean formatted text from table
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                # If no table, try to extract any text
                return soup.get_text(strip=True)
            
            # Extract table data in multiple formats for better context
            rows = table.find_all('tr')
            context_sentences = []
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    
                    if key and value:
                        # Multiple formats to help model understand
                        context_sentences.extend([
                            f"{key}: {value}",
                            f"{key} है {value}",
                            f"{value} {key} का मान है"
                        ])
            
            return ". ".join(context_sentences)
            
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return str(html_content)
    
    def create_focused_prompt(self, question, context):
        """
        Create a focused prompt based on question type
        
        Args:
            question (str): The question
            context (str): The context
            
        Returns:
            str: Optimized prompt
        """
        # Analyze question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['कौन', 'किस', 'कहाँ', 'कब']):
            # Who/What/Where/When questions - need specific answers
            prompt = f"प्रश्न: {question}\nसंदर्भ: {context}\nउत्तर (केवल मुख्य तथ्य):"
        elif any(word in question_lower for word in ['कितना', 'कितने', 'प्रतिशत', '%']):
            # Quantity/percentage questions
            prompt = f"प्रश्न: {question}\nसंदर्भ: {context}\nसंख्यात्मक उत्तर:"
        elif any(word in question_lower for word in ['क्या', 'कैसे', 'क्यों']):
            # What/How/Why questions
            prompt = f"प्रश्न: {question}\nसंदर्भ: {context}\nसीधा उत्तर:"
        else:
            # General questions
            prompt = f"प्रश्न: {question}\nसंदर्भ: {context}\nउत्तर:"
            
        return prompt
    
    def extract_clean_answer(self, raw_answer, question, context):
        """
        Extract and clean the answer using multiple strategies
        
        Args:
            raw_answer (str): Raw answer from model
            question (str): Original question
            context (str): Context used
            
        Returns:
            str: Cleaned answer
        """
        if not raw_answer or raw_answer.strip() == "":
            return "उत्तर नहीं मिला"
        
        answer = raw_answer.strip()
        
        # Strategy 1: Remove common Hindi particles and connectors
        particles_to_remove = [
            r'^(का|की|के|में|से|को|है|हैं|था|थे|थी|होना|होने|वाला|वाली|वाले)\s+',
            r'^(और|या|तथा|एवं|लेकिन|परन्तु|किन्तु)\s+',
            r'^(मान|मान्य|मूल्य|संख्या|डेटा|जानकारी)\s+',
            r'\s+(है|हैं|था|थे|थी|होना)$'
        ]
        
        for pattern in particles_to_remove:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Strategy 2: If answer is too generic, try to find specific values
        generic_words = ['मान', 'डेटा', 'संख्या', 'जानकारी', 'तथ्य']
        if any(word in answer for word in generic_words) and len(answer) < 10:
            # Try to extract numbers, percentages, or specific values from context
            numbers = re.findall(r'[\d,.]+ *%?', context)
            currencies = re.findall(r'₹[\d,.\s]+(?:करोड़|लाख)?', context)
            
            if currencies:
                answer = currencies[0]
            elif numbers:
                # Find the most relevant number based on question context
                if 'प्रतिशत' in question or '%' in question:
                    percentage_nums = [n for n in numbers if '%' in n]
                    if percentage_nums:
                        answer = percentage_nums[0]
                else:
                    answer = numbers[0]
        
        # Strategy 3: Handle incomplete answers
        if len(answer) < 3 and answer not in ['हां', 'नहीं', 'है', 'नहीं है']:
            # Try to find complete phrases from context
            sentences = context.split('.')
            for sentence in sentences:
                if any(word in sentence.lower() for word in question.lower().split()[-3:]):
                    # Extract key information from relevant sentence
                    values = re.findall(r'[\d,.]+ *%?|₹[\d,.\s]+(?:करोड़|लाख)?', sentence)
                    if values:
                        answer = values[0]
                        break
        
        # Strategy 4: Final cleanup
        answer = answer.strip(' ।.,;:')
        
        # If still too short or generic, return a fallback
        if len(answer) < 2:
            return "जानकारी उपलब्ध नहीं"
            
        return answer
    
    def predict_answer(self, question, html_content, max_answer_length=100):
        """
        Predict answer from HTML table content with robust processing
        
        Args:
            question (str): Question to answer
            html_content (str): HTML table content
            max_answer_length (int): Maximum length of answer
            
        Returns:
            dict: Prediction results
        """
        try:
            # Parse HTML table
            context = self.parse_html_table(html_content)
            
            # Try multiple approaches for better results
            approaches = [
                # Approach 1: Direct question-context
                {"question": question, "context": context},
                
                # Approach 2: With Hindi instruction
                {"question": f"निम्नलिखित प्रश्न का उत्तर दें: {question}", "context": context},
                
                # Approach 3: Focused prompt
                {"question": question, "context": self.create_focused_prompt(question, context)}
            ]
            
            best_result = None
            best_confidence = 0
            
            for approach in approaches:
                try:
                    result = self.qa_pipeline(
                        question=approach["question"],
                        context=approach["context"],
                        max_answer_len=max_answer_length,
                        doc_stride=64,
                        max_seq_len=512,
                        handle_impossible_answer=True
                    )
                    
                    if result['score'] > best_confidence:
                        best_confidence = result['score']
                        best_result = result
                        best_result['approach'] = approach
                        
                except Exception as e:
                    print(f"Error in approach: {e}")
                    continue
            
            if best_result is None:
                return {
                    'question': question,
                    'answer': "प्रसंस्करण त्रुटि",
                    'confidence': 0.0,
                    'context': context
                }
            
            # Extract and clean answer
            clean_answer = self.extract_clean_answer(
                best_result['answer'], 
                question, 
                context
            )
            
            return {
                'question': question,
                'answer': clean_answer,
                'confidence': best_result['score'],
                'raw_answer': best_result['answer'],
                'context': context
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'question': question,
                'answer': "Error in prediction",
                'confidence': 0.0,
                'error': str(e)
            }

def extract_answer(decoded_output, input_text):
    """
    Extract answer from model output
    """
    decoded_output = decoded_output.lower()
    if "### answer:" in decoded_output:
        start = decoded_output.find("### answer:") + len("### answer:")
        end = decoded_output.find("###", start)
        return decoded_output[start:end].strip() if end != -1 else decoded_output[start:].strip()
    else:
        return decoded_output.replace(input_text.lower(), "").strip()

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
    print("Initializing Hindi QA Inference Pipeline...")
    qa_inference = HindiQAInference()
    
    # Load the dataset
    data_path = "../torque_hindi_qa.json"
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
        ground_truth = entry["answer"].strip()
        html_content = entry["html"]
        
        # Get prediction
        result = qa_inference.predict_answer(question, html_content)
        predicted_answer = result['answer'].strip()
        
        # Calculate metrics
        lev_score = Levenshtein.ratio(predicted_answer.lower(), ground_truth.lower())
        is_exact = predicted_answer.lower() == ground_truth.lower()
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
            "index": entry.get("index", total),
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "confidence": result['confidence'],
            "levenshtein_score": lev_score,
            "exact_match": is_exact,
            "lenient_match": is_similar,
            "relieved_match": relieved_loose
        }
        predictions.append(prediction_data)
        
        # Print progress every 10 samples
        if total % 100 == 0:
            print(f"Processed {total} samples. Current accuracy: {exact_match/total*100:.2f}%")
    
    # Save predictions to JSONL file
    output_file = "mbert_results.jsonl"
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
    
    with open("mbert_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print("Statistics saved to mbert_stats.json")


if __name__ == "__main__":
    main()
    
    

