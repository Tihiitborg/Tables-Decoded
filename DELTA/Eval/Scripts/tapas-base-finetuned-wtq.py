#!/usr/bin/env python3
"""
TAPAS Hindi Table Question Answering Inference Script for Entire Dataset
Supports HTML table processing with Hindi questions and direct answer extraction
"""

import os
import pandas as pd
import torch
from transformers import TapasTokenizer, TapasForQuestionAnswering
from transformers import pipeline
import numpy as np
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm
import Levenshtein
import warnings
warnings.filterwarnings('ignore')

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TapasHindiQA:
    def __init__(self, model_path="../", model_name="google/tapas-base-finetuned-wtq"):
        """
        Initialize TAPAS model for Hindi table QA
        
        Args:
            model_path: Local path to store/load model
            model_name: Hugging Face model identifier
        """
        self.model_path = model_path
        self.model_name = model_name
        self.local_model_path = os.path.join(model_path, model_name.split('/')[-1])
        
        # Download and load model
        self.tokenizer, self.model = self._load_or_download_model()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create pipeline for easier inference
        self.qa_pipeline = pipeline(
            "table-question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def _load_or_download_model(self):
        """Load model from local path or download from HuggingFace"""
        try:
            # Try loading from local path first
            if os.path.exists(self.local_model_path):
                print(f"Loading model from local path: {self.local_model_path}")
                tokenizer = TapasTokenizer.from_pretrained(self.local_model_path)
                model = TapasForQuestionAnswering.from_pretrained(self.local_model_path)
            else:
                print(f"Downloading model: {self.model_name}")
                tokenizer = TapasTokenizer.from_pretrained(self.model_name)
                model = TapasForQuestionAnswering.from_pretrained(self.model_name)
                
                # Save locally for future use
                os.makedirs(self.local_model_path, exist_ok=True)
                tokenizer.save_pretrained(self.local_model_path)
                model.save_pretrained(self.local_model_path)
                print(f"Model saved to: {self.local_model_path}")
                
            return tokenizer, model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def html_to_dataframe(self, html_string):
        """
        Convert HTML table to pandas DataFrame with improved error handling
        
        Args:
            html_string: HTML table string
            
        Returns:
            pandas.DataFrame: Processed table
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_string, 'html.parser')
            
            # Find table
            table = soup.find('table')
            if not table:
                raise ValueError("No table found in HTML")
            
            # Extract all rows first
            all_rows = []
            for tr in table.find_all('tr'):
                cells = []
                for cell in tr.find_all(['td', 'th']):
                    cell_text = cell.get_text(strip=True)
                    # Handle rowspan - repeat cell value for spanned rows
                    rowspan = int(cell.get('rowspan', 1))
                    colspan = int(cell.get('colspan', 1))
                    
                    # For now, just add the cell value (handling rowspan/colspan is complex)
                    cells.append(cell_text)
                
                if cells:  # Only add non-empty rows
                    all_rows.append(cells)
            
            if not all_rows:
                raise ValueError("No rows found in table")
            
            # Find the maximum number of columns
            max_cols = max(len(row) for row in all_rows)
            
            # Pad all rows to have the same number of columns
            for row in all_rows:
                while len(row) < max_cols:
                    row.append("")
            
            # Try to determine if first row is header
            # If first row has different structure or seems like headers, use it
            if len(all_rows) > 1:
                # Check if first row seems like a header
                first_row_unique = len(set(all_rows[0])) == len(all_rows[0])
                if first_row_unique and max_cols <= 4:  # Reasonable header assumption
                    headers = all_rows[0]
                    data_rows = all_rows[1:]
                else:
                    # Generate generic headers
                    headers = [f"कॉलम_{i+1}" for i in range(max_cols)]
                    data_rows = all_rows
            else:
                headers = [f"कॉलम_{i+1}" for i in range(max_cols)]
                data_rows = all_rows
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Clean column names
            df.columns = [str(col).strip() if col else f"कॉलम_{i+1}" 
                         for i, col in enumerate(df.columns)]
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error converting HTML to DataFrame: {e}")
            # More robust fallback
            try:
                soup = BeautifulSoup(html_string, 'html.parser')
                all_text_elements = []
                
                # Extract all text from table cells
                for cell in soup.find_all(['td', 'th']):
                    text = cell.get_text(strip=True)
                    if text:
                        all_text_elements.append(text)
                
                if len(all_text_elements) >= 2:
                    # Create a simple 2-column table with alternating key-value pairs
                    rows = []
                    for i in range(0, len(all_text_elements)-1, 2):
                        if i+1 < len(all_text_elements):
                            rows.append([all_text_elements[i], all_text_elements[i+1]])
                    
                    if rows:
                        df = pd.DataFrame(rows, columns=['विवरण', 'मान'])
                        return df
                
                # Final fallback - create a single column table
                if all_text_elements:
                    df = pd.DataFrame(all_text_elements, columns=['डेटा'])
                    return df
                else:
                    raise ValueError("No data could be extracted from HTML table")
                    
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise ValueError("Could not parse HTML table at all")
    
    def preprocess_question(self, question):
        """
        Add Hindi prompt to question for better context
        
        Args:
            question: Original question
            
        Returns:
            str: Preprocessed question with prompt
        """
        # Clean the question first
        question = str(question).strip()
        # Remove any surrounding quotes
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        
        prompt = "तालिका से उत्तर दें: "
        return f"{prompt}{question}"
    
    def extract_direct_answer(self, prediction_result, original_question):
        """
        Extract direct answer from TAPAS prediction with better error handling
        
        Args:
            prediction_result: TAPAS pipeline result
            original_question: Original question for context
            
        Returns:
            str: Direct answer
        """
        try:
            # Handle None result
            if prediction_result is None:
                return "उत्तर नहीं मिला"
            
            # Get answer from pipeline result
            if isinstance(prediction_result, dict):
                answer = prediction_result.get('answer', '')
                cells = prediction_result.get('cells', [])
                coordinates = prediction_result.get('coordinates', [])
                
                # If we have cells and coordinates, extract relevant information
                if cells and coordinates:
                    # Join all selected cells
                    selected_cells = []
                    for cell in cells:
                        cell_str = str(cell) if cell is not None else ""
                        cell_str = cell_str.strip()
                        if cell_str:
                            selected_cells.append(cell_str)
                    
                    if selected_cells:
                        answer = ' '.join(selected_cells)
                
                # Clean and format answer
                answer = str(answer) if answer is not None else ""
                answer = answer.strip()
                
                # Post-process for specific patterns
                if answer:
                    # Remove extra spaces and clean formatting
                    answer = re.sub(r'\s+', ' ', answer)
                    answer = answer.strip()
                    
                    # Handle specific question types
                    if 'करोड़' in answer and 'कारोबार' in original_question:
                        # Extract the amount part for business turnover questions
                        match = re.search(r'₹\s*[\d,]+\s*करोड़', answer)
                        if match:
                            answer = match.group().strip()
                    
                    # Handle percentage questions
                    if 'प्रतिशत' in original_question and answer.isdigit():
                        return answer
                    
                    # Handle charge/fee questions
                    if 'शुल्क' in original_question and answer.isdigit():
                        return answer
                
                return answer if answer else "उत्तर नहीं मिला"
            
            # Handle string results
            result_str = str(prediction_result) if prediction_result is not None else ""
            return result_str.strip() if result_str.strip() else "उत्तर नहीं मिला"
            
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return "उत्तर निकालने में त्रुटि"
    
    def predict(self, html_table, question, max_length=512):
        """
        Main prediction function with comprehensive error handling
        
        Args:
            html_table: HTML table string
            question: Question in Hindi
            max_length: Maximum sequence length
            
        Returns:
            str: Direct answer
        """
        try:
            # Validate inputs
            if not html_table or not question:
                return "अवैध इनपुट: टेबल या प्रश्न अनुपस्थित"
            
            # Convert HTML to DataFrame
            df = self.html_to_dataframe(html_table)
            
            # Validate DataFrame
            if df.empty:
                return "खाली तालिका"
            
            # Preprocess question
            processed_question = self.preprocess_question(question)
            
            # Make prediction using pipeline with error handling
            try:
                result = self.qa_pipeline(
                    table=df,
                    query=processed_question
                )
            except Exception as pipeline_error:
                print(f"Pipeline error: {pipeline_error}")
                return f"मॉडल त्रुटि: {str(pipeline_error)[:50]}"
            
            # Extract direct answer
            direct_answer = self.extract_direct_answer(result, question)
            
            return direct_answer
            
        except Exception as e:
            error_msg = str(e)
            print(f"Prediction error: {error_msg}")
            return f"त्रुटि: {error_msg[:50]}"

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
    
    # Initialize QA system
    print("Initializing TAPAS QA system...")
    qa_system = TapasHindiQA()
    
    # Load the dataset
    data_path = "../torque_hindi_qa.json"
    print(f"Loading dataset from {data_path}...")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Dataset file not found: {data_path}")
        print("Please ensure the file exists and path is correct.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize metrics
    exact_match = 0
    similar_match = 0
    relieved_match = 0
    total = 0
    predictions = []
    
    # Clear previous results file
    output_file = "tapas_results.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Process each sample with progress tracking
    print("Starting inference...")
    for i, entry in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            question = entry.get("question", "")
            ground_truth = str(entry.get("answer", "")).strip().lower()
            html_content = entry.get("html", "")
            index = entry.get("index", i)
            
            # Skip if essential data is missing
            if not question or not html_content:
                print(f"Skipping sample {index}: Missing question or HTML")
                continue
            
            # Get prediction
            answer = qa_system.predict(html_content, question)
            predicted_answer = str(answer).strip().lower()
            
            # Calculate metrics
            try:
                lev_score = Levenshtein.ratio(predicted_answer, ground_truth)
            except:
                lev_score = 0.0
                
            is_exact = predicted_answer == ground_truth
            is_similar = lev_score >= 0.8
            
            # Relieved Accuracy (FinTabNet style)
            try:
                norm_pred, norm_preds = fintabnet_normalize(predicted_answer)
                norm_gt, norm_gts = fintabnet_normalize(ground_truth)
                relieved_loose = int(any(_p == _g for _p in norm_preds for _g in norm_gts))
            except:
                relieved_loose = 0
            
            # Update metrics
            exact_match += int(is_exact)
            similar_match += int(is_similar)
            relieved_match += relieved_loose
            total += 1
            
            # Save prediction
            prediction_data = {
                "index": index,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "levenshtein_score": lev_score,
                "exact_match": is_exact,
                "lenient_match": is_similar,
                "relieved_match": relieved_loose
            }
            predictions.append(prediction_data)
            
            # Append to JSONL file after each prediction for progress tracking
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction_data, ensure_ascii=False) + "\n")
                
            # Print progress for difficult cases
            if not is_exact and i % 50 == 0:
                print(f"\nSample {index}:")
                print(f"Q: {question[:50]}...")
                print(f"Expected: {ground_truth}")
                print(f"Got: {predicted_answer}")
                print(f"Match: {is_exact}, Similar: {is_similar}")
                
        except Exception as sample_error:
            print(f"Error processing sample {i}: {sample_error}")
            # Continue with next sample rather than crash
            continue
    
    # Print final statistics
    print("\n=== Final Evaluation ===")
    print(f"Total Samples             : {total}")
    if total > 0:
        print(f"Exact Match Accuracy      : {exact_match / total * 100:.2f}%")
        print(f"Levenshtein ≥ 0.8 Accuracy: {similar_match / total * 100:.2f}%")
        print(f"Relieved Accuracy         : {relieved_match / total * 100:.2f}%")
    else:
        print("No samples were processed successfully.")
    
    # Save summary statistics
    stats = {
        "total_samples": total,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match / total * 100 if total > 0 else 0,
        "lenient_match": similar_match,
        "lenient_match_accuracy": similar_match / total * 100 if total > 0 else 0,
        "relieved_match": relieved_match,
        "relieved_accuracy": relieved_match / total * 100 if total > 0 else 0
    }
    
    with open("tapas_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")
    print("Statistics saved to tapas_stats.json")

if __name__ == "__main__":
    main()