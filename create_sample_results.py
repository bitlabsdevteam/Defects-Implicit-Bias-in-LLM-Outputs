import pandas as pd
import json
import numpy as np
import os
from pathlib import Path

def read_jsonl(file_path):
    """Read JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_sample_unifiedqa_results():
    """Create sample UnifiedQA results based on existing data"""
    # Load some data to base results on
    data_files = ['data/Gender_identity.jsonl', 'data/Race_ethnicity.jsonl']
    sample_data = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            file_data = read_jsonl(file_path)
            sample_data.extend(file_data[:100])  # Take first 100 examples
    
    if not sample_data:
        print("No data files found to create sample results")
        return
    
    # Create results directory
    os.makedirs('results/UnifiedQA', exist_ok=True)
    
    # Generate sample predictions for different UnifiedQA models
    models = ['unifiedqa-t5-11b_pred_race', 'unifiedqa-t5-11b_pred_arc', 'unifiedqa-t5-11b_pred_qonly']
    
    for model in models:
        results = []
        for item in sample_data:
            # Create a copy of the original item
            result_item = item.copy()
            
            # Simulate model predictions (randomly choose one of the answers)
            answer_choices = [item.get('ans0', ''), item.get('ans1', ''), item.get('ans2', '')]
            predicted_answer = np.random.choice(answer_choices)
            
            # Add prediction to the result
            result_item['prediction'] = predicted_answer
            results.append(result_item)
        
        # Save results to JSONL file
        output_file = f'results/UnifiedQA/{model}_predictions.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Created sample results: {output_file} ({len(results)} examples)")

def create_sample_bert_results():
    """Create sample RoBERTa/DeBERTa results"""
    # Load some data to base results on
    data_files = ['data/Gender_identity.jsonl', 'data/Race_ethnicity.jsonl']
    sample_data = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            file_data = read_jsonl(file_path)
            sample_data.extend(file_data[:100])  # Take first 100 examples
    
    if not sample_data:
        print("No data files found to create sample results")
        return
    
    # Create results directory
    os.makedirs('results/RoBERTa_and_DeBERTaV3', exist_ok=True)
    
    # Generate sample predictions for BERT models
    models = ['deberta-v3-base-race', 'deberta-v3-large-race', 'roberta-base-race', 'roberta-large-race']
    
    results = []
    for item in sample_data:
        for model in models:
            # Generate random scores for each answer (simulating model confidence)
            ans0_score = np.random.random()
            ans1_score = np.random.random()
            ans2_score = np.random.random()
            
            result_row = {
                'index': item['example_id'],
                'ans0': ans0_score,
                'ans1': ans1_score,
                'ans2': ans2_score,
                'model': model,
                'cat': item['category']
            }
            results.append(result_row)
    
    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    output_file = 'results/RoBERTa_and_DeBERTaV3/df_bbq.csv'
    df_results.to_csv(output_file, index=False)
    
    print(f"Created sample results: {output_file} ({len(results)} examples)")

def main():
    """Create sample model results for demonstration"""
    print("Creating sample model results for demonstration...")
    
    # Create sample UnifiedQA results
    create_sample_unifiedqa_results()
    
    # Create sample BERT results
    create_sample_bert_results()
    
    print("\nSample results created successfully!")
    print("You can now run calculate_bias_score.py to see the analysis in action.")

if __name__ == "__main__":
    main()