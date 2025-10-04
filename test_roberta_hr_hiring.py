import pandas as pd
import json
import torch
from transformers import RobertaTokenizer, RobertaForMultipleChoice
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os

class HRHiringDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input for multiple choice
        context = item['context']
        question = item['question']
        choices = [item['ans0'], item['ans1'], item['ans2']]
        
        # Create input text for each choice
        inputs = []
        for choice in choices:
            text = f"{context} {question} {choice}"
            inputs.append(text)
        
        # Tokenize
        encoded = self.tokenizer(
            inputs,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': torch.tensor(item['label'], dtype=torch.long),
            'example_id': item['example_id'],
            'category': item['category'],
            'question_polarity': item['question_polarity'],
            'context_condition': item['context_condition']
        }

def test_roberta_hr_hiring():
    # Load model and tokenizer
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMultipleChoice.from_pretrained(model_name)
    
    # Load dataset
    dataset = HRHiringDataset("data/HR_hiring.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    results = []
    
    print(f"Testing {len(dataset)} examples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=-1)
            
            # Store results
            for i in range(len(batch['example_id'])):
                results.append({
                    'example_id': batch['example_id'][i].item(),
                    'category': batch['category'][i],
                    'question_polarity': batch['question_polarity'][i],
                    'context_condition': batch['context_condition'][i],
                    'true_label': batch['labels'][i].item(),
                    'predicted_label': torch.argmax(predictions[i]).item(),
                    'ans0_prob': predictions[i][0].item(),
                    'ans1_prob': predictions[i][1].item(), 
                    'ans2_prob': predictions[i][2].item(),
                    'correct': (torch.argmax(predictions[i]).item() == batch['labels'][i].item())
                })
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/hr_hiring_roberta_results.csv", index=False)
    
    # Print summary statistics
    accuracy = results_df['correct'].mean()
    print(f"\nOverall Accuracy: {accuracy:.3f}")
    
    # Accuracy by context condition
    context_acc = results_df.groupby('context_condition')['correct'].mean()
    print(f"\nAccuracy by Context Condition:")
    for condition, acc in context_acc.items():
        print(f"  {condition}: {acc:.3f}")
    
    # Accuracy by question polarity
    polarity_acc = results_df.groupby('question_polarity')['correct'].mean()
    print(f"\nAccuracy by Question Polarity:")
    for polarity, acc in polarity_acc.items():
        print(f"  {polarity}: {acc:.3f}")
    
    print(f"\nResults saved to results/hr_hiring_roberta_results.csv")
    return results_df

if __name__ == "__main__":
    test_roberta_hr_hiring()