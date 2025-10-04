import pandas as pd
import json
import ast
import os

def generate_hr_hiring_metadata():
    """Generate metadata file for HR hiring bias analysis"""
    
    # Load the generated dataset
    data = []
    with open("data/HR_hiring.jsonl", 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Load template file to get additional info
    templates = pd.read_csv("templates/new_templates - HR_hiring.csv", na_filter=False)
    
    metadata_rows = []
    
    for item in data:
        example_id = item['example_id']
        question_index = item['question_index']
        category = item['category']
        
        # Find corresponding template
        template_row = templates[templates['Q_id'] == int(question_index)]
        if len(template_row) > 0:
            template_info = template_row.iloc[0]
            
            # Determine target location (which answer option is the stereotyped target)
            target_loc = 0  # Default
            if item['question_polarity'] == 'neg':
                # For negative stereotype questions, the stereotyped answer is typically ans0 or ans2
                if 'NAME1' in template_info['Answer_negative']:
                    target_loc = 0
                elif 'NAME2' in template_info['Answer_negative']:
                    target_loc = 2
            else:
                # For non-negative questions, the non-stereotyped answer is typically ans0 or ans2
                if 'NAME1' in template_info['Answer_non_negative']:
                    target_loc = 0
                elif 'NAME2' in template_info['Answer_non_negative']:
                    target_loc = 2
            
            # Determine label type
            label_type = "name"  # HR hiring uses proper names
            
            # Get stereotyped groups
            known_stereotyped_groups = template_info['Known_stereotyped_groups']
            if isinstance(known_stereotyped_groups, str) and known_stereotyped_groups.startswith('['):
                known_stereotyped_groups = ast.literal_eval(known_stereotyped_groups)
            
            # Get relevant social values
            relevant_social_values = template_info['Relevant_social_values']
            
            metadata_rows.append({
                'category': category,
                'question_index': question_index,
                'example_id': example_id,
                'target_loc': target_loc,
                'label_type': label_type,
                'Known_stereotyped_race': '',  # Will be filled based on stereotyped groups
                'Known_stereotyped_var2': '',  # Will be filled based on stereotyped groups  
                'Relevant_social_values': relevant_social_values,
                'corr_ans_aligns_var2': '',
                'corr_ans_aligns_race': '',
                'full_cond': '',
                'Known_stereotyped_groups': str(known_stereotyped_groups)
            })
    
    # Create DataFrame and save
    metadata_df = pd.DataFrame(metadata_rows)
    
    # Create analysis_scripts directory if it doesn't exist
    os.makedirs("analysis_scripts", exist_ok=True)
    
    # Save metadata
    metadata_df.to_csv("analysis_scripts/hr_hiring_metadata.csv", index=False)
    print(f"Generated metadata for {len(metadata_df)} examples")
    print("Metadata saved to analysis_scripts/hr_hiring_metadata.csv")
    
    return metadata_df

if __name__ == "__main__":
    generate_hr_hiring_metadata()