import pandas as pd
import numpy as np
import io
import json
from utils import *
import random
import re
import ast

# HR hiring category
cat = "HR_hiring"

# read in vocabulary files
vocab = pd.read_csv("templates/vocabulary_hr_hiring.csv")
names_vocab = pd.read_csv("templates/vocabulary_proper_names.csv")

unknown_options = [
    "Unknown",
    "Cannot be determined", 
    "Can't be determined",
    "Not answerable",
    "Not known",
    "Not enough info",
    "Not enough information",
    "Cannot answer",
    "Can't answer",
    "Undetermined",
]

# Read template file
frames = pd.read_csv(
    "templates/new_templates - %s.csv" % cat, na_filter=False
)

# Open output file
dat_file = io.open(
    "data/%s.jsonl" % cat, "w"
)

# Get non-empty templates
the_frames = frames[
    frames.Ambiguous_Context != ""
].reset_index()
frame_cols = the_frames.columns

nn = 0

for i in range(len(the_frames)):
    print(f"Processing template {i+1}/{len(the_frames)}")
    
    # Get bias targets
    bias_targets = the_frames.Known_stereotyped_groups[i]
    if len(bias_targets) > 1:
        bias_targets = ast.literal_eval(the_frames.Known_stereotyped_groups[i])
    
    # Use proper names for HR hiring scenarios
    has_proper_name = True
    
    # Get names based on bias targets
    first_names_full = names_vocab[names_vocab.First_last == "first"]
    
    # For gender bias scenarios
    if "F" in bias_targets:
        # Get female names
        female_names = first_names_full[first_names_full.gender == "F"]
        female_word_list = female_names.Name.tolist()
        if len(female_word_list) > 5:
            female_word_list = random.sample(female_word_list, 5)
        
        # Get male names  
        male_names = first_names_full[first_names_full.gender == "M"]
        male_word_list = male_names.Name.tolist()
        if len(male_word_list) > 5:
            male_word_list = random.sample(male_word_list, 5)
            
        # Combine for word list
        word_list = female_word_list
        new_word_list = male_word_list
        
    elif "M" in bias_targets:
        # Get male names
        male_names = first_names_full[first_names_full.gender == "M"] 
        male_word_list = male_names.Name.tolist()
        if len(male_word_list) > 5:
            male_word_list = random.sample(male_word_list, 5)
        
        # Get female names
        female_names = first_names_full[first_names_full.gender == "F"]
        female_word_list = female_names.Name.tolist()
        if len(female_word_list) > 5:
            female_word_list = random.sample(female_word_list, 5)
            
        word_list = male_word_list
        new_word_list = female_word_list
        
    # For race bias scenarios
    elif any(race in bias_targets for race in ["Black", "Hispanic", "Asian"]):
        # Get names from stereotyped racial groups
        stereotyped_names = first_names_full[
            first_names_full.ethnicity.isin([t for t in bias_targets if t in ["Black", "Hispanic", "Asian"]])
        ]
        stereotyped_word_list = stereotyped_names.Name.tolist()
        if len(stereotyped_word_list) > 5:
            stereotyped_word_list = random.sample(stereotyped_word_list, 5)
            
        # Get names from non-stereotyped groups
        non_stereotyped_names = first_names_full[
            ~first_names_full.ethnicity.isin([t for t in bias_targets if t in ["Black", "Hispanic", "Asian"]])
        ]
        non_stereotyped_word_list = non_stereotyped_names.Name.tolist()
        if len(non_stereotyped_word_list) > 5:
            non_stereotyped_word_list = random.sample(non_stereotyped_word_list, 5)
            
        word_list = stereotyped_word_list
        new_word_list = non_stereotyped_word_list
        
    # For intersectional bias (gender + race)
    elif len([t for t in bias_targets if t in ["F", "M"]]) > 0 and len([t for t in bias_targets if t in ["Black", "Hispanic", "Asian"]]) > 0:
        gender_targets = [t for t in bias_targets if t in ["F", "M"]]
        race_targets = [t for t in bias_targets if t in ["Black", "Hispanic", "Asian"]]
        
        # Get names matching both gender and race criteria
        target_names = first_names_full[
            (first_names_full.gender.isin(gender_targets)) & 
            (first_names_full.ethnicity.isin(race_targets))
        ]
        target_word_list = target_names.Name.tolist()
        if len(target_word_list) > 5:
            target_word_list = random.sample(target_word_list, 5)
            
        # Get names not matching the criteria
        non_target_names = first_names_full[
            ~((first_names_full.gender.isin(gender_targets)) & 
              (first_names_full.ethnicity.isin(race_targets)))
        ]
        non_target_word_list = non_target_names.Name.tolist()
        if len(non_target_word_list) > 5:
            non_target_word_list = random.sample(non_target_word_list, 5)
            
        word_list = target_word_list
        new_word_list = non_target_word_list
    
    # Generate examples for each name combination
    for j in range(len(word_list)):
        this_word = word_list[j]
        
        # Get info for this name
        Name1_info = None
        if "F" in bias_targets or "M" in bias_targets:
            try:
                Name1_info = first_names_full.loc[
                    first_names_full["Name"] == this_word, "gender"
                ].iloc[0]
            except IndexError:
                Name1_info = this_word
                
        for k in range(len(new_word_list)):
            this_word_2 = new_word_list[k]
            this_frame_row = the_frames.iloc[[i]].reset_index()
            
            # Get info for second name
            Name2_info = None
            if "F" in bias_targets or "M" in bias_targets:
                try:
                    Name2_info = first_names_full.loc[
                        first_names_full["Name"] == this_word_2, "gender"
                    ].iloc[0]
                except IndexError:
                    Name2_info = this_word_2
            
            # Handle lexical diversity
            lex_div = this_frame_row.Lexical_diversity[0]
            rand_wrd1 = ""
            rand_wrd2 = ""
            
            if len(lex_div) > 1:
                wrdlist1, wrdlist2 = return_list_from_string(lex_div)
                rand_wrd1 = random.choice(wrdlist1)
                if len(wrdlist2) > 1:
                    rand_wrd2 = random.choice(wrdlist2)
            
            # Do slotting
            new_frame_row = do_slotting(
                this_frame_row,
                frame_cols,
                this_word,
                None,
                this_word_2,
                None,
                lex_div,
                rand_wrd1,
                rand_wrd2,
            )
            
            # Create templating dictionaries
            dat_formatted = create_templating_dicts(
                cat,
                new_frame_row,
                "None",  # subcategory
                unknown_options,
                frame_cols,
                bias_targets,
                this_word,
                this_word_2,
                Name1_info,
                Name2_info,
                nn,
            )
            nn += 4
            
            # Write to file
            for item in dat_formatted:
                dat_file.write(json.dumps(item, default=str))
                dat_file.write("\n")
            dat_file.flush()
            
            # Create reversed version for certain scenarios
            new_frame_row_reversed = do_slotting(
                this_frame_row,
                frame_cols,
                this_word_2,
                None,
                this_word,
                None,
                lex_div,
                rand_wrd1,
                rand_wrd2,
            )
            
            dat_formatted_reversed = create_templating_dicts(
                cat,
                new_frame_row_reversed,
                "None",
                unknown_options,
                frame_cols,
                bias_targets,
                this_word_2,
                this_word,
                Name2_info,
                Name1_info,
                nn,
            )
            nn += 4
            
            for item in dat_formatted_reversed:
                dat_file.write(json.dumps(item, default=str))
                dat_file.write("\n")
            dat_file.flush()

print("Generated %s sentences total for %s" % (str(nn), cat))
dat_file.close()