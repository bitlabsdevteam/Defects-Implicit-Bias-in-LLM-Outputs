#!/usr/bin/env python3
"""
Complete pipeline for HR Hiring bias evaluation
Runs: Templates + Vocabulary → Generation Script → Data Files → Model Testing → Bias Analysis
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False

def main():
    print("🚀 Starting HR Hiring Bias Evaluation Pipeline")
    
    # Step 1: Generate dataset from templates
    success = run_command(
        "python generate_hr_hiring.py",
        "Generating HR Hiring dataset from templates"
    )
    if not success:
        print("❌ Pipeline failed at dataset generation")
        return
    
    # Step 2: Test with RoBERTa model
    success = run_command(
        "python test_roberta_hr_hiring.py", 
        "Testing dataset with RoBERTa model"
    )
    if not success:
        print("❌ Pipeline failed at model testing")
        return
    
    # Step 3: Generate metadata
    success = run_command(
        "python generate_hr_hiring_metadata.py",
        "Generating metadata for bias analysis"
    )
    if not success:
        print("❌ Pipeline failed at metadata generation")
        return
    
    # Step 4: Run bias analysis
    success = run_command(
        "Rscript analyze_hr_hiring_bias.R",
        "Running bias analysis and generating visualizations"
    )
    if not success:
        print("❌ Pipeline failed at bias analysis")
        return
    
    print(f"\n{'='*50}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*50}")
    print("\nGenerated files:")
    print("📊 data/HR_hiring.jsonl - Generated dataset")
    print("📈 results/hr_hiring_roberta_results.csv - Model predictions")
    print("📋 analysis_scripts/hr_hiring_metadata.csv - Analysis metadata")
    print("📊 results/hr_hiring_bias_summary.csv - Bias analysis summary")
    print("📈 results/hr_hiring_*.png - Visualization plots")
    
    print("\nNext steps:")
    print("1. Review the bias analysis results in results/hr_hiring_bias_summary.csv")
    print("2. Examine the visualizations in results/hr_hiring_*.png")
    print("3. Use the generated dataset for testing other language models")

if __name__ == "__main__":
    main()