#!/usr/bin/env python3
# validate_datasets.py
# Validate all ABSA datasets for data leakage and quality issues

import sys
sys.path.append('.')
from absa_evaluation import DatasetValidator
import json

def main():
    print("🔍 MASCOT-2.0 Dataset Validation")
    print("=" * 50)
    
    validator = DatasetValidator()
    datasets = ['laptop14', 'rest14', 'rest15', 'rest16']
    
    all_results = {}
    critical_issues = []
    
    for dataset in datasets:
        print(f"\n📊 Validating {dataset}...")
        
        dataset_path = f"Datasets/aste/{dataset}"
        results = validator.validate_splits(
            f"{dataset_path}/train.txt",
            f"{dataset_path}/dev.txt",
            f"{dataset_path}/test.txt"
        )
        
        all_results[dataset] = results
        
        # Check for critical issues
        if results.get('data_leakage', False):
            critical_issues.append(f"{dataset}: Data leakage detected")
        
        if 'POOR' in results.get('split_quality', ''):
            critical_issues.append(f"{dataset}: Poor split quality")
        
        # Summary for this dataset
        quality = results.get('split_quality', 'UNKNOWN')
        leakage = "❌ YES" if results.get('data_leakage', False) else "✅ NO"
        
        print(f"   Quality: {quality}")
        print(f"   Data Leakage: {leakage}")
        print(f"   Train: {results.get('train_size', 0)} examples")
        print(f"   Dev: {results.get('dev_size', 0)} examples")
        print(f"   Test: {results.get('test_size', 0)} examples")
    
    # Overall assessment
    print("\n" + "=" * 50)
    print("📋 OVERALL ASSESSMENT")
    print("=" * 50)
    
    if critical_issues:
        print("🚨 CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"   - {issue}")
        print("\n❌ PUBLICATION BLOCKED - Fix these issues first!")
    else:
        print("✅ NO CRITICAL ISSUES FOUND")
        print("🚀 Datasets ready for publication-quality evaluation!")
    
    # Save results
    with open('dataset_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to dataset_validation_results.json")

if __name__ == "__main__":
    main()
