#!/usr/bin/env python
import os

def fix_dataset_paths():
    """Ensure all dataset files exist and are properly formatted"""
    datasets = ['laptop14', 'rest14', 'rest15', 'rest16']
    splits = ['train.txt', 'dev.txt', 'test.txt']
    
    for dataset in datasets:
        dataset_dir = f"Datasets/aste/{dataset}"
        if not os.path.exists(dataset_dir):
            print(f"ERROR: Missing dataset directory: {dataset_dir}")
            continue
            
        for split in splits:
            file_path = os.path.join(dataset_dir, split)
            if not os.path.exists(file_path):
                print(f"ERROR: Missing file: {file_path}")
            else:
                # Check file format
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if '####' not in first_line:
                        print(f"WARNING: {file_path} may have wrong format")
                    else:
                        print(f"✓ {file_path} looks good")

if __name__ == "__main__":
    fix_dataset_paths()