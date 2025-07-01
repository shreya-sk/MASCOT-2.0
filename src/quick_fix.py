#!/usr/bin/env python
"""Quick fixes for common issues"""
import torch
import os

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU available, using CPU")

def check_datasets():
    """Check dataset files"""
    datasets = ['laptop14', 'rest14', 'rest15', 'rest16']
    for dataset in datasets:
        path = f"Datasets/aste/{dataset}/train.txt"
        if os.path.exists(path):
            print(f"✓ {dataset} dataset found")
        else:
            print(f"✗ {dataset} dataset missing at {path}")

if __name__ == "__main__":
    check_gpu()
    check_datasets()