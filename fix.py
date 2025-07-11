#!/usr/bin/env python3
"""
Quick fix for dataset path issues
Run this to verify and fix dataset loading
"""

import os
import sys
from pathlib import Path

def check_dataset_structure():
    """Check the actual dataset structure"""
    print("🔍 Checking dataset structure...")
    
    datasets = ['laptop14', 'rest14', 'rest15', 'rest16']
    splits = ['train.txt', 'dev.txt', 'test.txt']
    
    base_path = "Datasets/aste"
    
    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        print(f"\n📁 {dataset}:")
        
        if os.path.exists(dataset_path):
            files = os.listdir(dataset_path)
            print(f"   Directory exists: {dataset_path}")
            print(f"   Files found: {files}")
            
            for split in splits:
                file_path = os.path.join(dataset_path, split)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        num_lines = len([l for l in lines if l.strip()])
                    print(f"   ✅ {split}: {num_lines} examples")
                    
                    # Show sample line format
                    if lines:
                        sample = lines[0].strip()
                        print(f"      Sample: {sample[:100]}...")
                else:
                    print(f"   ❌ Missing: {split}")
        else:
            print(f"   ❌ Directory not found: {dataset_path}")

def create_fixed_data_loader():
    """Create a fixed data loader function"""
    
    code = '''
def load_datasets_fixed(config):
    """Fixed dataset loading function"""
    from transformers import AutoTokenizer
    import torch
    from torch.utils.data import Dataset
    
    print(f"🔧 Using fixed dataset loader")
    print(f"   Looking in: {config.data_dir}/aste/")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
    
    class FixedABSADataset(Dataset):
        def __init__(self, data_dir, dataset_name, split, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.examples = []
            
            # Correct path construction
            file_path = os.path.join(data_dir, "aste", dataset_name, f"{split}.txt")
            print(f"   Loading from: {file_path}")
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse ASTE format: sentence####triplets
                        if '####' in line:
                            sentence, triplets_str = line.split('####', 1)
                            try:
                                triplets = eval(triplets_str) if triplets_str.strip() else []
                            except:
                                triplets = []
                        else:
                            sentence = line
                            triplets = []
                        
                        self.examples.append({
                            'sentence': sentence.strip(),
                            'triplets': triplets,
                            'id': f"{dataset_name}_{split}_{line_idx}"
                        })
                
                print(f"   ✅ Loaded {len(self.examples)} examples from {dataset_name}/{split}")
            else:
                print(f"   ❌ File not found: {file_path}")
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            example = self.examples[idx]
            
            # Tokenize
            encoding = self.tokenizer(
                example['sentence'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Create labels (simplified for now)
            seq_len = self.max_length
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': {
                    'aspect_labels': torch.zeros(seq_len, dtype=torch.long),
                    'opinion_labels': torch.zeros(seq_len, dtype=torch.long),
                    'sentiment_labels': torch.zeros(seq_len, dtype=torch.long)
                },
                'example_id': example['id'],
                'sentence': example['sentence'],
                'triplets': example['triplets']
            }
    
    datasets = {}
    for dataset_name in config.datasets:
        dataset_dict = {}
        for split in ['train', 'dev', 'test']:
            dataset = FixedABSADataset(
                data_dir=config.data_dir,
                dataset_name=dataset_name, 
                split=split,
                tokenizer=tokenizer,
                max_length=config.max_seq_length
            )
            if len(dataset) > 0:
                dataset_dict[split] = dataset
        
        if dataset_dict:
            datasets[dataset_name] = dataset_dict
    
    return datasets
'''
    
    # Write to file
    with open('fixed_data_loader.py', 'w') as f:
        f.write(code)
    
    print("✅ Created fixed_data_loader.py")

def test_data_loading():
    """Test loading a sample file"""
    test_file = "Datasets/aste/laptop14/train.txt"
    
    if os.path.exists(test_file):
        print(f"\n🧪 Testing data loading from {test_file}")
        
        with open(test_file, 'r') as f:
            lines = f.readlines()[:5]  # First 5 lines
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            print(f"   Line {i}: {line}")
            
            # Try to parse
            if '####' in line:
                sentence, triplets_str = line.split('####', 1)
                print(f"      Sentence: {sentence}")
                print(f"      Triplets: {triplets_str}")
                try:
                    triplets = eval(triplets_str) if triplets_str.strip() else []
                    print(f"      Parsed triplets: {triplets}")
                except Exception as e:
                    print(f"      Parse error: {e}")
            else:
                print(f"      No triplets found")
    else:
        print(f"❌ Test file not found: {test_file}")

if __name__ == "__main__":
    print("🔧 Dataset Path Diagnostic Tool")
    print("=" * 50)
    
    check_dataset_structure()
    print("\n" + "=" * 50)
    
    test_data_loading()
    print("\n" + "=" * 50)
    
    create_fixed_data_loader()
    print("\n✅ Run this to test the fix:")
    print("   python fixed_data_loader.py")