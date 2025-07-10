# src/data/clean_dataset.py
"""
Clean, simplified dataset handler
Replaces complex dataset implementations with working version
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import json
from typing import List, Dict, Tuple, Optional, Any

class ABSADataset(Dataset):
    """Clean ABSA dataset implementation"""
    
    def __init__(self, 
                 data_dir: str, 
                 dataset_name: str, 
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 128):
        
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.examples = self._load_data()
        print(f"✅ Loaded {len(self.examples)} examples from {dataset_name}/{split}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from ASTE format files"""
        file_path = os.path.join(self.data_dir, "aste", self.dataset_name, f"{self.split}.txt")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return []
        
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse ASTE format: sentence####triplets
                    if '####' in line:
                        sentence, triplets_str = line.split('####', 1)
                        triplets = eval(triplets_str) if triplets_str.strip() else []
                    else:
                        sentence = line
                        triplets = []
                    
                    # Create example
                    example = {
                        'id': f"{self.dataset_name}_{self.split}_{line_idx}",
                        'sentence': sentence.strip(),
                        'triplets': triplets
                    }
                    
                    examples.append(example)
                    
                except Exception as e:
                    print(f"⚠️ Error parsing line {line_idx}: {e}")
                    continue
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize sentence
        encoding = self.tokenizer(
            example['sentence'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels
        labels = self._create_labels(example, encoding)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'example_id': example['id'],
            'sentence': example['sentence'],
            'triplets': example['triplets']
        }
    
    def _create_labels(self, example: Dict, encoding) -> Dict[str, torch.Tensor]:
        """Create labels for training"""
        seq_len = encoding['input_ids'].size(1)
        
        # Initialize labels
        aspect_labels = torch.full((seq_len,), -100, dtype=torch.long)
        opinion_labels = torch.full((seq_len,), -100, dtype=torch.long)
        sentiment_labels = torch.full((seq_len,), 1, dtype=torch.long)  # Default neutral
        
        # Map triplets to token positions
        for triplet in example['triplets']:
            if len(triplet) >= 3:
                # Handle both string and list formats
                aspect_term = triplet[0] if isinstance(triplet[0], str) else str(triplet[0])
                opinion_term = triplet[1] if isinstance(triplet[1], str) else str(triplet[1])
                sentiment = triplet[2] if isinstance(triplet[2], str) else str(triplet[2])
                
                # Map sentiment to index
                sentiment_map = {'POS': 2, 'NEG': 0, 'NEU': 1}
                sentiment_idx = sentiment_map.get(sentiment, 1)
                
                # Find token positions (simplified)
                aspect_positions = self._find_token_positions(aspect_term, example['sentence'], encoding)
                opinion_positions = self._find_token_positions(opinion_term, example['sentence'], encoding)
                
                # Set aspect labels (BIO format)
                for i, pos in enumerate(aspect_positions):
                    if pos < seq_len:
                        aspect_labels[pos] = 1 if i == 0 else 2  # B or I
                
                # Set opinion labels (BIO format)
                for i, pos in enumerate(opinion_positions):
                    if pos < seq_len:
                        opinion_labels[pos] = 1 if i == 0 else 2  # B or I
                
                # Set sentiment for relevant positions
                all_positions = aspect_positions + opinion_positions
                for pos in all_positions:
                    if pos < seq_len:
                        sentiment_labels[pos] = sentiment_idx
        
        # Create implicit labels (simplified)
        implicit_aspect_labels = aspect_labels.clone()
        implicit_opinion_labels = opinion_labels.clone()
        
        return {
            'aspect_labels': aspect_labels,
            'opinion_labels': opinion_labels,
            'sentiment_labels': sentiment_labels,
            'implicit_aspect_labels': implicit_aspect_labels,
            'implicit_opinion_labels': implicit_opinion_labels
        }
    
    def _find_token_positions(self, term: str, sentence: str, encoding) -> List[int]:
        """Find token positions for a term in the sentence"""
        # Ensure term is a string
        if not isinstance(term, str):
            term = str(term)
        
        if not term or not sentence:
            return []
        
        # Simple word-based matching (can be improved)
        words = sentence.lower().split()
        term_words = term.lower().split()
        
        positions = []
        for i in range(len(words) - len(term_words) + 1):
            if words[i:i+len(term_words)] == term_words:
                # Found match, map to token positions
                for j in range(len(term_words)):
                    word_idx = i + j
                    # Approximate token position (simplified)
                    token_pos = word_idx + 1  # +1 for [CLS] token
                    if token_pos < self.max_length:
                        positions.append(token_pos)
                break
        
        return positions


def create_dataloader(dataset: ABSADataset, 
                     batch_size: int, 
                     shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """Create dataloader with proper collation"""
    
    def collate_fn(batch):
        """Custom collate function"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Collect all labels
        labels = {}
        label_keys = ['aspect_labels', 'opinion_labels', 'sentiment_labels', 
                     'implicit_aspect_labels', 'implicit_opinion_labels']
        
        for key in label_keys:
            if key in batch[0]['labels']:
                labels[key] = torch.stack([item['labels'][key] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'example_ids': [item['example_id'] for item in batch],
            'sentences': [item['sentence'] for item in batch],
            'triplets': [item['triplets'] for item in batch]
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )


def load_datasets(config) -> Dict[str, Dict[str, ABSADataset]]:
    """Load all datasets specified in config"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add special tokens if needed
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
    
    datasets = {}
    
    for dataset_name in config.datasets:
        dataset_dict = {}
        
        for split in ['train', 'dev', 'test']:
            try:
                dataset = ABSADataset(
                    data_dir=config.data_dir,
                    dataset_name=dataset_name,
                    split=split,
                    tokenizer=tokenizer,
                    max_length=config.max_seq_length
                )
                
                if len(dataset) > 0:
                    dataset_dict[split] = dataset
                    print(f"✅ {dataset_name}/{split}: {len(dataset)} examples")
                else:
                    print(f"⚠️ {dataset_name}/{split}: empty dataset")
                    
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}/{split}: {e}")
        
        if dataset_dict:
            datasets[dataset_name] = dataset_dict
    
    return datasets


def create_dataloaders(datasets: Dict[str, Dict[str, ABSADataset]], 
                      config) -> Dict[str, Dict[str, DataLoader]]:
    """Create dataloaders for all datasets"""
    dataloaders = {}
    
    for dataset_name, dataset_splits in datasets.items():
        dataloader_splits = {}
        
        for split, dataset in dataset_splits.items():
            shuffle = (split == 'train')
            dataloader = create_dataloader(
                dataset=dataset,
                batch_size=config.batch_size,
                shuffle=shuffle,
                num_workers=config.num_workers
            )
            dataloader_splits[split] = dataloader
        
        dataloaders[dataset_name] = dataloader_splits
    
    return dataloaders


# Quick dataset verification
def verify_datasets(config):
    """Verify dataset integrity"""
    print("🔍 Verifying datasets...")
    
    total_examples = 0
    for dataset_name in config.datasets:
        dataset_path = config.get_dataset_path(dataset_name)
        
        for split in ['train', 'dev', 'test']:
            file_path = os.path.join(dataset_path, f"{split}.txt")
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    examples = len([l for l in lines if l.strip()])
                    total_examples += examples
                    print(f"✅ {dataset_name}/{split}: {examples} examples")
            else:
                print(f"❌ Missing: {file_path}")
    
    print(f"📊 Total examples found: {total_examples}")
    return total_examples > 0