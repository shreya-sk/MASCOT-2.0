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

class ABSAConfig(Dataset):
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
                aspect_term = triplet[0]
                opinion_term = triplet[1]
                sentiment = triplet[2]
                
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
