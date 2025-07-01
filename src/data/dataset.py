# src/data/dataset.py
import os
from torch.utils.data import Dataset # type: ignore # type: ignore
from typing import Optional, Dict, Any
import torch # type: ignore
import numpy as np # type: ignore

class ABSADataset(Dataset):
    """ABSA dataset class supporting multiple datasets with preprocessor"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        preprocessor=None,
        split: str = 'train',
        dataset_name: str = 'rest15',
        max_length: int = 128,
        domain_id: Optional[int] = None
    ):
        """
        Args:
            data_dir: Base directory containing dataset folders
            tokenizer: Tokenizer for text encoding
            preprocessor: Optional custom preprocessor
            split: Data split ('train', 'dev', 'test')
            dataset_name: Dataset name ('laptop14', 'rest14', 'rest15', 'rest16')
            max_length: Maximum sequence length
            domain_id: Optional domain identifier for domain adaptation
        """
        # Construct file path using project root directory
        # Go up two levels from src/data to get to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_path = os.path.join(project_root, 'Datasets', 'aste', dataset_name, f'{split}.txt')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
            
        # Import here to avoid circular imports
        from .utils import read_aste_data
        
        # Load data
        self.data = read_aste_data(file_path)
        
        # Store preprocessor and domain id
        self.preprocessor = preprocessor
        self.domain_id = domain_id
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single instance"""
        text, span_labels = self.data[idx]
        
        # Regular preprocessing
        tokenized = self.preprocessor.preprocess(text, span_labels)
        
        # Add original text and domain ID if available
        tokenized['text'] = text
        if self.domain_id is not None:
            tokenized['domain_id'] = torch.tensor(self.domain_id)
            
        return tokenized
    # Add this to src/data/dataset.py or create a new file src/data/batch_utils.py

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized data"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # Initialize merged batch
    merged_batch = {}
    
    # Process each key separately
    for key in batch[0].keys():
        if key in ['aspect_labels', 'opinion_labels']:
            # These are the tensors causing issues - they have different first dimensions
            # Get max size
            max_size = max([item[key].size(0) for item in batch])
            
            # Pad each tensor to max size
            padded_tensors = []
            for item in batch:
                tensor = item[key]
                if tensor.size(0) < max_size:
                    # Create padding
                    padding = torch.zeros((max_size - tensor.size(0), tensor.size(1)), 
                                        dtype=tensor.dtype, device=tensor.device)
                    # Concatenate
                    padded = torch.cat([tensor, padding], dim=0)
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(tensor)
            
            # Stack
            merged_batch[key] = torch.stack(padded_tensors, dim=0)
        elif key == 'sentiment_labels':
            # Handle sentiment labels similarly
            max_size = max([item[key].size(0) for item in batch])
            padded_tensors = []
            for item in batch:
                tensor = item[key]
                if tensor.size(0) < max_size:
                    padding = torch.zeros(max_size - tensor.size(0), 
                                        dtype=tensor.dtype, device=tensor.device)
                    padded = torch.cat([tensor, padding], dim=0)
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(tensor)
            merged_batch[key] = torch.stack(padded_tensors, dim=0)
        elif key == 'num_spans':
            # Just convert to tensor
            merged_batch[key] = torch.tensor([item[key] for item in batch])
        elif key == 'text':
            # Keep text as list
            merged_batch[key] = [item[key] for item in batch]
        else:
            # Other tensors should be uniform size
            try:
                merged_batch[key] = torch.stack([item[key] for item in batch], dim=0)
            except:
                # If stacking fails, keep as list
                merged_batch[key] = [item[key] for item in batch]
    
    return merged_batch

def mixup_batch(batch, alpha=0.2):
    """Apply mixup augmentation to a batch to improve generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = batch['input_ids'].size(0)
    index = torch.randperm(batch_size)
    
    # FIXED: Don't apply mixup to input_ids because embeddings need integer inputs
    # Instead, we'll apply mixup at the embedding level in the forward pass
    mixed_input_ids = batch['input_ids']  # Keep input_ids as integers
    mixed_attention_mask = batch['attention_mask']  # Keep attention mask the same
    
    # Store mixup information for the model to use
    batch['mixup_lambda'] = lam
    batch['mixup_index'] = index
    
    # Mix labels for soft labels based on mixup (these will still be handled in loss calculation)
    batch['mixed_aspect_labels'] = torch.zeros_like(batch['aspect_labels'], dtype=torch.float)
    for b in range(batch_size):
        batch['mixed_aspect_labels'][b] = lam * F.one_hot(batch['aspect_labels'][b], num_classes=3) + \
                                        (1 - lam) * F.one_hot(batch['aspect_labels'][index[b]], num_classes=3)
    
    batch['mixed_opinion_labels'] = torch.zeros_like(batch['opinion_labels'], dtype=torch.float)
    for b in range(batch_size):
        batch['mixed_opinion_labels'][b] = lam * F.one_hot(batch['opinion_labels'][b], num_classes=3) + \
                                         (1 - lam) * F.one_hot(batch['opinion_labels'][index[b]], num_classes=3)
    
    # For sentiment labels
    if 'sentiment_labels' in batch:
        batch['mixed_sentiment_labels'] = torch.zeros((batch_size, 3), dtype=torch.float, device=batch['sentiment_labels'].device)
        for b in range(batch_size):
            batch['mixed_sentiment_labels'][b] = lam * F.one_hot(batch['sentiment_labels'][b], num_classes=3) + \
                                              (1 - lam) * F.one_hot(batch['sentiment_labels'][index[b]], num_classes=3)
    
    return batch