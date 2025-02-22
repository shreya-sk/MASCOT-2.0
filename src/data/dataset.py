import os
from torch.utils.data import Dataset
from typing import Optional, Dict
import torch

from .utils import read_aste_data, SpanLabel
from .preprocessor import ABSAPreprocessor

class ABSADataset(Dataset):
    """ABSA dataset class supporting multiple datasets"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        split: str = 'train',
        dataset_name: str = 'rest15',
        max_length: int = 128,
    ):
        """
        Args:
            data_dir: Base directory containing dataset folders
            tokenizer: Tokenizer for text encoding
            split: Data split ('train', 'dev', 'test')
            dataset_name: Dataset name ('laptop14', 'rest14', 'rest15', 'rest16')
            max_length: Maximum sequence length
        """
        # Construct file path
        file_path = os.path.join(data_dir, 'aste', dataset_name, f'{split}.txt')
        
        # Load and preprocess data
        self.data = read_aste_data(file_path)
        self.preprocessor = ABSAPreprocessor(tokenizer, max_length)
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single instance"""
        text, span_labels = self.data[idx]
        return self.preprocessor.preprocess(text, span_labels)