# src/data/dataset.py
import os
from torch.utils.data import Dataset # type: ignore # type: ignore
from typing import Optional, Dict, Any
import torch # type: ignore

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