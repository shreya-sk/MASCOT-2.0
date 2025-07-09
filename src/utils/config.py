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
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch
import os

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch
import os

@dataclass
class ABSAConfig:
    """ABSA Configuration"""
    
    # Model settings
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    num_classes: int = 3
    dropout: float = 0.1
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data settings
    max_seq_length: int = 128
    datasets: List[str] = field(default_factory=lambda: ['laptop14', 'rest14'])
    data_dir: str = "Datasets"
    
    # Feature toggles
    use_implicit_detection: bool = True
    use_few_shot_learning: bool = True
    use_generative_framework: bool = False
    use_contrastive_learning: bool = True
    
    # Few-shot settings
    few_shot_k: int = 5
    few_shot_episodes: int = 100
    
    # Training intervals (ADD THESE MISSING PARAMETERS)
    eval_interval: int = 100
    save_interval: int = 500
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42
    
    # Output settings
    output_dir: str = "outputs"
    experiment_name: str = "absa_experiment"


    
    def __post_init__(self):
        """Validate and adjust configuration"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate datasets
        self.datasets = self._validate_datasets()
    
    def _validate_datasets(self) -> List[str]:
        """Validate dataset availability"""
        valid_datasets = []
        
        for dataset in self.datasets:
            train_file = os.path.join(self.data_dir, "aste", dataset, "train.txt")
            if os.path.exists(train_file):
                valid_datasets.append(dataset)
                print(f"✅ Dataset found: {dataset}")
            else:
                print(f"❌ Dataset missing: {dataset} (looking for {train_file})")
        
        if not valid_datasets:
            print("⚠️ No valid datasets found, using default laptop14")
            return ['laptop14']
        
        return valid_datasets
    
    def get_dataset_path(self, dataset: str) -> str:
        """Get path for a specific dataset"""
        return os.path.join(self.data_dir, "aste", dataset)
    
    def get_experiment_dir(self) -> str:
        """Get experiment output directory"""
        exp_dir = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            key: getattr(self, key) for key in self.__dataclass_fields__.keys()
        }
def create_development_config():
    """Create development configuration"""
    return ABSAConfig(
        batch_size=4,
        num_epochs=5,
        learning_rate=1e-5,
        max_seq_length=64,
        eval_interval=50,
        save_interval=200,
        datasets=['laptop14'],
        experiment_name="absa_dev",
        num_workers=0
    )

def create_research_config():
    """Create research configuration"""
    return ABSAConfig(
        batch_size=8,
        num_epochs=25,
        learning_rate=3e-5,
        use_implicit_detection=True,
        use_few_shot_learning=True,
        use_generative_framework=True,
        use_contrastive_learning=True,
        datasets=['laptop14', 'rest14', 'rest15', 'rest16'],
        experiment_name="absa_research"
    )

def create_minimal_config():
    """Create minimal configuration for quick testing"""
    return ABSAConfig(
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-4,
        max_seq_length=32,
        datasets=['laptop14'],
        experiment_name="absa_minimal"
    )