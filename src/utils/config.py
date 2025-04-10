# src/utils/config.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class LLMABSAConfig:
    # Model settings - Use a model that definitely exists on HuggingFace
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"microsoft/phi-2"    # Fallback model that's guaranteed to exist
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1
    num_attention_heads: int = 12

    freeze_layers: bool = True 
    
    # Training settings
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 16
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    
    # Domain adaptation
    domain_adaptation: bool = False
    domain_mapping: Dict[str, int] = None  # Will be initialized in __post_init__
    
    # Data settings
    max_seq_length: int = 128
    datasets: List[str] = None  # Will be initialized in __post_init__
    dataset_paths: Dict[str, str] = None  # Will be initialized in __post_init__
    
    # Architectual choices
    use_syntax: bool = True
    use_fp16: bool = False
    confidence_threshold: float = 0.5
    
    # Loss weights
    aspect_loss_weight: float = 1.0
    opinion_loss_weight: float = 1.0
    sentiment_loss_weight: float = 1.0
    
    # Logging settings
    experiment_name: str = "stella-absa-v5"
    viz_interval: int = 5
    log_interval: int = 50
    eval_interval: int = 200
    save_interval: int = 500
    
    # Other settings
    num_workers: int = 4
    use_local: bool = False
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        # Initialize default values for lists and dicts
        if self.datasets is None:
            self.datasets = ["laptop14", "rest14", "rest15", "rest16"]
            
        if self.dataset_paths is None:
            # Set default dataset paths
            self.dataset_paths = {
                dataset: f"Datasets/aste/{dataset}" 
                for dataset in self.datasets
            }
            
        if self.domain_mapping is None:
            # Set default domain mapping
            self.domain_mapping = {
                "laptop14": 0,
                "rest14": 1,
                "rest15": 1,  # Same domain as rest14
                "rest16": 1,  # Same domain as rest14
            }