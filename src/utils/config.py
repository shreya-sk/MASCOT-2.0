# src/utils/config.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class LLMABSAConfig:
    # Model settings
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hidden_size: int = 768  # Set to a value divisible by most common head counts
    num_layers: int = 2
    dropout: float = 0.1
    num_attention_heads: int = 12  # Should divide hidden_size evenly
    
    # Generation settings
    num_decoder_layers: int = 2
    vocab_size: int = 32000  # Match your tokenizer's vocab size
    max_generation_length: int = 64
    use_generation: bool = True
    # Generative parameters
    generate_explanations: bool = True  # Set to True for training with generation
    generation_weight: float = 0.5  # Weight for generation loss
    
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
        
        # Ensure num_attention_heads divides hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            # Find closest number of heads that divides hidden_size
            for heads in [12, 8, 6, 4, 3, 2, 1]:
                if self.hidden_size % heads == 0:
                    self.num_attention_heads = heads
                    print(f"WARNING: Adjusted num_attention_heads to {heads} to be compatible with hidden_size={self.hidden_size}")
                    break