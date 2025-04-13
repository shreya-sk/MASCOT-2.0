# src/utils/config.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
# In src/utils/config.py - modify the config class
class LLMABSAConfig:
<<<<<<< Updated upstream
    # Model settings - Update hidden_size to match TinyLlama
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hidden_size: int = 2048  # Match TinyLlama's hidden size
    num_layers: int = 2
    dropout: float = 0.1
    num_attention_heads: int = 12
=======
    # Model settings - Using a tiny model to save memory
    model_name: str = "prajjwal1/bert-tiny"  # Tiny 4.4M parameter model
    hidden_size: int = 128  # Small hidden size
    num_layers: int = 2
    dropout: float = 0.1
    num_attention_heads: int = 4  # Should divide hidden_size evenly
>>>>>>> Stashed changes
    
    # Generation settings
    num_decoder_layers: int = 1  # Reduced from 2
    vocab_size: int = 30522  # BERT vocabulary size
    max_generation_length: int = 32  # Reduced from 64
    use_generation: bool = True
<<<<<<< Updated upstream
        # Generative parameters
    generate_explanations: bool = True  # Set to True for training with generation
    generation_weight: float = 0.5  # Weight for generation loss
    
    
    freeze_layers: bool = True 
=======
    # Generative parameters
    generate_explanations: bool = True
    generation_weight: float = 0.5
    
    # Freeze base model to save memory
    freeze_layers: bool = True
>>>>>>> Stashed changes
    
    # Training settings
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 4  # Small batch size
    num_epochs: int = 5  # Reduced epochs for testing
    gradient_accumulation_steps: int = 4  # Use gradient accumulation
    
    # Domain adaptation
    domain_adaptation: bool = False
    domain_mapping: Dict[str, int] = None  # Will be initialized in __post_init__
    
    # Data settings
    max_seq_length: int = 64  # Reduced from 128
    datasets: List[str] = None  # Will be initialized in __post_init__
    dataset_paths: Dict[str, str] = None  # Will be initialized in __post_init__
    
    # Architectural choices
    use_syntax: bool = False  # Disabled to simplify
    use_fp16: bool = False  # Disabled mixed precision
    confidence_threshold: float = 0.5
    
    # Loss weights
    aspect_loss_weight: float = 1.0
    opinion_loss_weight: float = 1.0
    sentiment_loss_weight: float = 1.0
    
    # Logging settings
    experiment_name: str = "minimal-absa"
    viz_interval: int = 5
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    
    # Other settings
    num_workers: int = 0  # No multiprocessing
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
<<<<<<< Updated upstream
            }
=======
            }
        
        # Ensure num_attention_heads divides hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            # Find closest number of heads that divides hidden_size
            for heads in [8, 4, 2, 1]:
                if self.hidden_size % heads == 0:
                    self.num_attention_heads = heads
                    print(f"Adjusted num_attention_heads to {heads} to be compatible with hidden_size={self.hidden_size}")
                    break
>>>>>>> Stashed changes
