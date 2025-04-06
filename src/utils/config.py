# src/utils/config.py
from dataclasses import dataclass, field
# src/utils/stella_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os

@dataclass
class LlamaABSAConfig:
    """Configuration for Llama-based ABSA model"""
    
    # Model settings
    model_name: str = "Llama-3.3-70B-Instruct"  # Path or name of model
    use_local: bool = True  # Use local model files
    # Add these lines to LlamaABSAConfig class
    use_online_model: bool = True
    model_name: str = "meta-llama/Llama-3-8B-Instruct"  # A smaller model is more responsive
    hf_api_token: str = os.environ.get("HF_TOKEN", None)
class StellaABSAConfig:
    """Configuration for Stella-based ABSA model"""
    
    # Model architecture
    model_name: str = "stanford-crfm/Stella-400M-v5"
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_layers: int = 3
    dropout: float = 0.1
    max_seq_length: int = 512
    
    # Novel architecture components
    use_aspect_first: bool = True  # Whether to prioritize aspect over opinion in joint classification
    use_syntax: bool = True  # Whether to use syntax-guided attention 
    freeze_layers: int = 8  # Number of layers to freeze in the base model
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 8  # Smaller batch size for large model
    gradient_accumulation_steps: int = 4  # Gradient accumulation for larger effective batch
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    batch_size: int = 16
    num_epochs: int = 10
    num_workers: int = 4
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # FP16 and optimization settings
    use_fp16: bool = True
    use_8bit: bool = False  # Whether to use 8-bit quantization (bitsandbytes)
    use_gradient_checkpointing: bool = True
    
    # Loss weights
    aspect_loss_weight: float = 1.0
    opinion_loss_weight: float = 1.0
    sentiment_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.5  # Weight for aspect-opinion consistency
    
    # Logging settings
    experiment_name: str = "llama-absa"
    viz_interval: int = 5
    experiment_name: str = "stella-absa-v5"
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500
    
    # Data settings
    datasets: List[str] = field(default_factory=lambda: ["laptop14", "rest14", "rest15", "rest16"])
    dataset_paths: Optional[Dict[str, str]] = None
    max_span_length: int = 128
    data_dir: str = "Datasets/aste"
    
    # Domain adaptation settings
    domain_adaptation: bool = True
    domain_mapping: Dict[str, int] = field(default_factory=lambda: {
        "laptop14": 0,
        "rest14": 1,
        "rest15": 1,
        "rest16": 1
    })
    
    # Inference settings
    confidence_threshold: float = 0.7  # Threshold for prediction confidence
    
    def __post_init__(self):
        """Initialize dataset paths and validate settings"""
        # Initialize dataset paths
        if self.dataset_paths is None:
            self.dataset_paths = {
                dataset: os.path.join(self.data_dir, dataset)
                for dataset in self.datasets
            }
        
        # Validate settings
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )
            
        if self.batch_size > 32 and not self.use_8bit:
            print("Warning: Large batch size without 8-bit quantization may cause OOM errors")
            
        if self.freeze_layers < 0:
            print("Warning: Negative freeze_layers will train all parameters")
            self.freeze_layers = 0