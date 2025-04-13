# src/utils/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class LLMABSAConfig:
    """
    Ultra-lightweight configuration for ABSA on severely constrained hardware
    
    This configuration is designed to run on systems with minimal resources,
    such as older laptops or low-end cloud instances.
    """
    # Use tiny models
    model_name: str = "prajjwal1/bert-tiny"  # 4.4M parameter model
    hidden_size: int = 128  # Very small hidden size
    num_layers: int = 1  # Single layer
    dropout: float = 0.1
    num_attention_heads: int = 4  # Should divide hidden_size evenly
    
    # No generation for maximum efficiency
    generate_explanations: bool = True  
    
    # Memory efficiency settings
    use_quantization: bool = False  # Skip quantization as it may cause errors
    freeze_layers: bool = True  # Freeze most encoder layers
    use_gradient_checkpointing: bool = False  # Skip gradient checkpointing
    use_gru: bool = True  # Use GRU instead of LSTM for memory efficiency
    
    # Simplified architecture
    use_focal_loss: bool = False  # Use simple cross-entropy loss
    use_boundary_loss: bool = False  # Skip boundary refinement
    use_contrastive_verification: bool = False  # Skip contrastive verification
    use_syntax: bool = False  # Skip syntax information
    
    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 8  # Small batch size
    num_epochs: int = 5
    gradient_accumulation_steps: int = 4  # Use gradient accumulation
    
    # Simplify domain handling
    domain_adaptation: bool = False
    domain_mapping: Dict[str, int] = field(default_factory=lambda: {
        "laptop14": 0,
        "rest14": 1,
        "rest15": 1,
        "rest16": 1,
    })
    
    # Data settings
    max_seq_length: int = 64  # Short sequences
    datasets: List[str] = field(default_factory=lambda: ["laptop14", "rest14", "rest15", "rest16"])
    dataset_paths: Dict[str, str] = field(default_factory=lambda: {})
    
    # Optimizer settings
    use_fp16: bool = False  # Skip mixed precision
    max_grad_norm: float = 1.0
    
    # Loss weights
    aspect_loss_weight: float = 1.0
    opinion_loss_weight: float = 1.0
    sentiment_loss_weight: float = 1.0
    
    # Shared projection to save parameters
    shared_projection: bool = True
    
    # Logging settings
    experiment_name: str = "ultra-lightweight-absa"
    viz_interval: int = 10
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    
    def __post_init__(self):
        # Initialize default values for paths
        if not self.dataset_paths:
            # Set default dataset paths
            self.dataset_paths = {
                dataset: f"Datasets/aste/{dataset}" 
                for dataset in self.datasets
            }
            
        # Ensure num_attention_heads divides hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            # Find closest number of heads that divides hidden_size
            for heads in [4, 2, 1]:
                if self.hidden_size % heads == 0:
                    self.num_attention_heads = heads
                    print(f"Adjusted num_attention_heads to {heads} to be compatible with hidden_size={self.hidden_size}")
                    break
        
        # Ensure consistent batch size with gradient accumulation
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        print(f"Effective batch size: {self.effective_batch_size}")

@dataclass
class MemoryConstrainedConfig(LLMABSAConfig):
    """Alias for the base config for compatibility"""
    pass