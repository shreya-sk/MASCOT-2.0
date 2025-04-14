# src/utils/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class LLMABSAConfig:
    """
    Improved memory-efficient configuration for ABSA that balances
    performance and resource constraints
    
    This configuration is designed to run on systems with minimal resources,
    while providing better performance than the ultra-lightweight config.
    """
    # IMPROVED: Better model with best performance-to-memory ratio
    model_name: str = "microsoft/deberta-v3-small"  # Better than BERT, similar memory footprint
    hidden_size: int = 384  # Matches DeBERTa-v3-small's hidden size
    num_layers: int = 2  # Use only 2 layers to save memory
    dropout: float = 0.2  # Moderate dropout for regularization
    num_attention_heads: int = 6  # Should divide hidden_size evenly
    
    # Enable features that help performance without significant memory cost
    use_focal_loss: bool = True  # Enable focal loss for class imbalance
    use_boundary_loss: bool = True  # Enable boundary refinement
    generate_explanations: bool = True  # Enable explanation generation
    
    # Memory efficiency settings
    use_quantization: bool = False  # Skip quantization as it may cause errors
    freeze_layers: bool = True  # Freeze most encoder layers
    use_gradient_checkpointing: bool = False  # Skip gradient checkpointing
    use_gru: bool = True  # Use GRU instead of LSTM for memory efficiency
    
    # Enable all core features
    use_focal_loss: bool = True  # Handle class imbalance
    use_boundary_loss: bool = True  # Improve span boundary detection
    use_contrastive_verification: bool = False  # Skip contrastive verification (memory intensive)
    use_syntax: bool = False  # Skip syntax information (memory intensive)
    
    # Training settings
    weight_decay: float = 0.01  # Standard weight decay
    warmup_ratio: float = 0.1  # Warm up for 10% of training steps
    learning_rate: float = 3e-4  # Conservative learning rate
    batch_size: int = 8  # Small batch size for memory efficiency
    num_epochs: int = 15  # Train for more epochs
    gradient_accumulation_steps: int = 4  # Use gradient accumulation
    
    # Simplified domain handling
    domain_adaptation: bool = False  # Disable domain adaptation
    domain_mapping: Dict[str, int] = field(default_factory=lambda: {
        "laptop14": 0,
        "rest14": 1,
        "rest15": 1,
        "rest16": 1,
    })
    
    # Data settings
    max_seq_length: int = 96  # Moderate sequence length
    datasets: List[str] = field(default_factory=lambda: ["laptop14", "rest14", "rest15", "rest16"])
    dataset_paths: Dict[str, str] = field(default_factory=lambda: {})
    
    # Optimizer settings
    use_fp16: bool = False  # Skip mixed precision
    max_grad_norm: float = 1.0  # Standard gradient clipping
    
    # Loss weights
    aspect_loss_weight: float = 1.5  # Emphasis on aspect extraction
    opinion_loss_weight: float = 1.5  # Emphasis on opinion extraction
    sentiment_loss_weight: float = 1.0  # Standard sentiment weight
    
    # Shared projection to save parameters
    shared_projection: bool = True  # Use shared projection
    
    # Logging settings
    experiment_name: str = "improved-absa"  # New experiment name
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
            for heads in [6, 4, 3, 2, 1]:
                if self.hidden_size % heads == 0:
                    self.num_attention_heads = heads
                    print(f"Adjusted num_attention_heads to {heads} to be compatible with hidden_size={self.hidden_size}")
                    break
        
        # Ensure consistent batch size with gradient accumulation
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        print(f"Effective batch size: {self.effective_batch_size}")

@dataclass
class MemoryConstrainedConfig(LLMABSAConfig):
    """
    Super lightweight configuration for extremely constrained environments
    
    This configuration uses the smallest possible models and disables
    memory-intensive features.
    """
    # Use tiny models
    model_name: str = "prajjwal1/bert-tiny"  # Absolute minimum model size
    hidden_size: int = 128  # Small hidden size
    num_layers: int = 1  # Single layer
    dropout: float = 0.1  # Light dropout
    num_attention_heads: int = 2  # Minimum heads
    
    # Disable memory-intensive features
    use_focal_loss: bool = False
    use_boundary_loss: bool = False
    use_contrastive_verification: bool = False
    use_syntax: bool = False
    
    # Minimal batch size
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    
    # Short sequences
    max_seq_length: int = 64
    
    # Simpler training
    learning_rate: float = 5e-4
    num_epochs: int = 8
    
    # Update experiment name
    experiment_name: str = "ultra-lightweight-absa"

@dataclass
class HighPerformanceConfig(LLMABSAConfig):
    """
    Configuration for higher performance when more memory is available
    
    This configuration uses a stronger model and enables more features,
    but requires more memory.
    """
    # Stronger model
    model_name: str = "microsoft/deberta-v3-base"  # Better model
    hidden_size: int = 768  # Match model hidden size
    num_layers: int = 3  # Use more layers
    dropout: float = 0.1  # Lighter dropout for better models
    num_attention_heads: int = 12  # Match model
    
    # Enable all features
    use_focal_loss: bool = True
    use_boundary_loss: bool = True
    use_contrastive_verification: bool = True  # Enable contrastive verification
    use_syntax: bool = True  # Enable syntax information
    
    # Less freezing
    freeze_layers: bool = 0.6  # Only freeze 60% of layers
    
    # Larger batch size
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    
    # Longer sequences
    max_seq_length: int = 128
    
    # More advanced training
    learning_rate: float = 2e-5
    use_fp16: bool = True  # Enable mixed precision
    
    # Update experiment name
    experiment_name: str = "high-performance-absa"