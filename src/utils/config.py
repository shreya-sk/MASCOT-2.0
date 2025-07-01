# src/utils/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class LLMABSAConfig:
    """
    Improved memory-efficient configuration for ABSA that addresses overfitting
    and enhances generation capabilities
    """
    # Model configuration
    model_name: str = "microsoft/deberta-v3-base"  # Use stronger model
    hidden_size: int = 768  # Match DeBERTa-v3-base's hidden size
    num_layers: int = 2  # Keep 2 layers for efficiency
    dropout: float = 0.3  # Increased dropout for better regularization
    num_attention_heads: int = 12  # Match model's attention heads
    
    # Memory efficiency settings
    use_quantization: bool = False  # Skip quantization as it may cause errors
    freeze_layers: float = 0.6  # Freeze 60% of encoder layers
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    use_gru: bool = True  # Use GRU instead of LSTM for memory efficiency
    shared_projection: bool = True  # Use shared projection to save parameters
    
    # Regularization
    weight_decay: float = 0.01  # Standard weight decay
    use_layernorm: bool = True  # Add layer normalization
    use_gradient_clipping: bool = True  # Enable gradient clipping
    max_grad_norm: float = 1.0  # Set max gradient norm
    spectral_norm: bool = True  # Apply spectral normalization
    label_smoothing: float = 0.1  # Label smoothing factor
    
    # Training dynamics
    learning_rate: float = 2e-5  # Lower learning rate
    scheduler_type: str = "cosine"  # Use cosine scheduler
    warmup_ratio: float = 0.1  # Warmup for 10% of steps
    batch_size: int = 8  # Reasonable batch size
    num_epochs: int = 15  # Train for more epochs
    gradient_accumulation_steps: int = 4  # Use gradient accumulation
    early_stopping_patience: int = 3  # Early stopping patience
    
    # Loss function
    aspect_loss_weight: float = 2.0  # Emphasis on aspect extraction
    opinion_loss_weight: float = 2.0  # Emphasis on opinion extraction
    sentiment_loss_weight: float = 1.0  # Standard sentiment weight
    boundary_weight: float = 0.5  # Weight for boundary refinement
    focal_gamma: float = 2.0  # Focal loss gamma parameter
    
    # Data augmentation - DISABLE mixup to avoid errors
    use_mixup: bool = False  # Disable mixup augmentation
    mixup_alpha: float = 0.2  # Mixup parameter
    use_augmentation: bool = True  # Enable data augmentation
    
    # Generation settings
    generate_explanations: bool = True  # Enable explanation generation
    use_beam_search: bool = True  # Use beam search for generation
    num_beams: int = 3  # Number of beams
    max_explanation_length: int = 64  # Reasonable explanation length
    explanation_diversity: bool = True  # Enable diverse explanations
    
    # Features control
    use_focal_loss: bool = True  # Enable focal loss
    use_boundary_loss: bool = True  # Enable boundary refinement
    use_contrastive_verification: bool = False  # Skip contrastive verification
    use_syntax: bool = True  # Enable syntax information
    
    # Domain adaptation
    domain_adaptation: bool = False  # Disable domain adaptation
    domain_mapping: Dict[str, int] = field(default_factory=lambda: {
        "laptop14": 0,
        "rest14": 1,
        "rest15": 1,
        "rest16": 1,
    })
    
    # Data settings
    max_seq_length: int = 128  # Moderate sequence length
    datasets: List[str] = field(default_factory=lambda: ["laptop14", "rest14", "rest15", "rest16"])
    dataset_paths: Dict[str, str] = field(default_factory=lambda: {})
    
    # Mixed precision
    use_fp16: bool = False  # Skip mixed precision
    
    # Logging settings
    experiment_name: str = "improved-absa-v2"  # New experiment name
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
            for heads in [12, 8, 6, 4, 3, 2, 1]:
                if self.hidden_size % heads == 0:
                    self.num_attention_heads = heads
                    print(f"Adjusted num_attention_heads to {heads} to be compatible with hidden_size={self.hidden_size}")
                    break
        
        # Ensure consistent batch size with gradient accumulation
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        print(f"Effective batch size: {self.effective_batch_size}")