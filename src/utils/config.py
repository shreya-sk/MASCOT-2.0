# src/utils/config.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class LLMABSAConfig:
    """
    Complete and robust configuration for ABSA training
    Optimized for stability and performance
    """
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    
    # Base model settings
    model_name: str = "microsoft/deberta-v3-base"  # Stable, well-tested model
    hidden_size: int = 768  # Match DeBERTa-v3-base
    num_layers: int = 2  # Lightweight for efficiency
    dropout: float = 0.2  # Moderate dropout for regularization
    num_attention_heads: int = 12  # Match DeBERTa default
    
    # Embedding configuration
    embedding_size: int = 768  # Will be set automatically
    freeze_layers: float = 0.5  # Freeze 50% of encoder layers
    use_quantization: bool = False  # Disabled for stability
    shared_projection: bool = True  # Share projections for efficiency

    use_instruction_following: bool = True
    instruction_model: str = "t5-small"  # Start small for testing
    max_instruction_length: int = 512
    max_generation_length: int = 256
    
    # Generation settings
    num_beams: int = 3
    early_stopping: bool = True
    
    # Training weights
    extraction_weight: float = 1.0  # Weight for your existing model
    generation_weight: float = 0.5  # Weight for instruction following
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    
    # Learning settings
    learning_rate: float = 2e-5  # Conservative learning rate
    weight_decay: float = 0.01  # Standard weight decay
    warmup_ratio: float = 0.1  # 10% warmup
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Batch settings
    batch_size: int = 8  # Conservative for memory
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    effective_batch_size: int = 32  # Will be calculated
    
    # Training dynamics
    num_epochs: int = 15  # Sufficient for convergence
    early_stopping_patience: int = 3  # Early stopping
    scheduler_type: str = "linear"  # Linear warmup + decay
    
    # Memory optimization
    use_fp16: bool = False  # Disabled for stability
    use_gradient_checkpointing: bool = True  # Save memory
    
    # ============================================================================
    # LOSS FUNCTION CONFIGURATION
    # ============================================================================
    
    # Loss weights
    aspect_loss_weight: float = 2.0  # Emphasize aspect detection
    opinion_loss_weight: float = 2.0  # Emphasize opinion detection
    sentiment_loss_weight: float = 1.0  # Standard sentiment weight
    boundary_weight: float = 0.5  # Boundary refinement weight
    
    # Loss function settings
    use_focal_loss: bool = True  # Better for imbalanced data
    focal_gamma: float = 2.0  # Focal loss parameter
    label_smoothing: float = 0.1  # Label smoothing for regularization
    
    # ============================================================================
    # DATA CONFIGURATION
    # ============================================================================
    
    # Sequence settings
    max_seq_length: int = 128  # Reasonable sequence length
    context_window: int = 2  # Context enhancement window
    
    # Dataset settings
    datasets: List[str] = field(default_factory=lambda: ["laptop14", "rest14", "rest15", "rest16"])
    dataset_paths: Dict[str, str] = field(default_factory=dict)
    
    # Data augmentation (DISABLED for stability)
    use_augmentation: bool = False  # Disable augmentation
    use_mixup: bool = False  # Disable mixup
    mixup_alpha: float = 0.2  # Mixup parameter (unused)
    
    # ============================================================================
    # FEATURE CONFIGURATION
    # ============================================================================
    
    # Advanced features (DISABLED for initial training)
    use_syntax: bool = True  # Enable syntax features
    use_boundary_loss: bool = True  # Enable boundary refinement
    use_contrastive_verification: bool = False  # DISABLED
    domain_adaptation: bool = False  # DISABLED
    generate_explanations: bool = False  # DISABLED initially
    
    # Domain adaptation (unused but defined)
    domain_mapping: Dict[str, int] = field(default_factory=lambda: {
        "laptop14": 0,
        "rest14": 1, 
        "rest15": 1,
        "rest16": 1,
    })
    
    # ============================================================================
    # GENERATION CONFIGURATION (for future use)
    # ============================================================================
    
    # Generation settings
    use_beam_search: bool = True
    num_beams: int = 3
    max_explanation_length: int = 64
    explanation_diversity: bool = True
    
    # ============================================================================
    # LOGGING AND MONITORING
    # ============================================================================
    
    # Experiment settings
    experiment_name: str = "improved-absa-stable"
    log_interval: int = 10  # Log every 10 batches
    eval_interval: int = 50  # Evaluate every 50 batches
    save_interval: int = 100  # Save every 100 batches
    viz_interval: int = 10  # Visualization interval
    
    # ============================================================================
    # SYSTEM CONFIGURATION
    # ============================================================================
    
    # Hardware settings
    num_workers: int = 0  # DataLoader workers (0 for debugging)
    pin_memory: bool = True  # Pin memory for GPU
    
    # Reproducibility
    seed: int = 42  # Random seed
    deterministic: bool = True  # Deterministic operations
    
    # ============================================================================
    # VALIDATION AND TESTING
    # ============================================================================
    
    # Evaluation settings
    confidence_threshold: float = 0.6  # Confidence threshold for predictions
    min_span_length: int = 1  # Minimum span length
    max_span_length: int = 10  # Maximum span length
    
    # Metrics configuration
    use_strict_evaluation: bool = True  # Strict span matching
    eval_batch_size: int = 16  # Batch size for evaluation
    
    def __post_init__(self):
        """Initialize derived values and validate configuration"""
        
        # ========================================================================
        # AUTOMATIC CONFIGURATION
        # ========================================================================
        
        # Calculate effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        
        # Set embedding size to match hidden size if not set
        if not hasattr(self, 'embedding_size') or self.embedding_size is None:
            self.embedding_size = self.hidden_size
        
        # Initialize dataset paths if empty
        if not self.dataset_paths:
            self.dataset_paths = {
                dataset: f"Datasets/aste/{dataset}" 
                for dataset in self.datasets
            }
        
        # ========================================================================
        # COMPATIBILITY CHECKS
        # ========================================================================
        
        # Ensure num_attention_heads divides hidden_size evenly
        if self.hidden_size % self.num_attention_heads != 0:
            # Find the largest divisor that works
            for heads in [12, 8, 6, 4, 3, 2, 1]:
                if self.hidden_size % heads == 0:
                    original_heads = self.num_attention_heads
                    self.num_attention_heads = heads
                    print(f"⚠ Adjusted attention heads: {original_heads} → {heads} (divisible by hidden_size={self.hidden_size})")
                    break
        
        # ========================================================================
        # VALIDATION
        # ========================================================================
        
        # Validate learning rate
        if self.learning_rate <= 0 or self.learning_rate > 1e-2:
            print(f"⚠ Learning rate {self.learning_rate} may be too high/low")
        
        # Validate batch size
        if self.effective_batch_size > 128:
            print(f"⚠ Large effective batch size ({self.effective_batch_size}) may slow training")
        
        # Validate sequence length
        if self.max_seq_length > 512:
            print(f"⚠ Long sequences ({self.max_seq_length}) require more memory")
        
        # ========================================================================
        # FEATURE COMPATIBILITY
        # ========================================================================
        
        # Disable conflicting features in stable mode
        if not self.use_fp16:
            # FP16 disabled, ensure compatibility
            pass
        
        if self.use_quantization and self.use_fp16:
            print("⚠ Quantization + FP16 may cause issues, consider disabling one")
        
        # ========================================================================
        # DATASET VALIDATION
        # ========================================================================
        
        # Check if dataset paths exist
        import os
        missing_datasets = []
        for dataset, path in self.dataset_paths.items():
            train_file = os.path.join(path, "train.txt")
            if not os.path.exists(train_file):
                missing_datasets.append(dataset)
        
        if missing_datasets:
            print(f"⚠ Missing datasets: {missing_datasets}")
            # Remove missing datasets from the list
            self.datasets = [d for d in self.datasets if d not in missing_datasets]
            if not self.datasets:
                print("❌ No valid datasets found!")
        
        # ========================================================================
        # MEMORY OPTIMIZATION
        # ========================================================================
        
        # Automatic memory optimization based on settings
        estimated_memory = self._estimate_memory_usage()
        if estimated_memory > 8:  # >8GB
            print(f"⚠ Estimated memory usage: {estimated_memory:.1f}GB")
            print("💡 Consider reducing batch_size or hidden_size")
        
        # ========================================================================
        # SUMMARY
        # ========================================================================
        
        print("\n" + "="*50)
        print("CONFIGURATION SUMMARY")
        print("="*50)
        print(f"Model: {self.model_name}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Batch Size: {self.batch_size} (effective: {self.effective_batch_size})")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Max Sequence Length: {self.max_seq_length}")
        print(f"Datasets: {len(self.datasets)} ({', '.join(self.datasets)})")
        print(f"Epochs: {self.num_epochs}")
        print(f"Features: Syntax={self.use_syntax}, Focal={self.use_focal_loss}")
        print(f"Estimated Memory: {estimated_memory:.1f}GB")
        print("="*50)
    
    def _estimate_memory_usage(self) -> float:
        """Estimate GPU memory usage in GB"""
        try:
            # Base model memory (rough estimates)
            model_params = {
                "microsoft/deberta-v3-base": 140,  # Million parameters
                "bert-base-uncased": 110,
                "roberta-base": 125,
            }
            
            # Get model parameter count (in millions)
            base_params = model_params.get(self.model_name, 120)  # Default estimate
            
            # Memory per parameter (FP32 = 4 bytes, FP16 = 2 bytes)
            bytes_per_param = 2 if self.use_fp16 else 4
            
            # Model memory
            model_memory = base_params * 1e6 * bytes_per_param / 1e9  # GB
            
            # Activation memory (depends on batch size and sequence length)
            activation_memory = (
                self.effective_batch_size * 
                self.max_seq_length * 
                self.hidden_size * 
                4 * bytes_per_param / 1e9
            )
            
            # Gradient memory (same as model for training)
            gradient_memory = model_memory
            
            # Optimizer memory (Adam uses 2x model params)
            optimizer_memory = model_memory * 2
            
            # Buffer memory
            buffer_memory = 1.0  # 1GB buffer
            
            total_memory = (
                model_memory + 
                activation_memory + 
                gradient_memory + 
                optimizer_memory + 
                buffer_memory
            )
            
            return total_memory
            
        except Exception:
            return 8.0  # Default estimate
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'dropout': self.dropout,
            'max_position_embeddings': self.max_seq_length,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'num_epochs': self.num_epochs,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'max_grad_norm': self.max_grad_norm,
        }
    
    def get_loss_config(self) -> Dict[str, Any]:
        """Get loss function configuration"""
        return {
            'aspect_loss_weight': self.aspect_loss_weight,
            'opinion_loss_weight': self.opinion_loss_weight,
            'sentiment_loss_weight': self.sentiment_loss_weight,
            'boundary_weight': self.boundary_weight,
            'use_focal_loss': self.use_focal_loss,
            'focal_gamma': self.focal_gamma,
            'label_smoothing': self.label_smoothing,
        }
    
    def validate_gpu_compatibility(self) -> bool:
        """Check if configuration is compatible with available GPU"""
        try:
            import torch
            if not torch.cuda.is_available():
                print("⚠ No GPU available, training will be slow")
                return False
            
            # Get GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            estimated_usage = self._estimate_memory_usage()
            
            if estimated_usage > gpu_memory * 0.9:  # Use 90% as safety margin
                print(f"❌ Estimated memory ({estimated_usage:.1f}GB) exceeds GPU memory ({gpu_memory:.1f}GB)")
                print("💡 Suggestions:")
                print(f"   - Reduce batch_size to {max(1, self.batch_size // 2)}")
                print(f"   - Enable FP16 training")
                print(f"   - Reduce max_seq_length to {self.max_seq_length // 2}")
                return False
            else:
                print(f"✓ GPU memory check passed ({estimated_usage:.1f}GB / {gpu_memory:.1f}GB)")
                return True
                
        except Exception as e:
            print(f"Warning: Could not validate GPU compatibility: {e}")
            return True  # Assume compatible if we can't check
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, (list, dict, str, int, float, bool)):
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMABSAConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LLMABSAConfig':
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)