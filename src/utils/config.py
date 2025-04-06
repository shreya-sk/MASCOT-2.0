# src/utils/config.py
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
    
    # Architecture settings
    hidden_size: int = 768  # Size after projection (can be smaller than model)
    num_layers: int = 2
    num_attention_heads: int = 12
    dropout: float = 0.1
    
    # Memory optimization
    use_8bit: bool = False  # Use 8-bit quantization (or 4-bit if False)
    use_fp16: bool = True  # Use mixed precision
    gradient_checkpointing: bool = True
    
    # Loss weights
    aspect_loss_weight: float = 1.0
    opinion_loss_weight: float = 1.0
    sentiment_loss_weight: float = 1.0
    
    # Training settings
    learning_rate: float = 5e-5  # Lower learning rate for Llama fine-tuning
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 8  # Smaller batch size for large model
    gradient_accumulation_steps: int = 4  # Gradient accumulation for larger effective batch
    num_epochs: int = 10
    num_workers: int = 4
    
    # Logging settings
    experiment_name: str = "llama-absa"
    viz_interval: int = 5
    
    # Data settings
    datasets: List[str] = field(default_factory=lambda: ["laptop14", "rest14", "rest15", "rest16"])
    dataset_paths: Optional[Dict[str, str]] = None
    max_span_length: int = 128
    
    def __post_init__(self):
        """Initialize dataset paths and validate settings"""
        # Initialize dataset paths
        if self.dataset_paths is None:
            self.dataset_paths = {
                dataset: f"Dataset/aste/{dataset}" 
                for dataset in self.datasets
            }
        
        # Validate settings
        if self.batch_size > 8 and not self.use_8bit and self.hidden_size > 1024:
            print("Warning: Reducing batch size to 8 to avoid OOM with large model")
            self.batch_size = 8
            
        # Check if we're actually using local files
        if self.use_local and not os.path.exists(self.model_name):
            print(f"Warning: Local model path {self.model_name} not found")
            print("Checking for model files in current directory...")
            
            if os.path.exists("Llama-3.3-70B-Instruct"):
                print("Found Llama-3.3-70B-Instruct directory, using it")
                self.model_name = "Llama-3.3-70B-Instruct"
            else:
                print("No local model files found, will try to download from Hugging Face")
                self.use_local = False