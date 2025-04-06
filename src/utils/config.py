from dataclasses import dataclass
from typing import List, Dict, Optional
import os
@dataclass
class LlamaABSAConfig:
    """Configuration for Llama-based ABSA model"""

    
    # Architecture settings
    hidden_size: int = 2048  # Reduced for testing
    num_attention_heads: int = 32
    max_seq_length: int = 512
    dropout: float = 0.1
    

    #model_path: str = "./Llama-3.3-70B-Instruct"
    tokenizer_path: str = "./Llama-3.3-70B-Instruct/original/tokenizer.model"
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    use_local: bool = False
    auth_token: str = os.environ.get("HF_TOKEN", None)



    
    
    # Optimization settings
    use_8bit: bool = False
    use_fp16: bool = True  
   
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    
    # Training settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 16
    num_epochs: int = 20
    num_workers: int = 4
    
    # Logging settings
    experiment_name: str = "llama-absa-baseline"
    viz_interval: int = 5
    
    # Dataset settings
    datasets: List[str] = ("laptop14", "rest14", "rest15", "rest16")
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
        if self.batch_size > 32 and not self.use_8bit:
            print("Warning: Large batch size without 8-bit quantization may cause OOM")
            
        if self.max_seq_length > 2048:
            print("Warning: Very large sequence length may impact performance")