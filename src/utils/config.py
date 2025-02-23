# # src/utils/config.py
from dataclasses import dataclass
from typing import List
@dataclass
class LlamaABSAConfig:
    # Model settings
    model_name: str = "meta-llama/Llama-2-7b"
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1

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

    # Data settings
    datasets: List[str] = ("laptop14", "rest14", "rest15", "rest16")  # All available datasets
    dataset_paths: dict = None  # Will be initialized in __post_init__
    max_span_length: int = 128
    
    def __post_init__(self):
        # Initialize paths for all datasets
        self.dataset_paths = {
            dataset: f"Dataset/aste/{dataset}" 
            for dataset in self.datasets
        }
