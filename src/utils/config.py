# src/utils/config.py
from dataclasses import dataclass  

@dataclass
class LlamaABSAConfig:
    model_name: str = "meta-llama/Llama-2-7b"
    experiment_name: str = "llama-absa-baseline"
    learning_rate: float = 1e-4
    max_span_length: int = 128
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 16
    num_epochs: int = 20
    num_workers: int = 4
    viz_interval: int = 5
    train_path: str = "data/train.txt"
    val_path: str = "data/val.txt"
    experiment_name: str = "llama-absa-baseline"