# src/utils/config.py
@dataclass
class LlamaABSAConfig:
    model_name: str = "meta-llama/Llama-2-7b"
    hidden_size: int = 768
    max_span_length: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000