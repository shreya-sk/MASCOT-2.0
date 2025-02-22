# src/utils/logger.py
# src/utils/logger.py
import wandb
from typing import Dict, Any
from src.utils.config import LlamaABSAConfig  # Add this import

class WandbLogger:
    """Enhanced Weights & Biases logger for ABSA"""
    
    def __init__(self, config: LlamaABSAConfig):  # Change type hint to LlamaABSAConfig
        """Initialize WandB run with config"""
        self.run = wandb.init(
            project="absa-llama",
            name=config.experiment_name,
            config={
                "model_name": config.model_name,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "warmup_steps": config.warmup_steps,
                "max_length": config.max_span_length,
                "hidden_size": config.hidden_size,
                "architecture": "LlamaABSA"
            }
        )
        
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log training/validation metrics"""
        wandb.log(metrics, step=step)
        
    def log_model(self, model, metrics: Dict[str, float]):
        """Log model checkpoint with metrics"""
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            metadata=metrics
        )
        # Save model checkpoint
        artifact.add_file("checkpoints/best_model.pt")
        self.run.log_artifact(artifact)
        
    def finish(self):
        """End the WandB run"""
        wandb.finish()