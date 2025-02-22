# src/utils/logger.py
import wandb
from typing import Dict, Any

class WandbLogger:
    """Weights & Biases logger for experiment tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.run = wandb.init(
            project="absa-llama",
            config=config,
            name=config.experiment_name
        )
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to W&B"""
        wandb.log(metrics, step=step)
        
    def log_model(self, model, metrics: Dict[str, float]):
        """Save model checkpoint with metrics"""
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            metadata=metrics
        )
        artifact.add_dir("checkpoints")
        self.run.log_artifact(artifact)
        
    def finish(self):
        """End the W&B run"""
        wandb.finish()