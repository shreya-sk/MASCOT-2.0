# src/utils/logger.py
import wandb
import os
from typing import Dict, Any, Optional

class WandbLogger:
    """Weights & Biases logger for experiment tracking with improved error handling"""
    
    def __init__(self, config: Any, use_wandb: bool = True):
        self.config = config
        self.use_wandb = use_wandb
        self.run = None
        
        if not self.use_wandb:
            print("Wandb logging disabled, running with local logging only")
            return
            
        try:
            # Try to initialize wandb
            print("Initializing wandb logger...")
            
            # Convert config to dict if it's not already
            config_dict = config.__dict__ if hasattr(config, "__dict__") else config
            
            # Initialize wandb without the timeout parameter
            self.run = wandb.init(
                project="absa-llama",
                config=config_dict,
                name=config.experiment_name if hasattr(config, "experiment_name") else "absa-experiment"
            )
            print("Successfully initialized wandb logger")
        except Exception as e:
            print(f"WARNING: Failed to initialize wandb: {e}")
            print("Continuing with training, but metrics won't be logged to wandb")
            self.use_wandb = False
        
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log metrics to W&B"""
        # Always print metrics to console for tracking
        print(f"Step {step} metrics:", ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]))
        
        # Log to wandb if available
        if self.use_wandb and self.run is not None:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"WARNING: Failed to log metrics to wandb: {e}")
        
    def log_model(self, model, metrics: Dict[str, Any]):
        """Save model checkpoint with metrics"""
        if not self.use_wandb or self.run is None:
            print("Skipping wandb model logging (wandb not available)")
            return
            
        try:
            artifact = wandb.Artifact(
                name=f"model-{self.run.id}",
                type="model",
                metadata=metrics
            )
            artifact.add_dir("checkpoints")
            self.run.log_artifact(artifact)
        except Exception as e:
            print(f"WARNING: Failed to log model to wandb: {e}")
        
    def finish(self):
        """End the W&B run"""
        if self.use_wandb and self.run is not None:
            try:
                wandb.finish()
            except Exception as e:
                print(f"WARNING: Failed to finish wandb run: {e}")