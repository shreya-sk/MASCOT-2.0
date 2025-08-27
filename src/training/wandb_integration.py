
# src/training/wandb_integration.py
"""
Weights & Biases integration for GRADIENT ABSA model
Comprehensive experiment tracking for ACL/EMNLP submission
"""
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

import torch
import numpy as np
from typing import Dict, List, Optional, Any
import os
import json
from pathlib import Path

class WandBIntegration:
    """
    Professional W&B integration for ABSA research
    """
    
    def __init__(self, config, project_name: str = "gradient-absa-2025"):
        self.config = config
        self.project_name = project_name
        self.run = None
        
        # Initialize W&B
        self.init_wandb()
    
    def init_wandb(self):
        """Initialize Weights & Biases tracking"""
        
        # Create experiment name
        
        experiment_name = f"gradient-{getattr(self.config, 'dataset_name', 'laptop14')}-{self.config.model_name.split('/')[-1]}"
        # Extract config dict
        config_dict = self._config_to_dict()
        
        # Initialize run
        self.run = wandb.init(
            project=self.project_name,
            name=experiment_name,
            config=config_dict,
            tags=self._generate_tags(),
            notes=f"GRADIENT model training on {self.config.dataset_name}",
            save_code=True,
            group=f"gradient-{self.config.dataset_name}"
        )
        
        print(f"🔗 W&B Run: {wandb.run.url}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to W&B compatible dict"""
        config_dict = {}
        
        # Core training parameters
        config_dict.update({
            'model_name': getattr(self.config, 'model_name', 'bert-base-uncased'),
            'dataset': getattr(self.config, 'dataset_name', 'laptop14'),
            'batch_size': getattr(self.config, 'batch_size', 8),
            'learning_rate': getattr(self.config, 'learning_rate', 3e-5),
            'num_epochs': getattr(self.config, 'num_epochs', 10),
            'max_seq_length': getattr(self.config, 'max_seq_length', 128),
            'dropout': getattr(self.config, 'dropout', 0.1),
            'warmup_steps': getattr(self.config, 'warmup_steps', 100),
            'weight_decay': getattr(self.config, 'weight_decay', 0.01),
        })
        
        # GRADIENT-specific parameters
        config_dict.update({
            'use_domain_adversarial': getattr(self.config, 'use_domain_adversarial', True),
            'use_gradient_reversal': getattr(self.config, 'use_gradient_reversal', True),
            'use_implicit_detection': getattr(self.config, 'use_implicit_detection', True),
            'use_contrastive_learning': getattr(self.config, 'use_contrastive_learning', True),
            'use_few_shot_learning': getattr(self.config, 'use_few_shot_learning', True),
            'domain_lambda': getattr(self.config, 'domain_lambda', 0.1),
            'contrastive_lambda': getattr(self.config, 'contrastive_lambda', 0.1),
            'implicit_lambda': getattr(self.config, 'implicit_lambda', 0.1),
        })
        
        # Architecture parameters
        config_dict.update({
            'hidden_size': getattr(self.config, 'hidden_size', 768),
            'num_attention_heads': getattr(self.config, 'num_attention_heads', 12),
            'intermediate_size': getattr(self.config, 'intermediate_size', 3072),
            'use_orthogonal_constraints': getattr(self.config, 'use_orthogonal_constraints', True),
        })
        
        return config_dict
    
    def _generate_tags(self) -> List[str]:
        """Generate relevant tags for the experiment"""
        tags = [
            'gradient-reversal',
            'domain-adversarial', 
            'absa',
            'sentiment-analysis',
            f"dataset-{getattr(self.config, 'dataset_name', 'laptop14')}",
            'nlp-2025',
            'absa-2025'
        ]
        
        # Add feature-specific tags
        if getattr(self.config, 'use_implicit_detection', False):
            tags.append('implicit-detection')
        if getattr(self.config, 'use_contrastive_learning', False):
            tags.append('contrastive-learning')
        if getattr(self.config, 'use_few_shot_learning', False):
            tags.append('few-shot-learning')
            
        return tags
    
    def log_model_architecture(self, model):
        """Log model architecture details"""
        if self.run is None:
            return
            
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log architecture info
        wandb.log({
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/parameter_efficiency': trainable_params / total_params,
        })
        
        # Log model graph (if small enough)
        if total_params < 500_000_000:  # Only for smaller models
            try:
                # Create dummy input
                dummy_input = {
                    'input_ids': torch.randint(0, 1000, (1, 128)),
                    'attention_mask': torch.ones(1, 128),
                }
                wandb.watch(model, log="all", log_freq=100, log_graph=True)
            except Exception as e:
                print(f"Note: Could not log model graph: {e}")
    
    def log_training_step(self, epoch: int, step: int, losses: Dict[str, float], 
                         learning_rates: Dict[str, float] = None, alpha: float = None):
        """Log training step metrics"""
        if self.run is None:
            return
            
        log_dict = {
            'epoch': epoch,
            'step': step,
        }
        
        # Log losses with proper prefixes
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            log_dict[f'train/{loss_name}'] = loss_value
        
        # Log learning rates
        if learning_rates:
            for lr_name, lr_value in learning_rates.items():
                log_dict[f'lr/{lr_name}'] = lr_value
        
        # Log gradient reversal alpha
        if alpha is not None:
            log_dict['train/gradient_reversal_alpha'] = alpha
        
        wandb.log(log_dict, step=step)
    
    def log_validation_metrics(self, epoch: int, metrics: Dict[str, float], 
                             prefix: str = 'val', step: int = None):
        """Log validation/test metrics"""
        if self.run is None:
            return
            
        log_dict = {'epoch': epoch}
        
        # Log all metrics with prefix
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            log_dict[f'{prefix}/{metric_name}'] = metric_value
        
        # Special handling for key ABSA metrics
        key_metrics = ['aspect_f1', 'opinion_f1', 'sentiment_f1', 'triplet_f1']
        for key_metric in key_metrics:
            if key_metric in metrics:
                log_dict[f'{prefix}/key_{key_metric}'] = metrics[key_metric]
        
        wandb.log(log_dict, step=step)
    
    def log_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], epoch_time: float):
        """Log comprehensive epoch summary"""
        if self.run is None:
            return
            
        log_dict = {
            'epoch': epoch,
            'epoch_time_minutes': epoch_time / 60,
        }
        
        # Combine train and val metrics
        for name, value in train_metrics.items():
            log_dict[f'epoch_summary/train_{name}'] = value
        for name, value in val_metrics.items():
            log_dict[f'epoch_summary/val_{name}'] = value
        
        # Calculate improvement metrics
        if epoch > 0:
            # You can track improvement from previous epochs here
            pass
        
        wandb.log(log_dict)
    
    def log_triplet_analysis(self, predicted_triplets: List, target_triplets: List, 
                           matches: int, epoch: int):
        """Log detailed triplet analysis for ABSA"""
        if self.run is None:
            return
            
        # Triplet statistics
        log_dict = {
            'triplets/predicted_count': len(predicted_triplets),
            'triplets/target_count': len(target_triplets),
            'triplets/matches': matches,
            'triplets/precision': matches / len(predicted_triplets) if predicted_triplets else 0,
            'triplets/recall': matches / len(target_triplets) if target_triplets else 0,
            'epoch': epoch
        }
        
        # Sentiment distribution analysis
        if predicted_triplets:
            pred_sentiments = [t[2] for t in predicted_triplets]
            sentiment_dist = {i: pred_sentiments.count(i) for i in range(1, 4)}
            for sentiment, count in sentiment_dist.items():
                log_dict[f'triplets/predicted_sentiment_{sentiment}'] = count
        
        wandb.log(log_dict)
    
    def log_gradient_analysis(self, model, epoch: int):
        """Log gradient norms and other training diagnostics"""
        if self.run is None:
            return
            
        log_dict = {'epoch': epoch}
        
        # Compute gradient norms
        total_norm = 0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log individual layer gradients for key components
                if any(key in name for key in ['domain_classifier', 'gradient_reversal', 'implicit']):
                    log_dict[f'gradients/{name}'] = param_norm.item()
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            log_dict['gradients/total_norm'] = total_norm
            log_dict['gradients/avg_norm'] = total_norm / param_count
        
        wandb.log(log_dict)
    
    def log_attention_analysis(self, attention_weights: torch.Tensor, epoch: int):
        """Log attention pattern analysis"""
        if self.run is None or attention_weights is None:
            return
            
        # Compute attention statistics
        attention_stats = {
            'attention/mean': attention_weights.mean().item(),
            'attention/std': attention_weights.std().item(),
            'attention/max': attention_weights.max().item(),
            'attention/entropy': self._compute_attention_entropy(attention_weights),
            'epoch': epoch
        }
        
        wandb.log(attention_stats)
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute attention entropy for analysis"""
        # Flatten and compute entropy
        flat_attention = attention_weights.view(-1)
        # Add small epsilon to avoid log(0)
        flat_attention = flat_attention + 1e-8
        entropy = -(flat_attention * torch.log(flat_attention)).sum()
        return entropy.item()
    
    def log_domain_confusion_matrix(self, domain_predictions: torch.Tensor, 
                                  domain_labels: torch.Tensor, epoch: int):
        """Log domain classification confusion matrix"""
        if self.run is None:
            return
            
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Convert to numpy
            domain_preds = domain_predictions.cpu().numpy()
            domain_true = domain_labels.cpu().numpy()
            
            # Create confusion matrix
            cm = confusion_matrix(domain_true, domain_preds)
            
            # Create figure
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Domain Confusion Matrix - Epoch {epoch}')
            plt.ylabel('True Domain')
            plt.xlabel('Predicted Domain')
            
            # Log to wandb
            wandb.log({
                f'domain_confusion_matrix_epoch_{epoch}': wandb.Image(plt),
                'epoch': epoch
            })
            plt.close()
            
        except ImportError:
            print("sklearn/matplotlib not available for confusion matrix")
    
    def save_model_artifact(self, model_path: str, epoch: int, metrics: Dict[str, float]):
        """Save model as W&B artifact"""
        if self.run is None:
            return
            
        # Create artifact
        model_artifact = wandb.Artifact(
            name=f"gradient-model-epoch-{epoch}",
            type="model",
            description=f"GRADIENT ABSA model after epoch {epoch}",
            metadata={
                'epoch': epoch,
                'dataset': self.config.dataset_name,
                **metrics
            }
        )
        
        # Add model file
        model_artifact.add_file(model_path)
        
        # Log artifact
        wandb.log_artifact(model_artifact)
    
    def log_final_results(self, results: Dict[str, Any]):
        """Log final training results and summary"""
        if self.run is None:
            return
            
        # Log final metrics
        final_metrics = {
            'final/best_aspect_f1': results.get('aspect_f1', 0),
            'final/best_opinion_f1': results.get('opinion_f1', 0),
            'final/best_sentiment_f1': results.get('sentiment_f1', 0),
            'final/best_triplet_f1': results.get('triplet_f1', 0),
            'final/training_time_hours': results.get('training_time_hours', 0),
        }
        
        # Add dataset-specific metrics
        dataset_name = getattr(self.config, 'dataset', 'unknown')
        for metric_name, value in final_metrics.items():
            if 'final/' in metric_name:
                final_metrics[f'{dataset_name}/{metric_name}'] = value
        
        wandb.log(final_metrics)
        
        # Create summary table
        self._create_results_table(results)
        
        # Mark run as finished
        wandb.finish()
    
    def _create_results_table(self, results: Dict[str, Any]):
        """Create formatted results table"""
        # Create table data
        table_data = [
            ['Aspect F1', f"{results.get('aspect_f1', 0):.4f}"],
            ['Opinion F1', f"{results.get('opinion_f1', 0):.4f}"],
            ['Sentiment F1', f"{results.get('sentiment_f1', 0):.4f}"],
            ['Triplet F1', f"{results.get('triplet_f1', 0):.4f}"],
            ['Training Time', f"{results.get('training_time_hours', 0):.2f} hours"],
        ]
        
        # Create W&B table
        results_table = wandb.Table(
            columns=['Metric', 'Score'],
            data=table_data
        )
        
        wandb.log({'final_results_table': results_table})


# Integration functions for your existing trainer

def setup_wandb_logging(config, project_name: str = "gradient-absa-2025") -> WandBIntegration:
    """
    Setup W&B logging for your trainer
    Call this in your train.py before training starts
    """
    return WandBIntegration(config, project_name)

def log_training_step_to_wandb(wandb_integration: WandBIntegration, 
                              epoch: int, step: int, losses: Dict[str, float],
                              alpha: float = None):
    """
    Call this in your training loop for each step
    """
    wandb_integration.log_training_step(epoch, step, losses, alpha=alpha)

def log_validation_to_wandb(wandb_integration: WandBIntegration,
                           epoch: int, metrics: Dict[str, float]):
    """
    Call this after validation
    """
    wandb_integration.log_validation_metrics(epoch, metrics)

def log_triplet_debug_to_wandb(wandb_integration: WandBIntegration,
                              predicted_triplets: List, target_triplets: List,
                              matches: int, epoch: int):
    """
    Call this when you do triplet analysis (like in your debug output)
    """
    wandb_integration.log_triplet_analysis(predicted_triplets, target_triplets, matches, epoch)


# Modified trainer integration class
class WandBTrainerMixin:
    """
    Mixin to add W&B logging to your existing trainer
    Add this to your UnifiedABSATrainer class
    """
    
    def __init_wandb__(self, project_name: str = "gradient-absa-2025"):
        """Initialize W&B in your trainer"""
        self.wandb_integration = WandBIntegration(self.config, project_name)
        self.wandb_integration.log_model_architecture(self.model)
    
    def __log_train_step_wandb__(self, epoch: int, step: int, losses: Dict[str, float], alpha: float = None):
        """Log training step to W&B"""
        if hasattr(self, 'wandb_integration'):
            self.wandb_integration.log_training_step(epoch, step, losses, alpha=alpha)
    
    def __log_validation_wandb__(self, epoch: int, metrics: Dict[str, float]):
        """Log validation to W&B"""
        if hasattr(self, 'wandb_integration'):
            self.wandb_integration.log_validation_metrics(epoch, metrics)
    
    def __finish_wandb__(self, final_results: Dict[str, Any]):
        """Finish W&B logging"""
        if hasattr(self, 'wandb_integration'):
            self.wandb_integration.log_final_results(final_results)


# Instructions for integration:
"""
TO INTEGRATE W&B INTO YOUR EXISTING TRAINER:

1. Add to your train.py imports:
   from training.wandb_integration import setup_wandb_logging, log_training_step_to_wandb, log_validation_to_wandb

2. In train.py, before starting training:
   wandb_integration = setup_wandb_logging(config, "gradient-absa-2025")

3. In your training loop, after computing losses:
   log_training_step_to_wandb(wandb_integration, epoch, step, losses, alpha=current_alpha)

4. After validation:
   log_validation_to_wandb(wandb_integration, epoch, val_metrics)

5. At the end of training:
   wandb_integration.log_final_results(final_results)

OR use the mixin approach:
1. Modify your UnifiedABSATrainer to inherit from WandBTrainerMixin
2. Call self.__init_wandb__() in your trainer's __init__
3. Call the appropriate logging methods in your training loop
"""