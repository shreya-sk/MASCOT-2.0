# src/training/domain_adversarial_trainer.py
"""
Domain Adversarial Trainer Integration
Fixed imports for standalone execution
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup
import os
import json
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
import numpy as np
import sys
from pathlib import Path

# Add src to path for absolute imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

# Fixed imports using absolute paths
try:
    from models.unified_absa_model import UnifiedABSAModel
    from models.domain_adversarial import get_domain_id
except ImportError:
    # Alternative import path
    try:
        from src.models.unified_absa_model import UnifiedABSAModel
        from src.models.domain_adversarial import get_domain_id
    except ImportError:
        # If running from project root
        import sys
        sys.path.append('src')
        from models.unified_absa_model import UnifiedABSAModel
        from models.domain_adversarial import get_domain_id


class DomainAdversarialABSATrainer:
    """
    ABSA Trainer with Domain Adversarial Training Integration
    Fixed import version for standalone execution
    """
    
    def __init__(self, model: UnifiedABSAModel, config, train_dataloader, eval_dataloader=None):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Domain adversarial specific config
        self.use_domain_adversarial = getattr(config, 'use_domain_adversarial', True)
        self.domain_loss_weight = getattr(config, 'domain_loss_weight', 0.1)
        self.orthogonal_loss_weight = getattr(config, 'orthogonal_loss_weight', 0.1)
        self.alpha_schedule = getattr(config, 'alpha_schedule', 'progressive')
        
        # Setup optimizer with domain adversarial components
        self.optimizer = self._setup_domain_adversarial_optimizer()
        
        # Setup scheduler
        total_steps = len(train_dataloader) * getattr(config, 'num_epochs', 10)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=getattr(config, 'warmup_steps', int(0.1 * total_steps)),
            num_training_steps=total_steps
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_f1 = 0.0
        self.training_history = []
        
        # Domain adversarial tracking
        self.domain_loss_history = []
        self.orthogonal_loss_history = []
        self.alpha_history = []
        
        # Output directory
        self.output_dir = getattr(config, 'output_dir', 'outputs/domain_adversarial')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.use_domain_adversarial:
            self.logger.info("🎯 Domain Adversarial Training Enabled")
            self.logger.info(f"   - Domain loss weight: {self.domain_loss_weight}")
            self.logger.info(f"   - Orthogonal loss weight: {self.orthogonal_loss_weight}")
            self.logger.info(f"   - Alpha schedule: {self.alpha_schedule}")
        
    def _setup_domain_adversarial_optimizer(self):
        """Setup optimizer with different learning rates for domain adversarial components"""
        
        # Separate parameters for different components
        backbone_params = []
        task_params = []  # Aspect, opinion, sentiment classifiers
        domain_params = []  # Domain adversarial components
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'domain_adversarial' in name:
                domain_params.append(param)
            else:
                task_params.append(param)
        
        # Different learning rates for different components
        param_groups = [
            {
                'params': backbone_params,
                'lr': getattr(self.config, 'learning_rate', 2e-5) * 0.1,  # Lower LR for backbone
                'name': 'backbone'
            },
            {
                'params': task_params,
                'lr': getattr(self.config, 'learning_rate', 2e-5),
                'name': 'task_specific'
            }
        ]
        
        # Add domain adversarial parameters if enabled
        if self.use_domain_adversarial and domain_params:
            param_groups.append({
                'params': domain_params,
                'lr': getattr(self.config, 'learning_rate', 2e-5) * 2.0,  # Higher LR for domain components
                'name': 'domain_adversarial'
            })
        
        # Filter out empty parameter groups
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        
        optimizer = AdamW(
            param_groups,
            lr=getattr(self.config, 'learning_rate', 2e-5),
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
            eps=1e-8
        )
        
        return optimizer
    
    def _get_current_alpha(self, epoch: int, total_epochs: int) -> float:
        """Get current alpha value for gradient reversal"""
        if self.alpha_schedule == 'fixed':
            return 1.0
        elif self.alpha_schedule == 'progressive':
            # Gradually increase from 0 to 1
            progress = epoch / total_epochs
            return 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
        elif self.alpha_schedule == 'cosine':
            # Cosine schedule
            progress = epoch / total_epochs
            return 0.5 * (1 + np.cos(np.pi * progress))
        else:
            return 1.0
    
    def train_step(self, batch: Dict[str, torch.Tensor], dataset_name: str) -> Dict[str, float]:
        """
        Single training step with domain adversarial training
        """
        self.model.train()
        
        # Update alpha for gradient reversal
        current_alpha = self._get_current_alpha(self.epoch, getattr(self.config, 'num_epochs', 10))
        if hasattr(self.model, 'domain_adversarial') and self.model.domain_adversarial:
            self.model.domain_adversarial.current_alpha = current_alpha
        
        # Forward pass with dataset name for domain identification
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch,
            dataset_name=dataset_name
        )
        
        # Compute losses
        if hasattr(self.model, 'compute_comprehensive_loss'):
            total_loss, loss_dict = self.model.compute_comprehensive_loss(outputs, batch, dataset_name)
        else:
            # Fallback loss computation
            total_loss = outputs.get('losses', {}).get('total_loss', torch.tensor(0.0))
            loss_dict = outputs.get('losses', {'total_loss': total_loss})
        
        # Add domain adversarial specific metrics
        if 'domain_outputs' in outputs and outputs['domain_outputs']:
            domain_outputs = outputs['domain_outputs']
            
            # Track domain classification accuracy
            if 'domain_logits' in domain_outputs:
                domain_logits = domain_outputs['domain_logits']
                domain_ids = torch.tensor([get_domain_id(dataset_name)] * len(batch['input_ids']), 
                                        device=domain_logits.device)
                domain_preds = domain_logits.argmax(dim=-1)
                domain_acc = (domain_preds == domain_ids).float().mean()
                loss_dict['domain_accuracy'] = domain_acc.item()
            
            # Track alpha value
            loss_dict['gradient_reversal_alpha'] = current_alpha
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        # Update global step
        self.global_step += 1
        
        # Convert tensor values to floats for logging
        log_dict = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                log_dict[key] = value.item()
            else:
                log_dict[key] = value
        
        return log_dict
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with domain adversarial training
        """
        self.epoch = epoch
        if hasattr(self.model, 'update_training_progress'):
            self.model.update_training_progress(epoch, getattr(self.config, 'num_epochs', 10))
        
        epoch_losses = {}
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{getattr(self.config, "num_epochs", 10)}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Get dataset name (this should be provided in your dataloader)
            dataset_name = batch.get('dataset_name', 'general')
            if isinstance(dataset_name, list):
                dataset_name = dataset_name[0]  # Take first if list
            
            # Training step
            step_losses = self.train_step(batch, dataset_name)
            
            # Accumulate losses
            for loss_name, loss_value in step_losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = []
                epoch_losses[loss_name].append(loss_value)
            
            num_batches += 1
            
            # Update progress bar
            if 'total_loss' in step_losses:
                current_loss = step_losses['total_loss']
                avg_loss = np.mean(epoch_losses['total_loss'])
                
                # Add domain adversarial info to progress bar
                postfix = {'Loss': f"{current_loss:.4f}", 'Avg': f"{avg_loss:.4f}"}
                if 'domain_accuracy' in step_losses:
                    postfix['DomAcc'] = f"{step_losses['domain_accuracy']:.3f}"
                if 'gradient_reversal_alpha' in step_losses:
                    postfix['Alpha'] = f"{step_losses['gradient_reversal_alpha']:.3f}"
                
                pbar.set_postfix(postfix)
        
        # Calculate average losses
        avg_losses = {name: np.mean(losses) for name, losses in epoch_losses.items()}
        
        # Track domain adversarial specific metrics
        if 'domain_loss' in avg_losses:
            self.domain_loss_history.append(avg_losses['domain_loss'])
        if 'orthogonal_loss' in avg_losses:
            self.orthogonal_loss_history.append(avg_losses['orthogonal_loss'])
        if 'gradient_reversal_alpha' in avg_losses:
            self.alpha_history.append(avg_losses['gradient_reversal_alpha'])
        
        self.training_history.append(avg_losses)
        
        return avg_losses
    
    def evaluate_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        Evaluate model with domain adversarial metrics
        """
        self.model.eval()
        
        eval_losses = {}
        domain_confusion_matrix = np.zeros((4, 4))  # 4 domains
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                # Move batch to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                dataset_name = batch.get('dataset_name', 'general')
                if isinstance(dataset_name, list):
                    dataset_name = dataset_name[0]
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch,
                    dataset_name=dataset_name
                )
                
                # Compute losses
                if hasattr(self.model, 'compute_comprehensive_loss'):
                    total_loss, loss_dict = self.model.compute_comprehensive_loss(outputs, batch, dataset_name)
                else:
                    loss_dict = outputs.get('losses', {'total_loss': torch.tensor(0.0)})
                
                # Accumulate losses
                for loss_name, loss_value in loss_dict.items():
                    if loss_name not in eval_losses:
                        eval_losses[loss_name] = []
                    if isinstance(loss_value, torch.Tensor):
                        eval_losses[loss_name].append(loss_value.item())
                    else:
                        eval_losses[loss_name].append(loss_value)
                
                # Track domain confusion for analysis
                if 'domain_outputs' in outputs and 'domain_logits' in outputs['domain_outputs']:
                    domain_logits = outputs['domain_outputs']['domain_logits']
                    true_domain_id = get_domain_id(dataset_name)
                    pred_domain_ids = domain_logits.argmax(dim=-1).cpu().numpy()
                    
                    for pred_id in pred_domain_ids:
                        if 0 <= true_domain_id < 4 and 0 <= pred_id < 4:
                            domain_confusion_matrix[true_domain_id, pred_id] += 1
        
        # Calculate average losses
        avg_eval_losses = {name: np.mean(losses) for name, losses in eval_losses.items()}
        
        # Calculate domain confusion metrics
        total_predictions = np.sum(domain_confusion_matrix)
        if total_predictions > 0:
            domain_accuracy = np.trace(domain_confusion_matrix) / total_predictions
            domain_confusion_score = 1.0 - domain_accuracy  # Higher confusion is better for adversarial training
        else:
            domain_accuracy = 0.0
            domain_confusion_score = 0.0
        
        # Add domain adversarial specific metrics
        avg_eval_losses['domain_confusion_score'] = domain_confusion_score
        avg_eval_losses['domain_accuracy'] = domain_accuracy
        
        return avg_eval_losses
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop with domain adversarial training
        """
        print("🚀 Starting Domain Adversarial ABSA Training...")
        print(f"   Model: {getattr(self.config, 'model_name', 'Unknown')}")
        print(f"   Epochs: {getattr(self.config, 'num_epochs', 10)}")
        print(f"   Batch size: {getattr(self.config, 'batch_size', 16)}")
        print(f"   Learning rate: {getattr(self.config, 'learning_rate', 2e-5)}")
        
        if self.use_domain_adversarial:
            print(f"   🎯 Domain Adversarial Features:")
            print(f"     - Gradient reversal: ✅ Enabled")
            print(f"     - Domain classifier: ✅ 4-domain output")
            print(f"     - Orthogonal constraints: ✅ Active")
            print(f"     - Alpha schedule: {self.alpha_schedule}")
        
        # Training results tracking
        training_results = {
            'best_f1': 0.0,
            'best_epoch': 0,
            'best_model_path': None,
            'training_history': [],
            'domain_adversarial_history': {
                'domain_losses': self.domain_loss_history,
                'orthogonal_losses': self.orthogonal_loss_history,
                'alpha_values': self.alpha_history,
                'confusion_scores': []
            }
        }
        
        # Training loop
        num_epochs = getattr(self.config, 'num_epochs', 10)
        for epoch in range(num_epochs):
            # Training phase
            train_losses = self.train_epoch(epoch)
            training_results['training_history'].append(train_losses)
            
            # Evaluation phase
            if self.eval_dataloader:
                eval_metrics = self.evaluate_epoch(self.eval_dataloader, epoch)
                
                # Check for best model (you may want to use different metric)
                current_f1 = eval_metrics.get('accuracy', eval_metrics.get('total_loss', 0.0))
                if current_f1 > training_results['best_f1']:
                    training_results['best_f1'] = current_f1
                    training_results['best_epoch'] = epoch
                    
                    # Save best model
                    best_model_path = os.path.join(self.output_dir, 'best_domain_adversarial_model.pt')
                    self.save_model(best_model_path)
                    training_results['best_model_path'] = best_model_path
                
                # Track domain confusion
                if 'domain_confusion_score' in eval_metrics:
                    training_results['domain_adversarial_history']['confusion_scores'].append(
                        eval_metrics['domain_confusion_score']
                    )
                
                # Log results
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
                self.logger.info(f"  Train Loss: {train_losses.get('total_loss', 0):.4f}")
                self.logger.info(f"  Eval Loss: {eval_metrics.get('total_loss', 0):.4f}")
                
                if self.use_domain_adversarial:
                    self.logger.info(f"  Domain Confusion: {eval_metrics.get('domain_confusion_score', 0):.4f}")
                    self.logger.info(f"  Alpha: {train_losses.get('gradient_reversal_alpha', 0):.4f}")
        
        # Save final results
        self._save_training_results(training_results)
        
        print("🎉 Domain Adversarial Training completed!")
        print(f"   Best F1: {training_results['best_f1']:.4f} (Epoch {training_results['best_epoch']})")
        print(f"   Best model: {training_results['best_model_path']}")
        
        return training_results
    
    def save_model(self, save_path: str):
        """Save model with domain adversarial components"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'domain_adversarial_enabled': self.use_domain_adversarial,
            'domain_loss_history': self.domain_loss_history,
            'orthogonal_loss_history': self.orthogonal_loss_history,
            'alpha_history': self.alpha_history
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        self.logger.info(f"✅ Model saved with domain adversarial components: {save_path}")
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save comprehensive training results"""
        results_path = os.path.join(self.output_dir, 'domain_adversarial_training_results.json')
        
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (list, dict)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"✅ Training results saved: {results_path}")


def create_domain_adversarial_trainer(model, config, train_dataloader, eval_dataloader=None):
    """Factory function to create domain adversarial trainer"""
    return DomainAdversarialABSATrainer(model, config, train_dataloader, eval_dataloader)