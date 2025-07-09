# src/training/enhanced_trainer.py - Complete Training Pipeline with Implicit Detection
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import wandb
from tqdm import tqdm
import logging
import os
from datetime import datetime

from .metrics import enhanced_compute_triplet_metrics_with_bench
from .losses import EnhancedABSALoss
from .metrics import compute_metrics, compute_triplet_recovery_score, generate_evaluation_report
from ..models.absa import LLMABSA
from ..utils.config import LLMABSAConfig


class EnhancedABSATrainer:
    """
    Complete ABSA trainer with implicit detection and 2024-2025 breakthrough features
    
    Integrates:
    - Explicit triplet extraction
    - Implicit sentiment detection (aspects + opinions)
    - Instruction-following generation
    - Contrastive learning
    - Advanced evaluation metrics
    """
    
    def __init__(self, 
                 model: LLMABSA,
                 config: LLMABSAConfig,
                 device: torch.device,
                 logger: Optional[logging.Logger] = None):
        
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or self._setup_logger()
        
        # Move model to device
        self.model.to(device)
        
        # Initialize enhanced loss function
        self.criterion = EnhancedABSALoss(config)
        
        # Initialize optimizer with different learning rates for components
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = None  # Will be set up after getting data loader
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_score = 0.0
        self.best_model_path = None
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.loss_history = defaultdict(list)
        
        # Gradient accumulation
        self.grad_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        
        # Mixed precision training
        self.use_fp16 = getattr(config, 'use_fp16', True)
        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Gradient clipping
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
        
        # Early stopping
        self.patience = getattr(config, 'patience', 5)
        self.patience_counter = 0
        
        self.logger.info("Enhanced ABSA Trainer initialized with implicit detection")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('EnhancedABSATrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with different learning rates for different components"""
        
        # Separate parameters for different components
        backbone_params = []
        implicit_params = []
        contrastive_params = []
        instruction_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'implicit' in name:
                    implicit_params.append(param)
                elif 'contrastive' in name or 'verification' in name:
                    contrastive_params.append(param)
                elif 't5_model' in name or 'instruction' in name:
                    instruction_params.append(param)
                else:
                    backbone_params.append(param)
        
        # Parameter groups with different learning rates
        param_groups = [
            {
                'params': backbone_params,
                'lr': self.config.learning_rate,
                'weight_decay': getattr(self.config, 'weight_decay', 0.01)
            }
        ]
        
        # Higher learning rate for implicit detection (new component)
        if implicit_params:
            param_groups.append({
                'params': implicit_params,
                'lr': self.config.learning_rate * 2.0,
                'weight_decay': getattr(self.config, 'weight_decay', 0.01)
            })
        
        # Lower learning rate for pre-trained instruction following
        if instruction_params:
            param_groups.append({
                'params': instruction_params,
                'lr': self.config.learning_rate * 0.5,
                'weight_decay': getattr(self.config, 'weight_decay', 0.01)
            })
        
        # Standard learning rate for contrastive components
        if contrastive_params:
            param_groups.append({
                'params': contrastive_params,
                'lr': self.config.learning_rate,
                'weight_decay': getattr(self.config, 'weight_decay', 0.01)
            })
        
        optimizer = optim.AdamW(param_groups)
        
        self.logger.info(f"Optimizer setup with {len(param_groups)} parameter groups")
        return optimizer
    
    def setup_scheduler(self, train_loader: DataLoader):
        """Setup learning rate scheduler"""
        total_steps = len(train_loader) * self.config.num_epochs // self.grad_accumulation_steps
        warmup_steps = int(total_steps * getattr(self.config, 'warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"Scheduler setup: {total_steps} total steps, {warmup_steps} warmup steps")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with implicit detection"""
        self.model.train()
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        # Setup progress bar
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        loss_dict = self._forward_step(batch)
                else:
                    loss_dict = self._forward_step(batch)
                
                # Get total loss
                total_loss = loss_dict['total_loss']
                
                # Scale loss for gradient accumulation
                scaled_loss = total_loss / self.grad_accumulation_steps
                
                # Backward pass
                if self.use_fp16:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Update weights every grad_accumulation_steps
                if (step + 1) % self.grad_accumulation_steps == 0:
                    # Gradient clipping
                    if self.use_fp16:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.current_step += 1
                
                # Log losses
                for loss_name, loss_value in loss_dict.items():
                    if torch.is_tensor(loss_value):
                        epoch_losses[loss_name].append(loss_value.item())
                
                # Compute metrics periodically
                if step % 100 == 0:
                    with torch.no_grad():
                        metrics = self._compute_training_metrics(batch, loss_dict)
                        for metric_name, metric_value in metrics.items():
                            epoch_metrics[metric_name].append(metric_value)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to wandb if enabled
                if hasattr(self, 'use_wandb') and self.use_wandb:
                    wandb_dict = {'step': self.current_step}
                    wandb_dict.update({f'train/{k}': v.item() if torch.is_tensor(v) else v 
                                     for k, v in loss_dict.items()})
                    wandb.log(wandb_dict)
                
            except Exception as e:
                self.logger.error(f"Error in training step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average losses and metrics
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items() if v}
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items() if v}
        
        # Combine results
        results = {**avg_losses, **avg_metrics}
        
        self.logger.info(f"Epoch {epoch} Training Results:")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return results
    
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward step with loss computation"""
        
        # Extract inputs
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        texts = batch.get('texts', None)
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=texts,
            task_type='triplet_extraction'
        )
        
        # Prepare targets for loss computation
        targets = {
            'aspect_labels': batch.get('aspect_labels'),
            'opinion_labels': batch.get('opinion_labels'),
            'sentiment_labels': batch.get('sentiment_labels'),
            # Implicit detection targets (if available)
            'implicit_aspect_labels': batch.get('implicit_aspect_labels'),
            'implicit_opinion_labels': batch.get('implicit_opinion_labels'),
            'sentiment_combination_labels': batch.get('sentiment_combination_labels'),
            'grid_labels': batch.get('grid_labels'),
            'combination_labels': batch.get('combination_labels'),
            'confidence_labels': batch.get('confidence_labels')
        }
        
        # Remove None targets
        targets = {k: v for k, v in targets.items() if v is not None}
        
        # Compute losses
        loss_dict = self.criterion(outputs, targets)
        
        return loss_dict
    
    def _compute_training_metrics(self, batch: Dict[str, torch.Tensor], 
                                loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute training metrics"""
        metrics = {}
        
        try:
            # Get model outputs for this batch
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    texts=batch.get('texts', None)
                )
            
            # Compute basic accuracy metrics
            if 'aspect_logits' in outputs and 'aspect_labels' in batch:
                aspect_preds = outputs['aspect_logits'].argmax(dim=-1)
                aspect_labels = batch['aspect_labels']
                
                valid_mask = aspect_labels != -100
                if valid_mask.any():
                    aspect_acc = (aspect_preds[valid_mask] == aspect_labels[valid_mask]).float().mean().item()
                    metrics['aspect_accuracy'] = aspect_acc
            
            # Similar for opinions and sentiments
            if 'opinion_logits' in outputs and 'opinion_labels' in batch:
                opinion_preds = outputs['opinion_logits'].argmax(dim=-1)
                opinion_labels = batch['opinion_labels']
                
                valid_mask = opinion_labels != -100
                if valid_mask.any():
                    opinion_acc = (opinion_preds[valid_mask] == opinion_labels[valid_mask]).float().mean().item()
                    metrics['opinion_accuracy'] = opinion_acc
            
            if 'sentiment_logits' in outputs and 'sentiment_labels' in batch:
                sentiment_preds = outputs['sentiment_logits'].argmax(dim=-1)
                sentiment_labels = batch['sentiment_labels']
                
                valid_mask = sentiment_labels != -100
                if valid_mask.any():
                    sentiment_acc = (sentiment_preds[valid_mask] == sentiment_labels[valid_mask]).float().mean().item()
                    metrics['sentiment_accuracy'] = sentiment_acc
            
            # Implicit detection metrics (if available)
            if 'implicit_aspect_scores' in outputs and 'implicit_aspect_labels' in batch:
                implicit_scores = torch.sigmoid(outputs['implicit_aspect_scores'])
                implicit_labels = batch['implicit_aspect_labels'].float()
                
                valid_mask = implicit_labels != -100
                if valid_mask.any():
                    implicit_preds = (implicit_scores[valid_mask] > 0.5).float()
                    implicit_acc = (implicit_preds == implicit_labels[valid_mask]).float().mean().item()
                    metrics['implicit_aspect_accuracy'] = implicit_acc
            
        except Exception as e:
            self.logger.warning(f"Error computing training metrics: {e}")
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = defaultdict(list)
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
            
            for batch in pbar:
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    loss_dict = self._forward_step(batch)
                    
                    # Get model outputs for metrics
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        texts=batch.get('texts', None)
                    )
                    
                    # Collect losses
                    for loss_name, loss_value in loss_dict.items():
                        if torch.is_tensor(loss_value):
                            epoch_losses[loss_name].append(loss_value.item())
                    
                    # Collect predictions and labels for detailed metrics
                    all_predictions.append(outputs)
                    all_labels.append(batch)
                    
                except Exception as e:
                    self.logger.error(f"Error in validation step: {e}")
                    continue
        
        # Compute average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items() if v}
        
        # Compute detailed metrics
        try:
            detailed_metrics = compute_metrics(all_predictions, all_labels)
            avg_losses.update(detailed_metrics)
        except Exception as e:
            self.logger.warning(f"Error computing detailed validation metrics: {e}")
        
        self.logger.info(f"Epoch {epoch} Validation Results:")
        for key, value in avg_losses.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        return avg_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              save_dir: str = 'checkpoints', use_wandb: bool = False) -> Dict[str, Any]:
        """Complete training loop"""
        
        self.use_wandb = use_wandb
        
        # Setup scheduler
        self.setup_scheduler(train_loader)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info("Starting enhanced ABSA training with implicit detection")
        self.logger.info(f"Total epochs: {self.config.num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info(f"Implicit detection: {self.config.use_implicit_detection}")
        self.logger.info(f"Contrastive learning: {self.config.use_contrastive_learning}")
        
        training_history = {
            'train_metrics': [],
            'val_metrics': [],
            'best_epoch': 0,
            'best_score': 0.0
        }
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            training_history['train_metrics'].append(train_metrics)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader, epoch)
            training_history['val_metrics'].append(val_metrics)
            
            # Determine validation score (use F1 or combined metric)
            val_score = val_metrics.get('overall_f1', 
                       val_metrics.get('aspect_f1', 
                       val_metrics.get('sentiment_accuracy', 0.0)))
            
            # Save checkpoint
            is_best = val_score > self.best_score
            if is_best:
                self.best_score = val_score
                self.best_model_path = self._save_checkpoint(
                    save_dir, epoch, train_metrics, val_metrics, is_best=True
                )
                training_history['best_epoch'] = epoch
                training_history['best_score'] = val_score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Regular checkpoint
            self._save_checkpoint(save_dir, epoch, train_metrics, val_metrics, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch} (patience: {self.patience})")
                break
            
            # Log to wandb
            if use_wandb:
                wandb_dict = {'epoch': epoch}
                wandb_dict.update({f'train/{k}': v for k, v in train_metrics.items()})
                wandb_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
                wandb.log(wandb_dict)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation score: {self.best_score:.4f} at epoch {training_history['best_epoch']}")
        self.logger.info(f"Best model saved at: {self.best_model_path}")
        
        return training_history
    
    def _save_checkpoint(self, save_dir: str, epoch: int, 
                        train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                        is_best: bool = False) -> str:
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.__dict__,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_score': self.best_score,
            'scaler_state_dict': self.scaler.state_dict() if self.use_fp16 else None
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            
            # Also save just the model state dict for easy loading
            model_path = os.path.join(save_dir, 'best_model_state.pt')
            torch.save(self.model.state_dict(), model_path)
            
            self.logger.info(f"💾 Best model saved: {best_path}")
            return best_path
        
        return checkpoint_path
    
    def evaluate_model(self, test_loader: DataLoader, 
                      save_predictions: bool = True, 
                      output_dir: str = 'results') -> Dict[str, Any]:
        """Comprehensive model evaluation with implicit detection"""
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_triplets = []
        all_implicit_results = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info("Starting comprehensive evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # Get model outputs
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        texts=batch.get('texts', None)
                    )
                    
                    # Extract triplets with implicit detection
                    if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
                        tokenizer = self.model.tokenizer
                    else:
                        # Use a default tokenizer if none is set
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
                    
                    triplet_results = self.model.extract_all_triplets_with_implicit(
                        batch['input_ids'],
                        batch['attention_mask'],
                        tokenizer=tokenizer,
                        texts=batch.get('texts', None)
                    )
                    
                    all_predictions.append(outputs)
                    all_labels.append(batch)
                    all_triplets.append(triplet_results)
                    
                    if 'implicit_results' in triplet_results:
                        all_implicit_results.append(triplet_results['implicit_results'])
                
                except Exception as e:
                    self.logger.error(f"Error in evaluation step: {e}")
                    continue
        
        # Compute comprehensive metrics
        try:
            # Traditional metrics
            traditional_metrics = compute_metrics(all_predictions, all_labels)
            
            # Triplet recovery metrics (if applicable)
            triplet_metrics = {}
            if all_triplets:
                try:
                    triplet_metrics = self._compute_triplet_metrics(all_triplets)
                except Exception as e:
                    self.logger.warning(f"Error computing triplet metrics: {e}")
            
            # Implicit detection metrics
            implicit_metrics = {}
            if all_implicit_results:
                try:
                    implicit_metrics = self._compute_implicit_metrics(all_implicit_results)
                except Exception as e:
                    self.logger.warning(f"Error computing implicit metrics: {e}")
            
            # Combine all metrics
            final_metrics = {
                **traditional_metrics,
                **triplet_metrics,
                **implicit_metrics
            }
            
            # Generate evaluation report
            report = generate_evaluation_report(final_metrics)
            
            # Save results
            if save_predictions:
                self._save_evaluation_results(
                    final_metrics, all_triplets, all_implicit_results, output_dir
                )
            
            self.logger.info("Evaluation completed!")
            self.logger.info(f"Final metrics: {final_metrics}")
            
            return {
                'metrics': final_metrics,
                'report': report,
                'triplets': all_triplets,
                'implicit_results': all_implicit_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {'metrics': {}, 'error': str(e)}
    
    def _compute_triplet_metrics(self, all_triplets: List[Dict]) -> Dict[str, float]:
        """Enhanced triplet metrics with ABSA-Bench integration"""
        return enhanced_compute_triplet_metrics_with_bench(all_triplets)

    def _compute_implicit_metrics(self, all_implicit_results: List[Dict]) -> Dict[str, float]:
        """Compute implicit detection specific metrics"""
        metrics = {}
        
        total_implicit_aspects = sum(len(result.get('implicit_aspects', [])) 
                                   for result in all_implicit_results)
        total_implicit_opinions = sum(len(result.get('implicit_opinions', [])) 
                                    for result in all_implicit_results)
        
        metrics.update({
            'total_implicit_aspects': total_implicit_aspects,
            'total_implicit_opinions': total_implicit_opinions,
            'avg_implicit_aspects_per_sample': total_implicit_aspects / (len(all_implicit_results) + 1e-8),
            'avg_implicit_opinions_per_sample': total_implicit_opinions / (len(all_implicit_results) + 1e-8)
        })
        
        return metrics
    
    def _save_evaluation_results(self, metrics: Dict[str, float], 
                               triplets: List[Dict], implicit_results: List[Dict],
                               output_dir: str):
        """Save evaluation results to files"""
        import json
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save triplets
        triplets_path = os.path.join(output_dir, 'extracted_triplets.json')
        with open(triplets_path, 'w') as f:
            json.dump(triplets, f, indent=2, default=str)
        
        # Save implicit results
        implicit_path = os.path.join(output_dir, 'implicit_results.json')
        with open(implicit_path, 'w') as f:
            json.dump(implicit_results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to {output_dir}")


def create_enhanced_trainer(config: LLMABSAConfig, device: torch.device) -> EnhancedABSATrainer:
    """Factory function to create enhanced trainer"""
    
    # Create model
    model = LLMABSA(config)
    
    # Create trainer
    trainer = EnhancedABSATrainer(model, config, device)
    
    return trainer