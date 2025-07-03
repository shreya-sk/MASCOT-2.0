# src/training/generative_trainer.py
"""
Generative Training Pipeline for ABSA
Specialized trainer for sequence-to-sequence generative models
"""

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
import json

from .trainer import ABSATrainer  # Your existing trainer
from .generative_losses import GenerativeLoss, TripletRecoveryLoss
from .generative_metrics import GenerativeMetrics
from ..models.unified_generative_absa import UnifiedGenerativeABSA


class GenerativeABSATrainer:
    """
    Specialized trainer for generative ABSA models
    Handles sequence-to-sequence training with ABSA-specific evaluation
    """
    
    def __init__(self, 
                 model: UnifiedGenerativeABSA,
                 config,
                 device: torch.device,
                 logger: Optional[logging.Logger] = None):
        
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or self._setup_logger()
        
        # Move model to device
        self.model.to(device)
        
        # Initialize generative losses
        self.criterion = GenerativeLoss(config)
        self.triplet_recovery_loss = TripletRecoveryLoss(config)
        
        # Initialize generative metrics
        self.metrics = GenerativeMetrics(config)
        
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
        
        # Generation parameters
        self.generation_config = {
            'max_length': getattr(config, 'max_generation_length', 128),
            'num_beams': getattr(config, 'num_beams', 4),
            'temperature': getattr(config, 'temperature', 1.0),
            'do_sample': getattr(config, 'do_sample', True),
            'early_stopping': True
        }
        
        # Training configuration
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
        self.warmup_steps = getattr(config, 'warmup_steps', 500)
        
        # Curriculum learning
        self.use_curriculum = getattr(config, 'use_curriculum_learning', True)
        self.curriculum_schedule = self._setup_curriculum_schedule()
        
        # Mixed precision training
        self.use_fp16 = getattr(config, 'use_fp16', True)
        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Evaluation configuration
        self.eval_steps = getattr(config, 'eval_steps', 500)
        self.save_steps = getattr(config, 'save_steps', 1000)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('GenerativeABSATrainer')
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
        """Setup optimizer with different learning rates for components"""
        
        # Separate parameters for different components
        generative_params = []
        existing_model_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'existing_model' in name:
                    existing_model_params.append(param)
                else:
                    generative_params.append(param)
        
        # Different learning rates for different components
        optimizer_grouped_parameters = [
            {
                'params': generative_params,
                'lr': self.config.learning_rate,
                'weight_decay': getattr(self.config, 'weight_decay', 0.01)
            }
        ]
        
        if existing_model_params:
            optimizer_grouped_parameters.append({
                'params': existing_model_params,
                'lr': self.config.learning_rate * 0.1,  # Lower LR for existing model
                'weight_decay': getattr(self.config, 'weight_decay', 0.01)
            })
        
        # Use AdamW optimizer
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            eps=getattr(self.config, 'adam_epsilon', 1e-8)
        )
        
        return optimizer
    
    def _setup_curriculum_schedule(self) -> Dict[str, Any]:
        """Setup curriculum learning schedule"""
        if not self.use_curriculum:
            return {}
        
        return {
            'phase_1': {
                'epochs': 2,
                'tasks': ['aspect_extraction', 'opinion_extraction'],
                'description': 'Simple extraction tasks'
            },
            'phase_2': {
                'epochs': 3,
                'tasks': ['sentiment_analysis', 'triplet_generation'],
                'description': 'Structured generation tasks'
            },
            'phase_3': {
                'epochs': 5,
                'tasks': ['explanation_generation', 'unified_generation'],
                'description': 'Complex generation tasks'
            }
        }
    
    def train(self, 
              train_dataloader: DataLoader,
              dev_dataloader: Optional[DataLoader] = None,
              test_dataloader: Optional[DataLoader] = None,
              num_epochs: int = 10) -> Dict[str, Any]:
        """Main training loop for generative ABSA"""
        
        self.logger.info("🚀 Starting Generative ABSA Training")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Training examples: {len(train_dataloader.dataset)}")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        
        # Setup scheduler
        total_steps = len(train_dataloader) * num_epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize wandb if available
        if getattr(self.config, 'use_wandb', False):
            wandb.init(
                project="generative-absa",
                name=f"generative-{self.config.dataset_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.__dict__
            )
        
        # Training loop
        training_results = {
            'best_dev_score': 0.0,
            'best_epoch': 0,
            'final_test_results': {},
            'training_history': {
                'train_loss': [],
                'dev_metrics': [],
                'learning_rates': []
            }
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Curriculum learning: adjust tasks based on epoch
            current_tasks = self._get_current_tasks(epoch)
            if current_tasks:
                self.logger.info(f"Epoch {epoch+1}: Curriculum tasks: {current_tasks}")
            
            # Training phase
            train_loss = self._train_epoch(train_dataloader, current_tasks)
            training_results['training_history']['train_loss'].append(train_loss)
            
            # Validation phase
            if dev_dataloader is not None:
                dev_metrics = self._evaluate(dev_dataloader, split='dev')
                training_results['training_history']['dev_metrics'].append(dev_metrics)
                
                # Check for best model
                current_score = dev_metrics.get('triplet_recovery_score', 0.0)
                if current_score > training_results['best_dev_score']:
                    training_results['best_dev_score'] = current_score
                    training_results['best_epoch'] = epoch
                    
                    # Save best model
                    if hasattr(self.config, 'output_dir'):
                        self.best_model_path = os.path.join(
                            self.config.output_dir, 
                            f'best_generative_model_{self.config.dataset_name}.pt'
                        )
                        self.model.save_generative_model(self.best_model_path)
                
                self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, "
                               f"Dev TRS: {current_score:.4f}, Best: {training_results['best_dev_score']:.4f}")
            
            # Log to wandb
            if getattr(self.config, 'use_wandb', False):
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                }
                if dev_dataloader is not None:
                    log_dict.update(dev_metrics)
                wandb.log(log_dict)
        
        # Final evaluation on test set
        if test_dataloader is not None:
            self.logger.info("🧪 Final evaluation on test set...")
            test_metrics = self._evaluate(test_dataloader, split='test')
            training_results['final_test_results'] = test_metrics
            
            self.logger.info(f"Final Test Results:")
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {metric}: {value:.4f}")
        
        # Save final model
        if hasattr(self.config, 'output_dir'):
            final_model_path = os.path.join(
                self.config.output_dir, 
                f'final_generative_model_{self.config.dataset_name}.pt'
            )
            self.model.save_generative_model(final_model_path)
        
        self.logger.info("✅ Generative training completed!")
        return training_results
    
    def _get_current_tasks(self, epoch: int) -> Optional[List[str]]:
        """Get current tasks based on curriculum learning schedule"""
        if not self.use_curriculum or not self.curriculum_schedule:
            return None
        
        cumulative_epochs = 0
        for phase_name, phase_config in self.curriculum_schedule.items():
            phase_epochs = phase_config['epochs']
            if epoch < cumulative_epochs + phase_epochs:
                return phase_config['tasks']
            cumulative_epochs += phase_epochs
        
        # Return all tasks if beyond curriculum
        return None
    
    def _train_epoch(self, dataloader: DataLoader, current_tasks: Optional[List[str]] = None) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Training Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Filter batch by curriculum tasks if specified
            if current_tasks:
                batch = self._filter_batch_by_tasks(batch, current_tasks)
                if not batch['input_ids'].size(0):  # Skip empty batch
                    continue
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.current_step += 1
            else:
                # Standard training
                loss = self._compute_loss(batch)
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.current_step += 1
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Periodic evaluation
            if self.current_step % self.eval_steps == 0 and hasattr(self, 'dev_dataloader'):
                eval_metrics = self._evaluate(self.dev_dataloader, split='dev')
                self.model.train()  # Return to training mode
        
        return total_loss / num_batches
    
    def _filter_batch_by_tasks(self, batch: Dict[str, Any], allowed_tasks: List[str]) -> Dict[str, Any]:
        """Filter batch to only include examples from allowed tasks"""
        if 'task_type' not in batch:
            return batch
        
        # Get task types for each example
        task_types = batch['task_type']
        
        # Create mask for allowed tasks
        mask = torch.tensor([task in allowed_tasks for task in task_types], dtype=torch.bool)
        
        # Filter batch
        filtered_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.size(0) == len(task_types):
                filtered_batch[key] = value[mask]
            elif isinstance(value, list) and len(value) == len(task_types):
                filtered_batch[key] = [value[i] for i in range(len(value)) if mask[i]]
            else:
                filtered_batch[key] = value
        
        return filtered_batch
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute generative loss for batch"""
        
        # Standard generation loss
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            target_ids=batch['target_ids']
        )
        
        generation_loss = outputs['loss']
        
        # Additional losses
        total_loss = generation_loss
        
        # Triplet recovery loss (if we have original triplets)
        if 'original_triplets' in batch and hasattr(self, 'triplet_recovery_loss'):
            # Generate predictions for triplet recovery
            with torch.no_grad():
                generated_outputs = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task_type='triplet_generation'
                )
            
            triplet_loss = self.triplet_recovery_loss(
                generated_outputs, 
                batch['original_triplets']
            )
            
            total_loss += 0.1 * triplet_loss  # Weight triplet recovery loss
        
        # Consistency loss (if using existing model)
        if hasattr(self.model, 'use_existing_backbone') and self.model.use_existing_backbone:
            consistency_weight = getattr(self.config, 'consistency_loss_weight', 0.05)
            total_loss += consistency_weight * generation_loss  # Simple consistency term
        
        return total_loss
    
    def _evaluate(self, dataloader: DataLoader, split: str = 'dev') -> Dict[str, float]:
        """Evaluate generative model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_original_triplets = []
        all_generated_texts = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate predictions
                generated_outputs = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task_type=batch.get('task_type', ['triplet_generation'])[0]
                )
                
                # Collect results
                if isinstance(generated_outputs, list):
                    for output in generated_outputs:
                        all_generated_texts.append(output.generated_text)
                        all_predictions.append(output.triplets)
                else:
                    all_generated_texts.append(generated_outputs.generated_text)
                    all_predictions.append(generated_outputs.triplets)
                
                # Collect targets
                if 'original_triplets' in batch:
                    for triplets in batch['original_triplets']:
                        all_original_triplets.append(triplets)
                
                if 'target_text' in batch:
                    all_targets.extend(batch['target_text'])
        
        # Compute metrics
        metrics = self.metrics.compute_all_metrics(
            predictions=all_predictions,
            targets=all_original_triplets,
            generated_texts=all_generated_texts,
            target_texts=all_targets
        )
        
        return metrics
    
    def predict(self, 
                text: str, 
                task_type: str = 'triplet_generation',
                return_explanation: bool = True) -> Dict[str, Any]:
        """Generate prediction for single text"""
        self.model.eval()
        
        with torch.no_grad():
            # Generate triplets
            triplet_output = self.model.generate_with_prompt(text, task_type='triplet_generation')
            
            # Generate explanation if requested
            explanation_output = None
            if return_explanation:
                explanation_output = self.model.generate_with_prompt(text, task_type='explanation_generation')
            
            # Unified analysis
            unified_output = self.model.unified_analysis(text)
        
        return {
            'input_text': text,
            'triplets': triplet_output.triplets,
            'explanation': explanation_output.explanations if explanation_output else None,
            'unified_analysis': unified_output,
            'generated_triplet_text': triplet_output.generated_text,
            'generated_explanation_text': explanation_output.generated_text if explanation_output else None
        }
    
    def batch_predict(self, 
                     texts: List[str], 
                     task_type: str = 'triplet_generation') -> List[Dict[str, Any]]:
        """Generate predictions for batch of texts"""
        results = []
        
        for text in tqdm(texts, desc="Generating predictions"):
            result = self.predict(text, task_type)
            results.append(result)
        
        return results
    
    def save_predictions(self, 
                        predictions: List[Dict[str, Any]], 
                        output_path: str):
        """Save predictions to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ Predictions saved to {output_path}")
    
    def load_best_model(self):
        """Load the best model from training"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model = UnifiedGenerativeABSA.load_generative_model(
                self.best_model_path, 
                device=self.device
            )
            self.logger.info(f"✅ Loaded best model from {self.best_model_path}")
        else:
            self.logger.warning("⚠️ No best model path available")


class HybridABSATrainer(GenerativeABSATrainer):
    """
    Hybrid trainer that combines classification and generation training
    Integrates your existing classification model with generative capabilities
    """
    
    def __init__(self, 
                 model: UnifiedGenerativeABSA,
                 classification_trainer: Optional[Any] = None,
                 config=None,
                 device: torch.device = None,
                 logger: Optional[logging.Logger] = None):
        
        super().__init__(model, config, device, logger)
        
        self.classification_trainer = classification_trainer
        self.training_mode = getattr(config, 'training_mode', 'hybrid')  # 'classification', 'generation', 'hybrid'
        
        # Hybrid training weights
        self.classification_weight = getattr(config, 'classification_weight', 0.5)
        self.generation_weight = getattr(config, 'generation_weight', 0.5)
        
        self.logger.info(f"✅ Hybrid trainer initialized")
        self.logger.info(f"   Training mode: {self.training_mode}")
        self.logger.info(f"   Classification weight: {self.classification_weight}")
        self.logger.info(f"   Generation weight: {self.generation_weight}")
    
    def _compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute hybrid loss combining classification and generation"""
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Generation loss
        if self.training_mode in ['generation', 'hybrid']:
            generation_loss = super()._compute_loss(batch)
            total_loss += self.generation_weight * generation_loss
        
        # Classification loss (if existing model is available)
        if (self.training_mode in ['classification', 'hybrid'] and 
            hasattr(self.model, 'existing_model') and 
            self.model.existing_model is not None):
            
            # Get classification outputs from existing model
            classification_outputs = self.model.existing_model(
                batch['input_ids'], 
                batch['attention_mask']
            )
            
            # Compute classification loss (simplified)
            if isinstance(classification_outputs, dict) and 'loss' in classification_outputs:
                classification_loss = classification_outputs['loss']
                total_loss += self.classification_weight * classification_loss
        
        return total_loss
    
    def set_training_mode(self, mode: str):
        """Set training mode for hybrid trainer"""
        if mode not in ['classification', 'generation', 'hybrid']:
            raise ValueError(f"Invalid training mode: {mode}")
        
        self.training_mode = mode
        self.model.set_training_mode(mode)
        self.logger.info(f"✅ Training mode set to: {mode}")


def create_generative_trainer(config, model: UnifiedGenerativeABSA, device: torch.device) -> GenerativeABSATrainer:
    """Factory function to create generative trainer"""
    
    training_type = getattr(config, 'training_type', 'generative')  # 'generative', 'hybrid'
    
    if training_type == 'hybrid':
        trainer = HybridABSATrainer(model, config=config, device=device)
    else:
        trainer = GenerativeABSATrainer(model, config, device)
    
    return trainer