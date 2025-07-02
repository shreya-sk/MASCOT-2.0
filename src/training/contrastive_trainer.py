"""
Contrastive Training Pipeline for ABSA
Integrates all contrastive learning components for 2024-2025 breakthrough training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import wandb
from tqdm import tqdm

from .contrastive_losses import (
    InfoNCELoss, 
    NTXentLoss, 
    EnhancedTripletLoss, 
    SupervisedContrastiveLoss
)
from .negative_sampling import NegativeSamplingManager
from ..models.implicit_sentiment_detector import ContrastiveImplicitABSA


class ContrastiveABSATrainer:
    """
    Complete contrastive training pipeline for ABSA
    Implements state-of-the-art 2024-2025 contrastive learning techniques
    """
    
    def __init__(self, 
                 model: ContrastiveImplicitABSA,
                 config,
                 device: torch.device,
                 logger=None):
        
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger
        
        # Initialize contrastive loss functions
        self.contrastive_losses = self._initialize_contrastive_losses()
        
        # Initialize negative sampling manager
        self.negative_sampler = NegativeSamplingManager(config)
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        
        # Scheduler setup  
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_score = 0.0
        
        # Loss tracking
        self.loss_history = defaultdict(list)
        self.metrics_history = defaultdict(list)
        
        # Contrastive learning parameters
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.5)
        self.temperature_schedule = getattr(config, 'temperature_schedule', 'constant')
        self.initial_temperature = getattr(config, 'initial_temperature', 0.07)
        self.final_temperature = getattr(config, 'final_temperature', 0.01)
        
        # Memory bank for global contrastive learning
        self.memory_bank = self._initialize_memory_bank()
        
    def _initialize_contrastive_losses(self) -> Dict[str, nn.Module]:
        """Initialize all contrastive loss functions"""
        losses = {}
        
        # InfoNCE Loss
        losses['infonce'] = InfoNCELoss(
            temperature=self.config.contrastive_temperature,
            reduction='mean'
        )
        
        # NT-Xent Loss
        losses['ntxent'] = NTXentLoss(
            temperature=self.config.contrastive_temperature,
            base_temperature=0.07
        )
        
        # Enhanced Triplet Loss
        losses['triplet'] = EnhancedTripletLoss(
            margin=getattr(self.config, 'triplet_margin', 0.3),
            mining_strategy=getattr(self.config, 'triplet_mining', 'hard'),
            distance_metric='euclidean'
        )
        
        # Supervised Contrastive Loss
        losses['supervised'] = SupervisedContrastiveLoss(
            temperature=self.config.contrastive_temperature,
            contrast_mode='all',
            base_temperature=0.07
        )
        
        return losses
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with different learning rates for different components"""
        
        # Separate parameters for different components
        backbone_params = list(self.model.backbone.parameters())
        implicit_params = (list(self.model.implicit_aspect_detector.parameters()) +
                          list(self.model.implicit_opinion_detector.parameters()) +
                          list(self.model.sentiment_combiner.parameters()))
        contrastive_params = []
        
        # Collect contrastive-specific parameters
        for loss_fn in self.contrastive_losses.values():
            if hasattr(loss_fn, 'parameters'):
                contrastive_params.extend(list(loss_fn.parameters()))
        
        # Parameter groups with different learning rates
        param_groups = [
            {
                'params': backbone_params,
                'lr': self.config.learning_rate * 0.1,  # Lower LR for backbone
                'name': 'backbone'
            },
            {
                'params': implicit_params,
                'lr': self.config.learning_rate,
                'name': 'implicit_detection'
            },
            {
                'params': contrastive_params,
                'lr': self.config.learning_rate * 2.0,  # Higher LR for contrastive components
                'name': 'contrastive'
            }
        ]
        
        # Filter out empty parameter groups
        param_groups = [group for group in param_groups if len(group['params']) > 0]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            weight_decay=getattr(self.config, 'weight_decay', 0.01),
            eps=1e-8
        )
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if hasattr(self.config, 'scheduler_type'):
            if self.config.scheduler_type == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                return CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.num_epochs,
                    eta_min=self.config.learning_rate * 0.01
                )
            elif self.config.scheduler_type == 'linear':
                from transformers import get_linear_schedule_with_warmup
                return get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=int(0.1 * self.config.num_epochs * 100),  # Estimate
                    num_training_steps=self.config.num_epochs * 100
                )
        
        return None
    
    def _initialize_memory_bank(self) -> Dict[str, torch.Tensor]:
        """Initialize memory bank for global contrastive learning"""
        memory_size = getattr(self.config, 'memory_bank_size', 4096)
        hidden_size = self.config.hidden_size
        
        memory_bank = {
            'features': torch.randn(memory_size, hidden_size, device=self.device),
            'labels': torch.zeros(memory_size, dtype=torch.long, device=self.device),
            'ptr': torch.zeros(1, dtype=torch.long, device=self.device)
        }
        
        # Normalize initial features
        memory_bank['features'] = F.normalize(memory_bank['features'], dim=1)
        
        return memory_bank
    
    def _update_memory_bank(self, features: torch.Tensor, labels: torch.Tensor):
        """Update memory bank with new features"""
        batch_size = features.size(0)
        ptr = int(self.memory_bank['ptr'])
        
        # Replace oldest features
        self.memory_bank['features'][ptr:ptr + batch_size] = F.normalize(features.detach(), dim=1)
        self.memory_bank['labels'][ptr:ptr + batch_size] = labels.detach()
        
        # Update pointer
        ptr = (ptr + batch_size) % self.memory_bank['features'].size(0)
        self.memory_bank['ptr'][0] = ptr
    
    def _get_current_temperature(self) -> float:
        """Get current temperature based on schedule"""
        if self.temperature_schedule == 'constant':
            return self.initial_temperature
        elif self.temperature_schedule == 'linear':
            progress = self.current_step / (self.config.num_epochs * 100)  # Estimate total steps
            return self.initial_temperature + progress * (self.final_temperature - self.initial_temperature)
        elif self.temperature_schedule == 'cosine':
            progress = self.current_step / (self.config.num_epochs * 100)
            return self.final_temperature + 0.5 * (self.initial_temperature - self.final_temperature) * (1 + np.cos(np.pi * progress))
        else:
            return self.initial_temperature
    
    def compute_contrastive_losses(self,
                                 outputs: Dict[str, torch.Tensor],
                                 batch: Dict[str, torch.Tensor],
                                 negative_samples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all contrastive losses
        
        Args:
            outputs: Model outputs
            batch: Input batch
            negative_samples: Sampled negative examples
            
        Returns:
            Dictionary of contrastive losses
        """
        contrastive_losses = {}
        
        # Extract features and labels
        fused_features = outputs['fused_features']
        attention_mask = batch['attention_mask']
        
        # Get valid tokens (non-padding)
        valid_mask = attention_mask.bool()
        batch_size, seq_len = valid_mask.shape
        
        # Flatten features and labels for contrastive learning
        flat_features = fused_features[valid_mask]  # [num_valid_tokens, hidden_dim]
        
        if 'aspect_labels' in batch:
            flat_aspect_labels = batch['aspect_labels'][valid_mask]
            flat_opinion_labels = batch['opinion_labels'][valid_mask] if 'opinion_labels' in batch else torch.zeros_like(flat_aspect_labels)
            flat_sentiment_labels = batch['sentiment_labels'][valid_mask] if 'sentiment_labels' in batch else torch.zeros_like(flat_aspect_labels)
        else:
            # Skip contrastive learning if no labels
            return {}
        
        # Remove padding tokens (-100 labels)
        non_pad_mask = flat_aspect_labels != -100
        if not non_pad_mask.any():
            return {}
        
        clean_features = flat_features[non_pad_mask]
        clean_aspect_labels = flat_aspect_labels[non_pad_mask]
        clean_opinion_labels = flat_opinion_labels[non_pad_mask]
        clean_sentiment_labels = flat_sentiment_labels[non_pad_mask]
        
        # Create combined labels for triplet-level contrastive learning
        combined_labels = (clean_aspect_labels * 100 + 
                          clean_opinion_labels * 10 + 
                          clean_sentiment_labels)
        
        # 1. InfoNCE Loss with memory bank
        if len(clean_features) > 0 and 'infonce' in self.contrastive_losses:
            try:
                # Sample from memory bank as negatives
                memory_features = self.memory_bank['features']
                memory_labels = self.memory_bank['labels']
                
                # For each clean feature, find positives and negatives
                infonce_losses = []
                for i, (feat, label) in enumerate(zip(clean_features, combined_labels)):
                    # Find positives in current batch
                    pos_mask = combined_labels == label
                    if pos_mask.sum() > 1:  # Need at least one other positive
                        pos_features = clean_features[pos_mask]
                        pos_features = pos_features[pos_features != feat]  # Remove self
                        
                        if len(pos_features) > 0:
                            # Find negatives in memory bank
                            neg_mask = memory_labels != label
                            if neg_mask.any():
                                neg_features = memory_features[neg_mask]
                                
                                # Sample negatives
                                num_negs = min(10, len(neg_features))
                                neg_indices = torch.randperm(len(neg_features))[:num_negs]
                                sampled_negs = neg_features[neg_indices]
                                
                                # Compute InfoNCE
                                query = feat.unsqueeze(0)
                                positives = pos_features[:1].unsqueeze(1)  # Take one positive
                                negatives = sampled_negs.unsqueeze(1)
                                
                                infonce_loss = self.contrastive_losses['infonce'](
                                    query, positives, negatives
                                )
                                infonce_losses.append(infonce_loss)
                
                if infonce_losses:
                    contrastive_losses['infonce'] = torch.stack(infonce_losses).mean()
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"InfoNCE loss computation failed: {e}")
        
        # 2. NT-Xent Loss
        if len(clean_features) > 1 and 'ntxent' in self.contrastive_losses:
            try:
                # Create augmented views (simple dropout)
                aug_features = F.dropout(clean_features, p=0.1, training=True)
                combined_features = torch.cat([clean_features, aug_features], dim=0)
                combined_lbls = torch.cat([combined_labels, combined_labels], dim=0)
                
                ntxent_loss = self.contrastive_losses['ntxent'](
                    combined_features, combined_lbls
                )
                contrastive_losses['ntxent'] = ntxent_loss
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"NT-Xent loss computation failed: {e}")
        
        # 3. Enhanced Triplet Loss
        if len(clean_features) > 2 and 'triplet' in self.contrastive_losses:
            try:
                triplet_outputs = self.contrastive_losses['triplet'](
                    clean_features, combined_labels
                )
                contrastive_losses['triplet'] = triplet_outputs['loss']
                contrastive_losses['triplet_stats'] = {
                    'num_triplets': triplet_outputs['num_triplets'],
                    'valid_triplets': triplet_outputs['valid_triplets']
                }
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Triplet loss computation failed: {e}")
        
        # 4. Supervised Contrastive Loss
        if len(clean_features) > 1 and 'supervised' in self.contrastive_losses:
            try:
                # Create multi-view features for supervised contrastive learning
                num_views = 2
                aug_views = []
                for _ in range(num_views - 1):
                    aug_view = F.dropout(clean_features, p=0.1, training=True)
                    aug_views.append(aug_view)
                
                # Stack views: [batch_size, num_views, hidden_dim]
                multi_view_features = torch.stack([clean_features] + aug_views, dim=1)
                
                supervised_losses = self.contrastive_losses['supervised'](
                    features=multi_view_features,
                    labels=combined_labels,
                    aspect_labels=clean_aspect_labels,
                    opinion_labels=clean_opinion_labels,
                    sentiment_labels=clean_sentiment_labels
                )
                
                # Add all supervised loss components
                for key, value in supervised_losses.items():
                    contrastive_losses[f'supervised_{key}'] = value
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Supervised contrastive loss computation failed: {e}")
        
        # 5. Implicit-Explicit Contrastive Loss
        if 'implicit_aspect_logits' in outputs and 'implicit_opinion_logits' in outputs:
            try:
                implicit_aspect_probs = F.softmax(outputs['implicit_aspect_logits'], dim=-1)
                implicit_opinion_probs = F.softmax(outputs['implicit_opinion_logits'], dim=-1)
                
                # Extract implicit vs explicit predictions
                implicit_aspect_preds = implicit_aspect_probs[:, :, 1]  # Implicit class
                implicit_opinion_preds = implicit_opinion_probs[:, :, 0]  # Implicit class (assuming 0=implicit, 1=explicit, 2=neutral)
                
                # Flatten
                flat_implicit_aspects = implicit_aspect_preds[valid_mask][non_pad_mask]
                flat_implicit_opinions = implicit_opinion_preds[valid_mask][non_pad_mask]
                
                # Contrastive loss between implicit and explicit
                implicit_features = clean_features * flat_implicit_aspects.unsqueeze(-1)
                explicit_features = clean_features * (1 - flat_implicit_aspects.unsqueeze(-1))
                
                if len(implicit_features) > 0 and len(explicit_features) > 0:
                    # Simple contrastive loss between implicit and explicit
                    implicit_mean = implicit_features.mean(dim=0, keepdim=True)
                    explicit_mean = explicit_features.mean(dim=0, keepdim=True)
                    
                    contrastive_sim = F.cosine_similarity(implicit_mean, explicit_mean, dim=1)
                    implicit_explicit_loss = torch.clamp(0.5 - contrastive_sim, min=0).mean()
                    
                    contrastive_losses['implicit_explicit'] = implicit_explicit_loss
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Implicit-explicit contrastive loss computation failed: {e}")
        
        # Update memory bank
        if len(clean_features) > 0:
            self._update_memory_bank(clean_features, combined_labels)
        
        return contrastive_losses
    
    def train_step(self, 
                  batch: Dict[str, torch.Tensor],
                  epoch: int,
                  step: int) -> Dict[str, float]:
        """
        Single training step with contrastive learning
        
        Args:
            batch: Input batch
            epoch: Current epoch
            step: Current step
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        self.current_epoch = epoch
        self.current_step = step
        
        # Update negative sampler
        self.negative_sampler.update_step(step)
        
        # Update temperature
        current_temp = self._get_current_temperature()
        for loss_fn in self.contrastive_losses.values():
            if hasattr(loss_fn, 'temperature'):
                loss_fn.temperature = current_temp
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            aspect_labels=batch.get('aspect_labels'),
            opinion_labels=batch.get('opinion_labels'),
            sentiment_labels=batch.get('sentiment_labels'),
            implicit_labels=batch.get('implicit_labels'),
            syntax_features=batch.get('syntax_features'),
            training=True
        )
        
        # Compute base ABSA losses
        base_losses = self._compute_base_losses(outputs, batch)
        
        # Sample negative examples for contrastive learning
        negative_samples = self._sample_negatives_for_batch(outputs, batch)
        
        # Compute contrastive losses
        contrastive_losses = self.compute_contrastive_losses(outputs, batch, negative_samples)
        
        # Combine all losses
        total_loss = 0.0
        loss_dict = {}
        
        # Base losses
        for name, loss in base_losses.items():
            if loss is not None and not torch.isnan(loss):
                total_loss += loss
                loss_dict[f'base_{name}'] = loss.item()
        
        # Contrastive losses with weighting
        contrastive_weight = self.contrastive_weight * self._get_contrastive_weight_schedule()
        for name, loss in contrastive_losses.items():
            if loss is not None and not torch.isnan(loss):
                weighted_loss = contrastive_weight * loss
                total_loss += weighted_loss
                loss_dict[f'contrastive_{name}'] = loss.item()
                loss_dict[f'weighted_contrastive_{name}'] = weighted_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        loss_dict['contrastive_weight'] = contrastive_weight
        loss_dict['temperature'] = current_temp
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update hard negatives for dynamic sampling
        if contrastive_losses and negative_samples:
            self._update_hard_negatives(outputs, negative_samples, contrastive_losses)
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        # Track losses
        for name, value in loss_dict.items():
            self.loss_history[name].append(value)
        
        return loss_dict
    
    def _compute_base_losses(self, 
                           outputs: Dict[str, torch.Tensor], 
                           batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute base ABSA losses (aspect, opinion, sentiment)"""
        losses = {}
        
        # Standard cross-entropy losses
        if 'aspect_labels' in batch and 'aspect_logits' in outputs:
            aspect_loss = F.cross_entropy(
                outputs['aspect_logits'].view(-1, outputs['aspect_logits'].size(-1)),
                batch['aspect_labels'].view(-1),
                ignore_index=-100
            )
            losses['aspect'] = aspect_loss
        
        if 'opinion_labels' in batch and 'opinion_logits' in outputs:
            opinion_loss = F.cross_entropy(
                outputs['opinion_logits'].view(-1, outputs['opinion_logits'].size(-1)),
                batch['opinion_labels'].view(-1),
                ignore_index=-100
            )
            losses['opinion'] = opinion_loss
        
        if 'sentiment_labels' in batch and 'sentiment_logits' in outputs:
            sentiment_loss = F.cross_entropy(
                outputs['sentiment_logits'].view(-1, outputs['sentiment_logits'].size(-1)),
                batch['sentiment_labels'].view(-1),
                ignore_index=-100
            )
            losses['sentiment'] = sentiment_loss
        
        # Implicit detection losses
        if 'implicit_labels' in batch:
            if 'implicit_aspect_logits' in outputs:
                implicit_aspect_loss = F.cross_entropy(
                    outputs['implicit_aspect_logits'].view(-1, outputs['implicit_aspect_logits'].size(-1)),
                    batch['implicit_labels'].view(-1),
                    ignore_index=-100
                )
                losses['implicit_aspect'] = implicit_aspect_loss
            
            if 'implicit_opinion_logits' in outputs:
                implicit_opinion_loss = F.cross_entropy(
                    outputs['implicit_opinion_logits'].view(-1, outputs['implicit_opinion_logits'].size(-1)),
                    batch['implicit_labels'].view(-1),
                    ignore_index=-100
                )
                losses['implicit_opinion'] = implicit_opinion_loss
        
        return losses
    
    def _sample_negatives_for_batch(self, 
                                   outputs: Dict[str, torch.Tensor],
                                   batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Sample negative examples for current batch"""
        if 'fused_features' not in outputs:
            return {}
        
        features = outputs['fused_features']
        attention_mask = batch['attention_mask']
        
        # Get valid features (non-padding)
        valid_mask = attention_mask.bool()
        flat_features = features[valid_mask]
        
        if 'aspect_labels' in batch:
            flat_labels = batch['aspect_labels'][valid_mask]
            non_pad_mask = flat_labels != -100
            
            if non_pad_mask.any():
                clean_features = flat_features[non_pad_mask]
                clean_labels = flat_labels[non_pad_mask]
                
                # Use negative sampling manager
                negative_results = self.negative_sampler.sample_negatives(
                    anchor_features=clean_features,
                    anchor_labels=clean_labels,
                    candidate_features=clean_features,  # Use same batch as candidates
                    candidate_labels=clean_labels,
                    num_negatives=10
                )
                
                return negative_results
        
        return {}
    
    def _update_hard_negatives(self,
                             outputs: Dict[str, torch.Tensor],
                             negative_samples: Dict[str, torch.Tensor],
                             contrastive_losses: Dict[str, torch.Tensor]):
        """Update hard negatives for dynamic sampling"""
        if ('negative_features' in negative_samples and 
            'negative_labels' in negative_samples and
            len(negative_samples['negative_features']) > 0):
            
            # Use contrastive loss values as difficulty scores
            if contrastive_losses:
                avg_loss = torch.mean(torch.stack([
                    loss for loss in contrastive_losses.values() 
                    if isinstance(loss, torch.Tensor) and loss.dim() == 0
                ]))
                
                # Create difficulty scores (higher loss = harder negative)
                num_negatives = len(negative_samples['negative_features'])
                difficulty_scores = torch.full((num_negatives,), avg_loss.item(), 
                                             device=self.device)
                
                self.negative_sampler.update_hard_negatives(
                    negative_samples['negative_features'],
                    negative_samples['negative_labels'],
                    difficulty_scores
                )
    
    def _get_contrastive_weight_schedule(self) -> float:
        """Get current contrastive loss weight based on schedule"""
        # Gradually increase contrastive weight during training
        progress = self.current_step / (self.config.num_epochs * 100)  # Estimate
        return min(1.0, progress * 2.0)  # Ramp up to full weight
    
    def validate_step(self, 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single validation step
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                aspect_labels=batch.get('aspect_labels'),
                opinion_labels=batch.get('opinion_labels'),
                sentiment_labels=batch.get('sentiment_labels'),
                implicit_labels=batch.get('implicit_labels'),
                syntax_features=batch.get('syntax_features'),
                training=False
            )
            
            # Compute base losses
            base_losses = self._compute_base_losses(outputs, batch)
            
            # Compute metrics
            metrics = self._compute_validation_metrics(outputs, batch)
            
            # Combine losses and metrics
            results = {}
            for name, loss in base_losses.items():
                if loss is not None:
                    results[f'val_{name}_loss'] = loss.item()
            
            for name, metric in metrics.items():
                results[f'val_{name}'] = metric
            
            return results
    
    def _compute_validation_metrics(self, 
                                   outputs: Dict[str, torch.Tensor],
                                   batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute validation metrics"""
        metrics = {}
        
        # Extract predictions and labels
        attention_mask = batch['attention_mask']
        valid_mask = attention_mask.bool()
        
        if 'aspect_labels' in batch and 'aspect_logits' in outputs:
            aspect_preds = outputs['aspect_logits'].argmax(dim=-1)
            aspect_labels = batch['aspect_labels']
            
            # Flatten and remove padding
            flat_preds = aspect_preds[valid_mask]
            flat_labels = aspect_labels[valid_mask]
            non_pad_mask = flat_labels != -100
            
            if non_pad_mask.any():
                clean_preds = flat_preds[non_pad_mask]
                clean_labels = flat_labels[non_pad_mask]
                
                # Accuracy
                aspect_acc = (clean_preds == clean_labels).float().mean().item()
                metrics['aspect_accuracy'] = aspect_acc
                
                # F1 score (binary case)
                if len(torch.unique(clean_labels)) == 2:
                    tp = ((clean_preds == 1) & (clean_labels == 1)).sum().item()
                    fp = ((clean_preds == 1) & (clean_labels == 0)).sum().item()
                    fn = ((clean_preds == 0) & (clean_labels == 1)).sum().item()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    metrics['aspect_precision'] = precision
                    metrics['aspect_recall'] = recall
                    metrics['aspect_f1'] = f1
        
        # Similar metrics for opinion and sentiment
        if 'opinion_labels' in batch and 'opinion_logits' in outputs:
            opinion_preds = outputs['opinion_logits'].argmax(dim=-1)
            opinion_labels = batch['opinion_labels']
            
            flat_preds = opinion_preds[valid_mask]
            flat_labels = opinion_labels[valid_mask]
            non_pad_mask = flat_labels != -100
            
            if non_pad_mask.any():
                clean_preds = flat_preds[non_pad_mask]
                clean_labels = flat_labels[non_pad_mask]
                
                opinion_acc = (clean_preds == clean_labels).float().mean().item()
                metrics['opinion_accuracy'] = opinion_acc
        
        if 'sentiment_labels' in batch and 'sentiment_logits' in outputs:
            sentiment_preds = outputs['sentiment_logits'].argmax(dim=-1)
            sentiment_labels = batch['sentiment_labels']
            
            flat_preds = sentiment_preds[valid_mask]
            flat_labels = sentiment_labels[valid_mask]
            non_pad_mask = flat_labels != -100
            
            if non_pad_mask.any():
                clean_preds = flat_preds[non_pad_mask]
                clean_labels = flat_labels[non_pad_mask]
                
                sentiment_acc = (clean_preds == clean_labels).float().mean().item()
                metrics['sentiment_accuracy'] = sentiment_acc
        
        return metrics
    
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses and metrics
        """
        self.model.train()
        epoch_losses = defaultdict(list)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for step, batch in enumerate(pbar):
            # Training step
            step_losses = self.train_step(batch, epoch, step)
            
            # Track losses
            for name, value in step_losses.items():
                epoch_losses[name].append(value)
            
            # Update progress bar
            if step % 10 == 0:
                current_loss = step_losses.get('total_loss', 0.0)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            # Log to wandb if available
            if hasattr(self.config, 'use_wandb') and self.config.use_wandb:
                wandb.log({
                    **{f'train_{k}': v for k, v in step_losses.items()},
                    'epoch': epoch,
                    'step': self.current_step
                })
        
        # Calculate average losses
        avg_losses = {}
        for name, values in epoch_losses.items():
            avg_losses[f'avg_{name}'] = np.mean(values)
        
        return avg_losses
    
    def validate_epoch(self, 
                      val_loader: DataLoader, 
                      epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(val_loader, desc=f'Validation {epoch}')
        for batch in pbar:
            # Validation step
            step_metrics = self.validate_step(batch)
            
            # Track metrics
            for name, value in step_metrics.items():
                epoch_metrics[name].append(value)
        
        # Calculate average metrics
        avg_metrics = {}
        for name, values in epoch_metrics.items():
            avg_metrics[name] = np.mean(values)
        
        return avg_metrics
    
    def save_checkpoint(self, 
                       checkpoint_path: str, 
                       epoch: int, 
                       metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'loss_history': dict(self.loss_history),
            'metrics_history': dict(self.metrics_history),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config,
            'memory_bank': self.memory_bank,
            'current_step': self.current_step,
            'best_score': self.best_score
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if self.logger:
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_step = checkpoint.get('current_step', 0)
        self.best_score = checkpoint.get('best_score', 0.0)
        self.loss_history = defaultdict(list, checkpoint.get('loss_history', {}))
        self.metrics_history = defaultdict(list, checkpoint.get('metrics_history', {}))
        
        if 'memory_bank' in checkpoint:
            self.memory_bank = checkpoint['memory_bank']
        
        if self.logger:
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress"""
        return {
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_score': self.best_score,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'loss_history_keys': list(self.loss_history.keys()),
            'metrics_history_keys': list(self.metrics_history.keys()),
            'memory_bank_size': self.memory_bank['features'].size(0),
            'contrastive_losses': list(self.contrastive_losses.keys())
        }