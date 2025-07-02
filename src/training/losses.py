# src/training/losses.py - Enhanced version with implicit detection integration
import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrastive_losses import ITSCLLoss, ContrastiveVerificationModule, MultiLevelContrastiveLoss
from typing import Dict, Any, Optional


class FocalLossWithLS(nn.Module):
    """Focal Loss with Label Smoothing for imbalanced classification"""
    
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            ce_loss = ce_loss * at
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImplicitDetectionLoss(nn.Module):
    """
    Specialized loss for implicit detection following 2024-2025 breakthrough standards
    Implements Grid Tagging Matching loss and sentiment combination vector loss
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Loss weights for different implicit components
        self.implicit_aspect_weight = getattr(config, 'implicit_aspect_weight', 1.0)
        self.implicit_opinion_weight = getattr(config, 'implicit_opinion_weight', 1.0)
        self.combination_weight = getattr(config, 'combination_weight', 0.5)
        self.grid_tagging_weight = getattr(config, 'grid_tagging_weight', 0.8)
        self.confidence_weight = getattr(config, 'confidence_weight', 0.3)
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        self.mse_loss = nn.MSELoss(reduction='none')
        
        # Grid tagging matrix loss (for relationship modeling)
        self.grid_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute implicit detection losses
        
        Args:
            outputs: Model outputs containing implicit detection results
            targets: Target labels for implicit elements
            
        Returns:
            Dictionary of computed losses
        """
        device = next(iter(outputs.values())).device
        losses = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 1. Implicit aspect detection loss
        if 'implicit_aspect_scores' in outputs and 'implicit_aspect_labels' in targets:
            implicit_aspect_loss = self._compute_implicit_aspect_loss(
                outputs['implicit_aspect_scores'],
                targets['implicit_aspect_labels']
            )
            losses['implicit_aspect_loss'] = implicit_aspect_loss
            total_loss = total_loss + self.implicit_aspect_weight * implicit_aspect_loss
        
        # 2. Implicit opinion detection loss
        if 'implicit_opinion_scores' in outputs and 'implicit_opinion_labels' in targets:
            implicit_opinion_loss = self._compute_implicit_opinion_loss(
                outputs['implicit_opinion_scores'],
                targets['implicit_opinion_labels']
            )
            losses['implicit_opinion_loss'] = implicit_opinion_loss
            total_loss = total_loss + self.implicit_opinion_weight * implicit_opinion_loss
        
        # 3. Sentiment combination vectors loss (EMNLP 2024 approach)
        if 'aspect_sentiment_combinations' in outputs and 'sentiment_combination_labels' in targets:
            combination_loss = self._compute_sentiment_combination_loss(
                outputs['aspect_sentiment_combinations'],
                targets['sentiment_combination_labels']
            )
            losses['sentiment_combination_loss'] = combination_loss
            total_loss = total_loss + self.combination_weight * combination_loss
        
        # 4. Grid tagging matrix loss (GM-GTM approach)
        if 'aspect_grid_logits' in outputs and 'grid_labels' in targets:
            grid_loss = self._compute_grid_tagging_loss(
                outputs['aspect_grid_logits'],
                targets['grid_labels']
            )
            losses['grid_tagging_loss'] = grid_loss
            total_loss = total_loss + self.grid_tagging_weight * grid_loss
        
        # 5. Implicit-explicit combination loss
        if 'combination_logits' in outputs and 'combination_labels' in targets:
            combination_class_loss = self._compute_combination_classification_loss(
                outputs['combination_logits'],
                targets['combination_labels']
            )
            losses['combination_classification_loss'] = combination_class_loss
            total_loss = total_loss + self.combination_weight * combination_class_loss
        
        # 6. Confidence scoring loss
        if 'confidence_scores' in outputs and 'confidence_labels' in targets:
            confidence_loss = self._compute_confidence_loss(
                outputs['confidence_scores'],
                targets['confidence_labels']
            )
            losses['confidence_loss'] = confidence_loss
            total_loss = total_loss + self.confidence_weight * confidence_loss
        
        losses['total_implicit_loss'] = total_loss
        return losses
    
    def _compute_implicit_aspect_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for implicit aspect detection"""
        # Binary classification loss for implicit vs explicit aspects
        valid_mask = labels != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask].float()
        
        loss = self.bce_loss(valid_scores, valid_labels)
        return loss.mean()
    
    def _compute_implicit_opinion_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for implicit opinion detection"""
        valid_mask = labels != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask].float()
        
        loss = self.bce_loss(valid_scores, valid_labels)
        return loss.mean()
    
    def _compute_sentiment_combination_loss(self, combinations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for sentiment combination vectors (4 fully connected layers approach)
        Following EMNLP 2024 breakthrough for implicit aspect detection
        """
        batch_size, seq_len, num_classes = combinations.shape
        
        # Reshape for loss computation
        combinations_flat = combinations.view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        # Filter out padding tokens
        valid_mask = labels_flat != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=combinations.device, requires_grad=True)
        
        valid_combinations = combinations_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]
        
        loss = self.cross_entropy(valid_combinations, valid_labels)
        return loss.mean()
    
    def _compute_grid_tagging_loss(self, grid_logits: torch.Tensor, grid_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute Grid Tagging Matching (GM-GTM) loss for relationship modeling
        Following the causality-compliant output template design
        """
        batch_size, seq_len, num_classes_1, num_classes_2 = grid_logits.shape
        
        # Reshape for loss computation
        grid_logits_flat = grid_logits.view(-1, num_classes_1 * num_classes_2)
        grid_labels_flat = grid_labels.view(-1)
        
        # Filter out padding
        valid_mask = grid_labels_flat != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=grid_logits.device, requires_grad=True)
        
        valid_logits = grid_logits_flat[valid_mask]
        valid_labels = grid_labels_flat[valid_mask]
        
        loss = self.cross_entropy(valid_logits, valid_labels)
        return loss.mean()
    
    def _compute_combination_classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for implicit-explicit combination classification"""
        batch_size, seq_len, num_classes = logits.shape
        
        logits_flat = logits.view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        valid_mask = labels_flat != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]
        
        loss = self.cross_entropy(valid_logits, valid_labels)
        return loss.mean()
    
    def _compute_confidence_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for confidence scoring"""
        valid_mask = labels != -100
        if not valid_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask].float()
        
        loss = self.mse_loss(valid_scores, valid_labels)
        return loss.mean()


class EnhancedABSALoss(nn.Module):
    """
    Complete ABSA loss function with implicit detection and contrastive learning
    
    2024-2025 breakthrough: Integrates explicit extraction, implicit detection,
    and instruction-following generation in a unified loss framework.
    """
    def __init__(self, config):
        super().__init__()
        
        # Traditional loss weights
        self.aspect_weight = getattr(config, 'aspect_loss_weight', 2.0)
        self.opinion_weight = getattr(config, 'opinion_loss_weight', 2.0)
        self.sentiment_weight = getattr(config, 'sentiment_loss_weight', 1.0)
        self.boundary_weight = getattr(config, 'boundary_weight', 0.5)
        
        # Instruction-following weights
        self.extraction_weight = getattr(config, 'extraction_weight', 1.0)
        self.generation_weight = getattr(config, 'generation_weight', 0.5)
        
        # Implicit detection weights (NEW 2024-2025)
        self.implicit_weight = getattr(config, 'implicit_weight', 0.8)
        
        # Contrastive learning weights
        self.contrastive_weight = getattr(config, 'contrastive_weight', 1.0)
        self.verification_weight = getattr(config, 'verification_weight', 0.3)
        self.multi_level_weight = getattr(config, 'multi_level_weight', 0.5)
        
        # Loss function parameters
        self.label_smoothing = getattr(config, 'label_smoothing', 0.1)
        self.gamma = getattr(config, 'focal_gamma', 2.0)
        self.use_focal_loss = getattr(config, 'use_focal_loss', True)
        
        # Initialize contrastive learning components
        self.use_contrastive = getattr(config, 'use_contrastive_learning', True)
        if self.use_contrastive:
            self.itscl_loss = ITSCLLoss(config)
            self.verification_module = ContrastiveVerificationModule(config)
            self.multi_level_contrastive = MultiLevelContrastiveLoss(config)
        
        # Initialize implicit detection loss
        self.use_implicit_detection = getattr(config, 'use_implicit_detection', True)
        if self.use_implicit_detection:
            self.implicit_loss = ImplicitDetectionLoss(config)
        
        # Class weights for imbalanced data
        aspect_weights = torch.tensor([0.1, 1.0, 0.8])  # O, B, I
        opinion_weights = torch.tensor([0.1, 1.0, 0.8])
        
        if self.use_focal_loss:
            self.span_criterion = FocalLossWithLS(
                gamma=self.gamma,
                alpha=aspect_weights,
                label_smoothing=self.label_smoothing
            )
            self.opinion_criterion = FocalLossWithLS(
                gamma=self.gamma,
                alpha=opinion_weights,
                label_smoothing=self.label_smoothing
            )
        else:
            self.span_criterion = nn.CrossEntropyLoss(
                weight=aspect_weights,
                label_smoothing=self.label_smoothing,
                ignore_index=-100
            )
            self.opinion_criterion = nn.CrossEntropyLoss(
                weight=opinion_weights,
                label_smoothing=self.label_smoothing,
                ignore_index=-100
            )
        
        # Sentiment loss
        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
            ignore_index=-100
        )
        
        # Boundary loss for better span detection
        self.boundary_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                generation_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute complete ABSA loss with implicit detection integration
        
        Args:
            outputs: Model outputs
            targets: Target labels
            generation_embeddings: Optional embeddings for contrastive learning
            
        Returns:
            Dictionary of computed losses
        """
        device = next(iter(outputs.values())).device
        
        # Validate inputs
        required_outputs = ['aspect_logits', 'opinion_logits', 'sentiment_logits']
        required_targets = ['aspect_labels', 'opinion_labels', 'sentiment_labels']
        
        for key in required_outputs:
            if key not in outputs or outputs[key] is None:
                raise ValueError(f"Missing required output: {key}")
        
        for key in required_targets:
            if key not in targets or targets[key] is None:
                raise ValueError(f"Missing required target: {key}")
        
        losses = {}
        
        # ============================================================================
        # EXPLICIT DETECTION LOSSES (Traditional ABSA)
        # ============================================================================
        
        # Extract components
        aspect_logits = outputs['aspect_logits']
        opinion_logits = outputs['opinion_logits']
        sentiment_logits = outputs['sentiment_logits']
        
        aspect_labels = targets['aspect_labels']
        opinion_labels = targets['opinion_labels']
        sentiment_labels = targets['sentiment_labels']
        
        # Compute traditional losses
        aspect_loss = self._compute_span_loss(aspect_logits, aspect_labels, self.span_criterion, "aspect")
        opinion_loss = self._compute_span_loss(opinion_logits, opinion_labels, self.opinion_criterion, "opinion")
        sentiment_loss = self._compute_sentiment_loss(sentiment_logits, sentiment_labels)
        
        losses.update({
            'aspect_loss': aspect_loss,
            'opinion_loss': opinion_loss,
            'sentiment_loss': sentiment_loss
        })
        
        # Boundary loss if available
        boundary_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if 'boundary_logits' in outputs and self.boundary_weight > 0:
            boundary_loss = self._compute_boundary_loss(
                outputs['boundary_logits'], aspect_labels, opinion_labels
            )
            losses['boundary_loss'] = boundary_loss
        
        # Generation loss if available
        generation_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if 'generation_loss' in outputs:
            generation_loss = outputs['generation_loss']
            losses['generation_loss'] = generation_loss
        
        # ============================================================================
        # IMPLICIT DETECTION LOSSES (2024-2025 BREAKTHROUGH)
        # ============================================================================
        
        implicit_loss_total = torch.tensor(0.0, device=device, requires_grad=True)
        
        if self.use_implicit_detection and self.implicit_loss is not None:
            try:
                implicit_losses = self.implicit_loss(outputs, targets)
                losses.update(implicit_losses)
                implicit_loss_total = implicit_losses.get('total_implicit_loss', torch.tensor(0.0, device=device))
            except Exception as e:
                print(f"Warning: Implicit detection loss computation failed: {e}")
                losses['total_implicit_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ============================================================================
        # CONTRASTIVE LEARNING LOSSES (2024-2025 BREAKTHROUGH)
        # ============================================================================
        
        contrastive_total = torch.tensor(0.0, device=device, requires_grad=True)
        contrastive_components = {}
        
        if self.use_contrastive:
            # 1. ITSCL Loss (InfoNCE + NT-Xent + Cross-Modal)
            try:
                itscl_results = self.itscl_loss(outputs, targets, generation_embeddings)
                contrastive_components.update(itscl_results)
                contrastive_total = contrastive_total + itscl_results.get('contrastive_loss', 0)
            except Exception as e:
                print(f"Warning: ITSCL loss computation failed: {e}")
                contrastive_components['contrastive_loss'] = torch.tensor(0.0, device=device)
            
            # 2. Multi-Level Contrastive Loss
            try:
                if 'span_features' in outputs or 'hidden_states' in outputs:
                    span_features = outputs.get('span_features', outputs.get('hidden_states'))
                    
                    # Extract embeddings for contrastive learning
                    aspect_embeddings = self._extract_embeddings_for_contrastive(
                        span_features, aspect_logits, aspect_labels
                    )
                    opinion_embeddings = self._extract_embeddings_for_contrastive(
                        span_features, opinion_logits, opinion_labels
                    )
                    sentiment_embeddings = self._extract_sentiment_embeddings(
                        span_features, sentiment_logits
                    )
                    
                    # Compute multi-level contrastive loss
                    multi_level_results = self.multi_level_contrastive(
                        aspect_embeddings, opinion_embeddings, sentiment_embeddings,
                        self._flatten_labels(aspect_labels),
                        self._flatten_labels(opinion_labels),
                        self._flatten_labels(sentiment_labels)
                    )
                    
                    contrastive_components.update({
                        f"ml_{k}": v for k, v in multi_level_results.items()
                    })
                    contrastive_total = contrastive_total + multi_level_results.get('multi_level_contrastive_loss', 0)
                    
            except Exception as e:
                print(f"Warning: Multi-level contrastive loss computation failed: {e}")
            
            # 3. Verification Loss
            try:
                verification_results = self.verification_module(outputs, targets)
                contrastive_components.update({
                    f"verify_{k}": v for k, v in verification_results.items()
                })
                contrastive_total = contrastive_total + verification_results.get('verification_loss', 0)
            except Exception as e:
                print(f"Warning: Verification loss computation failed: {e}")
        
        losses.update(contrastive_components)
        
        # ============================================================================
        # COMPUTE FINAL WEIGHTED LOSS
        # ============================================================================
        
        # Traditional extraction losses
        extraction_loss = (
            self.aspect_weight * aspect_loss +
            self.opinion_weight * opinion_loss +
            self.sentiment_weight * sentiment_loss +
            self.boundary_weight * boundary_loss
        )
        
        # Total loss combining all components
        total_loss = (
            self.extraction_weight * extraction_loss +
            self.generation_weight * generation_loss +
            self.implicit_weight * implicit_loss_total +
            self.contrastive_weight * contrastive_total
        )
        
        losses.update({
            'extraction_loss': extraction_loss,
            'total_contrastive_loss': contrastive_total,
            'total_loss': total_loss
        })
        
        return losses
    
    def _compute_span_loss(self, logits: torch.Tensor, labels: torch.Tensor, 
                          criterion, span_type: str) -> torch.Tensor:
        """Compute span detection loss with proper masking"""
        try:
            batch_size, seq_len, num_classes = logits.shape
            
            # Reshape for loss computation
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            # Filter out padding tokens
            valid_mask = labels_flat != -100
            
            if not valid_mask.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            valid_logits = logits_flat[valid_mask]
            valid_labels = labels_flat[valid_mask]
            
            loss = criterion(valid_logits, valid_labels)
            
            return loss
            
        except Exception as e:
            print(f"Error computing {span_type} loss: {e}")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    def _compute_sentiment_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute sentiment classification loss"""
        try:
            batch_size, seq_len, num_classes = logits.shape
            
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            valid_mask = labels_flat != -100
            
            if not valid_mask.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            valid_logits = logits_flat[valid_mask]
            valid_labels = labels_flat[valid_mask]
            
            loss = self.sentiment_criterion(valid_logits, valid_labels)
            
            return loss
            
        except Exception as e:
            print(f"Error computing sentiment loss: {e}")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    def _compute_boundary_loss(self, boundary_logits: torch.Tensor, 
                             aspect_labels: torch.Tensor, opinion_labels: torch.Tensor) -> torch.Tensor:
        """Compute boundary detection loss"""
        try:
            # Create boundary targets from span labels
            boundary_targets = self._create_boundary_targets(aspect_labels, opinion_labels)
            
            # Compute BCE loss
            loss = self.boundary_criterion(boundary_logits.squeeze(-1), boundary_targets.float())
            
            return loss.mean()
            
        except Exception as e:
            print(f"Error computing boundary loss: {e}")
            return torch.tensor(0.0, device=boundary_logits.device, requires_grad=True)
    
    def _create_boundary_targets(self, aspect_labels: torch.Tensor, opinion_labels: torch.Tensor) -> torch.Tensor:
        """Create boundary targets from span labels"""
        batch_size, seq_len = aspect_labels.shape
        boundary_targets = torch.zeros_like(aspect_labels, dtype=torch.float)
        
        for i in range(batch_size):
            for j in range(seq_len):
                # Mark boundaries where tags change from O to B or I to O
                if j > 0:
                    # Aspect boundaries
                    if (aspect_labels[i, j-1] == 0 and aspect_labels[i, j] == 1) or \
                       (aspect_labels[i, j-1] in [1, 2] and aspect_labels[i, j] == 0):
                        boundary_targets[i, j] = 1
                    
                    # Opinion boundaries
                    if (opinion_labels[i, j-1] == 0 and opinion_labels[i, j] == 1) or \
                       (opinion_labels[i, j-1] in [1, 2] and opinion_labels[i, j] == 0):
                        boundary_targets[i, j] = 1
        
        return boundary_targets
    
    def _extract_embeddings_for_contrastive(self, features: torch.Tensor, 
                                           logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for contrastive learning"""
        try:
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            # Find entity positions (B and I tags)
            entity_mask = (preds > 0) | (labels > 0)
            
            # Extract entity features
            batch_size, seq_len, hidden_size = features.shape
            entity_features = []
            
            for i in range(batch_size):
                sample_mask = entity_mask[i]
                if sample_mask.any():
                    sample_features = features[i][sample_mask]
                    # Average pool entity features
                    pooled_features = sample_features.mean(dim=0)
                    entity_features.append(pooled_features)
                else:
                    # Use global average if no entities found
                    entity_features.append(features[i].mean(dim=0))
            
            if entity_features:
                return torch.stack(entity_features)
            else:
                return features.mean(dim=1)  # Fallback to global pooling
                
        except Exception as e:
            print(f"Error extracting embeddings for contrastive learning: {e}")
            return features.mean(dim=1)
    
    def _extract_sentiment_embeddings(self, features: torch.Tensor, sentiment_logits: torch.Tensor) -> torch.Tensor:
        """Extract sentiment-aware embeddings"""
        try:
            # Use sentiment probabilities to weight features
            sentiment_probs = F.softmax(sentiment_logits, dim=-1)
            
            # Weight features by sentiment confidence
            sentiment_weights = sentiment_probs.max(dim=-1, keepdim=True)[0]
            weighted_features = features * sentiment_weights
            
            # Global average pooling
            pooled_features = weighted_features.mean(dim=1)
            
            return pooled_features
            
        except Exception as e:
            print(f"Error extracting sentiment embeddings: {e}")
            return features.mean(dim=1)
    
    def _flatten_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Flatten labels for contrastive learning"""
        return labels.view(-1)


# Additional helper functions for loss computation
def compute_boundary_targets(aspect_labels: torch.Tensor, opinion_labels: torch.Tensor) -> torch.Tensor:
    """Create boundary targets from aspect and opinion labels"""
    batch_size, seq_len = aspect_labels.shape
    boundary_targets = torch.zeros_like(aspect_labels, dtype=torch.float)
    
    for i in range(batch_size):
        for j in range(seq_len):
            if j > 0:
                # Aspect boundaries
                if (aspect_labels[i, j-1] == 0 and aspect_labels[i, j] == 1) or \
                   (aspect_labels[i, j-1] in [1, 2] and aspect_labels[i, j] == 0):
                    boundary_targets[i, j] = 1
                
                # Opinion boundaries  
                if (opinion_labels[i, j-1] == 0 and opinion_labels[i, j] == 1) or \
                   (opinion_labels[i, j-1] in [1, 2] and opinion_labels[i, j] == 0):
                    boundary_targets[i, j] = 1
    
    return boundary_targets


def compute_weighted_loss(losses: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    """Compute weighted combination of multiple losses"""
    total_loss = torch.tensor(0.0, requires_grad=True)
    device = next(iter(losses.values())).device
    total_loss = total_loss.to(device)
    
    for loss_name, loss_value in losses.items():
        if loss_name in weights and torch.is_tensor(loss_value):
            weight = weights[loss_name]
            total_loss = total_loss + weight * loss_value
    
    return total_loss


def create_loss_scheduler(config):
    """Create loss weight scheduler for curriculum learning"""
    
    class LossScheduler:
        def __init__(self, config):
            self.config = config
            self.step = 0
            
            # Initial weights
            self.initial_weights = {
                'extraction_weight': getattr(config, 'extraction_weight', 1.0),
                'implicit_weight': getattr(config, 'implicit_weight', 0.1),  # Start low
                'contrastive_weight': getattr(config, 'contrastive_weight', 0.1),  # Start low
                'generation_weight': getattr(config, 'generation_weight', 0.5)
            }
            
            # Final weights
            self.final_weights = {
                'extraction_weight': getattr(config, 'extraction_weight', 1.0),
                'implicit_weight': getattr(config, 'implicit_weight', 0.8),  # Increase
                'contrastive_weight': getattr(config, 'contrastive_weight', 1.0),  # Increase
                'generation_weight': getattr(config, 'generation_weight', 0.5)
            }
            
            # Warmup steps
            self.warmup_steps = getattr(config, 'loss_warmup_steps', 1000)
        
        def get_current_weights(self):
            """Get current loss weights based on training progress"""
            if self.step < self.warmup_steps:
                # Linear interpolation during warmup
                progress = self.step / self.warmup_steps
                current_weights = {}
                
                for key in self.initial_weights:
                    initial = self.initial_weights[key]
                    final = self.final_weights[key]
                    current_weights[key] = initial + progress * (final - initial)
                
                return current_weights
            else:
                return self.final_weights
        
        def step_update(self):
            """Update step counter"""
            self.step += 1
        
        def reset(self):
            """Reset step counter"""
            self.step = 0
    
    return LossScheduler(config)


# Loss utilities for evaluation and debugging
def analyze_loss_components(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyze loss components for debugging"""
    analysis = {}
    
    for loss_name, loss_value in loss_dict.items():
        if torch.is_tensor(loss_value):
            analysis[loss_name] = {
                'value': loss_value.item(),
                'requires_grad': loss_value.requires_grad,
                'is_finite': torch.isfinite(loss_value).all().item(),
                'magnitude': 'high' if loss_value.item() > 1.0 else 'normal' if loss_value.item() > 0.1 else 'low'
            }
    
    return analysis


def clip_loss_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """Clip gradients of loss-related parameters"""
    loss_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and any(loss_term in name for loss_term in 
                                     ['loss', 'criterion', 'contrastive', 'implicit']):
            loss_params.append(param)
    
    if loss_params:
        return torch.nn.utils.clip_grad_norm_(loss_params, max_norm)
    else:
        return 0.0


def get_loss_statistics(loss_history: List[Dict[str, float]], window_size: int = 100) -> Dict[str, Dict[str, float]]:
    """Get statistics for loss history"""
    if not loss_history:
        return {}
    
    # Get recent window
    recent_losses = loss_history[-window_size:] if len(loss_history) > window_size else loss_history
    
    # Collect all loss names
    all_loss_names = set()
    for loss_dict in recent_losses:
        all_loss_names.update(loss_dict.keys())
    
    statistics = {}
    
    for loss_name in all_loss_names:
        values = [loss_dict.get(loss_name, 0.0) for loss_dict in recent_losses if loss_name in loss_dict]
        
        if values:
            statistics[loss_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': 'decreasing' if len(values) > 10 and np.polyfit(range(len(values)), values, 1)[0] < 0 else 'stable'
            }
    
    return statistics