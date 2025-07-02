# src/training/implicit_losses.py
"""
Advanced Loss Functions for Implicit Sentiment Detection
Implements state-of-the-art loss functions for implicit-explicit alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class ImplicitDetectionLoss(nn.Module):
    """
    Complete loss function for implicit sentiment detection
    Combines multiple loss components for comprehensive training
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Loss weights
        self.implicit_aspect_weight = getattr(config, 'implicit_aspect_weight', 1.0)
        self.implicit_opinion_weight = getattr(config, 'implicit_opinion_weight', 1.0) 
        self.combination_weight = getattr(config, 'combination_weight', 0.5)
        self.grid_tagging_weight = getattr(config, 'grid_tagging_weight', 0.8)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 0.3)
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
        
        # 5. Contrastive alignment loss for implicit-explicit pairs
        if ('aspect_projections' in outputs and 'opinion_projections' in outputs and 
            'contrastive_labels' in targets):
            contrastive_loss = self._compute_contrastive_alignment_loss(
                outputs['aspect_projections'],
                outputs['opinion_projections'],
                targets['contrastive_labels']
            )
            losses['contrastive_alignment_loss'] = contrastive_loss
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
        
        # 6. Confidence scoring loss
        if 'confidence_scores' in outputs and 'confidence_labels' in targets:
            confidence_loss = self._compute_confidence_loss(
                outputs['confidence_scores'],
                targets['confidence_labels']
            )
            losses['confidence_loss'] = confidence_loss
            total_loss = total_loss + self.confidence_weight * confidence_loss
        
        # 7. Pattern-based sentiment loss for implicit opinions
        if 'pattern_outputs' in outputs and 'pattern_sentiment_labels' in targets:
            pattern_loss = self._compute_pattern_sentiment_loss(
                outputs['pattern_outputs'],
                targets['pattern_sentiment_labels']
            )
            losses['pattern_sentiment_loss'] = pattern_loss
            total_loss = total_loss + 0.4 * pattern_loss
        
        losses['total_implicit_loss'] = total_loss
        return losses
    
    def _compute_implicit_aspect_loss(self, 
                                    aspect_scores: torch.Tensor, 
                                    aspect_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for implicit aspect detection
        
        Args:
            aspect_scores: [batch_size, seq_len, 3] - (explicit, implicit, none)
            aspect_labels: [batch_size, seq_len] - target labels
            
        Returns:
            Computed loss
        """
        # Handle ignore labels (-100)
        valid_mask = (aspect_labels != -100)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=aspect_scores.device, requires_grad=True)
        
        # Flatten for loss computation
        flat_scores = aspect_scores.view(-1, 3)[valid_mask.view(-1)]
        flat_labels = aspect_labels.view(-1)[valid_mask.view(-1)]
        
        # Cross entropy loss with class weights for imbalanced data
        class_weights = torch.tensor([1.0, 2.0, 0.5], device=aspect_scores.device)  # Weight implicit class more
        loss = F.cross_entropy(flat_scores, flat_labels, weight=class_weights)
        
        return loss
    
    def _compute_implicit_opinion_loss(self,
                                     opinion_scores: torch.Tensor,
                                     opinion_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for implicit opinion detection
        
        Args:
            opinion_scores: [batch_size, seq_len, 3] - (explicit, implicit, none)
            opinion_labels: [batch_size, seq_len] - target labels
            
        Returns:
            Computed loss
        """
        # Handle ignore labels (-100)
        valid_mask = (opinion_labels != -100)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=opinion_scores.device, requires_grad=True)
        
        # Flatten for loss computation
        flat_scores = opinion_scores.view(-1, 3)[valid_mask.view(-1)]
        flat_labels = opinion_labels.view(-1)[valid_mask.view(-1)]
        
        # Cross entropy loss with class weights
        class_weights = torch.tensor([1.0, 2.5, 0.5], device=opinion_scores.device)  # Weight implicit class more
        loss = F.cross_entropy(flat_scores, flat_labels, weight=class_weights)
        
        return loss
    
    def _compute_sentiment_combination_loss(self,
                                          combination_scores: torch.Tensor,
                                          combination_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for aspect-sentiment combination vectors (EMNLP 2024)
        
        Args:
            combination_scores: [batch_size, seq_len, 3] - sentiment combination scores
            combination_labels: [batch_size, seq_len] - target sentiment labels
            
        Returns:
            Computed loss
        """
        valid_mask = (combination_labels != -100)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=combination_scores.device, requires_grad=True)
        
        flat_scores = combination_scores.view(-1, 3)[valid_mask.view(-1)]
        flat_labels = combination_labels.view(-1)[valid_mask.view(-1)]
        
        # Use focal loss to handle hard examples
        ce_loss = F.cross_entropy(flat_scores, flat_labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        
        return focal_loss.mean()
    
    def _compute_grid_tagging_loss(self,
                                 grid_logits: torch.Tensor,
                                 grid_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for grid-based tagging matrix (GM-GTM approach)
        
        Args:
            grid_logits: [batch_size, seq_len, 4] - grid tagging logits
            grid_labels: [batch_size, seq_len] - grid tagging labels
            
        Returns:
            Computed loss
        """
        valid_mask = (grid_labels != -100)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=grid_logits.device, requires_grad=True)
        
        flat_logits = grid_logits.view(-1, 4)[valid_mask.view(-1)]
        flat_labels = grid_labels.view(-1)[valid_mask.view(-1)]
        
        # Standard cross entropy for grid tagging
        loss = F.cross_entropy(flat_logits, flat_labels)
        
        return loss
    
    def _compute_contrastive_alignment_loss(self,
                                          aspect_projections: torch.Tensor,
                                          opinion_projections: torch.Tensor,
                                          contrastive_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive alignment loss for implicit-explicit pairs
        
        Args:
            aspect_projections: [batch_size, seq_len, 128] - aspect projections
            opinion_projections: [batch_size, seq_len, 128] - opinion projections
            contrastive_labels: [batch_size, seq_len] - alignment labels (1 for aligned, 0 for not)
            
        Returns:
            Computed contrastive loss
        """
        batch_size, seq_len, proj_dim = aspect_projections.shape
        
        # Normalize projections
        aspect_norm = F.normalize(aspect_projections, dim=-1)
        opinion_norm = F.normalize(opinion_projections, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.bmm(aspect_norm, opinion_norm.transpose(1, 2))  # [batch, seq_len, seq_len]
        
        # Create positive and negative pairs based on labels
        positive_mask = (contrastive_labels.unsqueeze(2) * contrastive_labels.unsqueeze(1)).bool()
        negative_mask = ~positive_mask
        
        # InfoNCE-style contrastive loss
        temperature = 0.07
        similarity_scaled = similarity / temperature
        
        # Positive pairs should have high similarity
        positive_loss = -torch.log(torch.exp(similarity_scaled[positive_mask]).sum() + 1e-8)
        
        # Negative pairs should have low similarity  
        negative_loss = torch.log(torch.exp(similarity_scaled[negative_mask]).sum() + 1e-8)
        
        total_loss = positive_loss + negative_loss
        
        return total_loss
    
    def _compute_confidence_loss(self,
                               confidence_scores: torch.Tensor,
                               confidence_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for confidence scoring
        
        Args:
            confidence_scores: [batch_size, seq_len] - predicted confidence scores
            confidence_labels: [batch_size, seq_len] - target confidence scores
            
        Returns:
            Computed loss
        """
        valid_mask = (confidence_labels != -100)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=confidence_scores.device, requires_grad=True)
        
        valid_pred = confidence_scores[valid_mask]
        valid_target = confidence_labels[valid_mask].float()
        
        # MSE loss for confidence regression
        loss = F.mse_loss(valid_pred, valid_target)
        
        return loss
    
    def _compute_pattern_sentiment_loss(self,
                                      pattern_outputs: torch.Tensor,
                                      pattern_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for pattern-based sentiment classification
        
        Args:
            pattern_outputs: [batch_size, seq_len, 3] - pattern sentiment logits
            pattern_labels: [batch_size, seq_len] - target sentiment labels
            
        Returns:
            Computed loss
        """
        valid_mask = (pattern_labels != -100)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pattern_outputs.device, requires_grad=True)
        
        flat_outputs = pattern_outputs.view(-1, 3)[valid_mask.view(-1)]
        flat_labels = pattern_labels.view(-1)[valid_mask.view(-1)]
        
        # Cross entropy loss
        loss = F.cross_entropy(flat_outputs, flat_labels)
        
        return loss


class ImplicitExplicitAlignmentLoss(nn.Module):
    """
    Specialized loss for aligning implicit and explicit sentiment elements
    Implements contrastive learning between implicit and explicit representations
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, 
                implicit_features: torch.Tensor,
                explicit_features: torch.Tensor,
                alignment_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between implicit and explicit features
        
        Args:
            implicit_features: [batch_size, seq_len, hidden_size]
            explicit_features: [batch_size, seq_len, hidden_size] 
            alignment_labels: [batch_size, seq_len] - 1 for aligned, 0 for not aligned
            
        Returns:
            Alignment loss
        """
        batch_size, seq_len, hidden_size = implicit_features.shape
        
        # Normalize features
        implicit_norm = F.normalize(implicit_features, dim=-1)
        explicit_norm = F.normalize(explicit_features, dim=-1)
        
        # Compute cosine similarity
        cosine_sim = torch.sum(implicit_norm * explicit_norm, dim=-1)  # [batch_size, seq_len]
        
        # Create positive and negative pairs
        positive_mask = (alignment_labels == 1)
        negative_mask = (alignment_labels == 0)
        
        # Contrastive loss
        positive_loss = torch.clamp(self.margin - cosine_sim[positive_mask], min=0).pow(2)
        negative_loss = torch.clamp(cosine_sim[negative_mask] - (-self.margin), min=0).pow(2)
        
        total_loss = positive_loss.mean() + negative_loss.mean()
        
        return total_loss


class HierarchicalImplicitLoss(nn.Module):
    """
    Hierarchical loss for implicit detection at multiple granularities
    Combines token-level, span-level, and sentence-level losses
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.token_weight = getattr(config, 'token_loss_weight', 1.0)
        self.span_weight = getattr(config, 'span_loss_weight', 0.8)
        self.sentence_weight = getattr(config, 'sentence_loss_weight', 0.6)
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self,
                token_logits: torch.Tensor,
                span_logits: torch.Tensor,
                sentence_logits: torch.Tensor,
                token_labels: torch.Tensor,
                span_labels: torch.Tensor,
                sentence_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical implicit detection losses
        
        Args:
            token_logits: [batch_size, seq_len, num_classes]
            span_logits: [batch_size, num_spans, num_classes]
            sentence_logits: [batch_size, num_classes]
            token_labels: [batch_size, seq_len]
            span_labels: [batch_size, num_spans]
            sentence_labels: [batch_size]
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        
        # Token-level loss
        valid_token_mask = (token_labels != -100)
        if valid_token_mask.sum() > 0:
            flat_token_logits = token_logits.view(-1, token_logits.size(-1))[valid_token_mask.view(-1)]
            flat_token_labels = token_labels.view(-1)[valid_token_mask.view(-1)]
            token_loss = self.cross_entropy(flat_token_logits, flat_token_labels).mean()
            losses['token_loss'] = token_loss
        else:
            losses['token_loss'] = torch.tensor(0.0, device=token_logits.device)
        
        # Span-level loss
        valid_span_mask = (span_labels != -100)
        if valid_span_mask.sum() > 0:
            flat_span_logits = span_logits.view(-1, span_logits.size(-1))[valid_span_mask.view(-1)]
            flat_span_labels = span_labels.view(-1)[valid_span_mask.view(-1)]
            span_loss = self.cross_entropy(flat_span_logits, flat_span_labels).mean()
            losses['span_loss'] = span_loss
        else:
            losses['span_loss'] = torch.tensor(0.0, device=span_logits.device)
        
        # Sentence-level loss
        valid_sentence_mask = (sentence_labels != -100)
        if valid_sentence_mask.sum() > 0:
            valid_sentence_logits = sentence_logits[valid_sentence_mask]
            valid_sentence_labels = sentence_labels[valid_sentence_mask]
            sentence_loss = self.cross_entropy(valid_sentence_logits, valid_sentence_labels).mean()
            losses['sentence_loss'] = sentence_loss
        else:
            losses['sentence_loss'] = torch.tensor(0.0, device=sentence_logits.device)
        
        # Total hierarchical loss
        total_loss = (self.token_weight * losses['token_loss'] + 
                     self.span_weight * losses['span_loss'] + 
                     self.sentence_weight * losses['sentence_loss'])
        
        losses['total_hierarchical_loss'] = total_loss
        
        return losses


def create_implicit_detection_loss(config) -> ImplicitDetectionLoss:
    """Factory function to create implicit detection loss"""
    return ImplicitDetectionLoss(config)