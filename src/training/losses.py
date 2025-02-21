
# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        return focal_loss.mean()

class ABSALoss(nn.Module):
    """Combined loss for ABSA tasks"""
    def __init__(self, config):
        super().__init__()
        # Span detection losses
        self.span_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Sentiment classification loss with focal loss
        self.sentiment_criterion = FocalLoss(gamma=2.0)
        
        # Loss weights
        self.aspect_weight = config.aspect_loss_weight
        self.opinion_weight = config.opinion_loss_weight
        self.sentiment_weight = config.sentiment_loss_weight
        
    def forward(self, outputs, targets):
        # Calculate span detection losses
        aspect_loss = self.span_criterion(
            outputs['aspect_logits'].view(-1, 3), 
            targets['aspect_labels'].view(-1)
        )
        
        opinion_loss = self.span_criterion(
            outputs['opinion_logits'].view(-1, 3),
            targets['opinion_labels'].view(-1)
        )
        
        # Calculate sentiment classification loss
        sentiment_loss = self.sentiment_criterion(
            outputs['sentiment_logits'],
            targets['sentiment_labels']
        )
        
        # Combine losses
        total_loss = (
            self.aspect_weight * aspect_loss +
            self.opinion_weight * opinion_loss +
            self.sentiment_weight * sentiment_loss
        )
        
        return {
            'loss': total_loss,
            'aspect_loss': aspect_loss.item(),
            'opinion_loss': opinion_loss.item(),
            'sentiment_loss': sentiment_loss.item()
        }