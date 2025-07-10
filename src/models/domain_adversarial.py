# src/models/domain_adversarial.py
"""
Domain Adversarial Training Components for ABSA
Implements gradient reversal layer and domain classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for Domain Adversarial Training
    Reverses gradients during backpropagation while keeping forward pass unchanged
    """
    
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        """Forward pass - identity function"""
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - reverse gradients"""
        return -ctx.alpha * grad_output, None


def gradient_reversal_layer(x, alpha=1.0):
    """Functional interface for gradient reversal"""
    return GradientReversalLayer.apply(x, alpha)


class DomainClassifier(nn.Module):
    """
    Domain classifier for adversarial training
    Predicts domain from reversed gradients
    """
    
    def __init__(self, hidden_size: int, num_domains: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_domains = num_domains
        
        # Domain classification layers
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_domains)
        )
        
        # Initialize weights for better domain confusion
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to encourage domain confusion"""
        for module in self.domain_classifier:
            if isinstance(module, nn.Linear):
                # Initialize with small weights to start confused
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Forward pass with gradient reversal
        
        Args:
            features: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
            alpha: Gradient reversal strength
            
        Returns:
            Domain logits: [batch_size, num_domains]
        """
        # Apply gradient reversal
        reversed_features = gradient_reversal_layer(features, alpha)
        
        # Handle sequence-level features (average pool if needed)
        if reversed_features.dim() == 3:
            # Average pool over sequence length
            reversed_features = reversed_features.mean(dim=1)  # [batch_size, hidden_size]
        
        # Domain classification
        domain_logits = self.domain_classifier(reversed_features)
        
        return domain_logits


class OrthogonalConstraint(nn.Module):
    """
    Orthogonal constraint loss for domain-invariant feature learning
    Encourages features to be orthogonal across domains
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonal constraint loss
        
        Args:
            features: [batch_size, feature_dim] - domain features
            domain_ids: [batch_size] - domain labels
            
        Returns:
            Orthogonal constraint loss
        """
        unique_domains = torch.unique(domain_ids)
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Compute domain-specific feature means
        domain_features = []
        for domain_id in unique_domains:
            domain_mask = (domain_ids == domain_id)
            if domain_mask.sum() > 0:
                domain_feat = features[domain_mask].mean(dim=0)  # [feature_dim]
                domain_features.append(domain_feat)
        
        if len(domain_features) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Stack domain features
        domain_matrix = torch.stack(domain_features)  # [num_domains, feature_dim]
        
        # Compute gram matrix (similarity between domain features)
        gram_matrix = torch.mm(domain_matrix, domain_matrix.t())  # [num_domains, num_domains]
        
        # Orthogonal loss - minimize off-diagonal elements
        eye = torch.eye(gram_matrix.size(0), device=features.device)
        orthogonal_loss = torch.sum((gram_matrix - eye) ** 2)
        
        return orthogonal_loss


class DomainAdversarialModule(nn.Module):
    """
    Complete Domain Adversarial Module
    Integrates gradient reversal, domain classifier, and orthogonal constraints
    """
    
    def __init__(self, 
                 hidden_size: int, 
                 num_domains: int = 4,
                 dropout: float = 0.1,
                 orthogonal_weight: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_domains = num_domains
        self.orthogonal_weight = orthogonal_weight
        
        # Domain classifier with gradient reversal
        self.domain_classifier = DomainClassifier(hidden_size, num_domains, dropout)
        
        # Orthogonal constraint
        self.orthogonal_constraint = OrthogonalConstraint()
        
        # Domain adaptation features
        self.domain_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Current alpha for gradient reversal (updated during training)
        self.current_alpha = 1.0
    
    def update_alpha(self, epoch: int, total_epochs: int):
        """Update gradient reversal strength during training"""
        # Gradually increase alpha from 0 to 1
        progress = epoch / total_epochs
        self.current_alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
    
    def forward(self, 
                features: torch.Tensor, 
                domain_ids: Optional[torch.Tensor] = None,
                return_losses: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with domain adversarial training
        
        Args:
            features: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
            domain_ids: [batch_size] - domain labels for training
            return_losses: Whether to compute losses
            
        Returns:
            Dictionary containing domain logits, adapted features, and losses
        """
        batch_size = features.size(0)
        
        # Adapt features for domain invariance
        adapted_features = self.domain_adapter(features)
        
        # Domain classification with gradient reversal
        domain_logits = self.domain_classifier(adapted_features, self.current_alpha)
        
        outputs = {
            'domain_logits': domain_logits,
            'adapted_features': adapted_features,
            'alpha': self.current_alpha
        }
        
        # Compute losses if domain_ids provided and training
        if return_losses and domain_ids is not None:
            losses = self.compute_losses(adapted_features, domain_logits, domain_ids)
            outputs.update(losses)
        
        return outputs
    
    def compute_losses(self, 
                      features: torch.Tensor,
                      domain_logits: torch.Tensor, 
                      domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute domain adversarial losses
        
        Args:
            features: Adapted features
            domain_logits: Domain classifier predictions
            domain_ids: True domain labels
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # 1. Domain classification loss (to confuse domain classifier)
        domain_loss = F.cross_entropy(domain_logits, domain_ids)
        losses['domain_loss'] = domain_loss
        
        # 2. Orthogonal constraint loss
        if features.dim() == 3:
            # Average pool sequence features for orthogonal constraint
            pooled_features = features.mean(dim=1)
        else:
            pooled_features = features
            
        orthogonal_loss = self.orthogonal_constraint(pooled_features, domain_ids)
        losses['orthogonal_loss'] = orthogonal_loss
        
        # 3. Total adversarial loss
        total_adversarial_loss = domain_loss + self.orthogonal_weight * orthogonal_loss
        losses['total_adversarial_loss'] = total_adversarial_loss
        
        return losses


# Domain mapping for common ABSA datasets
DOMAIN_MAPPING = {
    'restaurant': 0,
    'laptop': 1, 
    'hotel': 2,
    'general': 3,
    'rest14': 0,
    'rest15': 0,
    'rest16': 0,
    'laptop14': 1,
    'laptop15': 1,
    'laptop16': 1,
    'hotel_reviews': 2
}

def get_domain_id(dataset_name: str) -> int:
    """Get domain ID for dataset name"""
    return DOMAIN_MAPPING.get(dataset_name.lower(), 3)  # Default to general


def create_domain_adversarial_module(config) -> DomainAdversarialModule:
    """Factory function to create domain adversarial module"""
    return DomainAdversarialModule(
        hidden_size=config.hidden_size,
        num_domains=getattr(config, 'num_domains', 4),
        dropout=getattr(config, 'dropout', 0.1),
        orthogonal_weight=getattr(config, 'orthogonal_weight', 0.1)
    )