import torch
import torch.nn as nn
class ABSALoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Add global structure loss
        self.global_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Add local structure loss
        self.local_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Add consistency loss between global and local predictions
        self.consistency_weight = config.consistency_weight
        
    def forward(self, outputs, targets):
        # Calculate hierarchical losses
        global_loss = self.global_criterion(
            outputs['global_context'],
            targets['sentiment_labels']
        )
        
        local_loss = self.local_criterion(
            outputs['local_context'],
            targets['aspect_labels']
        )
        
        # Calculate consistency loss
        consistency_loss = self._compute_consistency_loss(
            outputs['global_context'],
            outputs['local_context']
        )
        
        # Original losses
        aspect_loss = self.span_criterion(
            outputs['aspect_logits'].view(-1, 3),
            targets['aspect_labels'].view(-1)
        )
        
        opinion_loss = self.span_criterion(
            outputs['opinion_logits'].view(-1, 3),
            targets['opinion_labels'].view(-1)
        )
        
        sentiment_loss = self.sentiment_criterion(
            outputs['sentiment_logits'],
            targets['sentiment_labels']
        )
        
        # Combine all losses
        total_loss = (
            self.aspect_weight * aspect_loss +
            self.opinion_weight * opinion_loss +
            self.sentiment_weight * sentiment_loss +
            global_loss + local_loss +
            self.consistency_weight * consistency_loss
        )
        
        return {
            'loss': total_loss,
            'aspect_loss': aspect_loss.item(),
            'opinion_loss': opinion_loss.item(),
            'sentiment_loss': sentiment_loss.item(),
            'global_loss': global_loss.item(),
            'local_loss': local_loss.item(),
            'consistency_loss': consistency_loss.item()
        }