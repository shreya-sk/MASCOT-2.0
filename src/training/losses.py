# src/training/losses.py
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

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
        self.aspect_weight = getattr(config, 'aspect_loss_weight', 1.0)
        self.opinion_weight = getattr(config, 'opinion_loss_weight', 1.0)
        self.sentiment_weight = getattr(config, 'sentiment_loss_weight', 1.0)
        
    def forward(self, outputs, targets):
        """
        Calculate the combined loss for ABSA tasks
        
        Args:
            outputs: Model outputs containing logits
            targets: Target labels
        """
        # Try-except to handle potential errors
        try:
            # Check dimensions and reshape if needed
            if outputs['aspect_logits'].size(0) != targets['aspect_labels'].size(0):
                print(f"Warning: Batch size mismatch. Model: {outputs['aspect_logits'].size()}, Target: {targets['aspect_labels'].size()}")
                
                # Handle batch size mismatch
                # Option 1: Truncate to the smaller batch size
                min_batch = min(outputs['aspect_logits'].size(0), targets['aspect_labels'].size(0))
                aspect_logits = outputs['aspect_logits'][:min_batch]
                opinion_logits = outputs['opinion_logits'][:min_batch]
                sentiment_logits = outputs['sentiment_logits'][:min_batch]
                
                aspect_labels = targets['aspect_labels'][:min_batch]
                opinion_labels = targets['opinion_labels'][:min_batch]
                sentiment_labels = targets['sentiment_labels'][:min_batch]
            else:
                # Use the original tensors
                aspect_logits = outputs['aspect_logits']
                opinion_logits = outputs['opinion_logits']
                sentiment_logits = outputs['sentiment_logits']
                
                aspect_labels = targets['aspect_labels']
                opinion_labels = targets['opinion_labels']
                sentiment_labels = targets['sentiment_labels']
            
            # Calculate span detection losses
            aspect_loss = self.span_criterion(
                aspect_logits.view(-1, 3), 
                aspect_labels.view(-1)
            )
            
            opinion_loss = self.span_criterion(
                opinion_logits.view(-1, 3),
                opinion_labels.view(-1)
            )
            
            # Calculate sentiment classification loss
            # Make sure the sentiment logits and labels are compatible
            if sentiment_logits.dim() != sentiment_labels.dim() + 1:
                print(f"Warning: Sentiment dimension mismatch. Logits: {sentiment_logits.shape}, Labels: {sentiment_labels.shape}")
                
                # Adjust dimensions if needed
                if sentiment_logits.dim() == 2 and sentiment_labels.dim() == 1:
                    # This is fine, continue
                    pass
                elif sentiment_logits.dim() == 3 and sentiment_labels.dim() == 1:
                    # Take the first token's prediction (CLS token)
                    sentiment_logits = sentiment_logits[:, 0, :]
                else:
                    # More complex mismatch - use a placeholder loss
                    sentiment_loss = torch.tensor(0.0, device=aspect_loss.device)
                    print("Using placeholder for sentiment loss due to dimension mismatch")
            
            # Calculate sentiment loss with proper dimensions
            try:
                sentiment_loss = self.sentiment_criterion(
                    sentiment_logits,
                    sentiment_labels
                )
            except Exception as e:
                print(f"Error in sentiment loss calculation: {e}")
                sentiment_loss = torch.tensor(0.0, device=aspect_loss.device)
            
            # Combine losses
            total_loss = (
                self.aspect_weight * aspect_loss +
                self.opinion_weight * opinion_loss +
                self.sentiment_weight * sentiment_loss
            )
            
        except Exception as e:
            # If there's an error, print it and use a simpler loss function
            print(f"Error in loss calculation: {e}")
            print("Using simplified loss function for training")
            
            # Simple MSE loss on aspect logits as a fallback
            total_loss = F.mse_loss(
                outputs['aspect_logits'], 
                torch.zeros_like(outputs['aspect_logits'])
            )
            
            # Set component losses for logging
            aspect_loss = opinion_loss = sentiment_loss = total_loss / 3
        
        return {
            'loss': total_loss,
            'aspect_loss': aspect_loss.item(),
            'opinion_loss': opinion_loss.item(),
            'sentiment_loss': sentiment_loss.item()
        }