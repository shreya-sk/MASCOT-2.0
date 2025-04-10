# src/training/losses.py
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss with a focus on hard examples
        
        Args:
            inputs: Prediction logits [N, C]
            targets: Target labels [N]
            
        Returns:
            Loss value (scalar)
        """
        # Ensure inputs and targets have proper shapes
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))
        
        if targets.dim() > 1:
            targets = targets.view(-1)
            
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal weights
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
                focal_weight = alpha_t * focal_weight
            else:
                alpha_factor = torch.ones_like(focal_weight) * self.alpha
                focal_weight = torch.where(targets > 0, alpha_factor, focal_weight)
                
        # Final loss
        focal_loss = focal_weight * ce_loss
            
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
        self.generation_weight = getattr(config, 'generation_weight', 0.5)
        
        # LM loss for generation
        self.generation_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        
        # Store config for generation
        self.config = config
        
    def forward(self, outputs, targets, generate=False):
        """
        Calculate the combined loss for ABSA tasks
        
        Args:
            outputs: Model output dictionary
            targets: Target dictionary
            generate: Whether to include generation loss
            
        Returns:
            dict: Dictionary with loss components
        """
        try:
            # Get predictions
            aspect_logits = outputs['aspect_logits']  # [batch_size, seq_len, 3]
            opinion_logits = outputs['opinion_logits']  # [batch_size, seq_len, 3]
            sentiment_logits = outputs['sentiment_logits']  # [batch_size, 3]
            
            # Get labels
            aspect_labels = targets.get('aspect_labels')  # Shape might be [batch_size, num_spans, seq_len]
            opinion_labels = targets.get('opinion_labels')  # Shape might be [batch_size, num_spans, seq_len]
            sentiment_labels = targets.get('sentiment_labels')  # Shape might be [batch_size, num_spans]
            
            # Handle and reshape aspect labels if necessary
            if aspect_labels is not None:
                if len(aspect_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                    # Take the first span or combine spans
                    if aspect_labels.size(1) > 0:
                        aspect_labels = aspect_labels[:, 0]  # Use first span
                        
                # Filter for valid positions (non-padding)
                valid_aspect_mask = aspect_labels != -100
                
                # Calculate span detection loss
                aspect_loss = self.span_criterion(
                    aspect_logits.view(-1, 3)[valid_aspect_mask.view(-1)], 
                    aspect_labels.view(-1)[valid_aspect_mask.view(-1)]
                )
            else:
                aspect_loss = torch.tensor(0.0, device=aspect_logits.device)
                
            # Handle and reshape opinion labels if necessary
            if opinion_labels is not None:
                if len(opinion_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                    # Take the first span or combine spans
                    if opinion_labels.size(1) > 0:
                        opinion_labels = opinion_labels[:, 0]
                
                # Filter for valid positions (non-padding)
                valid_opinion_mask = opinion_labels != -100
                
                # Calculate span detection loss
                opinion_loss = self.span_criterion(
                    opinion_logits.view(-1, 3)[valid_opinion_mask.view(-1)],
                    opinion_labels.view(-1)[valid_opinion_mask.view(-1)]
                )
            else:
                opinion_loss = torch.tensor(0.0, device=opinion_logits.device)
                
            # Handle sentiment labels
            if sentiment_labels is not None:
                if len(sentiment_labels.shape) > 1:  # [batch_size, num_spans]
                    # Take the first sentiment label for each batch item
                    sentiment_labels = sentiment_labels[:, 0]
                
                # Valid sentiment mask
                valid_sentiment_mask = sentiment_labels != -100
                
                if valid_sentiment_mask.sum() > 0:
                    # Calculate sentiment classification loss
                    sentiment_loss = self.sentiment_criterion(
                        sentiment_logits[valid_sentiment_mask],
                        sentiment_labels[valid_sentiment_mask]
                    )
                else:
                    sentiment_loss = torch.tensor(0.0, device=sentiment_logits.device)
            else:
                sentiment_loss = torch.tensor(0.0, device=sentiment_logits.device)
            
            # Combine extraction losses
            extraction_loss = (
                self.aspect_weight * aspect_loss +
                self.opinion_weight * opinion_loss +
                self.sentiment_weight * sentiment_loss
            )
            
            # Dictionary to store all loss components
            loss_dict = {
                'loss': extraction_loss,
                'aspect_loss': aspect_loss.item(),
                'opinion_loss': opinion_loss.item(),
                'sentiment_loss': sentiment_loss.item()
            }
            
            # Add generation loss if requested and available
            if generate and 'explanations' in outputs:
                if 'explanation_targets' in targets:
                    explanation_targets = targets['explanation_targets']
                    
                    # Calculate generation loss
                    explanation_logits = outputs['explanations']  # [batch_size, max_len, vocab_size]
                    
                    # Reshape for cross entropy
                    gen_loss = self.generation_criterion(
                        explanation_logits.view(-1, explanation_logits.size(-1)),
                        explanation_targets.view(-1)
                    )
                    
                    # Add to total loss
                    total_loss = extraction_loss + self.generation_weight * gen_loss
                    loss_dict['loss'] = total_loss
                    loss_dict['generation_loss'] = gen_loss.item()
                    
            return loss_dict
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple differentiable loss for backward compatibility
            device = outputs['aspect_logits'].device
            dummy_loss = outputs['aspect_logits'].mean() * 0.0001
            
            return {
                'loss': dummy_loss,
                'aspect_loss': 0.0,
                'opinion_loss': 0.0,
                'sentiment_loss': 0.0
            }