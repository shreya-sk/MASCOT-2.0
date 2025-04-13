# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABSALoss(nn.Module):
    """
    Enhanced ABSA loss function with stronger focus on aspect/opinion extraction
    """
    def __init__(self, config):
        super().__init__()
        
        # Stronger weights for aspect and opinion tasks
        self.aspect_weight = getattr(config, 'aspect_loss_weight', 2.0)  # Increased from 1.0
        self.opinion_weight = getattr(config, 'opinion_loss_weight', 2.0)  # Increased from 1.0
        self.sentiment_weight = getattr(config, 'sentiment_loss_weight', 1.0)
        self.boundary_weight = getattr(config, 'boundary_weight', 0.5)
        
        # Span detection loss with class weighting to handle imbalance
        # BIO tags are imbalanced (most tokens are O)
        self.span_criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            weight=torch.tensor([0.2, 1.0, 1.0])  # Lower weight for O tag, higher for B and I tags
        )
        
        # Standard sentiment classification loss
        self.sentiment_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, outputs, targets, generate=False):
        """
        Compute combined loss for ABSA
        
        Args:
            outputs: Model output dictionary
            targets: Target dictionary
            generate: Whether to include generation loss
            
        Returns:
            Dictionary with loss components
        """
        try:
            # Get predictions
            aspect_logits = outputs['aspect_logits']  # [batch_size, seq_len, 3]
            opinion_logits = outputs['opinion_logits']  # [batch_size, seq_len, 3]
            sentiment_logits = outputs['sentiment_logits']  # [batch_size, num_classes]
            
            # Initialize losses
            aspect_loss = torch.tensor(0.0, device=aspect_logits.device)
            opinion_loss = torch.tensor(0.0, device=opinion_logits.device)
            sentiment_loss = torch.tensor(0.0, device=sentiment_logits.device)
            boundary_loss = torch.tensor(0.0, device=aspect_logits.device)
            
            # Process aspect loss
            if 'aspect_labels' in targets:
                aspect_labels = targets['aspect_labels']
                
                # Handle multiple spans if needed
                if len(aspect_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                    # Take max across spans to get a single label per token
                    aspect_labels = aspect_labels.max(dim=1)[0]  # [batch_size, seq_len]
                aspect_labels = aspect_labels.long()
                # Compute aspect loss
                aspect_loss = self.span_criterion(
                    aspect_logits.view(-1, 3),
                    aspect_labels.view(-1)
                )
            
            # Process opinion loss
            if 'opinion_labels' in targets:
                opinion_labels = targets['opinion_labels']
                
                # Handle multiple spans if needed
                if len(opinion_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                    # Take max across spans to get a single label per token
                    opinion_labels = opinion_labels.max(dim=1)[0]  # [batch_size, seq_len]
                opinion_labels = opinion_labels.long()
                # Compute opinion loss
                opinion_loss = self.span_criterion(
                    opinion_logits.view(-1, 3),
                    opinion_labels.view(-1)
                )
            
            # Process sentiment loss
            if 'sentiment_labels' in targets:
                sentiment_labels = targets['sentiment_labels']
                
                # Handle multiple spans if needed
                if len(sentiment_labels.shape) > 1 and sentiment_labels.shape[1] > 0:
                    # Take the first sentiment label for each batch item
                    sentiment_labels = sentiment_labels[:, 0]
                sentiment_labels = sentiment_labels.long()
                # Compute sentiment loss
                sentiment_loss = self.sentiment_criterion(
                    sentiment_logits,
                    sentiment_labels
                )
            
            # Compute boundary refinement loss if available
            if 'boundary_logits' in outputs and self.boundary_weight > 0:
                boundary_logits = outputs['boundary_logits']
                
                # Create boundary labels from aspect and opinion labels
                if 'aspect_labels' in targets and 'opinion_labels' in targets:
                    aspect_labels_for_boundary = targets.get('aspect_labels')
                    if len(aspect_labels_for_boundary.shape) == 3:
                        aspect_labels_for_boundary = aspect_labels_for_boundary.max(dim=1)[0]
                        
                    opinion_labels_for_boundary = targets.get('opinion_labels')
                    if len(opinion_labels_for_boundary.shape) == 3:
                        opinion_labels_for_boundary = opinion_labels_for_boundary.max(dim=1)[0]
                    
                    # Simple boundary loss - binary cross entropy
                    boundary_target = torch.zeros_like(boundary_logits)
                    
                    # Mark start positions (B tags) in the target
                    batch_size, seq_len = aspect_labels_for_boundary.shape
                    for b in range(batch_size):
                        for s in range(seq_len):
                            if aspect_labels_for_boundary[b, s] == 1 or opinion_labels_for_boundary[b, s] == 1:
                                boundary_target[b, s, 0] = 1.0  # Start boundary
                            if s > 0 and (aspect_labels_for_boundary[b, s-1] > 0 and aspect_labels_for_boundary[b, s] == 0 or
                                         opinion_labels_for_boundary[b, s-1] > 0 and opinion_labels_for_boundary[b, s] == 0):
                                boundary_target[b, s-1, 1] = 1.0  # End boundary
                                
                    # Compute boundary loss
                    boundary_loss = F.binary_cross_entropy_with_logits(
                        boundary_logits,
                        boundary_target
                    )
            
            # Combine all losses with their weights
            total_loss = (
                self.aspect_weight * aspect_loss +
                self.opinion_weight * opinion_loss +
                self.sentiment_weight * sentiment_loss +
                self.boundary_weight * boundary_loss
            )
            
            # Return dictionary with all loss components
            return {
                'loss': total_loss,
                'aspect_loss': aspect_loss.item(),
                'opinion_loss': opinion_loss.item(),
                'sentiment_loss': sentiment_loss.item(),
                'boundary_loss': boundary_loss.item()
            }
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple differentiable loss for backward compatibility
            device = outputs['aspect_logits'].device
            
            # Create a dummy loss that requires gradient
            dummy_loss = (
                outputs['aspect_logits'].sum() * 0.0001 +
                outputs['opinion_logits'].sum() * 0.0001 +
                outputs['sentiment_logits'].sum() * 0.0001
            )
            
            return {
                'loss': dummy_loss,
                'aspect_loss': 0.0,
                'opinion_loss': 0.0,
                'sentiment_loss': 0.0,
                'boundary_loss': 0.0
            }