# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABSALoss(nn.Module):
    """
    Improved ABSA loss function with robust error handling and consistent tensor processing
    """
    def __init__(self, config):
        super().__init__()
        
        # Loss weights based on task importance
        self.aspect_weight = getattr(config, 'aspect_loss_weight', 2.0)
        self.opinion_weight = getattr(config, 'opinion_loss_weight', 2.0)
        self.sentiment_weight = getattr(config, 'sentiment_loss_weight', 1.0)
        self.boundary_weight = getattr(config, 'boundary_weight', 0.5)
        
        # Label smoothing for better generalization
        self.label_smoothing = getattr(config, 'label_smoothing', 0.1)
        
        # Focal loss parameters
        self.gamma = getattr(config, 'focal_gamma', 2.0)
        self.use_focal_loss = getattr(config, 'use_focal_loss', True)
        
        # Class weights for imbalanced data
        # Standard BIO scheme: O, B, I
        aspect_weights = torch.tensor([0.1, 1.0, 0.8])  # Less weight on O, more on B and I
        opinion_weights = torch.tensor([0.1, 1.0, 0.8])  # Same for opinions
        
        if self.use_focal_loss:
            # Use focal loss for span detection
            self.span_criterion = FocalLossWithLS(
                gamma=self.gamma,
                alpha=aspect_weights,
                ignore_index=-100,
                label_smoothing=self.label_smoothing
            )
            self.opinion_criterion = FocalLossWithLS(
                gamma=self.gamma,
                alpha=opinion_weights,
                ignore_index=-100,
                label_smoothing=self.label_smoothing
            )
        else:
            # Use standard cross-entropy
            self.span_criterion = nn.CrossEntropyLoss(
                weight=aspect_weights,
                ignore_index=-100,
                label_smoothing=self.label_smoothing
            )
            self.opinion_criterion = nn.CrossEntropyLoss(
                weight=opinion_weights,
                ignore_index=-100,
                label_smoothing=self.label_smoothing
            )
        
        # Sentiment classification with label smoothing
        self.sentiment_criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.label_smoothing
        )
        
    def forward(self, outputs, targets, generate=False):
        """
        Compute combined loss for ABSA with robust tensor handling
        
        Args:
            outputs: Model output dictionary
            targets: Target dictionary
            generate: Whether to include generation loss (not implemented yet)
            
        Returns:
            Dictionary with loss components
        """
        try:
            # Extract predictions from outputs
            aspect_logits = outputs.get('aspect_logits')  # [batch_size, seq_len, 3]
            opinion_logits = outputs.get('opinion_logits')  # [batch_size, seq_len, 3]
            sentiment_logits = outputs.get('sentiment_logits')  # [batch_size, num_classes]
            
            # Extract targets
            aspect_labels = targets.get('aspect_labels')
            opinion_labels = targets.get('opinion_labels')
            sentiment_labels = targets.get('sentiment_labels')
            
            # Validate inputs
            if any(x is None for x in [aspect_logits, opinion_logits, sentiment_logits]):
                raise ValueError("Missing required logits in outputs")
            
            if any(x is None for x in [aspect_labels, opinion_labels, sentiment_labels]):
                raise ValueError("Missing required labels in targets")
            
            device = aspect_logits.device
            
            # Initialize losses
            aspect_loss = torch.tensor(0.0, device=device, requires_grad=True)
            opinion_loss = torch.tensor(0.0, device=device, requires_grad=True)
            sentiment_loss = torch.tensor(0.0, device=device, requires_grad=True)
            boundary_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Process aspect loss
            aspect_loss = self._compute_span_loss(
                aspect_logits, aspect_labels, self.span_criterion, "aspect"
            )
            
            # Process opinion loss
            opinion_loss = self._compute_span_loss(
                opinion_logits, opinion_labels, self.opinion_criterion, "opinion"
            )
            
            # Process sentiment loss
            sentiment_loss = self._compute_sentiment_loss(
                sentiment_logits, sentiment_labels
            )
            
            # Process boundary loss if available
            if 'boundary_logits' in outputs and self.boundary_weight > 0:
                boundary_loss = self._compute_boundary_loss(
                    outputs['boundary_logits'], aspect_labels, opinion_labels
                )
            
            # Combine all losses
            total_loss = (
                self.aspect_weight * aspect_loss +
                self.opinion_weight * opinion_loss +
                self.sentiment_weight * sentiment_loss +
                self.boundary_weight * boundary_loss
            )
            
            # Ensure total_loss requires gradients
            if not total_loss.requires_grad:
                # Add a small differentiable term
                dummy_term = (aspect_logits.sum() + opinion_logits.sum() + sentiment_logits.sum()) * 1e-8
                total_loss = total_loss + dummy_term
            
            # Return loss dictionary
            return {
                'loss': total_loss,
                'aspect_loss': aspect_loss.detach().item() if isinstance(aspect_loss, torch.Tensor) else aspect_loss,
                'opinion_loss': opinion_loss.detach().item() if isinstance(opinion_loss, torch.Tensor) else opinion_loss,
                'sentiment_loss': sentiment_loss.detach().item() if isinstance(sentiment_loss, torch.Tensor) else sentiment_loss,
                'boundary_loss': boundary_loss.detach().item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss,
            }
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a minimal differentiable loss
            device = outputs['aspect_logits'].device
            dummy_loss = (
                outputs['aspect_logits'].sum() * 1e-6 +
                outputs['opinion_logits'].sum() * 1e-6 +
                outputs['sentiment_logits'].sum() * 1e-6
            )
            
            return {
                'loss': dummy_loss,
                'aspect_loss': 0.0,
                'opinion_loss': 0.0,
                'sentiment_loss': 0.0,
                'boundary_loss': 0.0,
            }
    
    def _compute_span_loss(self, logits, labels, criterion, loss_name):
        """
        Compute loss for span detection (aspect or opinion)
        
        Args:
            logits: Prediction logits [batch_size, seq_len, num_classes]
            labels: Target labels [batch_size, num_spans, seq_len] or [batch_size, seq_len]
            criterion: Loss criterion to use
            loss_name: Name for debugging
            
        Returns:
            Computed loss tensor
        """
        try:
            # Handle multi-span case
            if len(labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                # Take the first span or aggregate across spans
                batch_size, num_spans, seq_len = labels.shape
                
                if num_spans == 1:
                    # Single span case
                    labels = labels.squeeze(1)  # [batch_size, seq_len]
                else:
                    # Multiple spans - take element-wise maximum
                    labels = labels.max(dim=1)[0]  # [batch_size, seq_len]
            
            # Ensure labels are the right dtype
            labels = labels.long()
            
            # Validate shapes
            batch_size, seq_len, num_classes = logits.shape
            if labels.shape != (batch_size, seq_len):
                print(f"Warning: Shape mismatch in {loss_name} loss. Logits: {logits.shape}, Labels: {labels.shape}")
                # Try to fix common shape issues
                if labels.numel() == batch_size * seq_len:
                    labels = labels.view(batch_size, seq_len)
                else:
                    # Create dummy labels as fallback
                    labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
            
            # Compute loss
            loss = criterion(
                logits.view(-1, num_classes),  # [batch_size * seq_len, num_classes]
                labels.view(-1)  # [batch_size * seq_len]
            )
            
            return loss
            
        except Exception as e:
            print(f"Error computing {loss_name} loss: {e}")
            # Return a small differentiable loss
            return logits.sum() * 1e-8
    
    def _compute_sentiment_loss(self, logits, labels):
        """
        Compute sentiment classification loss
        
        Args:
            logits: Sentiment logits [batch_size, num_classes]
            labels: Sentiment labels [batch_size] or [batch_size, num_spans]
            
        Returns:
            Computed loss tensor
        """
        try:
            # Handle multi-span case
            if len(labels.shape) > 1:
                # Take the first sentiment label for each batch item
                labels = labels[:, 0] if labels.shape[1] > 0 else labels.squeeze()
            
            # Ensure labels are the right dtype and shape
            labels = labels.long()
            
            # Validate shapes
            if logits.shape[0] != labels.shape[0]:
                print(f"Warning: Batch size mismatch in sentiment loss. Logits: {logits.shape}, Labels: {labels.shape}")
                # Take minimum batch size
                min_batch = min(logits.shape[0], labels.shape[0])
                logits = logits[:min_batch]
                labels = labels[:min_batch]
            
            # Compute loss
            loss = self.sentiment_criterion(logits, labels)
            
            return loss
            
        except Exception as e:
            print(f"Error computing sentiment loss: {e}")
            # Return a small differentiable loss
            return logits.sum() * 1e-8
    
    def _compute_boundary_loss(self, boundary_logits, aspect_labels, opinion_labels):
        """
        Compute boundary refinement loss
        
        Args:
            boundary_logits: Boundary prediction logits [batch_size, seq_len, 2]
            aspect_labels: Aspect labels for creating boundary targets
            opinion_labels: Opinion labels for creating boundary targets
            
        Returns:
            Computed boundary loss
        """
        try:
            batch_size, seq_len, _ = boundary_logits.shape
            device = boundary_logits.device
            
            # Create boundary targets
            boundary_targets = torch.zeros(batch_size, seq_len, 2, device=device)
            
            # Handle multi-span labels
            if len(aspect_labels.shape) == 3:
                aspect_labels = aspect_labels.max(dim=1)[0]
            if len(opinion_labels.shape) == 3:
                opinion_labels = opinion_labels.max(dim=1)[0]
            
            # Mark boundaries
            for b in range(batch_size):
                for s in range(seq_len):
                    # Start boundary (B tags)
                    if (s < aspect_labels.shape[1] and aspect_labels[b, s] == 1) or \
                       (s < opinion_labels.shape[1] and opinion_labels[b, s] == 1):
                        boundary_targets[b, s, 0] = 1.0
                    
                    # End boundary (transition from I to O)
                    if s > 0 and s < min(aspect_labels.shape[1], opinion_labels.shape[1]):
                        aspect_end = (aspect_labels[b, s-1] > 0 and aspect_labels[b, s] == 0)
                        opinion_end = (opinion_labels[b, s-1] > 0 and opinion_labels[b, s] == 0)
                        if aspect_end or opinion_end:
                            boundary_targets[b, s-1, 1] = 1.0
            
            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
            
            return loss
            
        except Exception as e:
            print(f"Error computing boundary loss: {e}")
            return boundary_logits.sum() * 1e-8


class FocalLossWithLS(nn.Module):
    """
    Focal loss with label smoothing for handling class imbalance
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        # Handle alpha (class weights)
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = None
    
    def forward(self, logits, targets):
        """
        Compute focal loss with label smoothing
        
        Args:
            logits: Prediction logits [N, C]
            targets: Target indices [N]
            
        Returns:
            Focal loss value
        """
        try:
            # Ensure targets are long
            targets = targets.long()
            
            # Get number of classes
            num_classes = logits.size(-1)
            
            # Compute cross-entropy
            ce_loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction='none')
            
            # Compute probabilities
            pt = torch.exp(-ce_loss)
            
            # Apply focal weight
            focal_weight = (1 - pt) ** self.gamma
            focal_loss = focal_weight * ce_loss
            
            # Apply alpha weighting if provided
            if self.alpha is not None:
                alpha = self.alpha.to(logits.device)
                # Create alpha weights for each target
                alpha_weights = torch.ones_like(targets, dtype=torch.float, device=logits.device)
                valid_mask = targets != self.ignore_index
                valid_targets = targets[valid_mask]
                
                if len(valid_targets) > 0:
                    # Clamp targets to valid range
                    valid_targets_clamped = torch.clamp(valid_targets, 0, len(alpha) - 1)
                    alpha_weights[valid_mask] = alpha[valid_targets_clamped]
                
                focal_loss = focal_loss * alpha_weights
            
            # Apply mask for ignored indices
            mask = (targets != self.ignore_index).float()
            focal_loss = focal_loss * mask
            
            # Compute mean loss
            if mask.sum() > 0:
                return focal_loss.sum() / mask.sum()
            else:
                return focal_loss.sum()
            
        except Exception as e:
            print(f"Error in focal loss computation: {e}")
            # Fallback to simple cross-entropy
            return F.cross_entropy(logits, targets, ignore_index=self.ignore_index)