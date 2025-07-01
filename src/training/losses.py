# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABSALoss(nn.Module):
    """
    Improved ABSA loss function with focal loss and label smoothing
    for better handling of class imbalance and regularization
    """
    def __init__(self, config):
        super().__init__()
        
        # Weights based on task importance
        self.aspect_weight = getattr(config, 'aspect_loss_weight', 2.0)
        self.opinion_weight = getattr(config, 'opinion_loss_weight', 2.0)
        self.sentiment_weight = getattr(config, 'sentiment_loss_weight', 1.0)
        self.boundary_weight = getattr(config, 'boundary_weight', 0.5)
        
        # Label smoothing for better generalization
        self.label_smoothing = getattr(config, 'label_smoothing', 0.1)
        
        # Focal loss parameters
        self.gamma = getattr(config, 'focal_gamma', 2.0)
        
        # Span detection with focal loss
        self.span_criterion = FocalLossWithLS(
            gamma=self.gamma,
            alpha=torch.tensor([0.1, 0.45, 0.45]),  # Class weights
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
            aspect_loss = torch.tensor(0.0, device=aspect_logits.device, requires_grad=True)
            opinion_loss = torch.tensor(0.0, device=opinion_logits.device, requires_grad=True)
            sentiment_loss = torch.tensor(0.0, device=sentiment_logits.device, requires_grad=True)
            boundary_loss = torch.tensor(0.0, device=aspect_logits.device, requires_grad=True)
            explanation_loss = torch.tensor(0.0, device=aspect_logits.device, requires_grad=True)
            
            # Process aspect loss
            if 'aspect_labels' in targets:
                aspect_labels = targets['aspect_labels']
                
                # Handle mixed labels from mixup
                if 'mixed_aspect_labels' in targets:
                    try:
                        # Use soft labels from mixup
                        aspect_loss = self._compute_soft_loss(aspect_logits, targets['mixed_aspect_labels'])
                    except Exception as e:
                        print(f"Error in mixed aspect loss: {e}. Falling back to standard loss.")
                        # Fall back to standard loss
                        if len(aspect_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                            aspect_labels = aspect_labels.max(dim=1)[0]  # [batch_size, seq_len]
                        aspect_labels = aspect_labels.long()
                        aspect_loss = self.span_criterion(
                            aspect_logits.view(-1, 3),
                            aspect_labels.view(-1)
                        )
                else:
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
                
                # Handle mixed labels from mixup
                if 'mixed_opinion_labels' in targets:
                    try:
                        # Use soft labels from mixup
                        opinion_loss = self._compute_soft_loss(opinion_logits, targets['mixed_opinion_labels'])
                    except Exception as e:
                        print(f"Error in mixed opinion loss: {e}. Falling back to standard loss.")
                        # Fall back to standard loss
                        if len(opinion_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                            opinion_labels = opinion_labels.max(dim=1)[0]  # [batch_size, seq_len]
                        opinion_labels = opinion_labels.long()
                        opinion_loss = self.span_criterion(
                            opinion_logits.view(-1, 3),
                            opinion_labels.view(-1)
                        )
                else:
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
                
                # Handle mixed labels from mixup
                if 'mixed_sentiment_labels' in targets:
                    try:
                        # Use soft labels from mixup
                        sentiment_loss = self._compute_soft_sentiment_loss(
                            sentiment_logits, targets['mixed_sentiment_labels']
                        )
                    except Exception as e:
                        print(f"Error in mixed sentiment loss: {e}. Falling back to standard loss.")
                        # Fall back to standard loss
                        if len(sentiment_labels.shape) > 1 and sentiment_labels.shape[1] > 0:
                            sentiment_labels = sentiment_labels[:, 0]
                        sentiment_labels = sentiment_labels.long()
                        sentiment_loss = self.sentiment_criterion(
                            sentiment_logits,
                            sentiment_labels
                        )
                else:
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
                    try:
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
                    except Exception as e:
                        print(f"Error in boundary loss: {e}")
                        boundary_loss = torch.tensor(0.0, device=aspect_logits.device, requires_grad=True)
            
            # Handle explanation generation loss if available and requested
            if generate and 'explanations' in outputs:
                explanation_logits = outputs['explanations']
                
                if 'explanation_targets' in targets:
                    # Compute generation loss
                    try:
                        explanation_targets = targets['explanation_targets']
                        explanation_loss = self._compute_generation_loss(
                            explanation_logits, explanation_targets
                        )
                    except Exception as e:
                        print(f"Error in explanation loss: {e}")
                        explanation_loss = torch.tensor(0.0, device=aspect_logits.device, requires_grad=True)
            
            # Combine all losses with their weights
            total_loss = (
                self.aspect_weight * aspect_loss +
                self.opinion_weight * opinion_loss +
                self.sentiment_weight * sentiment_loss +
                self.boundary_weight * boundary_loss
            )
            
            # Add explanation loss if available
            if generate and explanation_loss > 0:
                total_loss = total_loss + explanation_loss
            
            # Make sure total_loss is a scalar and requires_grad
            if not total_loss.requires_grad:
                # Create a small dummy loss that requires grad
                dummy_term = (aspect_logits.sum() + opinion_logits.sum() + sentiment_logits.sum()) * 0.0001
                total_loss = total_loss + dummy_term
            
            # Return dictionary with all loss components
            return {
                'loss': total_loss,
                'aspect_loss': aspect_loss.detach().item(),
                'opinion_loss': opinion_loss.detach().item(),
                'sentiment_loss': sentiment_loss.detach().item(),
                'boundary_loss': boundary_loss.detach().item(),
                'explanation_loss': explanation_loss.detach().item() if generate else 0.0
            }
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple differentiable loss for backward compatibility
            device = outputs['aspect_logits'].device
            
            # Create a dummy loss that definitely requires gradient
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
                'boundary_loss': 0.0,
                'explanation_loss': 0.0
            }
    
    def _compute_soft_loss(self, logits, soft_targets):
        """Compute loss with soft labels from mixup"""
        # Ensure dimensions match
        if soft_targets.dim() == logits.dim():
            # Dimensions already match
            pass
        elif soft_targets.dim() == 3 and logits.dim() == 3:
            # Check if the class dimension matches
            if soft_targets.size(-1) != logits.size(-1):
                # Reshape soft_targets to match logits class dimension
                if soft_targets.size(-1) > logits.size(-1):
                    # Truncate extra dimensions
                    soft_targets = soft_targets[..., :logits.size(-1)]
                else:
                    # Pad with zeros
                    padding = torch.zeros(*soft_targets.shape[:-1], logits.size(-1) - soft_targets.size(-1), 
                                        device=soft_targets.device)
                    soft_targets = torch.cat([soft_targets, padding], dim=-1)
        elif soft_targets.dim() < logits.dim():
            # Expand soft_targets to match logits dimensions
            soft_targets = soft_targets.view(*soft_targets.shape, 1).expand(*soft_targets.shape, logits.size(-1))
        
        # Apply softmax and compute cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Ensure shapes match for element-wise multiplication
        if log_probs.shape != soft_targets.shape:
            # Reshape log_probs to match the expected shape
            log_probs = log_probs.view(soft_targets.shape)
        
        # Compute loss
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss
    
    def _compute_soft_sentiment_loss(self, logits, soft_targets):
        """Compute sentiment loss with soft labels from mixup"""
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss
    
    def _compute_generation_loss(self, logits, targets):
        """Compute cross-entropy loss for generation"""
        # Shift targets to align with logits
        shifted_targets = targets[:, 1:]
        shifted_logits = logits[:, :-1, :]
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_targets.reshape(-1)
        )
        
        return loss


class FocalLossWithLS(nn.Module):
    """
    Focal loss with label smoothing for balanced classification
    
    Args:
        gamma: Focusing parameter for focal loss
        alpha: Optional tensor of class weights
        ignore_index: Index to ignore in the target
        label_smoothing: Label smoothing factor (0 to disable)
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        
        # Ensure alpha is a tensor with proper dtype if provided
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()  # Ensure alpha is float tensor
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = None
            
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, targets):
        """
        Compute focal loss with label smoothing
        
        Args:
            logits: [N, C] tensor of logits
            targets: [N] tensor of target indices
            
        Returns:
            Loss tensor
        """
        # Ensure targets are long dtype
        if targets.dtype != torch.long:
            targets = targets.long()
            
        # Get number of classes
        num_classes = logits.size(-1)
        
        # Create one-hot encoding for targets
        targets_one_hot = torch.zeros_like(logits, dtype=torch.float).scatter_(
            -1, targets.unsqueeze(-1), 1.0
        )
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            # Create mask for valid targets
            mask = (targets != self.ignore_index).unsqueeze(-1)
            
            # Apply label smoothing only to valid targets
            targets_one_hot = torch.where(
                mask,
                targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes,
                targets_one_hot
            )
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Create mask for valid targets
        mask = (targets != self.ignore_index).float()
        
        # Calculate focal weights (1 - p_t)^gamma
        # For each target class, get the predicted probability
        target_probs = torch.sum(targets_one_hot * probs, dim=-1)
        focal_weights = (1 - target_probs) ** self.gamma
        
        # Apply focal weights to cross-entropy loss
        ce_loss = -torch.sum(targets_one_hot * log_probs, dim=-1)  # Cross-entropy
        focal_loss = focal_weights * ce_loss  # Apply focal scaling
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Get device of logits
            device = logits.device
            
            # Make sure alpha is on the correct device
            alpha = self.alpha.to(device)
            
            # Initialize alpha weights with zeros (same dtype as focal_loss)
            alpha_weights = torch.zeros_like(targets, device=device, dtype=torch.float)
            
            # Get valid targets
            valid_indices = targets != self.ignore_index
            valid_targets = targets[valid_indices]
            
            # Apply alpha weighting only to valid targets (with proper indexing)
            if len(valid_targets) > 0:
                # Make sure valid_targets is in bounds of alpha
                valid_targets_clamped = torch.clamp(valid_targets, 0, len(alpha) - 1)
                # Use float tensor for alpha_weights to match alpha dtype
                alpha_weights[valid_indices] = alpha[valid_targets_clamped]
            
            focal_loss = focal_loss * alpha_weights
        
        # Apply mask and compute mean loss
        masked_loss = (focal_loss * mask).sum() / mask.sum().clamp(min=1e-6)
        
        return masked_loss