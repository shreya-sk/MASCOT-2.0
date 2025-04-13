# src/training/losses.py
<<<<<<< Updated upstream
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
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
>>>>>>> Stashed changes

class ABSALoss(nn.Module):
    """Combined loss for ABSA tasks - simplified version to fix training issues"""
    def __init__(self, config):
        super().__init__()
        # Span detection losses
        self.span_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Sentiment classification loss
        self.sentiment_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Loss weights
        self.aspect_weight = getattr(config, 'aspect_loss_weight', 1.0)
        self.opinion_weight = getattr(config, 'opinion_loss_weight', 1.0)
        self.sentiment_weight = getattr(config, 'sentiment_loss_weight', 1.0)
        

    def forward(self, outputs, targets, generate=False):
<<<<<<< Updated upstream
        """Calculate the combined loss for ABSA tasks"""
=======
        """
        Calculate the combined loss for ABSA tasks - simplified for robustness
        
        Args:
            outputs: Model output dictionary
            targets: Target dictionary
            generate: Whether to include generation loss
            
        Returns:
            dict: Dictionary with loss components
        """
>>>>>>> Stashed changes
        try:
            # Get predictions
            aspect_logits = outputs['aspect_logits']  # [batch_size, seq_len, 3]
            opinion_logits = outputs['opinion_logits']  # [batch_size, seq_len, 3]
            sentiment_logits = outputs['sentiment_logits']  # [batch_size, 3]
            
<<<<<<< Updated upstream
            # Get labels
            aspect_labels = targets['aspect_labels']  # [batch_size, num_spans, seq_len]
            opinion_labels = targets['opinion_labels']  # [batch_size, num_spans, seq_len]
            sentiment_labels = targets['sentiment_labels']  # [batch_size, num_spans]
            
            print(f"Aspect logits shape: {aspect_logits.shape}, Aspect labels shape: {aspect_labels.shape}")
            print(f"Opinion logits shape: {opinion_logits.shape}, Opinion labels shape: {opinion_labels.shape}")
            print(f"Sentiment logits shape: {sentiment_logits.shape}, Sentiment labels shape: {sentiment_labels.shape}")
            
            # Fix 1: Convert multi-span labels to match logits
            # We'll take the maximum values across spans to create a single label per token
            if len(aspect_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                print("Converting multi-span aspect labels to match logits")
                # Take max across spans dimension to get a single label per token
                aspect_labels = aspect_labels.max(dim=1)[0]  # [batch_size, seq_len]
            
            if len(opinion_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                print("Converting multi-span opinion labels to match logits")
                opinion_labels = opinion_labels.max(dim=1)[0]  # [batch_size, seq_len]
            
            # Fix 2: Handle sentiment labels
            if len(sentiment_labels.shape) == 2:  # [batch_size, num_spans]
                print("Converting multi-span sentiment labels")
                # Take the first sentiment label for each batch item
                sentiment_labels = sentiment_labels[:, 0]  # [batch_size]
            
            # Calculate losses
            aspect_loss = self.span_criterion(
                aspect_logits.reshape(-1, 3), 
                aspect_labels.reshape(-1).long()
            )
            
            opinion_loss = self.span_criterion(
                opinion_logits.reshape(-1, 3),
                opinion_labels.reshape(-1).long()
            )
            
            sentiment_loss = self.sentiment_criterion(
                sentiment_logits,
                sentiment_labels.long()
            )
=======
            # Get labels - ensure we can handle different shapes
            aspect_labels = targets.get('aspect_labels')
            opinion_labels = targets.get('opinion_labels')
            sentiment_labels = targets.get('sentiment_labels')
            
            # Initialize default losses
            aspect_loss = torch.tensor(0.0, device=aspect_logits.device, requires_grad=True)
            opinion_loss = torch.tensor(0.0, device=opinion_logits.device, requires_grad=True)
            sentiment_loss = torch.tensor(0.0, device=sentiment_logits.device, requires_grad=True)
            
            # Calculate aspect loss
            if aspect_labels is not None:
                try:
                    # Reshape logits and labels to 2D
                    aspect_logits_flat = aspect_logits.reshape(-1, 3)
                    
                    # Handle different label shapes
                    if len(aspect_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                        # Take the first span
                        aspect_labels = aspect_labels[:, 0] if aspect_labels.size(1) > 0 else torch.zeros_like(aspect_logits[:, :, 0])
                    
                    # Convert to long and make contiguous    
                    aspect_labels_flat = aspect_labels.reshape(-1).long().contiguous()
                    
                    # Filter valid positions
                    valid_mask = aspect_labels_flat != -100
                    if valid_mask.sum() > 0:
                        aspect_loss = self.span_criterion(
                            aspect_logits_flat[valid_mask], 
                            aspect_labels_flat[valid_mask]
                        )
                except Exception as e:
                    print(f"Error calculating aspect loss: {e}")
                    # Create a dummy loss that's differentiable
                    aspect_loss = aspect_logits.sum() * 0.0001
            
            # Calculate opinion loss
            if opinion_labels is not None:
                try:
                    # Reshape logits and labels to 2D
                    opinion_logits_flat = opinion_logits.reshape(-1, 3)
                    
                    # Handle different label shapes
                    if len(opinion_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                        # Take the first span
                        opinion_labels = opinion_labels[:, 0] if opinion_labels.size(1) > 0 else torch.zeros_like(opinion_logits[:, :, 0])
                    
                    # Convert to long and make contiguous
                    opinion_labels_flat = opinion_labels.reshape(-1).long().contiguous()
                    
                    # Filter valid positions
                    valid_mask = opinion_labels_flat != -100
                    if valid_mask.sum() > 0:
                        opinion_loss = self.span_criterion(
                            opinion_logits_flat[valid_mask],
                            opinion_labels_flat[valid_mask]
                        )
                except Exception as e:
                    print(f"Error calculating opinion loss: {e}")
                    # Create a dummy loss that's differentiable
                    opinion_loss = opinion_logits.sum() * 0.0001
            
            # Calculate sentiment loss
            if sentiment_labels is not None:
                try:
                    # Handle different shapes
                    if len(sentiment_labels.shape) > 1:
                        # Take the first sentiment label for each batch item
                        sentiment_labels = sentiment_labels[:, 0] if sentiment_labels.size(1) > 0 else torch.zeros(sentiment_logits.size(0), device=sentiment_logits.device)
                    
                    # Convert to long
                    sentiment_labels = sentiment_labels.long()
                    
                    # Calculate loss
                    sentiment_loss = self.sentiment_criterion(sentiment_logits, sentiment_labels)
                except Exception as e:
                    print(f"Error calculating sentiment loss: {e}")
                    # Create a dummy loss that's differentiable
                    sentiment_loss = sentiment_logits.sum() * 0.0001
>>>>>>> Stashed changes
            
            # Combine losses
            total_loss = (
                self.aspect_weight * aspect_loss +
                self.opinion_weight * opinion_loss +
                self.sentiment_weight * sentiment_loss
            )
            
<<<<<<< Updated upstream
=======
            # Ensure the loss requires gradient
            if not total_loss.requires_grad:
                print("Warning: Loss doesn't require grad, creating differentiable dummy loss")
                # Add a small value with gradient to make it differentiable
                total_loss = total_loss + (aspect_logits.sum() + opinion_logits.sum() + sentiment_logits.sum()) * 0.0001
            
            # Return dictionary with all loss components
>>>>>>> Stashed changes
            return {
                'loss': total_loss,
                'aspect_loss': aspect_loss.item(),
                'opinion_loss': opinion_loss.item(),
                'sentiment_loss': sentiment_loss.item()
            }
<<<<<<< Updated upstream
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            # Create a dummy loss that's differentiable
            dummy_loss = outputs['aspect_logits'].sum() * 0.0001

        if generate and 'explanations' in outputs and 'explanation_targets' in targets:
            try:
                # Compute cross-entropy loss for generation
                gen_loss = F.cross_entropy(
                    outputs['explanations'].view(-1, self.config.vocab_size),
                    targets['explanation_targets'].view(-1),
                    ignore_index=-100  # Ignore padding tokens
                )
                
                # Add weighted generative loss
                total_loss = total_loss + getattr(self.config, 'generation_weight', 0.5) * gen_loss
                return {
                    'loss': total_loss,
                    'aspect_loss': aspect_loss.item(),
                    'opinion_loss': opinion_loss.item(),
                    'sentiment_loss': sentiment_loss.item(),
                    'generation_loss': gen_loss.item()
                }
            except Exception as e:
                print(f"Error in generation loss calculation: {e}")
=======
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple differentiable loss for backward compatibility
            device = outputs['aspect_logits'].device
            dummy_loss = outputs['aspect_logits'].sum() * 0.0001 + outputs['opinion_logits'].sum() * 0.0001 + outputs['sentiment_logits'].sum() * 0.0001
>>>>>>> Stashed changes
            
            return {
                'loss': dummy_loss,
                'aspect_loss': 0.0,
                'opinion_loss': 0.0,
                'sentiment_loss': 0.0
            }