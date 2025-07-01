# src/training/losses.py - Enhanced version with contrastive learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrastive_losses import ITSCLLoss, ContrastiveVerificationModule, MultiLevelContrastiveLoss

class ABSALoss(nn.Module):
    """
    Enhanced ABSA loss function with contrastive learning integration
    
    2024-2025 breakthrough: Combines traditional extraction losses with advanced
    contrastive learning for unified extraction-generation training.
    """
    def __init__(self, config):
        super().__init__()
        
        # Traditional loss weights
        self.aspect_weight = getattr(config, 'aspect_loss_weight', 2.0)
        self.opinion_weight = getattr(config, 'opinion_loss_weight', 2.0)
        self.sentiment_weight = getattr(config, 'sentiment_loss_weight', 1.0)
        self.boundary_weight = getattr(config, 'boundary_weight', 0.5)
        
        # Instruction-following weights
        self.extraction_weight = getattr(config, 'extraction_weight', 1.0)
        self.generation_weight = getattr(config, 'generation_weight', 0.5)
        
        # Contrastive learning weights (NEW 2024-2025)
        self.contrastive_weight = getattr(config, 'contrastive_weight', 1.0)
        self.verification_weight = getattr(config, 'verification_weight', 0.3)
        self.multi_level_weight = getattr(config, 'multi_level_weight', 0.5)
        
        # Label smoothing and focal loss parameters
        self.label_smoothing = getattr(config, 'label_smoothing', 0.1)
        self.gamma = getattr(config, 'focal_gamma', 2.0)
        self.use_focal_loss = getattr(config, 'use_focal_loss', True)
        
        # Initialize contrastive learning components
        self.use_contrastive = getattr(config, 'use_contrastive_learning', True)
        if self.use_contrastive:
            self.itscl_loss = ITSCLLoss(config)
            self.verification_module = ContrastiveVerificationModule(config)
            self.multi_level_contrastive = MultiLevelContrastiveLoss(config)
        
        # Class weights for imbalanced data
        aspect_weights = torch.tensor([0.1, 1.0, 0.8])  # O, B, I
        opinion_weights = torch.tensor([0.1, 1.0, 0.8])
        
        if self.use_focal_loss:
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
        
        self.sentiment_criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.label_smoothing
        )
        
    def forward(self, outputs, targets, generation_embeddings=None):
        """
        Enhanced forward pass with contrastive learning integration
        """
        try:
            device = outputs['aspect_logits'].device
            
            # ============================================================================
            # TRADITIONAL EXTRACTION LOSSES
            # ============================================================================
            
            # Extract predictions and targets
            aspect_logits = outputs.get('aspect_logits')
            opinion_logits = outputs.get('opinion_logits')
            sentiment_logits = outputs.get('sentiment_logits')
            
            aspect_labels = targets.get('aspect_labels')
            opinion_labels = targets.get('opinion_labels')
            sentiment_labels = targets.get('sentiment_labels')
            
            # Validate inputs
            if any(x is None for x in [aspect_logits, opinion_logits, sentiment_logits]):
                raise ValueError("Missing required logits in outputs")
            if any(x is None for x in [aspect_labels, opinion_labels, sentiment_labels]):
                raise ValueError("Missing required labels in targets")
            
            # Compute traditional losses
            aspect_loss = self._compute_span_loss(aspect_logits, aspect_labels, self.span_criterion, "aspect")
            opinion_loss = self._compute_span_loss(opinion_logits, opinion_labels, self.opinion_criterion, "opinion")
            sentiment_loss = self._compute_sentiment_loss(sentiment_logits, sentiment_labels)
            
            # Boundary loss if available
            boundary_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if 'boundary_logits' in outputs and self.boundary_weight > 0:
                boundary_loss = self._compute_boundary_loss(
                    outputs['boundary_logits'], aspect_labels, opinion_labels
                )
            
            # Generation loss if available
            generation_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if 'generation_loss' in outputs:
                generation_loss = outputs['generation_loss']
            
            # ============================================================================
            # CONTRASTIVE LEARNING LOSSES (2024-2025 BREAKTHROUGH)
            # ============================================================================
            
            contrastive_total = torch.tensor(0.0, device=device, requires_grad=True)
            contrastive_components = {}
            
            if self.use_contrastive:
                # 1. ITSCL Loss (InfoNCE + NT-Xent + Cross-Modal)
                try:
                    itscl_results = self.itscl_loss(outputs, targets, generation_embeddings)
                    contrastive_components.update(itscl_results)
                    contrastive_total = contrastive_total + itscl_results.get('contrastive_loss', 0)
                except Exception as e:
                    print(f"Warning: ITSCL loss computation failed: {e}")
                    contrastive_components['contrastive_loss'] = torch.tensor(0.0, device=device)
                
                # 2. Multi-Level Contrastive Loss
                try:
                    if 'span_features' in outputs or 'hidden_states' in outputs:
                        span_features = outputs.get('span_features', outputs.get('hidden_states'))
                        
                        # Extract embeddings for contrastive learning
                        aspect_embeddings = self._extract_embeddings_for_contrastive(
                            span_features, aspect_logits, aspect_labels
                        )
                        opinion_embeddings = self._extract_embeddings_for_contrastive(
                            span_features, opinion_logits, opinion_labels
                        )
                        sentiment_embeddings = self._extract_sentiment_embeddings(
                            span_features, sentiment_logits
                        )
                        
                        # Compute multi-level contrastive loss
                        multi_level_results = self.multi_level_contrastive(
                            aspect_embeddings, opinion_embeddings, sentiment_embeddings,
                            self._flatten_labels(aspect_labels),
                            self._flatten_labels(opinion_labels),
                            self._flatten_labels(sentiment_labels)
                        )
                        
                        contrastive_components.update({
                            f"ml_{k}": v for k, v in multi_level_results.items()
                        })
                        contrastive_total = contrastive_total + multi_level_results.get('multi_level_contrastive_loss', 0)
                        
                except Exception as e:
                    print(f"Warning: Multi-level contrastive loss computation failed: {e}")
                
                # 3. Contrastive Verification (if generation embeddings available)
                try:
                    if generation_embeddings is not None and 'span_features' in outputs:
                        span_features = outputs['span_features']
                        batch_size = span_features.size(0)
                        
                        # Create triplet embeddings
                        triplet_embeddings = self._create_triplet_embeddings(
                            span_features, aspect_logits, opinion_logits, sentiment_logits
                        )
                        
                        # Ensure generation_embeddings has correct batch dimension
                        if generation_embeddings.size(0) != batch_size:
                            # Repeat or truncate to match batch size
                            if generation_embeddings.size(0) == 1:
                                generation_embeddings = generation_embeddings.repeat(batch_size, 1)
                            else:
                                generation_embeddings = generation_embeddings[:batch_size]
                        
                        verification_results = self.verification_module(
                            triplet_embeddings, generation_embeddings
                        )
                        
                        contrastive_components.update({
                            f"verify_{k}": v for k, v in verification_results.items()
                        })
                        contrastive_total = contrastive_total + verification_results.get('verification_loss', 0)
                        
                except Exception as e:
                    print(f"Warning: Contrastive verification computation failed: {e}")
            
            # ============================================================================
            # TOTAL LOSS COMBINATION
            # ============================================================================
            
            # Traditional extraction loss
            extraction_total = (
                self.aspect_weight * aspect_loss +
                self.opinion_weight * opinion_loss +
                self.sentiment_weight * sentiment_loss +
                self.boundary_weight * boundary_loss
            )
            
            # Total unified loss
            total_loss = (
                self.extraction_weight * extraction_total +
                self.generation_weight * generation_loss +
                self.contrastive_weight * contrastive_total
            )
            
            # Ensure total_loss requires gradients
            if not total_loss.requires_grad:
                dummy_term = (aspect_logits.sum() + opinion_logits.sum() + sentiment_logits.sum()) * 1e-8
                total_loss = total_loss + dummy_term
            
            # Return comprehensive loss dictionary
            loss_dict = {
                'loss': total_loss,
                'aspect_loss': aspect_loss.detach().item() if isinstance(aspect_loss, torch.Tensor) else aspect_loss,
                'opinion_loss': opinion_loss.detach().item() if isinstance(opinion_loss, torch.Tensor) else opinion_loss,
                'sentiment_loss': sentiment_loss.detach().item() if isinstance(sentiment_loss, torch.Tensor) else sentiment_loss,
                'boundary_loss': boundary_loss.detach().item() if isinstance(boundary_loss, torch.Tensor) else boundary_loss,
                'generation_loss': generation_loss.detach().item() if isinstance(generation_loss, torch.Tensor) else generation_loss,
                'extraction_loss': extraction_total.detach().item() if isinstance(extraction_total, torch.Tensor) else extraction_total,
                'contrastive_total': contrastive_total.detach().item() if isinstance(contrastive_total, torch.Tensor) else contrastive_total,
            }
            
            # Add contrastive component losses
            for k, v in contrastive_components.items():
                if isinstance(v, torch.Tensor):
                    loss_dict[k] = v.detach().item()
                else:
                    loss_dict[k] = v
            
            return loss_dict
            
        except Exception as e:
            print(f"Error in enhanced loss calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal differentiable loss
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
                'generation_loss': 0.0,
                'extraction_loss': 0.0,
                'contrastive_total': 0.0,
            }
    
    def _compute_span_loss(self, logits, labels, criterion, loss_name):
        """Compute loss for span detection with robust handling"""
        try:
            # Handle multi-span case
            if len(labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                batch_size, num_spans, seq_len = labels.shape
                if num_spans == 1:
                    labels = labels.squeeze(1)
                else:
                    labels = labels.max(dim=1)[0]
            
            labels = labels.long()
            
            # Validate shapes
            batch_size, seq_len, num_classes = logits.shape
            if labels.shape != (batch_size, seq_len):
                if labels.numel() == batch_size * seq_len:
                    labels = labels.view(batch_size, seq_len)
                else:
                    labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=logits.device)
            
            # Compute loss
            loss = criterion(
                logits.view(-1, num_classes),
                labels.view(-1)
            )
            
            return loss
            
        except Exception as e:
            print(f"Error computing {loss_name} loss: {e}")
            return logits.sum() * 1e-8
    
    def _compute_sentiment_loss(self, logits, labels):
        """Compute sentiment classification loss with robust handling"""
        try:
            if len(labels.shape) > 1:
                labels = labels[:, 0] if labels.shape[1] > 0 else labels.squeeze()
            
            labels = labels.long()
            
            if logits.shape[0] != labels.shape[0]:
                min_batch = min(logits.shape[0], labels.shape[0])
                logits = logits[:min_batch]
                labels = labels[:min_batch]
            
            loss = self.sentiment_criterion(logits, labels)
            return loss
            
        except Exception as e:
            print(f"Error computing sentiment loss: {e}")
            return logits.sum() * 1e-8
    
    def _compute_boundary_loss(self, boundary_logits, aspect_labels, opinion_labels):
        """Compute boundary refinement loss"""
        try:
            batch_size, seq_len, _ = boundary_logits.shape
            device = boundary_logits.device
            
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
                    
                    # End boundary
                    if s > 0 and s < min(aspect_labels.shape[1], opinion_labels.shape[1]):
                        aspect_end = (aspect_labels[b, s-1] > 0 and aspect_labels[b, s] == 0)
                        opinion_end = (opinion_labels[b, s-1] > 0 and opinion_labels[b, s] == 0)
                        if aspect_end or opinion_end:
                            boundary_targets[b, s-1, 1] = 1.0
            
            loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
            return loss
            
        except Exception as e:
            print(f"Error computing boundary loss: {e}")
            return boundary_logits.sum() * 1e-8
    
    def _extract_embeddings_for_contrastive(self, span_features, logits, labels):
        """Extract embeddings for contrastive learning using attention pooling"""
        try:
            batch_size, seq_len, hidden_size = span_features.shape
            
            # Get attention weights from B+I tags
            attention_weights = F.softmax(logits[:, :, 1:].sum(dim=-1), dim=-1)
            
            # Weighted pooling
            embeddings = torch.sum(span_features * attention_weights.unsqueeze(-1), dim=1)
            
            return embeddings
            
        except Exception as e:
            print(f"Error extracting embeddings for contrastive learning: {e}")
            # Return mean pooled embeddings as fallback
            return span_features.mean(dim=1)
    
    def _extract_sentiment_embeddings(self, span_features, sentiment_logits):
        """Extract sentiment embeddings using global pooling"""
        try:
            # Global average pooling for sentiment representation
            sentiment_embeddings = span_features.mean(dim=1)
            return sentiment_embeddings
            
        except Exception as e:
            print(f"Error extracting sentiment embeddings: {e}")
            return span_features.mean(dim=1)
    
    def _flatten_labels(self, labels):
        """Flatten multi-dimensional labels for contrastive learning"""
        try:
            if len(labels.shape) == 3:
                # Convert multi-span labels to single labels by taking max
                labels = labels.max(dim=1)[0]
            
            if len(labels.shape) == 2:
                # Convert sequence labels to single label by taking mode
                labels = torch.mode(labels, dim=1)[0]
            
            return labels.long()
            
        except Exception as e:
            print(f"Error flattening labels: {e}")
            batch_size = labels.size(0)
            return torch.zeros(batch_size, dtype=torch.long, device=labels.device)
    
    def _create_triplet_embeddings(self, span_features, aspect_logits, opinion_logits, sentiment_logits):
        """Create triplet embeddings for verification"""
        try:
            batch_size = span_features.size(0)
            
            # Extract aspect and opinion embeddings
            aspect_embeddings = self._extract_embeddings_for_contrastive(
                span_features, aspect_logits, None
            )
            opinion_embeddings = self._extract_embeddings_for_contrastive(
                span_features, opinion_logits, None
            )
            
            # Get sentiment embeddings
            sentiment_embeddings = sentiment_logits.mean(dim=1) if len(sentiment_logits.shape) > 1 else sentiment_logits
            
            # Expand sentiment to match hidden size
            if sentiment_embeddings.size(-1) != span_features.size(-1):
                hidden_size = span_features.size(-1)
                sentiment_expanded = sentiment_embeddings.unsqueeze(-1).expand(-1, hidden_size)
            else:
                sentiment_expanded = sentiment_embeddings
            
            # Concatenate all embeddings
            triplet_embeddings = torch.cat([
                aspect_embeddings,
                opinion_embeddings,
                sentiment_expanded
            ], dim=-1)
            
            return triplet_embeddings
            
        except Exception as e:
            print(f"Error creating triplet embeddings: {e}")
            # Return concatenated mean pooled features
            hidden_size = span_features.size(-1)
            return torch.cat([
                span_features.mean(dim=1),
                span_features.mean(dim=1),
                span_features.mean(dim=1)
            ], dim=-1)


class FocalLossWithLS(nn.Module):
    """Focal loss with label smoothing for handling class imbalance"""
    
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha.float()
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = None
    
    def forward(self, logits, targets):
        """Compute focal loss with label smoothing"""
        try:
            targets = targets.long()
            
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
                alpha_weights = torch.ones_like(targets, dtype=torch.float, device=logits.device)
                valid_mask = targets != self.ignore_index
                valid_targets = targets[valid_mask]
                
                if len(valid_targets) > 0:
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
            return F.cross_entropy(logits, targets, ignore_index=self.ignore_index)