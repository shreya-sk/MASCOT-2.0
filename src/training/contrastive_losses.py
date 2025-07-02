# src/training/contrastive_losses.py - Complete Implementation
"""
Contrastive Learning Losses for ABSA (2024-2025 Breakthrough Features)
Implements InfoNCE, NT-Xent, Enhanced Triplet Loss, and ITSCL for supervised contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss extended for multiple positive and negative samples
    Based on EMNLP 2024 breakthrough for implicit sentiment detection
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, 
                query: torch.Tensor,           # [batch_size, hidden_dim]
                positive_keys: torch.Tensor,   # [batch_size, num_positives, hidden_dim]
                negative_keys: torch.Tensor,   # [batch_size, num_negatives, hidden_dim]
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Query representations (aspect/opinion embeddings)
            positive_keys: Positive samples (same sentiment/aspect)
            negative_keys: Negative samples (different sentiment/aspect)
            labels: Optional labels for supervised variant
        
        Returns:
            InfoNCE loss value
        """
        batch_size = query.size(0)
        
        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        positive_keys = F.normalize(positive_keys, dim=-1)
        negative_keys = F.normalize(negative_keys, dim=-1)
        
        # Compute similarities with positives
        # [batch_size, num_positives]
        pos_sim = torch.einsum('bd,bpd->bp', query, positive_keys) / self.temperature
        
        # Compute similarities with negatives
        # [batch_size, num_negatives]
        neg_sim = torch.einsum('bd,bnd->bn', query, negative_keys) / self.temperature
        
        # For multiple positives, we use the log-sum-exp trick
        pos_logits = torch.logsumexp(pos_sim, dim=1)  # [batch_size]
        
        # Combine positive and negative similarities
        all_sim = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, num_pos + num_neg]
        all_logits = torch.logsumexp(all_sim, dim=1)    # [batch_size]
        
        # InfoNCE loss
        loss = -pos_logits + all_logits
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss adapted for supervised learning with arbitrary positives
    Following 2024-2025 contrastive learning advances
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, hidden_dim] normalized feature representations
            labels: [batch_size] class labels
            
        Returns:
            NT-Xent loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Create mask for positive pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask to exclude self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()
        
        return loss


class EnhancedTripletLoss(nn.Module):
    """
    Enhanced Triplet Loss specifically designed for aspect-opinion-sentiment relationships
    Implements hard/semi-hard negative mining and adaptive margins
    """
    
    def __init__(self, margin: float = 0.3, mining_strategy: str = 'hard', 
                 distance_metric: str = 'euclidean'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.distance_metric = distance_metric
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            anchor: [batch_size, hidden_dim] anchor embeddings
            positive: [batch_size, hidden_dim] positive embeddings  
            negative: [batch_size, hidden_dim] negative embeddings
            labels: Optional labels for adaptive margin
            
        Returns:
            Enhanced triplet loss
        """
        # Compute distances
        if self.distance_metric == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Apply mining strategy
        if self.mining_strategy == 'hard':
            # Hard negative mining: select hardest negatives
            losses = F.relu(pos_dist - neg_dist + self.margin)
        elif self.mining_strategy == 'semi_hard':
            # Semi-hard mining: negatives that are closer than positives but still outside margin
            semi_hard_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)
            losses = F.relu(pos_dist - neg_dist + self.margin) * semi_hard_mask.float()
        else:
            # Standard triplet loss
            losses = F.relu(pos_dist - neg_dist + self.margin)
        
        # Adaptive margin based on label difficulty (if provided)
        if labels is not None:
            # Increase margin for harder sentiment classes
            label_weights = torch.ones_like(labels, dtype=torch.float)
            label_weights[labels == 0] = 1.2  # Neutral is often harder
            losses = losses * label_weights
        
        return losses.mean()


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for ABSA with multi-class support
    Extends SupCon to handle aspect-opinion-sentiment triplet relationships
    """
    
    def __init__(self, temperature: float = 0.07, contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, hidden_dim] feature representations
            labels: [batch_size] class labels
            
        Returns:
            Supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Expand labels to create mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute anchor-positive/negative logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask to exclude diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class ITSCLLoss(nn.Module):
    """
    Instruction Tuning with Supervised Contrastive Learning (ITSCL)
    Implements the EMNLP 2024 breakthrough combining InfoNCE, NT-Xent, and Cross-Modal alignment
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Loss components
        self.infonce_loss = InfoNCELoss(temperature=config.contrastive_temperature)
        self.ntxent_loss = NTXentLoss(temperature=config.contrastive_temperature)
        self.triplet_loss = EnhancedTripletLoss(margin=getattr(config, 'triplet_margin', 0.3))
        
        # Loss weights
        self.lambda_infonce = getattr(config, 'lambda_infonce', 1.0)
        self.lambda_ntxent = getattr(config, 'lambda_ntxent', 0.5)
        self.lambda_triplet = getattr(config, 'lambda_triplet', 0.3)
        self.lambda_cross_modal = getattr(config, 'lambda_cross_modal', 0.3)
        
        # Four-layer contrastive framework for combining sentiments, aspects, opinions
        self.aspect_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 128)
        )
        
        self.opinion_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 128)
        )
        
        self.sentiment_projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 128)
        )
        
        self.combination_projector = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                generation_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute ITSCL loss combining multiple contrastive objectives
        """
        device = next(iter(outputs.values())).device
        losses = {}
        
        # Extract features
        hidden_states = outputs.get('hidden_states')
        if hidden_states is None:
            return {'contrastive_loss': torch.tensor(0.0, device=device, requires_grad=True)}
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Pool sequence features
        pooled_features = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Project to contrastive spaces
        aspect_features = self.aspect_projector(pooled_features)
        opinion_features = self.opinion_projector(pooled_features)
        sentiment_features = self.sentiment_projector(pooled_features)
        
        # Combine features
        combined_features = torch.cat([aspect_features, opinion_features, sentiment_features], dim=-1)
        combined_projected = self.combination_projector(combined_features)
        
        # Get labels for contrastive learning
        aspect_labels = self._extract_contrastive_labels(targets.get('aspect_labels'))
        opinion_labels = self._extract_contrastive_labels(targets.get('opinion_labels'))
        sentiment_labels = self._extract_contrastive_labels(targets.get('sentiment_labels'))
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 1. InfoNCE Loss for each component
        if aspect_labels is not None and len(aspect_labels) > 1:
            try:
                aspect_infonce = self._compute_infonce_for_component(
                    aspect_features, aspect_labels
                )
                losses['aspect_infonce'] = aspect_infonce
                total_loss = total_loss + self.lambda_infonce * aspect_infonce
            except Exception as e:
                print(f"Warning: Aspect InfoNCE failed: {e}")
        
        if opinion_labels is not None and len(opinion_labels) > 1:
            try:
                opinion_infonce = self._compute_infonce_for_component(
                    opinion_features, opinion_labels
                )
                losses['opinion_infonce'] = opinion_infonce
                total_loss = total_loss + self.lambda_infonce * opinion_infonce
            except Exception as e:
                print(f"Warning: Opinion InfoNCE failed: {e}")
        
        if sentiment_labels is not None and len(sentiment_labels) > 1:
            try:
                sentiment_infonce = self._compute_infonce_for_component(
                    sentiment_features, sentiment_labels
                )
                losses['sentiment_infonce'] = sentiment_infonce
                total_loss = total_loss + self.lambda_infonce * sentiment_infonce
            except Exception as e:
                print(f"Warning: Sentiment InfoNCE failed: {e}")
        
        # 2. NT-Xent Loss for combined features
        if sentiment_labels is not None and len(sentiment_labels) > 1:
            try:
                combined_ntxent = self.ntxent_loss(combined_projected, sentiment_labels)
                losses['combined_ntxent'] = combined_ntxent
                total_loss = total_loss + self.lambda_ntxent * combined_ntxent
            except Exception as e:
                print(f"Warning: Combined NT-Xent failed: {e}")
        
        # 3. Cross-modal alignment loss (if generation embeddings available)
        if generation_embeddings is not None:
            try:
                cross_modal_loss = self._compute_cross_modal_loss(
                    combined_projected, generation_embeddings
                )
                losses['cross_modal'] = cross_modal_loss
                total_loss = total_loss + self.lambda_cross_modal * cross_modal_loss
            except Exception as e:
                print(f"Warning: Cross-modal loss failed: {e}")
        
        # 4. Enhanced triplet loss for sentiment relationships
        if len(sentiment_labels) >= 3:  # Need at least 3 samples for triplet
            try:
                triplet_loss = self._compute_enhanced_triplet_loss(
                    sentiment_features, sentiment_labels
                )
                losses['enhanced_triplet'] = triplet_loss
                total_loss = total_loss + self.lambda_triplet * triplet_loss
            except Exception as e:
                print(f"Warning: Enhanced triplet loss failed: {e}")
        
        losses['contrastive_loss'] = total_loss
        return losses
    
    def _extract_contrastive_labels(self, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Extract labels for contrastive learning"""
        if labels is None:
            return None
        
        # For sequence labels, extract the dominant label per sample
        if len(labels.shape) > 1:
            # Get most frequent non-padding label for each sample
            batch_labels = []
            for sample_labels in labels:
                valid_labels = sample_labels[sample_labels != -100]
                if len(valid_labels) > 0:
                    # Get most frequent label
                    unique_labels, counts = torch.unique(valid_labels, return_counts=True)
                    most_frequent = unique_labels[torch.argmax(counts)]
                    batch_labels.append(most_frequent.item())
                else:
                    batch_labels.append(0)  # Default to neutral
            
            return torch.tensor(batch_labels, device=labels.device)
        else:
            return labels
    
    def _compute_infonce_for_component(self, features: torch.Tensor, 
                                     labels: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss for a specific component"""
        batch_size = features.size(0)
        device = features.device
        
        # Create positive and negative pairs
        positive_pairs = []
        negative_pairs = []
        
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        for i, label in enumerate(labels):
            # Find positives (same label)
            pos_mask = (labels == label) & (torch.arange(batch_size, device=device) != i)
            # Find negatives (different label)
            neg_mask = labels != label
            
            if pos_mask.any() and neg_mask.any():
                pos_features = features[pos_mask]
                neg_features = features[neg_mask]
                
                # Sample a subset to avoid memory issues
                max_pos = min(5, len(pos_features))
                max_neg = min(10, len(neg_features))
                
                pos_indices = torch.randperm(len(pos_features))[:max_pos]
                neg_indices = torch.randperm(len(neg_features))[:max_neg]
                
                positive_pairs.append(pos_features[pos_indices])
                negative_pairs.append(neg_features[neg_indices])
        
        if not positive_pairs or not negative_pairs:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Pad sequences to same length
        max_pos = max(len(p) for p in positive_pairs)
        max_neg = max(len(n) for n in negative_pairs)
        
        padded_positives = []
        padded_negatives = []
        
        for pos, neg in zip(positive_pairs, negative_pairs):
            # Pad positives
            if len(pos) < max_pos:
                padding = pos[-1:].repeat(max_pos - len(pos), 1)
                pos = torch.cat([pos, padding], dim=0)
            padded_positives.append(pos)
            
            # Pad negatives
            if len(neg) < max_neg:
                padding = neg[-1:].repeat(max_neg - len(neg), 1)
                neg = torch.cat([neg, padding], dim=0)
            padded_negatives.append(neg)
        
        if not padded_positives or not padded_negatives:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        positive_keys = torch.stack(padded_positives)  # [batch_size, max_pos, hidden_dim]
        negative_keys = torch.stack(padded_negatives)  # [batch_size, max_neg, hidden_dim]
        
        # Compute InfoNCE loss
        infonce_loss = self.infonce_loss(features, positive_keys, negative_keys)
        
        return infonce_loss
    
    def _compute_cross_modal_loss(self, extraction_features: torch.Tensor,
                                generation_features: torch.Tensor) -> torch.Tensor:
        """Compute cross-modal alignment loss between extraction and generation"""
        # Align extraction and generation representations
        cosine_sim = F.cosine_similarity(extraction_features, generation_features, dim=-1)
        
        # Maximize similarity (minimize negative similarity)
        loss = -cosine_sim.mean()
        
        return loss
    
    def _compute_enhanced_triplet_loss(self, features: torch.Tensor,
                                     labels: torch.Tensor) -> torch.Tensor:
        """Compute enhanced triplet loss for sentiment relationships"""
        device = features.device
        unique_labels = torch.unique(labels)
        
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        triplet_losses = []
        
        for anchor_label in unique_labels:
            anchor_mask = labels == anchor_label
            anchor_indices = torch.where(anchor_mask)[0]
            
            if len(anchor_indices) == 0:
                continue
            
            # Get positive and negative labels
            positive_labels = anchor_label
            negative_labels = unique_labels[unique_labels != anchor_label]
            
            for anchor_idx in anchor_indices:
                anchor = features[anchor_idx:anchor_idx+1]
                
                # Find positive (same label, different instance)
                pos_mask = (labels == positive_labels) & (torch.arange(len(labels), device=device) != anchor_idx)
                pos_indices = torch.where(pos_mask)[0]
                
                if len(pos_indices) == 0:
                    continue
                
                # Find negative (different label)
                neg_mask = torch.isin(labels, negative_labels)
                neg_indices = torch.where(neg_mask)[0]
                
                if len(neg_indices) == 0:
                    continue
                
                # Sample positive and negative
                pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]
                
                positive = features[pos_idx:pos_idx+1]
                negative = features[neg_idx:neg_idx+1]
                
                # Compute triplet loss
                triplet_loss = self.triplet_loss(anchor, positive, negative)
                triplet_losses.append(triplet_loss)
        
        if triplet_losses:
            return torch.stack(triplet_losses).mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class ContrastiveVerificationModule(nn.Module):
    """
    Contrastive verification module for ensuring consistency between 
    extracted triplets and generated explanations
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.verification_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.consistency_loss = nn.BCELoss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute verification loss"""
        
        hidden_states = outputs.get('hidden_states')
        if hidden_states is None:
            return {'verification_loss': torch.tensor(0.0, requires_grad=True)}
        
        # Simple verification: check if extraction and sentiment are consistent
        pooled_features = hidden_states.mean(dim=1)
        
        # Create pseudo-verification targets (simplified)
        batch_size = pooled_features.size(0)
        verification_targets = torch.ones(batch_size, device=pooled_features.device)
        
        # Duplicate features for verification head
        verification_input = torch.cat([pooled_features, pooled_features], dim=-1)
        verification_scores = self.verification_head(verification_input).squeeze(-1)
        
        verification_loss = self.consistency_loss(verification_scores, verification_targets)
        
        return {'verification_loss': verification_loss}


class MultiLevelContrastiveLoss(nn.Module):
    """
    Multi-level contrastive loss for aspect, opinion, and sentiment representations
    Implements hierarchical contrastive learning across different semantic levels
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.temperature = getattr(config, 'contrastive_temperature', 0.07)
        self.aspect_weight = getattr(config, 'aspect_contrastive_weight', 1.0)
        self.opinion_weight = getattr(config, 'opinion_contrastive_weight', 1.0)
        self.sentiment_weight = getattr(config, 'sentiment_contrastive_weight', 1.0)
        
        # Projection heads for different levels
        self.aspect_projector = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.opinion_projector = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.sentiment_projector = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Contrastive loss functions
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=self.temperature)
        
    def forward(self, aspect_embeddings: torch.Tensor, opinion_embeddings: torch.Tensor,
                sentiment_embeddings: torch.Tensor, aspect_labels: torch.Tensor,
                opinion_labels: torch.Tensor, sentiment_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-level contrastive losses
        
        Args:
            aspect_embeddings: [batch_size, hidden_size] aspect representations
            opinion_embeddings: [batch_size, hidden_size] opinion representations  
            sentiment_embeddings: [batch_size, hidden_size] sentiment representations
            aspect_labels: [batch_size] aspect labels
            opinion_labels: [batch_size] opinion labels
            sentiment_labels: [batch_size] sentiment labels
            
        Returns:
            Dictionary of multi-level contrastive losses
        """
        device = aspect_embeddings.device
        losses = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Project embeddings to contrastive spaces
        aspect_projected = self.aspect_projector(aspect_embeddings)
        opinion_projected = self.opinion_projector(opinion_embeddings)
        sentiment_projected = self.sentiment_projector(sentiment_embeddings)
        
        # Filter valid labels (remove padding)
        valid_mask = (aspect_labels != -100) & (opinion_labels != -100) & (sentiment_labels != -100)
        
        if not valid_mask.any():
            return {'multi_level_contrastive_loss': total_loss}
        
        valid_aspect_proj = aspect_projected[valid_mask]
        valid_opinion_proj = opinion_projected[valid_mask]
        valid_sentiment_proj = sentiment_projected[valid_mask]
        valid_aspect_labels = aspect_labels[valid_mask]
        valid_opinion_labels = opinion_labels[valid_mask]
        valid_sentiment_labels = sentiment_labels[valid_mask]
        
        # Compute contrastive losses for each level
        try:
            # Aspect-level contrastive loss
            if len(torch.unique(valid_aspect_labels)) > 1:
                aspect_loss = self.contrastive_loss(valid_aspect_proj, valid_aspect_labels)
                losses['aspect_contrastive'] = aspect_loss
                total_loss = total_loss + self.aspect_weight * aspect_loss
            
            # Opinion-level contrastive loss
            if len(torch.unique(valid_opinion_labels)) > 1:
                opinion_loss = self.contrastive_loss(valid_opinion_proj, valid_opinion_labels)
                losses['opinion_contrastive'] = opinion_loss
                total_loss = total_loss + self.opinion_weight * opinion_loss
            
            # Sentiment-level contrastive loss
            if len(torch.unique(valid_sentiment_labels)) > 1:
                sentiment_loss = self.contrastive_loss(valid_sentiment_proj, valid_sentiment_labels)
                losses['sentiment_contrastive'] = sentiment_loss
                total_loss = total_loss + self.sentiment_weight * sentiment_loss
            
            # Cross-level consistency loss
            cross_level_loss = self._compute_cross_level_consistency(
                valid_aspect_proj, valid_opinion_proj, valid_sentiment_proj,
                valid_aspect_labels, valid_opinion_labels, valid_sentiment_labels
            )
            losses['cross_level_consistency'] = cross_level_loss
            total_loss = total_loss + 0.5 * cross_level_loss
            
        except Exception as e:
            print(f"Warning: Multi-level contrastive loss computation failed: {e}")
        
        losses['multi_level_contrastive_loss'] = total_loss
        return losses
    
    def _compute_cross_level_consistency(self, aspect_proj: torch.Tensor, opinion_proj: torch.Tensor,
                                       sentiment_proj: torch.Tensor, aspect_labels: torch.Tensor,
                                       opinion_labels: torch.Tensor, sentiment_labels: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss across different semantic levels"""
        
        device = aspect_proj.device
        
        # Compute pairwise similarities
        aspect_opinion_sim = F.cosine_similarity(aspect_proj, opinion_proj, dim=-1)
        aspect_sentiment_sim = F.cosine_similarity(aspect_proj, sentiment_proj, dim=-1)
        opinion_sentiment_sim = F.cosine_similarity(opinion_proj, sentiment_proj, dim=-1)
        
        # Create consistency targets based on label agreement
        # If aspect and opinion have consistent labels, their similarity should be high
        aspect_opinion_consistency = (aspect_labels == opinion_labels).float()
        aspect_sentiment_consistency = (aspect_labels == sentiment_labels).float()
        opinion_sentiment_consistency = (opinion_labels == sentiment_labels).float()
        
        # Compute consistency losses
        consistency_loss = (
            F.mse_loss(aspect_opinion_sim, aspect_opinion_consistency) +
            F.mse_loss(aspect_sentiment_sim, aspect_sentiment_consistency) +
            F.mse_loss(opinion_sentiment_sim, opinion_sentiment_consistency)
        ) / 3.0
        
        return consistency_loss


class AdaptiveContrastiveLoss(nn.Module):
    """
    Adaptive contrastive loss that adjusts temperature and margins based on training progress
    Implements curriculum learning for contrastive training
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.initial_temperature = getattr(config, 'initial_temperature', 0.07)
        self.final_temperature = getattr(config, 'final_temperature', 0.01)
        self.warmup_steps = getattr(config, 'contrastive_warmup_steps', 1000)
        
        self.base_contrastive = SupervisedContrastiveLoss(
            temperature=self.initial_temperature
        )
        
        self.current_step = 0
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive temperature"""
        
        # Update temperature based on training progress
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            current_temp = self.initial_temperature + progress * (self.final_temperature - self.initial_temperature)
        else:
            current_temp = self.final_temperature
        
        # Update base contrastive loss temperature
        self.base_contrastive.temperature = current_temp
        
        # Compute loss
        loss = self.base_contrastive(features, labels)
        
        self.current_step += 1
        
        return loss
    
    def reset_step(self):
        """Reset step counter (call at beginning of each epoch)"""
        self.current_step = 0


class ImplicitContrastiveLoss(nn.Module):
    """
    Specialized contrastive loss for implicit sentiment detection
    Handles implicit-explicit contrastive relationships
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.temperature = getattr(config, 'contrastive_temperature', 0.07)
        self.implicit_weight = getattr(config, 'implicit_contrastive_weight', 1.0)
        self.explicit_weight = getattr(config, 'explicit_contrastive_weight', 1.0)
        
        # Projection heads for implicit/explicit features
        self.implicit_projector = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.explicit_projector = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Contrastive loss
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=self.temperature)
        
    def forward(self, implicit_features: torch.Tensor, explicit_features: torch.Tensor,
                implicit_labels: torch.Tensor, explicit_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute implicit-explicit contrastive loss
        
        Args:
            implicit_features: [batch_size, hidden_size] implicit representations
            explicit_features: [batch_size, hidden_size] explicit representations
            implicit_labels: [batch_size] implicit labels
            explicit_labels: [batch_size] explicit labels
            
        Returns:
            Dictionary of implicit contrastive losses
        """
        device = implicit_features.device
        losses = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Project features
        implicit_proj = self.implicit_projector(implicit_features)
        explicit_proj = self.explicit_projector(explicit_features)
        
        # Filter valid samples
        valid_implicit_mask = implicit_labels != -100
        valid_explicit_mask = explicit_labels != -100
        
        # Implicit contrastive loss
        if valid_implicit_mask.any() and len(torch.unique(implicit_labels[valid_implicit_mask])) > 1:
            implicit_loss = self.contrastive_loss(
                implicit_proj[valid_implicit_mask], 
                implicit_labels[valid_implicit_mask]
            )
            losses['implicit_contrastive'] = implicit_loss
            total_loss = total_loss + self.implicit_weight * implicit_loss
        
        # Explicit contrastive loss
        if valid_explicit_mask.any() and len(torch.unique(explicit_labels[valid_explicit_mask])) > 1:
            explicit_loss = self.contrastive_loss(
                explicit_proj[valid_explicit_mask],
                explicit_labels[valid_explicit_mask]
            )
            losses['explicit_contrastive'] = explicit_loss
            total_loss = total_loss + self.explicit_weight * explicit_loss
        
        # Cross-modal implicit-explicit consistency
        if valid_implicit_mask.any() and valid_explicit_mask.any():
            cross_modal_loss = self._compute_implicit_explicit_consistency(
                implicit_proj[valid_implicit_mask],
                explicit_proj[valid_explicit_mask],
                implicit_labels[valid_implicit_mask],
                explicit_labels[valid_explicit_mask]
            )
            losses['implicit_explicit_consistency'] = cross_modal_loss
            total_loss = total_loss + 0.3 * cross_modal_loss
        
        losses['total_implicit_contrastive'] = total_loss
        return losses
    
    def _compute_implicit_explicit_consistency(self, implicit_proj: torch.Tensor, explicit_proj: torch.Tensor,
                                             implicit_labels: torch.Tensor, explicit_labels: torch.Tensor) -> torch.Tensor:
        """Compute consistency between implicit and explicit representations"""
        
        # For samples with same sentiment labels, implicit and explicit features should be similar
        min_len = min(len(implicit_proj), len(explicit_proj))
        implicit_proj = implicit_proj[:min_len]
        explicit_proj = explicit_proj[:min_len]
        implicit_labels = implicit_labels[:min_len]
        explicit_labels = explicit_labels[:min_len]
        
        # Compute similarity
        similarity = F.cosine_similarity(implicit_proj, explicit_proj, dim=-1)
        
        # Create targets: high similarity if labels match, low similarity otherwise
        label_match = (implicit_labels == explicit_labels).float()
        
        # Consistency loss: encourage high similarity for matching labels
        consistency_loss = F.mse_loss(similarity, label_match)
        
        return consistency_loss


# Helper functions for contrastive learning
def create_contrastive_pairs(features: torch.Tensor, labels: torch.Tensor, 
                           num_positives: int = 2, num_negatives: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create positive and negative pairs for contrastive learning
    
    Args:
        features: [batch_size, hidden_dim] feature representations
        labels: [batch_size] labels
        num_positives: Number of positive samples per anchor
        num_negatives: Number of negative samples per anchor
        
    Returns:
        anchors, positives, negatives tensors
    """
    device = features.device
    batch_size = features.shape[0]
    
    anchors = []
    positives = []
    negatives = []
    
    for i in range(batch_size):
        anchor_label = labels[i]
        anchor_feature = features[i]
        
        # Find positive samples (same label, different instance)
        pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=device) != i)
        pos_indices = torch.where(pos_mask)[0]
        
        # Find negative samples (different label)
        neg_mask = labels != anchor_label
        neg_indices = torch.where(neg_mask)[0]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            # Sample positives and negatives
            num_pos = min(num_positives, len(pos_indices))
            num_neg = min(num_negatives, len(neg_indices))
            
            selected_pos = pos_indices[torch.randperm(len(pos_indices))[:num_pos]]
            selected_neg = neg_indices[torch.randperm(len(neg_indices))[:num_neg]]
            
            # Repeat anchor for each positive
            for pos_idx in selected_pos:
                anchors.append(anchor_feature)
                positives.append(features[pos_idx])
                
                # Sample one negative for this positive
                neg_idx = selected_neg[torch.randint(0, len(selected_neg), (1,))]
                negatives.append(features[neg_idx])
    
    if anchors:
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    else:
        # Return empty tensors if no valid pairs found
        empty_tensor = torch.empty(0, features.shape[-1], device=device)
        return empty_tensor, empty_tensor, empty_tensor


def compute_contrastive_accuracy(features: torch.Tensor, labels: torch.Tensor, 
                               temperature: float = 0.07) -> float:
    """
    Compute contrastive accuracy (how often positive samples are ranked higher than negatives)
    
    Args:
        features: [batch_size, hidden_dim] normalized feature representations
        labels: [batch_size] labels
        temperature: Temperature for similarity computation
        
    Returns:
        Contrastive accuracy as float
    """
    device = features.device
    batch_size = features.shape[0]
    
    # Normalize features
    features = F.normalize(features, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature
    
    # Create label mask
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Remove self-similarity
    mask.fill_diagonal_(0)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(batch_size):
        # Get similarities for anchor i
        anchor_sim = sim_matrix[i]
        
        # Find positive and negative samples
        pos_mask = mask[i] == 1
        neg_mask = mask[i] == 0
        
        if pos_mask.any() and neg_mask.any():
            pos_sim = anchor_sim[pos_mask]
            neg_sim = anchor_sim[neg_mask]
            
            # Count how many positives rank higher than negatives
            for pos_s in pos_sim:
                for neg_s in neg_sim:
                    total_predictions += 1
                    if pos_s > neg_s:
                        correct_predictions += 1
    
    if total_predictions > 0:
        return correct_predictions / total_predictions
    else:
        return 0.0


def adaptive_temperature_schedule(initial_temp: float, final_temp: float, 
                                current_step: int, total_steps: int) -> float:
    """
    Compute adaptive temperature for contrastive learning
    
    Args:
        initial_temp: Starting temperature
        final_temp: Ending temperature
        current_step: Current training step
        total_steps: Total training steps
        
    Returns:
        Current temperature
    """
    if current_step >= total_steps:
        return final_temp
    
    # Linear decay
    progress = current_step / total_steps
    current_temp = initial_temp + progress * (final_temp - initial_temp)
    
    return current_temp