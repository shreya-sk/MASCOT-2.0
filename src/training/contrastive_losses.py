"""
Contrastive Learning Losses for ABSA (2024-2025 Breakthrough Features)
Implements InfoNCE, NT-Xent, and Enhanced Triplet Loss for supervised contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


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
    Enhanced version supporting aspect-opinion-sentiment relationships
    """
    
    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, 
                features: torch.Tensor,        # [batch_size, hidden_dim]
                labels: torch.Tensor,          # [batch_size] - sentiment/aspect labels
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Supervised NT-Xent loss for contrastive learning
        
        Args:
            features: Normalized feature embeddings
            labels: Class labels for supervision
            mask: Optional mask for valid samples
            
        Returns:
            NT-Xent loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        if mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create positive mask: samples with same label
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-contrast (diagonal)
        mask_pos = mask_pos * (1 - torch.eye(batch_size).to(device))
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size).to(device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class EnhancedTripletLoss(nn.Module):
    """
    Enhanced Triplet Loss for aspect-opinion-sentiment relationships
    Supports multiple positive/negative mining strategies
    """
    
    def __init__(self, 
                 margin: float = 0.3,
                 mining_strategy: str = 'hard',
                 distance_metric: str = 'euclidean'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
        self.distance_metric = distance_metric
        
    def compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute distance between embeddings"""
        if self.distance_metric == 'euclidean':
            return F.pairwise_distance(x1, x2, p=2)
        elif self.distance_metric == 'cosine':
            return 1 - F.cosine_similarity(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def mine_triplets(self, 
                     embeddings: torch.Tensor, 
                     labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine hard triplets for training
        
        Args:
            embeddings: [batch_size, hidden_dim]
            labels: [batch_size] - aspect/sentiment labels
            
        Returns:
            anchor_indices, positive_indices, negative_indices
        """
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        
        # Remove diagonal (self-pairs)
        eye_mask = torch.eye(batch_size, device=embeddings.device).bool()
        labels_equal = labels_equal & ~eye_mask
        
        anchors, positives, negatives = [], [], []
        
        for i in range(batch_size):
            # Find positive samples (same label)
            pos_mask = labels_equal[i]
            if not pos_mask.any():
                continue
                
            # Find negative samples (different label)
            neg_mask = labels_not_equal[i]
            if not neg_mask.any():
                continue
            
            if self.mining_strategy == 'hard':
                # Hard positive: farthest positive sample
                pos_distances = distances[i][pos_mask]
                hardest_pos_idx = pos_distances.argmax()
                pos_idx = torch.where(pos_mask)[0][hardest_pos_idx]
                
                # Hard negative: closest negative sample
                neg_distances = distances[i][neg_mask]
                hardest_neg_idx = neg_distances.argmin()
                neg_idx = torch.where(neg_mask)[0][hardest_neg_idx]
                
            elif self.mining_strategy == 'semi_hard':
                # Semi-hard negative mining
                pos_distances = distances[i][pos_mask]
                pos_idx = torch.where(pos_mask)[0][pos_distances.argmin()]  # Closest positive
                
                # Semi-hard negatives: closer than positive but still negative
                anchor_pos_dist = distances[i][pos_idx]
                neg_distances = distances[i][neg_mask]
                semi_hard_mask = (neg_distances > anchor_pos_dist) & (neg_distances < anchor_pos_dist + self.margin)
                
                if semi_hard_mask.any():
                    semi_hard_neg_indices = torch.where(neg_mask)[0][semi_hard_mask]
                    neg_idx = semi_hard_neg_indices[torch.randint(len(semi_hard_neg_indices), (1,))]
                else:
                    # Fallback to hardest negative
                    neg_idx = torch.where(neg_mask)[0][neg_distances.argmin()]
                    
            else:  # random
                pos_idx = torch.where(pos_mask)[0][torch.randint(pos_mask.sum(), (1,))]
                neg_idx = torch.where(neg_mask)[0][torch.randint(neg_mask.sum(), (1,))]
            
            anchors.append(i)
            positives.append(pos_idx.item())
            negatives.append(neg_idx.item())
        
        if not anchors:
            # Return dummy triplets if no valid triplets found
            return torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
            
        return (torch.tensor(anchors, device=embeddings.device),
                torch.tensor(positives, device=embeddings.device),
                torch.tensor(negatives, device=embeddings.device))
    
    def forward(self, 
                embeddings: torch.Tensor, 
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced triplet loss
        
        Args:
            embeddings: [batch_size, hidden_dim]
            labels: [batch_size] - aspect/sentiment labels
            
        Returns:
            Dictionary with loss and mining statistics
        """
        # Mine triplets
        anchor_indices, pos_indices, neg_indices = self.mine_triplets(embeddings, labels)
        
        if len(anchor_indices) <= 1:
            return {'loss': torch.tensor(0.0, device=embeddings.device),
                   'num_triplets': 0}
        
        # Get anchor, positive, negative embeddings
        anchor_emb = embeddings[anchor_indices]
        pos_emb = embeddings[pos_indices]
        neg_emb = embeddings[neg_indices]
        
        # Compute distances
        pos_dist = self.compute_distance(anchor_emb, pos_emb)
        neg_dist = self.compute_distance(anchor_emb, neg_emb)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # Statistics
        valid_triplets = (loss > 0).sum().item()
        
        return {
            'loss': loss.mean(),
            'num_triplets': len(anchor_indices),
            'valid_triplets': valid_triplets,
            'avg_pos_dist': pos_dist.mean().item(),
            'avg_neg_dist': neg_dist.mean().item()
        }


class SupervisedContrastiveLoss(nn.Module):
    """
    Unified supervised contrastive loss combining multiple strategies
    Implements the ITSCL framework from EMNLP 2024
    """
    
    def __init__(self,
                 temperature: float = 0.07,
                 contrast_mode: str = 'all',
                 base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self,
                features: torch.Tensor,           # [batch_size, num_views, hidden_dim]
                labels: torch.Tensor,             # [batch_size]
                aspect_labels: torch.Tensor,      # [batch_size] - aspect labels
                opinion_labels: torch.Tensor,     # [batch_size] - opinion labels
                sentiment_labels: torch.Tensor,   # [batch_size] - sentiment labels
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Multi-level supervised contrastive loss for ABSA
        
        Args:
            features: Feature representations with multiple views
            labels: Combined triplet labels
            aspect_labels: Aspect-specific labels
            opinion_labels: Opinion-specific labels  
            sentiment_labels: Sentiment labels
            mask: Optional mask for valid samples
            
        Returns:
            Dictionary with different loss components
        """
        device = features.device
        batch_size, num_views = features.shape[:2]
        
        if mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        
        # Reshape features: [batch_size * num_views, hidden_dim]
        contrast_features = features.view(batch_size * num_views, -1)
        contrast_features = F.normalize(contrast_features, dim=1)
        
        # Create extended labels for all views
        contrast_labels = labels.repeat(num_views)
        contrast_aspect = aspect_labels.repeat(num_views)
        contrast_opinion = opinion_labels.repeat(num_views)
        contrast_sentiment = sentiment_labels.repeat(num_views)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_features, contrast_features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create positive masks for different levels
        contrast_labels = contrast_labels.contiguous().view(-1, 1)
        contrast_aspect = contrast_aspect.contiguous().view(-1, 1)
        contrast_opinion = contrast_opinion.contiguous().view(-1, 1)
        contrast_sentiment = contrast_sentiment.contiguous().view(-1, 1)
        
        # Triplet-level positives (exact match)
        mask_triplet = torch.eq(contrast_labels, contrast_labels.T).float().to(device)
        
        # Aspect-level positives
        mask_aspect = torch.eq(contrast_aspect, contrast_aspect.T).float().to(device)
        
        # Opinion-level positives  
        mask_opinion = torch.eq(contrast_opinion, contrast_opinion.T).float().to(device)
        
        # Sentiment-level positives
        mask_sentiment = torch.eq(contrast_sentiment, contrast_sentiment.T).float().to(device)
        
        # Remove self-contrast
        num_contrast = batch_size * num_views
        eye_mask = torch.eye(num_contrast).to(device)
        
        mask_triplet = mask_triplet * (1 - eye_mask)
        mask_aspect = mask_aspect * (1 - eye_mask)
        mask_opinion = mask_opinion * (1 - eye_mask)
        mask_sentiment = mask_sentiment * (1 - eye_mask)
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * (1 - eye_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute losses for different levels
        losses = {}
        
        # Triplet-level loss (strongest constraint)
        if mask_triplet.sum() > 0:
            mean_log_prob_triplet = (mask_triplet * log_prob).sum(1) / (mask_triplet.sum(1) + 1e-8)
            losses['triplet'] = -(self.temperature / self.base_temperature) * mean_log_prob_triplet.mean()
        else:
            losses['triplet'] = torch.tensor(0.0, device=device)
        
        # Aspect-level loss
        if mask_aspect.sum() > 0:
            mean_log_prob_aspect = (mask_aspect * log_prob).sum(1) / (mask_aspect.sum(1) + 1e-8)
            losses['aspect'] = -(self.temperature / self.base_temperature) * mean_log_prob_aspect.mean()
        else:
            losses['aspect'] = torch.tensor(0.0, device=device)
        
        # Opinion-level loss
        if mask_opinion.sum() > 0:
            mean_log_prob_opinion = (mask_opinion * log_prob).sum(1) / (mask_opinion.sum(1) + 1e-8)
            losses['opinion'] = -(self.temperature / self.base_temperature) * mean_log_prob_opinion.mean()
        else:
            losses['opinion'] = torch.tensor(0.0, device=device)
        
        # Sentiment-level loss
        if mask_sentiment.sum() > 0:
            mean_log_prob_sentiment = (mask_sentiment * log_prob).sum(1) / (mask_sentiment.sum(1) + 1e-8)
            losses['sentiment'] = -(self.temperature / self.base_temperature) * mean_log_prob_sentiment.mean()
        else:
            losses['sentiment'] = torch.tensor(0.0, device=device)
        
        # Combined loss with adaptive weighting
        total_loss = (losses['triplet'] + 
                     0.7 * losses['aspect'] + 
                     0.7 * losses['opinion'] + 
                     0.5 * losses['sentiment'])
        
        losses['total'] = total_loss
        
        return losses


class NegativeSampler:
    """
    Advanced negative sampling strategies for contrastive learning
    """
    
    def __init__(self, strategy: str = 'hard', num_negatives: int = 5):
        self.strategy = strategy
        self.num_negatives = num_negatives
        
    def sample_negatives(self,
                        anchor_features: torch.Tensor,
                        anchor_labels: torch.Tensor,
                        candidate_features: torch.Tensor,
                        candidate_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negative examples for contrastive learning
        
        Args:
            anchor_features: [batch_size, hidden_dim]
            anchor_labels: [batch_size]
            candidate_features: [num_candidates, hidden_dim]
            candidate_labels: [num_candidates]
            
        Returns:
            negative_features, negative_labels
        """
        batch_size = anchor_features.size(0)
        device = anchor_features.device
        
        if self.strategy == 'random':
            return self._random_sampling(anchor_labels, candidate_features, candidate_labels)
        elif self.strategy == 'hard':
            return self._hard_negative_sampling(anchor_features, anchor_labels, 
                                              candidate_features, candidate_labels)
        elif self.strategy == 'focal':
            return self._focal_sampling(anchor_features, anchor_labels,
                                      candidate_features, candidate_labels)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
    
    def _random_sampling(self, anchor_labels, candidate_features, candidate_labels):
        """Random negative sampling"""
        negatives = []
        neg_labels = []
        
        for anchor_label in anchor_labels:
            # Find candidates with different labels
            neg_mask = candidate_labels != anchor_label
            neg_candidates = candidate_features[neg_mask]
            neg_cand_labels = candidate_labels[neg_mask]
            
            if len(neg_candidates) == 0:
                continue
                
            # Random sample
            num_samples = min(self.num_negatives, len(neg_candidates))
            indices = torch.randperm(len(neg_candidates))[:num_samples]
            
            negatives.append(neg_candidates[indices])
            neg_labels.append(neg_cand_labels[indices])
        
        if not negatives:
            return torch.empty(0, candidate_features.size(1)), torch.empty(0, dtype=torch.long)
            
        return torch.cat(negatives, dim=0), torch.cat(neg_labels, dim=0)
    
    def _hard_negative_sampling(self, anchor_features, anchor_labels, 
                               candidate_features, candidate_labels):
        """Hard negative sampling (closest negatives)"""
        negatives = []
        neg_labels = []
        
        for i, anchor_label in enumerate(anchor_labels):
            anchor_emb = anchor_features[i:i+1]  # [1, hidden_dim]
            
            # Find candidates with different labels
            neg_mask = candidate_labels != anchor_label
            neg_candidates = candidate_features[neg_mask]
            neg_cand_labels = candidate_labels[neg_mask]
            
            if len(neg_candidates) == 0:
                continue
            
            # Compute distances to anchor
            distances = torch.cdist(anchor_emb, neg_candidates, p=2).squeeze(0)
            
            # Select hardest (closest) negatives
            num_samples = min(self.num_negatives, len(neg_candidates))
            _, hard_indices = distances.topk(num_samples, largest=False)
            
            negatives.append(neg_candidates[hard_indices])
            neg_labels.append(neg_cand_labels[hard_indices])
        
        if not negatives:
            return torch.empty(0, candidate_features.size(1)), torch.empty(0, dtype=torch.long)
            
        return torch.cat(negatives, dim=0), torch.cat(neg_labels, dim=0)
    
    def _focal_sampling(self, anchor_features, anchor_labels,
                       candidate_features, candidate_labels):
        """Focal sampling (probability-based on difficulty)"""
        negatives = []
        neg_labels = []
        
        for i, anchor_label in enumerate(anchor_labels):
            anchor_emb = anchor_features[i:i+1]
            
            # Find candidates with different labels
            neg_mask = candidate_labels != anchor_label
            neg_candidates = candidate_features[neg_mask]
            neg_cand_labels = candidate_labels[neg_mask]
            
            if len(neg_candidates) == 0:
                continue
            
            # Compute similarities (higher = harder negative)
            similarities = F.cosine_similarity(anchor_emb, neg_candidates, dim=1)
            
            # Convert to probabilities (higher similarity = higher probability)
            probs = F.softmax(similarities * 2.0, dim=0)  # Temperature scaling
            
            # Sample based on probabilities
            num_samples = min(self.num_negatives, len(neg_candidates))
            indices = torch.multinomial(probs, num_samples, replacement=False)
            
            negatives.append(neg_candidates[indices])
            neg_labels.append(neg_cand_labels[indices])
        
        if not negatives:
            return torch.empty(0, candidate_features.size(1)), torch.empty(0, dtype=torch.long)
            
        return torch.cat(negatives, dim=0), torch.cat(neg_labels, dim=0)