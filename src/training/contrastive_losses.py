# src/training/contrastive_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ITSCLLoss(nn.Module):
    """
    Instruction Tuning with Supervised Contrastive Learning (ITSCL)
    
    2024-2025 breakthrough: Combines InfoNCE and NT-Xent losses for unified
    extraction-generation alignment with contrastive verification.
    """
    
    def __init__(self, config):
        super().__init__()
        self.temperature = getattr(config, 'contrastive_temperature', 0.07)
        self.margin = getattr(config, 'contrastive_margin', 0.2)
        self.lambda_infonce = getattr(config, 'lambda_infonce', 1.0)
        self.lambda_ntxent = getattr(config, 'lambda_ntxent', 0.5)
        self.lambda_cross_modal = getattr(config, 'lambda_cross_modal', 0.3)
        
        # Projection heads for contrastive learning
        hidden_size = config.hidden_size
        self.aspect_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 128),
            nn.L2Norm(dim=1)
        )
        
        self.opinion_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 128),
            nn.L2Norm(dim=1)
        )
        
        self.sentiment_combination_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.L2Norm(dim=1)
        )
        
        # Cross-modal alignment projection
        self.cross_modal_projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.L2Norm(dim=1)
        )
        
    def forward(self, outputs, targets, generation_embeddings=None):
        """
        Compute ITSCL loss combining multiple contrastive objectives
        
        Args:
            outputs: Model outputs with embeddings
            targets: Target labels
            generation_embeddings: Optional embeddings from generation model
            
        Returns:
            Dictionary with contrastive loss components
        """
        device = outputs['aspect_logits'].device
        
        # Extract embeddings from span features
        span_features = outputs.get('span_features')
        if span_features is None:
            span_features = outputs.get('hidden_states')
        
        if span_features is None:
            return {'contrastive_loss': torch.tensor(0.0, device=device)}
        
        # Create aspect and opinion embeddings through attention pooling
        aspect_embeddings = self._extract_span_embeddings(
            span_features, outputs['aspect_logits'], targets['aspect_labels']
        )
        opinion_embeddings = self._extract_span_embeddings(
            span_features, outputs['opinion_logits'], targets['opinion_labels']
        )
        
        # Project embeddings to contrastive space
        aspect_proj = self.aspect_projection(aspect_embeddings)
        opinion_proj = self.opinion_projection(opinion_embeddings)
        
        # 1. InfoNCE Loss for Aspect-Opinion Alignment
        infonce_loss = self._compute_infonce_loss(aspect_proj, opinion_proj, targets['sentiment_labels'])
        
        # 2. NT-Xent Loss for Sentiment Consistency
        sentiment_combinations = torch.cat([aspect_embeddings, opinion_embeddings], dim=-1)
        sentiment_proj = self.sentiment_combination_projection(sentiment_combinations)
        ntxent_loss = self._compute_ntxent_loss(sentiment_proj, targets['sentiment_labels'])
        
        # 3. Cross-Modal Contrastive Learning (if generation embeddings available)
        cross_modal_loss = torch.tensor(0.0, device=device)
        if generation_embeddings is not None:
            extraction_repr = torch.cat([aspect_proj, opinion_proj], dim=-1)
            cross_modal_loss = self._compute_cross_modal_loss(extraction_repr, generation_embeddings)
        
        # Combine all contrastive losses
        total_contrastive_loss = (
            self.lambda_infonce * infonce_loss +
            self.lambda_ntxent * ntxent_loss +
            self.lambda_cross_modal * cross_modal_loss
        )
        
        return {
            'contrastive_loss': total_contrastive_loss,
            'infonce_loss': infonce_loss,
            'ntxent_loss': ntxent_loss,
            'cross_modal_loss': cross_modal_loss
        }
    
    def _extract_span_embeddings(self, span_features, logits, labels):
        """Extract span embeddings using attention pooling"""
        batch_size, seq_len, hidden_size = span_features.shape
        device = span_features.device
        
        # Get attention weights from logits (softmax over sequence)
        attention_weights = F.softmax(logits[:, :, 1:].sum(dim=-1), dim=-1)  # B+I tags
        
        # Attention pooling
        span_embeddings = torch.sum(span_features * attention_weights.unsqueeze(-1), dim=1)
        
        return span_embeddings
    
    def _compute_infonce_loss(self, aspect_embeddings, opinion_embeddings, sentiment_labels):
        """
        Compute InfoNCE loss for aspect-opinion alignment
        Multi-positive contrastive learning considering sentiment compatibility
        """
        batch_size = aspect_embeddings.size(0)
        device = aspect_embeddings.device
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(aspect_embeddings, opinion_embeddings.transpose(0, 1)) / self.temperature
        
        # Create positive mask based on sentiment compatibility
        sentiment_labels = sentiment_labels.squeeze() if len(sentiment_labels.shape) > 1 else sentiment_labels
        positive_mask = torch.eye(batch_size, device=device)
        
        # Enhanced positive pairs: same sentiment = additional positive signal
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and sentiment_labels[i] == sentiment_labels[j]:
                    positive_mask[i, j] = 0.5  # Weaker positive for same sentiment
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        pos_sim = exp_sim * positive_mask
        
        loss = -torch.log(pos_sim.sum(dim=1) / exp_sim.sum(dim=1) + 1e-8).mean()
        
        return loss
    
    def _compute_ntxent_loss(self, sentiment_embeddings, sentiment_labels):
        """
        Compute NT-Xent loss for sentiment consistency
        Supervised contrastive learning for sentiment clustering
        """
        batch_size = sentiment_embeddings.size(0)
        device = sentiment_embeddings.device
        
        # Normalize embeddings
        sentiment_embeddings = F.normalize(sentiment_embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(sentiment_embeddings, sentiment_embeddings.transpose(0, 1)) / self.temperature
        
        # Create mask for same sentiment labels
        sentiment_labels = sentiment_labels.squeeze() if len(sentiment_labels.shape) > 1 else sentiment_labels
        mask = torch.eq(sentiment_labels.unsqueeze(0), sentiment_labels.unsqueeze(1)).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=device)
        
        # Compute NT-Xent loss
        exp_sim = torch.exp(similarity_matrix)
        
        # Positive pairs (same sentiment)
        pos_sim = exp_sim * mask
        
        # Negative pairs (different sentiment + self)
        neg_mask = 1.0 - mask
        neg_sim = exp_sim * neg_mask
        
        # Compute loss for each positive pair
        loss = 0.0
        num_positives = 0
        
        for i in range(batch_size):
            pos_count = mask[i].sum()
            if pos_count > 0:
                pos_log_prob = torch.log(pos_sim[i].sum() / (pos_sim[i].sum() + neg_sim[i].sum()) + 1e-8)
                loss += -pos_log_prob / pos_count
                num_positives += 1
        
        return loss / max(num_positives, 1)
    
    def _compute_cross_modal_loss(self, extraction_embeddings, generation_embeddings):
        """
        Compute cross-modal contrastive loss between extraction and generation
        """
        batch_size = extraction_embeddings.size(0)
        
        # Project to same dimensionality
        extraction_proj = self.cross_modal_projection(extraction_embeddings)
        generation_proj = self.cross_modal_projection(generation_embeddings)
        
        # Compute similarity
        similarity = F.cosine_similarity(extraction_proj, generation_proj, dim=1)
        
        # Target similarity = 1 (perfect alignment)
        target = torch.ones_like(similarity)
        
        # MSE loss for alignment
        loss = F.mse_loss(similarity, target)
        
        return loss


class ContrastiveVerificationModule(nn.Module):
    """
    Contrastive verification module for sentiment consistency checking
    
    Verifies that extracted triplets are consistent with generated explanations
    using contrastive learning principles.
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.verification_threshold = 0.7
        
        # Triplet encoder
        self.triplet_encoder = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),  # aspect + opinion + sentiment
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 256)
        )
        
        # Explanation encoder
        self.explanation_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 256)
        )
        
        # Contrastive classifier
        self.consistency_classifier = nn.Sequential(
            nn.Linear(512, 256),  # Concatenated triplet + explanation
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Consistent / Inconsistent
        )
        
    def forward(self, triplet_embeddings, explanation_embeddings):
        """
        Verify consistency between triplets and explanations
        
        Args:
            triplet_embeddings: Embeddings from extracted triplets
            explanation_embeddings: Embeddings from generated explanations
            
        Returns:
            Consistency scores and verification loss
        """
        # Encode triplets and explanations
        triplet_repr = self.triplet_encoder(triplet_embeddings)
        explanation_repr = self.explanation_encoder(explanation_embeddings)
        
        # Concatenate representations
        combined_repr = torch.cat([triplet_repr, explanation_repr], dim=-1)
        
        # Classify consistency
        consistency_logits = self.consistency_classifier(combined_repr)
        consistency_probs = F.softmax(consistency_logits, dim=-1)
        
        # Contrastive verification loss
        verification_loss = self._compute_verification_loss(triplet_repr, explanation_repr)
        
        return {
            'consistency_scores': consistency_probs[:, 1],  # Probability of being consistent
            'consistency_logits': consistency_logits,
            'verification_loss': verification_loss
        }
    
    def _compute_verification_loss(self, triplet_repr, explanation_repr):
        """Compute contrastive verification loss"""
        # Positive pairs: corresponding triplet-explanation pairs
        pos_similarity = F.cosine_similarity(triplet_repr, explanation_repr, dim=1)
        
        # Negative pairs: mismatched triplet-explanation pairs
        batch_size = triplet_repr.size(0)
        if batch_size > 1:
            # Create negative pairs by shifting
            shifted_explanation = torch.roll(explanation_repr, shifts=1, dims=0)
            neg_similarity = F.cosine_similarity(triplet_repr, shifted_explanation, dim=1)
            
            # Contrastive loss: maximize positive similarity, minimize negative similarity
            contrastive_loss = torch.clamp(self.verification_threshold - pos_similarity + neg_similarity, min=0.0).mean()
        else:
            # Single sample - just maximize positive similarity
            contrastive_loss = torch.clamp(self.verification_threshold - pos_similarity, min=0.0).mean()
        
        return contrastive_loss


class MultiLevelContrastiveLoss(nn.Module):
    """
    4-layer contrastive framework combining sentiments, aspects, opinions, and their combinations
    """
    
    def __init__(self, config):
        super().__init__()
        self.temperature = getattr(config, 'contrastive_temperature', 0.07)
        
        # Layer 1: Aspect-level contrastive learning
        self.aspect_contrastive = SupConLoss(temperature=self.temperature)
        
        # Layer 2: Opinion-level contrastive learning  
        self.opinion_contrastive = SupConLoss(temperature=self.temperature)
        
        # Layer 3: Sentiment-level contrastive learning
        self.sentiment_contrastive = SupConLoss(temperature=self.temperature)
        
        # Layer 4: Combined triplet-level contrastive learning
        self.triplet_contrastive = SupConLoss(temperature=self.temperature)
        
    def forward(self, aspect_embeddings, opinion_embeddings, sentiment_embeddings, 
                aspect_labels, opinion_labels, sentiment_labels):
        """Compute multi-level contrastive loss"""
        
        # Layer 1: Aspect contrastive loss
        aspect_loss = self.aspect_contrastive(aspect_embeddings, aspect_labels)
        
        # Layer 2: Opinion contrastive loss
        opinion_loss = self.opinion_contrastive(opinion_embeddings, opinion_labels)
        
        # Layer 3: Sentiment contrastive loss
        sentiment_loss = self.sentiment_contrastive(sentiment_embeddings, sentiment_labels)
        
        # Layer 4: Combined triplet contrastive loss
        triplet_embeddings = torch.cat([aspect_embeddings, opinion_embeddings, sentiment_embeddings], dim=-1)
        combined_labels = self._combine_labels(aspect_labels, opinion_labels, sentiment_labels)
        triplet_loss = self.triplet_contrastive(triplet_embeddings, combined_labels)
        
        total_loss = aspect_loss + opinion_loss + sentiment_loss + triplet_loss
        
        return {
            'multi_level_contrastive_loss': total_loss,
            'aspect_contrastive_loss': aspect_loss,
            'opinion_contrastive_loss': opinion_loss,
            'sentiment_contrastive_loss': sentiment_loss,
            'triplet_contrastive_loss': triplet_loss
        }
    
    def _combine_labels(self, aspect_labels, opinion_labels, sentiment_labels):
        """Combine labels for triplet-level contrastive learning"""
        # Create combined labels by concatenating string representations
        batch_size = aspect_labels.size(0)
        combined_labels = []
        
        for i in range(batch_size):
            # Convert to string representation for hashing
            aspect_str = str(aspect_labels[i].tolist())
            opinion_str = str(opinion_labels[i].tolist())
            sentiment_str = str(sentiment_labels[i].item())
            
            combined_str = f"{aspect_str}_{opinion_str}_{sentiment_str}"
            combined_labels.append(hash(combined_str) % 10000)  # Modulo to keep reasonable range
        
        return torch.tensor(combined_labels, device=aspect_labels.device)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    From "Supervised Contrastive Learning" by Khosla et al.
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
            
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
            
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss