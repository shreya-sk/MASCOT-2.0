
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss

class ITSCLLoss(nn.Module):
    """Inter-Task Supervised Contrastive Learning Loss"""
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

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
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class ContrastiveVerificationModule(nn.Module):
    """Contrastive Verification Module for ABSA"""
    
    def __init__(self, hidden_size=768, projection_dim=128):
        super().__init__()
        self.projection = nn.Linear(hidden_size, projection_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
    def forward(self, features, labels=None):
        """Forward pass for contrastive verification"""
        projected = self.projection(features)
        projected = F.normalize(projected, dim=-1)
        
        if labels is not None:
            loss_fn = SupervisedContrastiveLoss(temperature=self.temperature)
            loss = loss_fn(projected, labels)
            return {'loss': loss, 'features': projected}
        
        return {'features': projected}
    
# Add this to the end of your src/training/contrastive_losses.py file:

class MultiLevelContrastiveLoss(nn.Module):
    """Multi-level contrastive loss for different granularities"""
    
    def __init__(self, temperature=0.1, levels=['word', 'phrase', 'sentence']):
        super().__init__()
        self.temperature = temperature
        self.levels = levels
        self.level_weights = nn.Parameter(torch.ones(len(levels)))
        
    def forward(self, multi_level_features, labels):
        """
        Args:
            multi_level_features: Dict with keys like 'word', 'phrase', 'sentence'
            labels: Corresponding labels for each level
        """
        total_loss = 0.0
        valid_levels = 0
        
        for i, level in enumerate(self.levels):
            if level in multi_level_features:
                features = multi_level_features[level]
                level_labels = labels.get(level, labels) if isinstance(labels, dict) else labels
                
                if features.size(0) > 1:  # Need at least 2 samples for contrastive loss
                    contrastive_loss = SupervisedContrastiveLoss(self.temperature)
                    loss = contrastive_loss(features, level_labels)
                    total_loss += self.level_weights[i] * loss
                    valid_levels += 1
        
        if valid_levels > 0:
            return total_loss / valid_levels
        else:
            return torch.tensor(0.0, device=next(iter(multi_level_features.values())).device)