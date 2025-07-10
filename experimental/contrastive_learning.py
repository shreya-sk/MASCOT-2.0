# src/models/contrastive_learning.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLearning(nn.Module):
    """
    Advanced supervised contrastive learning for ABSA
    Based on ITSCL framework from EMNLP 2024
    """
    def __init__(self, config):
        super().__init__()
        self.temperature = getattr(config, 'contrastive_temperature', 0.07)
        self.hidden_dim = config.hidden_size
        
        # Four-layer contrastive framework
        self.sentiment_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 128)
        )
        
        self.aspect_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 128)
        )
        
        self.opinion_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 128)
        )
        
        # Combined representation projector
        self.combined_projector = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self, aspect_repr, opinion_repr, sentiment_repr, labels):
        """
        Compute supervised contrastive loss
        
        Args:
            aspect_repr: [batch_size, hidden_dim]
            opinion_repr: [batch_size, hidden_dim] 
            sentiment_repr: [batch_size, hidden_dim]
            labels: [batch_size] - combined labels for sentiment-aspect-opinion
        """
        # Project representations
        aspect_proj = F.normalize(self.aspect_projector(aspect_repr), dim=1)
        opinion_proj = F.normalize(self.opinion_projector(opinion_repr), dim=1)
        sentiment_proj = F.normalize(self.sentiment_projector(sentiment_repr), dim=1)
        
        # Combine representations
        combined = torch.cat([aspect_proj, opinion_proj, sentiment_proj], dim=1)
        combined_proj = F.normalize(self.combined_projector(combined), dim=1)
        
        # Compute contrastive loss
        batch_size = combined_proj.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(combined_proj, combined_proj.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity
        mask.fill_diagonal_(0)
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        
        # Sum of exponentials for each row (denominator)
        sum_exp = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # Positive pairs only
        pos_sim = exp_sim * mask
        
        # Compute loss for each positive pair
        losses = []
        for i in range(batch_size):
            if mask[i].sum() > 0:  # If there are positive pairs
                pos_sum = torch.sum(pos_sim[i])
                if pos_sum > 0:
                    loss = -torch.log(pos_sum / sum_exp[i])
                    losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=combined_proj.device, requires_grad=True)

class EnhancedTripletLoss(nn.Module):
    """Enhanced triplet loss for aspect-opinion-sentiment relationships"""
    
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor: [batch_size, embed_dim] - aspect representations
            positive: [batch_size, embed_dim] - matching opinion representations  
            negative: [batch_size, embed_dim] - non-matching opinion representations
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class MultiTaskContrastiveLoss(nn.Module):
    """
    Multi-task contrastive loss combining different ABSA subtasks
    """
    def __init__(self, config):
        super().__init__()
        self.scl = SupervisedContrastiveLearning(config)
        self.triplet_loss = EnhancedTripletLoss()
        
        # Task weights
        self.scl_weight = getattr(config, 'scl_weight', 1.0)
        self.triplet_weight = getattr(config, 'triplet_weight', 0.5)
        
    def forward(self, representations, labels, triplets=None):
        """
        Compute combined contrastive loss
        
        Args:
            representations: Dict with 'aspect', 'opinion', 'sentiment' keys
            labels: Combined labels for contrastive learning
            triplets: Optional triplet data for triplet loss
        """
        total_loss = 0.0
        
        # Supervised contrastive loss
        scl_loss = self.scl(
            representations['aspect'],
            representations['opinion'], 
            representations['sentiment'],
            labels
        )
        total_loss += self.scl_weight * scl_loss
        
        # Triplet loss if provided
        if triplets is not None:
            triplet_loss = self.triplet_loss(
                triplets['anchor'],
                triplets['positive'],
                triplets['negative']
            )
            total_loss += self.triplet_weight * triplet_loss
            
        return total_loss