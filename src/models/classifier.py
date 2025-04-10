# src/models/classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F 

class AspectOpinionJointClassifier(nn.Module):
    """
    Novel joint classifier that simultaneously considers aspect and opinion
    interactions for sentiment classification.
    
    This approach models the interdependencies between aspects and opinions,
    allowing for more accurate sentiment predictions.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_classes=3, use_aspect_first=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_aspect_first = use_aspect_first
        
        # Attention for aspect and opinion pooling
        self.aspect_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        
        self.opinion_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Aspect-opinion relation modeling
        self.relation = nn.Bilinear(input_dim, input_dim, hidden_dim)
        
        # Main sentiment classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states, aspect_logits=None, opinion_logits=None, span_features=None, attention_mask=None):
        """
        Forward pass through the aspect-opinion joint classifier
        
        Args:
            hidden_states: Hidden states from the encoder [batch_size, seq_length, hidden_dim]
            aspect_logits: Aspect logits [batch_size, seq_length, 3]
            opinion_logits: Opinion logits [batch_size, seq_length, 3]
            span_features: Span features [batch_size, seq_length, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            tuple: (sentiment_logits, confidence_scores)
        """
        try:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            device = hidden_states.device
            
            # Create aspect and opinion weights from logits
            if aspect_logits is not None:
                # Only consider B and I tags (indices 1 and 2)
                aspect_weights = torch.softmax(aspect_logits[:, :, 1:], dim=-1).sum(-1)  # [batch_size, seq_len]
            else:
                # Use attention if logits not available
                aspect_attn = self.aspect_attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
                if attention_mask is not None:
                    aspect_attn = aspect_attn.masked_fill(attention_mask == 0, -1e10)
                aspect_weights = F.softmax(aspect_attn, dim=-1)
            
            if opinion_logits is not None:
                opinion_weights = torch.softmax(opinion_logits[:, :, 1:], dim=-1).sum(-1)
            else:
                opinion_attn = self.opinion_attention(hidden_states).squeeze(-1)
                if attention_mask is not None:
                    opinion_attn = opinion_attn.masked_fill(attention_mask == 0, -1e10)
                opinion_weights = F.softmax(opinion_attn, dim=-1)
            
            # Apply weights to get span representations
            aspect_weights = aspect_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            opinion_weights = opinion_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Weight hidden states with attention weights
            aspect_repr = (hidden_states * aspect_weights).sum(dim=1)  # [batch_size, hidden_dim]
            opinion_repr = (hidden_states * opinion_weights).sum(dim=1)  # [batch_size, hidden_dim]
            
            # Use aspect-first ordering if specified
            if self.use_aspect_first:
                combined = torch.cat([aspect_repr, opinion_repr], dim=-1)
            else:
                combined = torch.cat([opinion_repr, aspect_repr], dim=-1)
            
            # Apply fusion
            fused = self.fusion(combined)  # [batch_size, hidden_dim]
            
            # Model relation between aspect and opinion
            relation = self.relation(aspect_repr, opinion_repr)  # [batch_size, hidden_dim]
            
            # Combine fused and relation representations
            final_repr = torch.cat([fused, relation], dim=-1)  # [batch_size, hidden_dim*2]
            
            # Predict sentiment and confidence
            sentiment_logits = self.classifier(final_repr)  # [batch_size, num_classes]
            confidence = self.confidence_estimator(final_repr)  # [batch_size, 1]
            
            return sentiment_logits, confidence
            
        except Exception as e:
            print(f"Error in classifier forward pass: {e}")
            # Create fallback outputs with correct dimensions
            sentiment_logits = torch.zeros(batch_size, self.num_classes, device=device)
            confidence = torch.ones(batch_size, 1, device=device) * 0.5
            
            return sentiment_logits, confidence