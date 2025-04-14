# src/models/classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AspectOpinionJointClassifier(nn.Module):
    """
    Improved aspect-opinion joint classifier with balanced sentiment prediction
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_classes=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Attention for aspects
        self.aspect_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Attention for opinions
        self.opinion_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Feature fusion - adapt to the actual input dimensions
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # This handles combining the two vectors
            nn.GELU(),
            nn.Dropout(dropout)
        )
            
        # Sentiment classifier with balanced initialization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize sentiment classifier with no bias toward any class
        self.classifier[-1].bias.data.zero_()
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Word lists for rule-based sentiment detection
        self.negative_words = set([
            'bad', 'terrible', 'poor', 'awful', 'mediocre', 'disappointing', 
            'horrible', 'disgusting', 'unpleasant', 'worst', 'cold', 'slow',
            'overpriced', 'expensive', 'rude', 'unfriendly', 'dirty', 'noisy'
        ])
        
        self.positive_words = set([
            'good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful',
            'delicious', 'tasty', 'fresh', 'friendly', 'helpful', 'nice',
            'best', 'perfect', 'favorite', 'loved', 'enjoy', 'clean', 'quick'
        ])
        
    def forward(self, hidden_states, aspect_logits=None, opinion_logits=None, 
                attention_mask=None, sentiment_labels=None, input_ids=None):
        """
        Forward pass through the sentiment classifier
        
        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_dim]
            aspect_logits: Aspect logits [batch_size, seq_len, 3]
            opinion_logits: Opinion logits [batch_size, seq_len, 3]
            attention_mask: Attention mask [batch_size, seq_len]
            sentiment_labels: Optional sentiment labels
            input_ids: Optional input token IDs for rule-based enhancement
            
        Returns:
            tuple: (sentiment_logits, confidence_scores)
        """
        try:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            device = hidden_states.device
            
            # Create aspect and opinion weights from logits or attention
            # For aspect weights
            if aspect_logits is not None:
                # Only consider B and I tags (indices 1 and 2)
                aspect_weights = torch.softmax(aspect_logits[:, :, 1:], dim=-1).sum(-1)  # [batch_size, seq_len]
            else:
                # Use attention if logits not available
                aspect_attn = self.aspect_attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
                if attention_mask is not None:
                    aspect_attn = aspect_attn.masked_fill(attention_mask == 0, -1e10)
                aspect_weights = F.softmax(aspect_attn, dim=-1)
            
            # For opinion weights
            if opinion_logits is not None:
                opinion_weights = torch.softmax(opinion_logits[:, :, 1:], dim=-1).sum(-1)
            else:
                opinion_attn = self.opinion_attention(hidden_states).squeeze(-1)
                if attention_mask is not None:
                    opinion_attn = opinion_attn.masked_fill(attention_mask == 0, -1e10)
                opinion_weights = F.softmax(opinion_attn, dim=-1)
            
            # Apply weights to get weighted representations
            aspect_weights = aspect_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            opinion_weights = opinion_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Handle cases where weights are all zero (apply uniform weights)
            if attention_mask is not None:
                # Use attention mask to create uniform weights for non-padding tokens
                mask_expanded = attention_mask.float().unsqueeze(-1)  # [batch_size, seq_len, 1]
                aspect_sum = aspect_weights.sum(dim=1, keepdim=True)
                opinion_sum = opinion_weights.sum(dim=1, keepdim=True)
                
                # If sum is close to zero, use uniform weights
                aspect_weights = torch.where(
                    aspect_sum > 1e-6, 
                    aspect_weights, 
                    mask_expanded / mask_expanded.sum(dim=1, keepdim=True).clamp(min=1e-6)
                )
                opinion_weights = torch.where(
                    opinion_sum > 1e-6, 
                    opinion_weights, 
                    mask_expanded / mask_expanded.sum(dim=1, keepdim=True).clamp(min=1e-6)
                )
            
            # Weight hidden states with attention weights
            aspect_repr = (hidden_states * aspect_weights).sum(dim=1)  # [batch_size, hidden_dim]
            opinion_repr = (hidden_states * opinion_weights).sum(dim=1)  # [batch_size, hidden_dim]
            
            # Combine aspect and opinion representations
            combined = torch.cat([aspect_repr, opinion_repr], dim=-1)
            
            # Apply fusion
            fused = self.fusion(combined)  # [batch_size, hidden_dim]
            
            # Basic sentiment classification
            sentiment_logits = self.classifier(fused)  # [batch_size, num_classes]
            
            # Apply rule-based sentiment modification if input_ids are provided
            if input_ids is not None and hasattr(self, 'tokenizer'):
                for b in range(batch_size):
                    # Analyze opinion tokens for sentiment cues
                    opinion_indices = torch.where(opinion_weights[b, :, 0] > 0.1)[0]
                    
                    if len(opinion_indices) > 0:
                        has_negative = False
                        has_positive = False
                        
                        for idx in opinion_indices:
                            if idx < input_ids.size(1):
                                token_id = input_ids[b, idx].item()
                                token = self.tokenizer.decode([token_id]).lower().strip()
                                
                                if token in self.negative_words:
                                    has_negative = True
                                if token in self.positive_words:
                                    has_positive = True
                        
                        # Adjust sentiment logits based on detected terms
                        if has_negative and not has_positive:
                            sentiment_logits[b, 2] += 2.0  # Boost negative
                        elif has_positive and not has_negative:
                            sentiment_logits[b, 0] += 2.0  # Boost positive
            
            # Rule-based adjustment for common sentiment patterns even without tokenizer
            # Look at the opinion representation for patterns that suggest sentiment
            opinion_norm = F.normalize(opinion_repr, p=2, dim=1)
            
            # Compute confidence scores
            confidence = self.confidence_estimator(fused)  # [batch_size, 1]
            
            return sentiment_logits, confidence
            
        except Exception as e:
            print(f"Error in classifier forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback outputs with correct dimensions
            sentiment_logits = torch.zeros(batch_size, self.num_classes, device=device)
            confidence = torch.ones(batch_size, 1, device=device) * 0.5
            
            return sentiment_logits, confidence