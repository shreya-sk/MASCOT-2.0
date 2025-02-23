# src/models/classifier.py
import torch.nn as nn
class SentimentClassifier(nn.Module):
    """Classifies sentiment for detected spans"""
    def __init__(self, config):
        super().__init__()
        
        # Span representation fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Sentiment classification
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 3)  # POS, NEU, NEG
        )
        
    def forward(self, hidden_states, aspect_mask, opinion_mask):
        # Get span representations
        aspect_repr = self._get_span_repr(hidden_states, aspect_mask)
        opinion_repr = self._get_span_repr(hidden_states, opinion_mask)
        
        # Fuse aspect and opinion representations
        fused_repr = self.fusion(
            torch.cat([aspect_repr, opinion_repr], dim=-1)
        )
        
        # Classify sentiment
        sentiment_logits = self.classifier(fused_repr)
        
        return sentiment_logits
    
    def _get_span_repr(self, hidden_states, span_mask):
        """Extract span representation using attention"""
        span_scores = torch.softmax(span_mask, dim=1)
        span_repr = torch.bmm(
            span_scores.unsqueeze(1),
            hidden_states
        ).squeeze(1)
        return span_repr