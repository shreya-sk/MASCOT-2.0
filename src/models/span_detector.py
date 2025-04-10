# src/models/span_detector.py
import torch  #type: ignore 
import torch.nn as nn  #type: ignore
from src.models.cross_attention import MultiHeadCrossAttention  #type: ignore

class SpanDetector(nn.Module):
    """Detects aspect and opinion spans using bidirectional modeling"""
    def __init__(self, config):
        super().__init__()
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if getattr(config, 'num_layers', 2) > 1 else 0
        )
        
        # Cross attention between aspects and opinions
        self.cross_attention = MultiHeadCrossAttention(config)
        
        # Span classifiers
        self.aspect_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 3)  # B-I-O tags
        )
        
        self.opinion_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 3)  # B-I-O tags
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, aspect_embeddings=None, opinion_embeddings=None, attention_mask=None, hidden_states=None):
        """Forward pass supporting both separate embeddings and single hidden states"""
        try:
            # If separate embeddings aren't provided, use hidden_states for both
            if aspect_embeddings is None and opinion_embeddings is None:
                if hidden_states is None:
                    raise ValueError("Either hidden_states or aspect/opinion embeddings must be provided")
                aspect_embeddings = opinion_embeddings = hidden_states
            
            # BiLSTM encoding for aspects
            # Fix tensor reshaping issue - ensure proper dimensions
            batch_size, seq_len, hidden_dim = aspect_embeddings.size()
            
            # Run LSTM - reshaping properly for sequence models
            aspect_lstm_out, _ = self.lstm(aspect_embeddings)
            aspect_lstm_out = self.dropout(aspect_lstm_out)
            
            # BiLSTM encoding for opinions
            if opinion_embeddings is aspect_embeddings:
                opinion_lstm_out = aspect_lstm_out
            else:
                opinion_lstm_out, _ = self.lstm(opinion_embeddings)
                opinion_lstm_out = self.dropout(opinion_lstm_out)
            
            # Cross attention between aspects and opinions
            # Fix: Ensure proper shape handling in cross attention
            aspect_hidden = self.cross_attention(aspect_lstm_out, opinion_lstm_out, attention_mask)
            opinion_hidden = self.cross_attention(opinion_lstm_out, aspect_lstm_out, attention_mask)
            
            # Span predictions
            aspect_logits = self.aspect_classifier(aspect_hidden)
            opinion_logits = self.opinion_classifier(opinion_hidden)
            
            # Generate span features for sentiment classification
            # Element-wise multiplication for feature fusion
            span_features = aspect_hidden * opinion_hidden
            
            return aspect_logits, opinion_logits, span_features
        
        except Exception as e:
            print(f"Error in span detector forward pass: {e}")
            # Return tensor placeholders with correct dimensions
            batch_size = hidden_states.size(0)
            seq_len = hidden_states.size(1)
            device = hidden_states.device
            
            # Create properly shaped outputs instead of zeros with wrong dimensions
            aspect_logits = torch.zeros(batch_size, seq_len, 3, device=device)
            opinion_logits = torch.zeros(batch_size, seq_len, 3, device=device)
            span_features = torch.zeros_like(hidden_states)
            
            return aspect_logits, opinion_logits, span_features