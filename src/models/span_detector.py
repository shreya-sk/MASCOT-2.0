# src/models/span_detector.py
from torch.nn import nn
from src.models.cross_attention import MultiHeadCrossAttention

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
            dropout=config.dropout if config.num_layers > 1 else 0
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
        
    def forward(self, hidden_states, attention_mask=None):
        # BiLSTM encoding
        lstm_out, _ = self.lstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        
        # Cross attention between aspects and opinions
        aspect_hidden = self.cross_attention(lstm_out, lstm_out, attention_mask)
        opinion_hidden = self.cross_attention(lstm_out, lstm_out, attention_mask)
        
        # Span predictions
        aspect_logits = self.aspect_classifier(aspect_hidden)
        opinion_logits = self.opinion_classifier(opinion_hidden)
        
        return aspect_logits, opinion_logits