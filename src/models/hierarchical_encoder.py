import torch
import torch.nn as nn
from src.models.cross_attention import MultiHeadCrossAttention

class HierarchicalEncoder(nn.Module):
    """Two-stage encoder for global and local features"""
    def __init__(self, config):
        super().__init__()
        
        # Global context encoder
        self.global_encoder = nn.LSTM(
            config.hidden_size,
            config.hidden_size // 2,
            bidirectional=True,
            batch_first=True
        )
        
        # Local span encoder  
        self.span_encoder = nn.LSTM(
            config.hidden_size * 2,
            config.hidden_size // 2,
            bidirectional=True,
            batch_first=True
        )

        self.attention = MultiHeadCrossAttention(config)
        
    def forward(self, embeddings, attention_mask):
        # Get global context
        global_out, _ = self.global_encoder(embeddings)
        
        # Get local context with attention
        local_out, _ = self.span_encoder(
            torch.cat([embeddings, global_out], dim=-1)
        )
        
        # Cross attention between global and local
        final_repr = self.attention(
            local_out, global_out, attention_mask
        )
        
        return final_repr