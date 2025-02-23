import torch
import torch.nn as nn

class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder for global sentiment and fine-grained aspect extraction"""
    def __init__(self, config):
        super().__init__()
        
        # Global sentiment encoder
        self.global_encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Local aspect-opinion encoder
        self.local_encoder = nn.LSTM(
            input_size=config.hidden_size * 2,  # Concatenated with global context
            hidden_size=config.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention layers
        self.global_attention = nn.MultiheadAttention(
            config.hidden_size * 2,  # Bidirectional
            num_heads=config.num_attention_heads
        )
        
        self.local_attention = nn.MultiheadAttention(
            config.hidden_size * 2,
            num_heads=config.num_attention_heads
        )
        
        self.fusion_layer = nn.Linear(config.hidden_size * 4, config.hidden_size)
        
    def forward(self, hidden_states, attention_mask=None):
        # Stage 1: Global sentiment encoding
        global_out, _ = self.global_encoder(hidden_states)
        global_context, _ = self.global_attention(
            global_out, global_out, global_out,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        # Stage 2: Local aspect-opinion encoding
        # Concatenate global context with input
        local_input = torch.cat([hidden_states, global_context], dim=-1)
        local_out, _ = self.local_encoder(local_input)
        local_context, _ = self.local_attention(
            local_out, local_out, local_out,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        # Fuse global and local features
        fused_features = self.fusion_layer(
            torch.cat([global_context, local_context], dim=-1)
        )
        
        return fused_features, global_context, local_context