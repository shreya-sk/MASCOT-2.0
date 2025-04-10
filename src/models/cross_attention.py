# src/models/cross_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    """Enhanced cross-attention between aspect and opinion spans"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = getattr(config, 'num_attention_heads', 8)
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dynamic attention mask
        self.span_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

    def forward(self, queries, keys, attention_mask=None):
        """
        Forward pass for cross-attention
        
        Args:
            queries: Query tensor [batch_size, seq_len, hidden_size]
            keys: Key tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, q_len, _ = queries.size()
        _, k_len, _ = keys.size()
        
        # Project queries, keys and values
        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(keys)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Ensure proper shape for mask: [batch_size, 1, 1, seq_len]
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(extended_mask == 0, -10000.0)
        
        # Calculate attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
        output = self.out_proj(context)
        
        return output