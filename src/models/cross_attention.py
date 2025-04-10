# src/models/cross_attention.py
import torch # type: ignore # type: ignore
import torch.nn as nn # type: ignore

class MultiHeadCrossAttention(nn.Module):
    """Enhanced cross-attention between aspect and opinion spans"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = getattr(config, 'num_attention_heads', 8)
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dynamic attention mask
        self.span_bilinear = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

    def forward(self, aspect_hidden, opinion_hidden, attention_mask=None):
        batch_size = aspect_hidden.size(0)
        
        # Project queries, keys and values
        q = self.q_proj(aspect_hidden)
        k = self.k_proj(opinion_hidden)
        v = self.v_proj(opinion_hidden)
        
        # Make sure everything has the same sequence length
        seq_len = min(q.size(1), k.size(1), v.size(1))
        q = q[:, :seq_len, :]
        k = k[:, :seq_len, :]
        v = v[:, :seq_len, :]
        
        # Reshape for multi-head attention
        head_dim = self.head_dim
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len].unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e10)
        
        # Calculate attention weights
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.out_proj(context)
        
        return output