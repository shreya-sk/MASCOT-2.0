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
        q = self.q_proj(aspect_hidden).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(opinion_hidden).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(opinion_hidden).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Reshape for attention computation
        q = q.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Dynamic span-aware attention weights
        span_scores = self.span_bilinear(aspect_hidden, opinion_hidden).unsqueeze(1)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add span-aware scores (reshaped appropriately)
        span_scores_expanded = span_scores.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attention_scores = attention_scores + span_scores_expanded
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, 
                float('-inf')
            )
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attention_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.out_proj(context)