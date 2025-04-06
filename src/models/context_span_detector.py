# src/models/context_span_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SyntaxGuidedAttention(nn.Module):
    """
    Novel attention mechanism that incorporates syntactic information
    to improve aspect and opinion detection
    """
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Syntax integration
        self.syntax_proj = nn.Linear(hidden_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, query, key, value, attention_mask=None, syntax_info=None):
        batch_size = query.size(0)
        
        # Project query, key, value
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Reshape for attention computation
        q = q.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Incorporate syntax information if available
        if syntax_info is not None:
            # Project syntax information to attention space
            syntax_attention = self.syntax_proj(syntax_info).unsqueeze(2)  # [batch, seq_len, 1, heads]
            syntax_attention = syntax_attention.permute(0, 3, 1, 2)  # [batch, heads, seq_len, 1]
            
            # Combine with semantic attention
            attention_scores = attention_scores + torch.matmul(syntax_attention, syntax_attention.transpose(-2, -1))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attention_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, heads, head_dim]
        context = context.view(batch_size, -1, self.hidden_dim)
        
        # Project to output
        output = self.out_proj(context)
        
        return output, attention_weights

class ContextSpanDetector(nn.Module):
    """
    Novel context-aware span detector with bidirectional influence
    between aspects and opinions
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1, use_syntax=True):
        super().__init__()
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Novel: Syntax-guided attention
        self.aspect_attention = SyntaxGuidedAttention(hidden_dim)
        self.opinion_attention = SyntaxGuidedAttention(hidden_dim)
        
        # Novel: Bidirectional interaction between aspects and opinions
        self.aspect_to_opinion = nn.Linear(hidden_dim, hidden_dim)
        self.opinion_to_aspect = nn.Linear(hidden_dim, hidden_dim)
        
        # Novel: Contextual span boundary refinement
        # This helps in precise detection of aspect and opinion boundaries
        self.boundary_refinement = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # Start and end logits
        )
        
        # Span classifiers with syntax integration
        self.aspect_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # B-I-O tags
        )
        
        self.opinion_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # B-I-O tags
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Whether to use syntax information
        self.use_syntax = use_syntax
    
    def forward(self, aspect_embeddings, opinion_embeddings, attention_mask=None, syntax_features=None):
        # Process both aspect and opinion embeddings
        aspect_lstm_out, _ = self.lstm(aspect_embeddings)
        opinion_lstm_out, _ = self.lstm(opinion_embeddings)
        
        aspect_lstm_out = self.dropout(aspect_lstm_out)
        opinion_lstm_out = self.dropout(opinion_lstm_out)
        
        # Novel: Cross-attention between aspects and opinions
        aspect_attn, _ = self.aspect_attention(
            aspect_lstm_out, 
            opinion_lstm_out, 
            opinion_lstm_out,
            attention_mask,
            syntax_features if self.use_syntax else None
        )
        
        opinion_attn, _ = self.opinion_attention(
            opinion_lstm_out,
            aspect_lstm_out,
            aspect_lstm_out,
            attention_mask,
            syntax_features if self.use_syntax else None
        )
        
        # Novel: Bidirectional influence - aspects affect opinions and vice versa
        aspect_to_opinion_influence = self.aspect_to_opinion(aspect_attn)
        opinion_to_aspect_influence = self.opinion_to_aspect(opinion_attn)
        
        # Enhanced representations with bidirectional influence
        enhanced_aspect = aspect_attn + opinion_to_aspect_influence
        enhanced_opinion = opinion_attn + aspect_to_opinion_influence
        
        # Concatenate original and enhanced representations
        aspect_features = torch.cat([aspect_lstm_out, enhanced_aspect], dim=-1)
        opinion_features = torch.cat([opinion_lstm_out, enhanced_opinion], dim=-1)
        
        # Span boundary refinement
        boundary_logits = self.boundary_refinement(torch.cat([aspect_features, opinion_features], dim=-1))
        
        # Span predictions
        aspect_logits = self.aspect_classifier(aspect_features)
        opinion_logits = self.opinion_classifier(opinion_features)
        
        # Generate span features for sentiment classification
        span_features = enhanced_aspect * enhanced_opinion
        
        return aspect_logits, opinion_logits, span_features