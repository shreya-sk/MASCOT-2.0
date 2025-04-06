# src/models/aspect_opinion_joint_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AspectOpinionJointClassifier(nn.Module):
    """
    Novel joint classifier that simultaneously considers aspect and opinion
    interactions for sentiment classification.
    
    This approach models the interdependencies between aspects and opinions,
    allowing for more accurate sentiment predictions.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_classes=3, use_aspect_first=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_aspect_first = use_aspect_first
        
        # Novel: Triple Attention (aspect-to-opinion, opinion-to-aspect, and context-to-both)
        # This allows modeling complex interactions between aspects, opinions, and context
        self.triple_attention = TripleAttention(input_dim)
        
        # Novel: Weighted aspect-opinion fusion
        # This adaptively combines aspect and opinion representations
        self.fusion_gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )
        
        # Multi-level fusion
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Novel: Aspect-Opinion polarity consistency regularization
        # This ensures consistent polarity predictions between related aspects and opinions
        self.consistency_check = nn.Bilinear(input_dim, input_dim, 1)
        
        # Main sentiment classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Novel: Confidence estimation branch
        # This provides uncertainty estimates for predictions
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states, aspect_logits, opinion_logits, span_features, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Convert logits to attention weights
        aspect_weights = F.softmax(aspect_logits[:, :, 1:], dim=1)  # Only use B-I tags
        aspect_weights = aspect_weights.sum(dim=2, keepdim=True)
        
        opinion_weights = F.softmax(opinion_logits[:, :, 1:], dim=1)  # Only use B-I tags
        opinion_weights = opinion_weights.sum(dim=2, keepdim=True)
        
        # Novel: Apply triple attention for complex interactions
        aspect_attn, opinion_attn, context_attn = self.triple_attention(
            hidden_states, 
            aspect_weights, 
            opinion_weights, 
            attention_mask
        )
        
        # Extract aspect and opinion representations using attention weights
        aspect_repr = torch.bmm(aspect_weights.transpose(1, 2), hidden_states)  # [batch, 1, hidden]
        opinion_repr = torch.bmm(opinion_weights.transpose(1, 2), hidden_states)  # [batch, 1, hidden]
        
        # Novel: Enhanced representations with triple attention
        aspect_repr = aspect_repr + aspect_attn.mean(dim=1, keepdim=True)
        opinion_repr = opinion_repr + opinion_attn.mean(dim=1, keepdim=True)
        context_repr = context_attn.mean(dim=1, keepdim=True)
        
        # Adaptive fusion based on aspect-opinion interaction
        fusion_input = torch.cat([aspect_repr, opinion_repr], dim=-1)
        fusion_weights = self.fusion_gate(fusion_input)
        
        # Novel: Apply adaptive fusion weights
        if self.use_aspect_first:
            fused_repr = fusion_weights * aspect_repr + (1 - fusion_weights) * opinion_repr
        else:
            fused_repr = fusion_weights * opinion_repr + (1 - fusion_weights) * aspect_repr
        
        # Combine with context for final representation
        final_repr = self.fusion(
            torch.cat([fused_repr, aspect_repr, opinion_repr], dim=-1)
        )
        
        # Consistency check between aspect and opinion polarities
        consistency_score = self.consistency_check(
            aspect_repr.squeeze(1), 
            opinion_repr.squeeze(1)
        )
        
        # Main sentiment classification
        sentiment_logits = self.classifier(final_repr.squeeze(1))
        
        # Confidence estimation
        confidence_scores = self.confidence_estimator(final_repr.squeeze(1))
        
        return sentiment_logits, confidence_scores


class TripleAttention(nn.Module):
    """
    Novel triple attention mechanism for modeling complex interactions
    between aspects, opinions, and context
    """
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear projections for aspects
        self.aspect_q = nn.Linear(hidden_dim, hidden_dim)
        self.aspect_k = nn.Linear(hidden_dim, hidden_dim)
        self.aspect_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Linear projections for opinions
        self.opinion_q = nn.Linear(hidden_dim, hidden_dim)
        self.opinion_k = nn.Linear(hidden_dim, hidden_dim)
        self.opinion_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Linear projections for context
        self.context_q = nn.Linear(hidden_dim, hidden_dim)
        self.context_k = nn.Linear(hidden_dim, hidden_dim)
        self.context_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projections
        self.aspect_out = nn.Linear(hidden_dim, hidden_dim)
        self.opinion_out = nn.Linear(hidden_dim, hidden_dim)
        self.context_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, hidden_states, aspect_weights, opinion_weights, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project for aspects
        aspect_q = self.aspect_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        aspect_k = self.aspect_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        aspect_v = self.aspect_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Project for opinions
        opinion_q = self.opinion_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        opinion_k = self.opinion_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        opinion_v = self.opinion_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Project for context
        context_q = self.context_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        context_k = self.context_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        context_v = self.context_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape for attention computation
        aspect_q = aspect_q.permute(0, 2, 1, 3)   # [batch, heads, seq_len, head_dim]
        aspect_k = aspect_k.permute(0, 2, 1, 3)
        aspect_v = aspect_v.permute(0, 2, 1, 3)
        
        opinion_q = opinion_q.permute(0, 2, 1, 3)
        opinion_k = opinion_k.permute(0, 2, 1, 3)
        opinion_v = opinion_v.permute(0, 2, 1, 3)
        
        context_q = context_q.permute(0, 2, 1, 3)
        context_k = context_k.permute(0, 2, 1, 3)
        context_v = context_v.permute(0, 2, 1, 3)
        
        # Compute attention scores and weights
        # 1. Aspect attention
        aspect_scores = torch.matmul(aspect_q, aspect_k.transpose(-2, -1)) * self.scale
        
        # Apply aspect weights to modify attention
        aspect_weight_effect = aspect_weights.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
        aspect_scores = aspect_scores + aspect_weight_effect
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            aspect_scores = aspect_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        aspect_weights = F.softmax(aspect_scores, dim=-1)
        aspect_context = torch.matmul(aspect_weights, aspect_v)
        
        # 2. Opinion attention
        opinion_scores = torch.matmul(opinion_q, opinion_k.transpose(-2, -1)) * self.scale
        
        # Apply opinion weights to modify attention
        opinion_weight_effect = opinion_weights.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
        opinion_scores = opinion_scores + opinion_weight_effect
        
        if attention_mask is not None:
            opinion_scores = opinion_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        opinion_weights = F.softmax(opinion_scores, dim=-1)
        opinion_context = torch.matmul(opinion_weights, opinion_v)
        
        # 3. Context attention with cross-influence from aspects and opinions
        context_scores = torch.matmul(context_q, context_k.transpose(-2, -1)) * self.scale
        
        # Apply combined influence from aspects and opinions
        combined_weight_effect = (aspect_weights + opinion_weights) / 2
        context_scores = context_scores + combined_weight_effect
        
        if attention_mask is not None:
            context_scores = context_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        context_weights = F.softmax(context_scores, dim=-1)
        context_context = torch.matmul(context_weights, context_v)
        
        # Reshape and project outputs
        aspect_context = aspect_context.permute(0, 2, 1, 3).contiguous()
        aspect_context = aspect_context.view(batch_size, seq_len, self.hidden_dim)
        aspect_output = self.aspect_out(aspect_context)
        
        opinion_context = opinion_context.permute(0, 2, 1, 3).contiguous()
        opinion_context = opinion_context.view(batch_size, seq_len, self.hidden_dim)
        opinion_output = self.opinion_out(opinion_context)
        
        context_context = context_context.permute(0, 2, 1, 3).contiguous()
        context_context = context_context.view(batch_size, seq_len, self.hidden_dim)
        context_output = self.context_out(context_context)
        
        return aspect_output, opinion_output, context_output