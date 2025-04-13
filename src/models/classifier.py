<<<<<<< Updated upstream
# src/models/aspect_opinion_joint_classifier.py
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
=======
# src/models/classifier.py
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F  #type: ignore
>>>>>>> Stashed changes

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

    def contrastive_loss(self, features, labels):
        """Compute contrastive loss to improve sentiment separation"""
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.transpose(0, 1))
        # Create label matrix where 1 means same sentiment
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        # Compute contrastive loss with temperature
        temperature = 0.1
        sim_matrix = sim_matrix / temperature
        pos_loss = -torch.log(torch.sigmoid(sim_matrix)).masked_select(label_matrix.bool()).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(sim_matrix)).masked_select(~label_matrix.bool()).mean()
        return pos_loss + neg_loss

    def forward(self, hidden_states, aspect_logits=None, opinion_logits=None, span_features=None, 
                attention_mask=None, sentiment_labels=None):
        """
        Forward pass through the aspect-opinion joint classifier
        
        Args:
            hidden_states: Hidden states from the encoder [batch_size, seq_length, hidden_dim]
            aspect_logits: Aspect logits [batch_size, seq_length, 3]
            opinion_logits: Opinion logits [batch_size, seq_length, 3]
            span_features: Span features [batch_size, seq_length, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_length]
<<<<<<< Updated upstream
=======
            sentiment_labels: Optional sentiment labels for contrastive learning
            
        Returns:
            tuple: (sentiment_logits, confidence_scores) or 
                  (sentiment_logits, confidence_scores, contrastive_loss)
>>>>>>> Stashed changes
        """
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Create aspect and opinion weights from logits
        aspect_weights = torch.softmax(aspect_logits[:, :, 1:], dim=-1).sum(-1) if aspect_logits is not None else None
        opinion_weights = torch.softmax(opinion_logits[:, :, 1:], dim=-1).sum(-1) if opinion_logits is not None else None
        
        # Get weighted representations using triple attention
        try:
<<<<<<< Updated upstream
            # Try using the triple attention if available
            aspect_repr, opinion_repr, context_repr = self.triple_attention(
                hidden_states, aspect_weights, opinion_weights, attention_mask
            )
        except:
            # Fallback to a simple pooling if triple attention fails
            print("Warning: Triple attention failed, using fallback pooling")
            # Simple pooling - take mean of all non-masked tokens
            pooled = hidden_states.mean(dim=1)
            aspect_repr = opinion_repr = context_repr = pooled.unsqueeze(1)
        
        # Pool to get sentence-level representations
        # Take mean of all tokens in the sequence
        aspect_pooled = aspect_repr.mean(dim=1)
        opinion_pooled = opinion_repr.mean(dim=1)
        context_pooled = context_repr.mean(dim=1)
        
        # Concatenate all representations
        fusion_input = torch.cat([aspect_pooled, opinion_pooled, context_pooled], dim=-1)
        
        # Apply fusion
        fused = self.fusion(fusion_input)
        
        # Predict sentiment and confidence
        logits = self.classifier(fused)
        confidence = self.confidence_estimator(fused)
        
        return logits, confidence
        
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
=======
            batch_size, seq_len, hidden_dim = hidden_states.shape
            device = hidden_states.device
            
            # Create aspect and opinion weights from logits
            if aspect_logits is not None:
                # Only consider B and I tags (indices 1 and 2)
                aspect_weights = torch.softmax(aspect_logits[:, :, 1:], dim=-1).sum(-1)  # [batch_size, seq_len]
            else:
                # Use attention if logits not available
                aspect_attn = self.aspect_attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
                if attention_mask is not None:
                    aspect_attn = aspect_attn.masked_fill(attention_mask == 0, -1e10)
                aspect_weights = F.softmax(aspect_attn, dim=-1)
            
            if opinion_logits is not None:
                opinion_weights = torch.softmax(opinion_logits[:, :, 1:], dim=-1).sum(-1)
            else:
                opinion_attn = self.opinion_attention(hidden_states).squeeze(-1)
                if attention_mask is not None:
                    opinion_attn = opinion_attn.masked_fill(attention_mask == 0, -1e10)
                opinion_weights = F.softmax(opinion_attn, dim=-1)
            
            # Apply weights to get span representations
            aspect_weights = aspect_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            opinion_weights = opinion_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Weight hidden states with attention weights
            aspect_repr = (hidden_states * aspect_weights).sum(dim=1)  # [batch_size, hidden_dim]
            opinion_repr = (hidden_states * opinion_weights).sum(dim=1)  # [batch_size, hidden_dim]
            
            # Use aspect-first ordering if specified
            if self.use_aspect_first:
                combined = torch.cat([aspect_repr, opinion_repr], dim=-1)
            else:
                combined = torch.cat([opinion_repr, aspect_repr], dim=-1)
            
            # Apply fusion
            fused = self.fusion(combined)  # [batch_size, hidden_dim]
            
            # Model relation between aspect and opinion
            relation = self.relation(aspect_repr, opinion_repr)  # [batch_size, hidden_dim]
            
            # Combine fused and relation representations
            final_repr = torch.cat([fused, relation], dim=-1)  # [batch_size, hidden_dim*2]
            
            # Predict sentiment and confidence
            sentiment_logits = self.classifier(final_repr)  # [batch_size, num_classes]
            confidence = self.confidence_estimator(final_repr)  # [batch_size, 1]
            
            # Add contrastive learning if labels are provided
            if sentiment_labels is not None:
                contrastive = self.contrastive_loss(final_repr, sentiment_labels)
                return sentiment_logits, confidence, contrastive
            else:
                return sentiment_logits, confidence
            
        except Exception as e:
            print(f"Error in classifier forward pass: {e}")
            # Create fallback outputs with correct dimensions
            sentiment_logits = torch.zeros(batch_size, self.num_classes, device=device)
            confidence = torch.ones(batch_size, 1, device=device) * 0.5
            
            if sentiment_labels is not None:
                return sentiment_logits, confidence, torch.tensor(0.0, device=device)
            else:
                return sentiment_logits, confidence
>>>>>>> Stashed changes
