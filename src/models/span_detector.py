# src/models/span_detector.py - Simplified Rule-Based Detector
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class SpanDetector(nn.Module):
    """Rule-enhanced span detector for restaurant reviews"""
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.hidden_dim = config.hidden_size
        self.dropout_rate = getattr(config, 'dropout', 0.1)
        
        # Simple recurrent layer (GRU is more memory efficient)
        self.rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0
        )
        
        # Aspect classifier with strong bias for aspect detection
        self.aspect_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3)  # B-I-O tags
        )
        
        # Opinion classifier with strong bias for opinion detection
        self.opinion_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3)  # B-I-O tags
        )
        
        # Initialize with strong bias for aspects and opinions
        self._initialize_with_bias()
        
        # Common food and restaurant terms for rule-based enhancement
        self.food_terms = {'food', 'meal', 'dish', 'pizza', 'pasta', 'sushi', 'burger', 
                          'salad', 'appetizer', 'dessert', 'lunch', 'dinner', 'breakfast', 
                          'menu', 'restaurant', 'service', 'staff', 'waiter', 'waitress',
                          'ambiance', 'atmosphere', 'price', 'value', 'taste', 'flavor'}
        
        self.opinion_terms = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'delicious',
                             'tasty', 'fantastic', 'bad', 'terrible', 'awful', 'horrible',
                             'poor', 'mediocre', 'disappointing', 'fresh', 'stale', 'expensive',
                             'cheap', 'overpriced', 'reasonable', 'friendly', 'rude', 'slow',
                             'fast', 'efficient', 'inefficient', 'clean', 'dirty'}
        
    def _initialize_with_bias(self):
        """Initialize with strong bias for aspects and opinions"""
        # Bias aspect classifier to detect more aspects
        # B tag (index 1) gets a positive bias
        self.aspect_classifier[-1].bias.data[1] = 0.5
        
        # Bias opinion classifier to detect more opinions
        # B tag (index 1) gets a positive bias 
        self.opinion_classifier[-1].bias.data[1] = 0.5
    
    def forward(self, hidden_states, attention_mask=None, texts=None):
        """
        Forward pass with rule-based enhancement
        
        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            texts: Optional raw text for rule-based enhancement
            
        Returns:
            aspect_logits, opinion_logits, span_features, boundary_logits
        """
        try:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            device = hidden_states.device
            
            # Apply RNN for sequential context
            rnn_out, _ = self.rnn(hidden_states)
            
            # Get base logits from neural network
            aspect_logits = self.aspect_classifier(rnn_out)
            opinion_logits = self.opinion_classifier(rnn_out)
            
            # If texts are provided, enhance with rule-based detection
            if texts is not None and isinstance(texts, list):
                # Apply rule-based enhancement
                for i, text in enumerate(texts):
                    if i >= batch_size:
                        break
                        
                    # Apply rule-based enhancement for this sample
                    aspect_logits[i], opinion_logits[i] = self._apply_rules(
                        text, aspect_logits[i], opinion_logits[i]
                    )
            
            # Apply attention mask
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                aspect_logits = aspect_logits * mask
                opinion_logits = opinion_logits * mask
                
                # Large negative for padding
                padding_mask = (1 - mask) * -10000.0
                aspect_logits = aspect_logits + padding_mask
                opinion_logits = opinion_logits + padding_mask
            
            # Create boundary logits (just placeholders)
            boundary_logits = torch.zeros(batch_size, seq_len, 2, device=device)
            
            # Use RNN output as span features
            span_features = rnn_out
            
            return aspect_logits, opinion_logits, span_features, boundary_logits
        
        except Exception as e:
            print(f"Error in span detector: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback tensors
            device = hidden_states.device
            batch_size, seq_len = hidden_states.shape[:2]
            
            aspect_logits = torch.zeros(batch_size, seq_len, 3, device=device)
            opinion_logits = torch.zeros(batch_size, seq_len, 3, device=device)
            span_features = torch.zeros_like(hidden_states)
            boundary_logits = torch.zeros(batch_size, seq_len, 2, device=device)
            
            return aspect_logits, opinion_logits, span_features, boundary_logits
    
    def _apply_rules(self, text, aspect_logits, opinion_logits):
        """Apply rule-based enhancement to logits"""
        # Split text into tokens
        tokens = text.lower().split()
        
        # Don't modify if length mismatch
        if len(tokens) > aspect_logits.size(0):
            return aspect_logits, opinion_logits
            
        # Apply food/restaurant term rules for aspects
        for i, token in enumerate(tokens):
            if i >= aspect_logits.size(0):
                break
                
            # Check if token is a food/restaurant term
            if token in self.food_terms:
                # Boost B tag for aspect (more than I and O tags)
                aspect_logits[i, 1] += 2.0  # Big boost for B tag
                
            # Check if part of multi-word food item
            if i < len(tokens) - 1:
                bigram = f"{token} {tokens[i+1]}"
                if any(term in bigram for term in self.food_terms):
                    aspect_logits[i, 1] += 1.5  # Boost for B tag (beginning)
                    if i+1 < aspect_logits.size(0):
                        aspect_logits[i+1, 2] += 1.5  # Boost for I tag (inside)
        
        # Apply opinion term rules
        for i, token in enumerate(tokens):
            if i >= opinion_logits.size(0):
                break
                
            # Check if token is an opinion term
            if token in self.opinion_terms:
                # Boost B tag for opinion
                opinion_logits[i, 1] += 2.0
                
            # Boost adjectives and adverbs (simple pattern matching)
            if token.endswith('ly') or token.endswith('ful') or token.endswith('ous'):
                opinion_logits[i, 1] += 1.0
        
        return aspect_logits, opinion_logits