#!/usr/bin/env python3
"""
GRADIENT Core Model
Simple model implementation for GRADIENT framework
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class GRADIENTModel(nn.Module):
    """
    Simple GRADIENT model for basic functionality
    Replace with unified_absa_model.py for full features
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load base model
        self.backbone = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.backbone.config.hidden_size
        
        # Simple classifier layers
        self.aspect_classifier = nn.Linear(hidden_size, config.num_classes)
        self.opinion_classifier = nn.Linear(hidden_size, config.num_classes) 
        self.sentiment_classifier = nn.Linear(hidden_size, config.num_classes)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Fixed forward pass - all token-level predictions"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use token-level features for ALL classifiers
        token_features = outputs.last_hidden_state              # [batch, seq_len, hidden]
        token_features = self.dropout(token_features)
        
        # ALL predictions are now token-level
        aspect_logits = self.aspect_classifier(token_features)   # [batch, seq_len, 3]
        opinion_logits = self.opinion_classifier(token_features) # [batch, seq_len, 3] 
        sentiment_logits = self.sentiment_classifier(token_features) # [batch, seq_len, 4]
        
        return {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits,
            'last_hidden_state': token_features
        }
# Backward compatibility
GRADIENTModel = GRADIENTModel
