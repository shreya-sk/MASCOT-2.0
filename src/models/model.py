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
        """Simple forward pass"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pool the sequence
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        
        # Classifications
        aspect_logits = self.aspect_classifier(pooled_output)
        opinion_logits = self.opinion_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        return {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits,
            'last_hidden_state': outputs.last_hidden_state
        }

# Backward compatibility
GRADIENTModel = GRADIENTModel
