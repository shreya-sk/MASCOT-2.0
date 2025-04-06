# src/models/stella_absa.py
import torch
import torch.nn as nn
from src.models.stella_embedding import StellaEmbedding
from src.models.context_span_detector import ContextSpanDetector
from src.models.aspect_opinion_joint_classifier import AspectOpinionJointClassifier

class StellaABSA(nn.Module):
    """
    Novel ABSA model using Stella v5 embeddings with multi-focal attention
    
    Key innovations:
    1. Aspect-Opinion Joint Learning with bidirectional influence
    2. Context-aware span detection with focal attention
    3. Multi-domain knowledge transfer adapter
    4. Hierarchical fusion of syntactic and semantic features
    """
    def __init__(self, config):
        super().__init__()
        
        # Stella embeddings with dual projection
        self.embeddings = StellaEmbedding(config)
        
        # Novel: Context-aware Span Detector with syntax-guided attention
        self.span_detector = ContextSpanDetector(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_syntax=config.use_syntax
        )
        
        # Novel: Joint Aspect-Opinion Sentiment Classifier
        # This models the interdependence between aspects and opinions
        self.sentiment_classifier = AspectOpinionJointClassifier(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            dropout=config.dropout,
            num_classes=3,  # POS, NEU, NEG
            use_aspect_first=config.use_aspect_first
        )
        
        # Novel: Dynamic Focal Weighting module
        # This allows the model to focus on the most relevant parts of the input
        self.focal_weighting = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Novel: Confidence estimation for predictions
        # This provides uncertainty estimates for model predictions
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask, domain_id=None, syntax_features=None):
        """Forward pass through the model"""
        # Get embeddings from Stella
        embeddings_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_id=domain_id
        )
        
        aspect_embeddings = embeddings_output['aspect_embeddings']
        opinion_embeddings = embeddings_output['opinion_embeddings']
        hidden_states = embeddings_output['hidden_states']
        
        # Apply focal weighting 
        focal_weights = self.focal_weighting(hidden_states)
        weighted_embeddings = hidden_states * focal_weights
        
        # Novel: Context-aware span detection with bidirectional influence
        # Aspects influence opinions and vice versa
        aspect_logits, opinion_logits, span_features = self.span_detector(
            aspect_embeddings=aspect_embeddings,
            opinion_embeddings=opinion_embeddings,
            attention_mask=attention_mask,
            syntax_features=syntax_features
        )
        
        # Novel: Joint sentiment classification
        sentiment_logits, confidence_scores = self.sentiment_classifier(
            hidden_states=weighted_embeddings,
            aspect_logits=aspect_logits,
            opinion_logits=opinion_logits,
            span_features=span_features,
            attention_mask=attention_mask
        )
        
        # Compute confidence estimates for predictions
        combined_features = torch.cat([
            span_features, 
            weighted_embeddings[:, 0, :]  # Use CLS token representation
        ], dim=-1)
        confidence = self.confidence_estimator(combined_features)
        
        return {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits,
            'confidence_scores': confidence,
            'focal_weights': focal_weights,
            'span_features': span_features
        }