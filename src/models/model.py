from src.models.embedding import LlamaEmbedding
from src.models.span_detector import SpanDetector 
from src.models.hierarchical_encoder import HierarchicalEncoder
from src.models.classifier import SentimentClassifier
import torch.nn as nn
class LlamaABSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = LlamaEmbedding(config)
        
        # Add hierarchical encoder
        self.hierarchical_encoder = HierarchicalEncoder(config)
        
        # Modify span detector to use hierarchical features
        self.span_detector = SpanDetector(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            num_layers=config.num_layers
        )
        
        self.sentiment_classifier = SentimentClassifier(
            input_dim=config.hidden_size,
            num_classes=3
        )

    def forward(self, input_ids, attention_mask, span_positions=None):
        # Get embeddings
        embeddings = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_positions=span_positions
        )
        
        # Apply hierarchical encoding
        fused_features, global_context, local_context = self.hierarchical_encoder(
            embeddings,
            attention_mask
        )
        
        # Detect spans using hierarchical features
        aspect_logits, opinion_logits = self.span_detector(fused_features)
        
        # Classify sentiment using both global and local context
        sentiment_logits = self.sentiment_classifier(
            fused_features,
            global_context=global_context,
            local_context=local_context
        )
        
        return {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits,
            'global_context': global_context,
            'local_context': local_context
        }