# src/models/model.py
from src.models.embedding import LlamaEmbedding
import torch.nn as nn
class LlamaABSA(nn.Module):
    """ABSA model using Llama embeddings"""
    def __init__(self, config):
        super().__init__()
        self.embeddings = LlamaEmbedding(config)
        
        # Aspect-Opinion span detection
        self.span_detector = MultiAspectSpanDetector(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            num_layers=config.num_layers
        )
        
        # Sentiment classification
        self.sentiment_classifier = SentimentClassifier(
            input_dim=config.hidden_size,
            num_classes=3  # POS, NEU, NEG
        )

    def forward(self, input_ids, attention_mask, span_positions=None):
        # Get Llama embeddings
        embeddings = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_positions=span_positions
        )
        
        # Detect aspect-opinion spans
        aspect_logits, opinion_logits = self.span_detector(embeddings)
        
        # Classify sentiment
        sentiment_logits = self.sentiment_classifier(
            embeddings,
            aspect_logits,
            opinion_logits
        )
        
        return {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits
        }