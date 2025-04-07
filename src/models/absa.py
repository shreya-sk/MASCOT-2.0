# src/models/absa.py
import torch.nn as nn # type: ignore

class StellaABSA(nn.Module):
    """ABSA model using Stella embeddings
       
    Novel ABSA model using Stella v5 embeddings with multi-focal attention
    
    Key innovations:
    1. Aspect-Opinion Joint Learning with bidirectional influence
    2. Context-aware span detection with focal attention
    3. Multi-domain knowledge transfer adapter
    4. Hierarchical fusion of syntactic and semantic features
    """
    def __init__(self, config):
        super().__init__()
        # Import locally to avoid circular imports
        from src.models.embedding import StellaEmbedding
        from src.models.span_detector import SpanDetector
        from src.models.classifier import AspectOpinionJointClassifier
        
        # Use Stella embeddings
        self.embeddings = StellaEmbedding(config)
        
        # Aspect-Opinion span detection
        self.span_detector = SpanDetector(config)
        
        # Sentiment classification using joint classifier
        self.sentiment_classifier = AspectOpinionJointClassifier(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            dropout=config.dropout,
            num_classes=3,
            use_aspect_first=getattr(config, 'use_aspect_first', True)
        )

    def forward(self, input_ids, attention_mask, span_positions=None, domain_id=None):
        """
        Forward pass through the ABSA model
        
        Args:
            input_ids: Input token ids [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            span_positions: Optional span position ids
            domain_id: Optional domain identifier for domain adaptation
        """
        # Get embeddings from Stella
        embeddings_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_id=domain_id
        )
        
        # Get aspect and opinion embeddings
        aspect_embeddings = embeddings_output['aspect_embeddings']
        opinion_embeddings = embeddings_output['opinion_embeddings']
        hidden_states = embeddings_output['hidden_states']
        
        # Detect aspect-opinion spans
        aspect_logits, opinion_logits, span_features = self.span_detector(
            aspect_embeddings=aspect_embeddings,
            opinion_embeddings=opinion_embeddings,
            attention_mask=attention_mask
        )
        
        # Classify sentiment using joint classifier
        sentiment_logits, confidence_scores = self.sentiment_classifier(
            hidden_states=hidden_states,
            aspect_logits=aspect_logits,
            opinion_logits=opinion_logits,
            span_features=span_features,
            attention_mask=attention_mask
        )
        
        return {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits,
            'confidence_scores': confidence_scores,
            'span_features': span_features
        }