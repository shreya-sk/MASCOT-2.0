# In src/models/model.py
from src.models.embedding import ModernEmbedding

class LlamaABSA(nn.Module):
    """ABSA model using modern embeddings"""
    def __init__(self, config):
        super().__init__()
        self.embeddings = ModernEmbedding(config)
        
        # The rest of your model remains the same
        self.span_detector = SpanDetector(config)
        self.sentiment_classifier = SentimentClassifier(config)
        
    def forward(self, input_ids, attention_mask, texts=None, **kwargs):
        # Get embeddings
        embeddings = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            texts=texts
        )
        
        # Rest of your model
        aspect_logits, opinion_logits = self.span_detector(embeddings, attention_mask)
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