# src/models/absa.py
import torch # type: ignore
import torch.nn as nn # type: ignore
from src.models.cross_attention import MultiHeadCrossAttention
from src.models.span_detector import SpanDetector
from src.models.embedding import LLMEmbedding
from src.models.classifier import AspectOpinionJointClassifier
from transformers import AutoTokenizer
from src.models.explanation_generator import ExplanationGenerator
class LLMABSA(nn.Module):
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
        from src.models.embedding import LLMEmbedding
        from src.models.span_detector import SpanDetector
        from src.models.classifier import AspectOpinionJointClassifier
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.explanation_generator = ExplanationGenerator(config, tokenizer)
        self.generate_explanations = getattr(config, 'generate_explanations', False)
        
        # Use Stella embeddings
        self.embeddings = LLMEmbedding(config)
        
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
        self._initialize_weights()

    # The **kwargs will capture any additional parameters like aspect_labels
    def _initialize_weights(self):
        """Initialize weights with small values to prevent exploding gradients"""
        for name, param in self.named_parameters():
            if 'embeddings' not in name:  # Don't initialize pretrained weights
                if 'weight' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
    # Add to your ABSA model class
    def _extract_triplets(self, aspect_logits, opinion_logits, sentiment_logits, input_ids, tokenizer):
        """Extract triplets from model predictions for generative explanation"""
        batch_size = aspect_logits.size(0)
        triplets = []
        
        for b in range(batch_size):
            batch_triplets = []
            
            # Convert logits to predictions
            aspect_preds = aspect_logits[b].argmax(dim=-1)  # [seq_len]
            opinion_preds = opinion_logits[b].argmax(dim=-1)  # [seq_len]
            sentiment_pred = sentiment_logits[b].argmax(dim=-1).item()  # scalar
            
            # Map sentiment ID to label
            sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
            sentiment = sentiment_map[sentiment_pred]
            
            # Extract aspect spans
            aspect_spans = []
            current_span = []
            for i, pred in enumerate(aspect_preds):
                if pred == 1:  # B tag
                    if current_span:
                        aspect_spans.append(current_span)
                    current_span = [i]
                elif pred == 2:  # I tag
                    if current_span:
                        current_span.append(i)
                else:  # O tag
                    if current_span:
                        aspect_spans.append(current_span)
                        current_span = []
            if current_span:
                aspect_spans.append(current_span)
            
            # Extract opinion spans
            opinion_spans = []
            current_span = []
            for i, pred in enumerate(opinion_preds):
                if pred == 1:  # B tag
                    if current_span:
                        opinion_spans.append(current_span)
                    current_span = [i]
                elif pred == 2:  # I tag
                    if current_span:
                        current_span.append(i)
                else:  # O tag
                    if current_span:
                        opinion_spans.append(current_span)
                        current_span = []
            if current_span:
                opinion_spans.append(current_span)
            
            # Create triplets from spans
            for aspect_span in aspect_spans:
                for opinion_span in opinion_spans:
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
                    
                    aspect_text = ' '.join([tokens[i] for i in aspect_span 
                                        if i < len(tokens)])
                    opinion_text = ' '.join([tokens[i] for i in opinion_span
                                        if i < len(tokens)])
                    
                    triplet = {
                        'aspect': aspect_text,
                        'aspect_indices': aspect_span,
                        'opinion': opinion_text, 
                        'opinion_indices': opinion_span,
                        'sentiment': sentiment
                    }
                    batch_triplets.append(triplet)
            
            triplets.append(batch_triplets)
        
        return triplets
                    
# In src/models/absa.py - modify the forward method 

    def forward(self, input_ids, attention_mask,generate=False, **kwargs):
        # Get embeddings
        embeddings_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract hidden states
        hidden_states = self._extract_hidden_states(embeddings_output)
        
        # Create simple linear layers if needed
        if not hasattr(self, 'aspect_linear'):
            emb_size = hidden_states.size(-1)
            self.aspect_linear = nn.Linear(emb_size, 3).to(hidden_states.device)
            self.opinion_linear = nn.Linear(emb_size, 3).to(hidden_states.device)
            self.pooler = nn.Linear(emb_size, emb_size).to(hidden_states.device)
            self.sentiment_linear = nn.Linear(emb_size, 3).to(hidden_states.device)
        
        # Generate output tensors that have gradients
        aspect_logits = self.aspect_linear(hidden_states)
        opinion_logits = self.opinion_linear(hidden_states)
        
        # Pool for sentiment classification
        pooled = torch.tanh(self.pooler(hidden_states.mean(dim=1)))
        sentiment_logits = self.sentiment_linear(pooled)
        
        # Make sure tensors require gradients
        aspect_logits.requires_grad_(True)
        opinion_logits.requires_grad_(True)
        sentiment_logits.requires_grad_(True)

        outputs = {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits
            
        }
        
        # Add explanation generation if requested
        if self.generate_explanations or generate:
            # Extract triplets from predictions
            batch_size = input_ids.size(0)
            aspect_preds = aspect_logits.argmax(dim=-1)  # [batch_size, seq_len]
            opinion_preds = opinion_logits.argmax(dim=-1)  # [batch_size, seq_len]
            sentiment_preds = sentiment_logits.argmax(dim=-1)  # [batch_size]
            
            # Generate explanations
            explanation_logits = self.explanation_generator(
                hidden_states,
                aspect_preds, 
                opinion_preds,
                sentiment_preds,
                attention_mask
            )
            
            outputs['explanations'] = explanation_logits
        
        return outputs
        
        
    
# In src/models/absa.py - add a generative component
class GenerativeLLMABSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your existing code for span detection and sentiment classification
        self.embeddings = LLMEmbedding(config)
        self.span_detector = SpanDetector(config)
        self.sentiment_classifier = AspectOpinionJointClassifier(...)
        
        # Add the new generative component
        self.explanation_generator = ExplanationGenerator(config)
        
    def forward(self, input_ids, attention_mask, generate=False, **kwargs):
        # First stage: Extract triplets (your existing forward method)
        embeddings_output = self.embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = self._extract_hidden_states(embeddings_output)
        
        aspect_logits, opinion_logits, span_features = self.span_detector(
            hidden_states=hidden_states, 
            attention_mask=attention_mask
        )
        
        sentiment_logits, confidence_scores = self.sentiment_classifier(
            hidden_states=hidden_states,
            aspect_logits=aspect_logits, 
            opinion_logits=opinion_logits
        )
        
        # Results from first stage
        outputs = {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits,
            'confidence_scores': confidence_scores
        }
        
        # Second stage: Generate explanations (if requested)
        if generate:
            # Extract triplets from predictions
            triplets = self._extract_triplets(
                aspect_logits, 
                opinion_logits, 
                sentiment_logits, 
                input_ids, 
                self.tokenizer
            )
            
            # Generate explanations for each triplet
            explanations = self.explanation_generator(
                hidden_states, 
                triplets, 
                input_ids, 
                attention_mask
            )
            
            outputs['explanations'] = explanations
            
        return outputs