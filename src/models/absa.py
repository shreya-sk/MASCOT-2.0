# src/models/absa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LLMABSA(nn.Module):
    """
    Memory-efficient ABSA model for triplet extraction
    
    This 2025 implementation uses lightweight components and advanced
    training techniques for resource-constrained environments.
    """
    def __init__(self, config):
        super().__init__()
        # Import components locally to avoid circular imports
        from src.models.embedding import LLMEmbedding
        from src.models.span_detector import SpanDetector
        from src.models.classifier import AspectOpinionJointClassifier
        
        # Initialize components
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Initialize embeddings
        self.embeddings = LLMEmbedding(config)
        
        # Get the actual embedding output dimension
        embedding_size = getattr(self.embeddings, 'model_hidden_size', config.hidden_size)
        
        # Set the embedding size in config for other components to use
        config.embedding_size = embedding_size
        
        # Update hidden size to match embedding size - IMPORTANT!
        config.hidden_size = embedding_size
        self.hidden_size = embedding_size
        
        print(f"Using embedding size {embedding_size} as the hidden size")
        
        # No dimension adapter needed anymore
        self.dim_adapter = None
        
        # Initialize span detector for aspect and opinion extraction
        self.span_detector = SpanDetector(config)
        
        # Initialize sentiment classifier
        self.sentiment_classifier = AspectOpinionJointClassifier(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            dropout=config.dropout,
            num_classes=3
        )
        # Safe initialization of weights
        self._initialize_weights()
        
        # Store configuration for saving/loading
        self.config = config
        
    def _initialize_weights(self):
        """Initialize weights with safe initialization for different tensor dimensions"""
        for name, param in self.named_parameters():
            if 'embeddings.encoder' not in name:  # Don't initialize pretrained weights
                if 'weight' in name and len(param.shape) >= 2:
                    # Use Xavier initialization only for 2D+ tensors
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name or len(param.shape) < 2:
                    # Use zeros initialization for biases and 1D tensors
                    nn.init.zeros_(param.data)
    
    def _extract_hidden_states(self, embeddings_output):
        """Extract hidden states from embeddings output"""
        if isinstance(embeddings_output, dict):
            if 'hidden_states' in embeddings_output:
                return embeddings_output['hidden_states']
            elif 'last_hidden_state' in embeddings_output:
                return embeddings_output['last_hidden_state']
            elif 'aspect_embeddings' in embeddings_output:
                # Return aspect embeddings if hidden states not available
                return embeddings_output['aspect_embeddings']
            else:
                # Return the first value in the dict as a fallback
                return list(embeddings_output.values())[0]
        else:
            return embeddings_output
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """Forward pass for triplet extraction"""
        try:
            # Get embeddings
            embeddings_output = self.embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract hidden states
            hidden_states = self._extract_hidden_states(embeddings_output)
            
            # No dimension adapter needed
            
            # Get aspect and opinion spans
            aspect_logits, opinion_logits, span_features, boundary_logits = self.span_detector(
                hidden_states,
                attention_mask
            )
            
            # Get sentiment and confidence
            sentiment_logits, confidence_scores = self.sentiment_classifier(
                hidden_states,
                aspect_logits,
                opinion_logits,
                attention_mask,
                input_ids=input_ids  # Pass input_ids for additional rule-based processing
            )
            
            # Return all outputs
            outputs = {
                'aspect_logits': aspect_logits,
                'opinion_logits': opinion_logits,
                'sentiment_logits': sentiment_logits,
                'confidence_scores': confidence_scores
            }
            
            # Add boundary logits if available
            if boundary_logits is not None:
                outputs['boundary_logits'] = boundary_logits
            
            return outputs
            
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback outputs with correct shapes
            batch_size, seq_len = input_ids.size()
            device = input_ids.device
            
            # Return tensor placeholders with correct dimensions
            return {
                'aspect_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                'opinion_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                'sentiment_logits': torch.zeros(batch_size, 3, device=device),
                'confidence_scores': torch.ones(batch_size, 1, device=device)
            }
    def save(self, save_path):
        """Save model with proper metadata"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create a state dictionary with configuration
        state_dict = {
            'model_state_dict': self.state_dict(),
            'config': {k: v for k, v in self.config.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        }
        
        # Save to file
        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, load_path, config=None, device='cpu'):
        """Load model with proper configuration"""
        if not os.path.exists(load_path):
            print(f"Warning: Model file not found at {load_path}, initializing with default weights")
            if config is None:
                from src.utils.config import LLMABSAConfig
                config = LLMABSAConfig()
            return cls(config)
            
        # Load state dictionary with configuration
        state_dict = torch.load(load_path, map_location=device)
        
        # Handle both formats (direct state dict or dict with model_state_dict)
        if 'model_state_dict' in state_dict:
            model_state = state_dict['model_state_dict']
            # If config was loaded and not provided, use it
            if config is None and 'config' in state_dict:
                from src.utils.config import LLMABSAConfig
                loaded_config = LLMABSAConfig()
                for k, v in state_dict['config'].items():
                    if hasattr(loaded_config, k):
                        setattr(loaded_config, k, v)
                config = loaded_config
        else:
            # Direct state dict
            model_state = state_dict
            
        # Ensure config is provided
        if config is None:
            from src.utils.config import LLMABSAConfig
            config = LLMABSAConfig()
            
        # Create model instance
        model = cls(config)
        
        # Load state dictionary
        # Use strict=False to allow missing keys, which may happen due to model changes
        model.load_state_dict(model_state, strict=False)
        
        print(f"Model loaded from {load_path}")
        return model
        
    def extract_triplets(self, input_ids, attention_mask, tokenizer=None):
        """Extract triplets from model predictions"""
        # Get model predictions
        outputs = self.forward(input_ids, attention_mask)
        
        # Extract predictions
        aspect_preds = outputs['aspect_logits'].argmax(dim=-1)  # [batch_size, seq_len]
        opinion_preds = outputs['opinion_logits'].argmax(dim=-1)  # [batch_size, seq_len]
        sentiment_preds = outputs['sentiment_logits'].argmax(dim=-1)  # [batch_size]
        confidence = outputs['confidence_scores']  # [batch_size, 1]
        
        # Convert to triplets
        batch_size = input_ids.size(0)
        all_triplets = []
        
        for b in range(batch_size):
            # Get predictions for this item
            aspect_pred = aspect_preds[b]  # [seq_len]
            opinion_pred = opinion_preds[b]  # [seq_len]
            sentiment = sentiment_preds[b].item()  # scalar
            conf = confidence[b].item()  # scalar
            
            # Map sentiment to label
            sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
            sentiment_label = sentiment_map.get(sentiment, 'NEU')
            
            # Extract aspect spans
            aspect_spans = self._extract_spans(aspect_pred, attention_mask[b])
            
            # Extract opinion spans
            opinion_spans = self._extract_spans(opinion_pred, attention_mask[b])
            
            # Create triplets
            triplets = []
            
            # If we have a tokenizer, decode spans
            if tokenizer is not None:
                # Decode spans to text
                aspect_texts = [self._decode_span(input_ids[b], span, tokenizer) for span in aspect_spans]
                opinion_texts = [self._decode_span(input_ids[b], span, tokenizer) for span in opinion_spans]
                
                # Create triplets (all combinations of aspects and opinions)
                for aspect_text, aspect_span in zip(aspect_texts, aspect_spans):
                    for opinion_text, opinion_span in zip(opinion_texts, opinion_spans):
                        triplets.append({
                            'aspect': aspect_text,
                            'aspect_indices': aspect_span,
                            'opinion': opinion_text,
                            'opinion_indices': opinion_span,
                            'sentiment': sentiment_label,
                            'confidence': conf
                        })
            else:
                # Just use indices if no tokenizer
                for aspect_span in aspect_spans:
                    for opinion_span in opinion_spans:
                        triplets.append({
                            'aspect_indices': aspect_span,
                            'opinion_indices': opinion_span,
                            'sentiment': sentiment_label,
                            'confidence': conf
                        })
            
            all_triplets.append(triplets)
        
        return all_triplets
    
    def _extract_spans(self, predictions, attention_mask):
        """Extract token spans from BIO predictions"""
        # Get valid predictions (exclude padding)
        valid_mask = attention_mask.bool()
        valid_preds = predictions[valid_mask]
        
        # Extract spans
        spans = []
        current_span = []
        
        for i, pred in enumerate(valid_preds):
            if pred == 1:  # B tag
                if current_span:
                    spans.append(current_span)
                current_span = [i]
            elif pred == 2:  # I tag
                if current_span:
                    current_span.append(i)
            else:  # O tag
                if current_span:
                    spans.append(current_span)
                    current_span = []
        
        # Add last span if it exists
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def _decode_span(self, input_ids, span_indices, tokenizer):
        """Decode a token span to text"""
        # Get token IDs for this span
        span_token_ids = [input_ids[i].item() for i in span_indices]
        
        # Decode to text
        try:
            # Use tokenizer to decode
            text = tokenizer.decode(span_token_ids, skip_special_tokens=True)
            return text.strip()
        except:
            # Fallback if decoding fails
            return f"Span at indices {span_indices}"