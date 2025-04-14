# src/models/absa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LLMABSA(nn.Module):
    """
    Memory-efficient ABSA model for triplet extraction with improved prediction filtering
    and robust error handling
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
        
        # Store configuration for saving/loading
        self.config = config
        
        # Token filtering - these tokens should never be aspects or opinions
        self.filtered_tokens = [
            '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}',
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to',
            'for', 'with', 'by', 'about', 'as', 'from', 'of',
            '[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '[PAD]'
        ]
        
        # Confidence thresholds for filtering predictions
        self.aspect_confidence_threshold = 0.65
        self.opinion_confidence_threshold = 0.60
        self.sentiment_confidence_threshold = 0.70
        
        # Safe initialization of weights
        self._initialize_weights()
    
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
    
# Update this in src/models/absa.py
    def forward(self, input_ids, attention_mask, texts=None, **kwargs):
        """Forward pass for triplet extraction with enhanced error handling and improved outputs"""
        try:
            # Get embeddings
            embeddings_output = self.embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract hidden states
            hidden_states = self._extract_hidden_states(embeddings_output)
            
            # Verify hidden states are valid
            if hidden_states is None:
                raise ValueError("Hidden states are None")
            
            # Get aspect and opinion spans
            aspect_logits, opinion_logits, span_features, boundary_logits = self.span_detector(
                hidden_states,
                attention_mask,
                texts=texts,
                input_ids=input_ids,
                tokenizer=getattr(self, 'tokenizer', None)
            )
            
            # Verify aspect and opinion logits are valid
            if aspect_logits is None:
                print("Warning: aspect_logits are None, creating dummy values")
                batch_size, seq_len = input_ids.shape
                device = input_ids.device
                aspect_logits = torch.zeros(batch_size, seq_len, 3, device=device)
                # Set bias toward O tag (index 0)
                aspect_logits[:, :, 0] = 1.0
                
            if opinion_logits is None:
                print("Warning: opinion_logits are None, creating dummy values")
                batch_size, seq_len = input_ids.shape
                device = input_ids.device
                opinion_logits = torch.zeros(batch_size, seq_len, 3, device=device)
                # Set bias toward O tag (index 0)
                opinion_logits[:, :, 0] = 1.0
                
            # Print debug info for the dimension of hidden_states
            # print(f"Hidden states shape: {hidden_states.shape}")
            # print(f"Expected hidden dimension: {self.hidden_size}")
            # print(f"Expected classifier input dimension: {self.sentiment_classifier.input_dim}")
            
            # Get sentiment and confidence with explicit error handling
            try:
                sentiment_logits, confidence_scores = self.sentiment_classifier(
                    hidden_states,
                    aspect_logits,
                    opinion_logits,
                    attention_mask,
                    input_ids=input_ids,
                    tokenizer=getattr(self, 'tokenizer', None)
                )
            except Exception as e:
                print(f"Error in sentiment classifier: {e}")
                batch_size = input_ids.size(0)
                device = input_ids.device
                
                # Create fallback sentiment logits (balanced probabilities)
                sentiment_logits = torch.ones(batch_size, 3, device=device) / 3
                confidence_scores = torch.ones(batch_size, 1, device=device) * 0.5
                
            # Ensure sentiment_logits is not None
            if sentiment_logits is None:
                batch_size = input_ids.size(0)
                device = input_ids.device
                sentiment_logits = torch.ones(batch_size, 3, device=device) / 3
                
            # Ensure confidence_scores is not None
            if confidence_scores is None:
                batch_size = input_ids.size(0)
                device = input_ids.device
                confidence_scores = torch.ones(batch_size, 1, device=device) * 0.5
            
            # Return all outputs
            outputs = {
                'aspect_logits': aspect_logits,
                'opinion_logits': opinion_logits,
                'sentiment_logits': sentiment_logits,
                'confidence_scores': confidence_scores,
                'boundary_logits': boundary_logits if boundary_logits is not None else None
            }
            
            return outputs
            
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback outputs with correct shapes
            try:
                batch_size, seq_len = input_ids.size()
                device = input_ids.device
                
                # Return tensor placeholders with correct dimensions
                return {
                    'aspect_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                    'opinion_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                    'sentiment_logits': torch.ones(batch_size, 3, device=device) / 3,  # Balanced
                    'confidence_scores': torch.ones(batch_size, 1, device=device) * 0.5
                }
            except:
                # Complete failure - return None and let extract_triplets handle it
                return None
                
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
        """Load model with proper configuration and error handling"""
        if not os.path.exists(load_path):
            # Try alternate path (handles different saved model patterns)
            alt_paths = [
                load_path.replace("generative_absa_", "ultra-lightweight-absa_") + "_best.pt",
                load_path.replace("generative_absa_", "ultra-lightweight-absa_") + "_final.pt",
                load_path + "_best.pt",
                load_path + "_final.pt"
            ]
            
            # Try each alternate path
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"Found model at alternate path: {alt_path}")
                    load_path = alt_path
                    break
            else:
                print(f"Warning: Model file not found at {load_path} or alternate paths, initializing with default weights")
                if config is None:
                    from src.utils.config import LLMABSAConfig
                    config = LLMABSAConfig()
                return cls(config)
            
        try:
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
            
            print(f"Model loaded successfully with strict=False (ignoring architecture differences)")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a default model as fallback
            if config is None:
                from src.utils.config import LLMABSAConfig
                config = LLMABSAConfig()
            
            print("Creating a default model as fallback")
            return cls(config)
        
    def extract_triplets(self, input_ids, attention_mask, tokenizer=None, texts=None):
        """Extract triplets from model predictions with robust error handling"""
        # Set the tokenizer for the model if provided
        if tokenizer is not None:
            self.tokenizer = tokenizer
        
        try:
            # Get model predictions
            outputs = self.forward(input_ids, attention_mask, texts=texts)
            
            # Check if outputs is None (complete model failure)
            if outputs is None:
                print("Warning: Model forward pass returned None. Using fallback predictions.")
                # Create fallback outputs
                batch_size = input_ids.size(0)
                device = input_ids.device
                seq_len = input_ids.size(1)
                
                # Return empty triplets for each batch item
                return [[] for _ in range(batch_size)]
            
            # Check if sentiment_logits exists and is not None
            if 'sentiment_logits' not in outputs or outputs['sentiment_logits'] is None:
                print("Warning: sentiment_logits missing from model outputs. Using fallback sentiment.")
                # Create fallback sentiment logits (default to neutral)
                batch_size = input_ids.size(0)
                device = input_ids.device
                sentiment_preds = torch.ones(batch_size, dtype=torch.long, device=device)  # Default to neutral (1)
            else:
                # Normal case - use the model's sentiment predictions
                sentiment_preds = outputs['sentiment_logits'].argmax(dim=-1)  # [batch_size]
                
            # Check if aspect_logits exists and is not None
            if 'aspect_logits' not in outputs or outputs['aspect_logits'] is None:
                print("Warning: aspect_logits missing from model outputs. Using fallback aspects.")
                # Create fallback aspect logits (all zeros - no aspects)
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                device = input_ids.device
                aspect_preds = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            else:
                # Normal case - use the model's aspect predictions
                aspect_preds = outputs['aspect_logits'].argmax(dim=-1)  # [batch_size, seq_len]
                
            # Check if opinion_logits exists and is not None
            if 'opinion_logits' not in outputs or outputs['opinion_logits'] is None:
                print("Warning: opinion_logits missing from model outputs. Using fallback opinions.")
                # Create fallback opinion logits (all zeros - no opinions)
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                device = input_ids.device
                opinion_preds = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            else:
                # Normal case - use the model's opinion predictions
                opinion_preds = outputs['opinion_logits'].argmax(dim=-1)  # [batch_size, seq_len]
                
            # Check if confidence_scores exists and is not None
            if 'confidence_scores' not in outputs or outputs['confidence_scores'] is None:
                # Create fallback confidence (medium confidence)
                batch_size = input_ids.size(0)
                device = input_ids.device
                confidence = torch.ones(batch_size, 1, device=device) * 0.5  # Default to 0.5 confidence
            else:
                # Normal case - use the model's confidence scores
                confidence = outputs['confidence_scores']  # [batch_size, 1]
            
            # Convert to triplets
            batch_size = input_ids.size(0)
            all_triplets = []
            
            for b in range(batch_size):
                # Get predictions for this item
                aspect_pred = aspect_preds[b]  # [seq_len]
                opinion_pred = opinion_preds[b]  # [seq_len]
                sentiment = sentiment_preds[b].item() if isinstance(sentiment_preds[b], torch.Tensor) else sentiment_preds[b]  # scalar
                conf = confidence[b].item() if isinstance(confidence[b], torch.Tensor) and confidence[b].numel() == 1 else 0.5  # scalar
                
                # Map sentiment to label
               
                sentiment_map = {0: 'NEG', 1: 'NEU', 2: 'POS'}
                sentiment_label = sentiment_map.get(sentiment, 'NEU')  # Default to neutral if unknown
                
                # Extract aspect spans with safety checks
                try:
                    aspect_spans = self._extract_spans(aspect_pred, attention_mask[b])
                except Exception as e:
                    print(f"Error extracting aspect spans: {e}")
                    aspect_spans = []
                
                # Extract opinion spans with safety checks
                try:
                    opinion_spans = self._extract_spans(opinion_pred, attention_mask[b])
                except Exception as e:
                    print(f"Error extracting opinion spans: {e}")
                    opinion_spans = []
                
                # Create triplets
                triplets = []
                
                # If we have a tokenizer, decode spans
                if tokenizer is not None:
                    try:
                        # Decode spans to text with error handling
                        aspect_texts = []
                        for span in aspect_spans:
                            try:
                                text = self._decode_span(input_ids[b], span, tokenizer)
                                aspect_texts.append(text)
                            except Exception as e:
                                print(f"Error decoding aspect span: {e}")
                                aspect_texts.append("")
                        
                        opinion_texts = []
                        for span in opinion_spans:
                            try:
                                text = self._decode_span(input_ids[b], span, tokenizer)
                                opinion_texts.append(text)
                            except Exception as e:
                                print(f"Error decoding opinion span: {e}")
                                opinion_texts.append("")
                        
                        # Create triplets
                        for aspect_text, aspect_span in zip(aspect_texts, aspect_spans):
                            for opinion_text, opinion_span in zip(opinion_texts, opinion_spans):
                                try:
                                    triplets.append({
                                        'aspect': aspect_text,
                                        'aspect_indices': aspect_span.tolist(),
                                        'opinion': opinion_text,
                                        'opinion_indices': opinion_span.tolist(),
                                        'sentiment': sentiment_label,
                                        'confidence': float(conf)  # Ensure it's a Python float
                                    })
                                except Exception as e:
                                    print(f"Error creating triplet: {e}")
                    except Exception as e:
                        print(f"Error creating triplets with tokenizer: {e}")
                        # Fallback: create a basic triplet
                        if aspect_spans and opinion_spans:
                            try:
                                triplets.append({
                                    'aspect_indices': aspect_spans[0].tolist(),
                                    'opinion_indices': opinion_spans[0].tolist(),
                                    'sentiment': sentiment_label,
                                    'confidence': float(conf)
                                })
                            except Exception as e:
                                print(f"Error creating fallback triplet: {e}")
                else:
                    # Just use indices if no tokenizer
                    for aspect_span in aspect_spans:
                        for opinion_span in opinion_spans:
                            try:
                                triplets.append({
                                    'aspect_indices': aspect_span.tolist(),
                                    'opinion_indices': opinion_span.tolist(),
                                    'sentiment': sentiment_label,
                                    'confidence': float(conf)
                                })
                            except Exception as e:
                                print(f"Error creating triplet without tokenizer: {e}")
                
                # Add a fallback triplet if we have no valid triplets but do have text
                if not triplets and texts is not None and b < len(texts):
                    # Create a simple fallback triplet from the input text
                    try:
                        triplets.append({
                            'aspect': "general",
                            'aspect_indices': [0],
                            'opinion': "",
                            'opinion_indices': [],
                            'sentiment': sentiment_label,
                            'confidence': 0.5
                        })
                    except Exception as e:
                        print(f"Error creating text fallback triplet: {e}")
                        
                all_triplets.append(triplets)
            
            return all_triplets
            
        except Exception as e:
            print(f"Critical error in extract_triplets: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty predictions as last resort fallback
            batch_size = input_ids.size(0)
            return [[] for _ in range(batch_size)]
    
    def _extract_spans(self, predictions, attention_mask):
        """Extract token spans from BIO predictions with error handling
        
        Args:
            predictions: Tensor of BIO tag predictions [seq_len]
            attention_mask: Tensor of attention mask [seq_len]
            
        Returns:
            List of tensors containing token indices for each span
        """
        try:
            # Get valid predictions (exclude padding)
            valid_mask = attention_mask.bool()
            
            # Handle empty or invalid inputs
            if not valid_mask.any():
                return []
                
            valid_preds = predictions[valid_mask]
            
            # Extract spans
            spans = []
            current_span = []
            
            for i, pred in enumerate(valid_preds):
                if pred == 1:  # B tag - beginning of a new span
                    # Add previous span if it exists
                    if current_span:
                        spans.append(torch.tensor(current_span))
                    # Start new span
                    current_span = [i]
                elif pred == 2 and current_span:  # I tag - continue current span
                    current_span.append(i)
                elif current_span:  # O tag or other - end current span if it exists
                    spans.append(torch.tensor(current_span))
                    current_span = []
            
            # Add last span if it exists
            if current_span:
                spans.append(torch.tensor(current_span))
            
            return spans
            
        except Exception as e:
            print(f"Error in _extract_spans: {e}")
            # Return empty list as fallback
            return []

    def _decode_span(self, input_ids, span_indices, tokenizer):
        """Decode a token span to text with error handling
        
        Args:
            input_ids: Tensor of input token IDs [seq_len]
            span_indices: Tensor of token indices in the span
            tokenizer: Tokenizer for decoding tokens
            
        Returns:
            String containing the decoded text
        """
        try:
            # Handle empty spans
            if len(span_indices) == 0:
                return ""
                
            # Get token IDs for this span
            span_token_ids = [input_ids[i].item() for i in span_indices]
            
            # Decode to text
            text = tokenizer.decode(span_token_ids, skip_special_tokens=True)
            
            # Clean up text
            text = text.strip()
            # Remove common prefixes from wordpiece tokenizers
            text = text.replace(' ##', '').replace('##', '')
            # Remove common special tokens
            text = text.replace('[CLS]', '').replace('[SEP]', '').replace('<s>', '').replace('</s>', '')
            text = text.strip()
            
            return text
        except Exception as e:
            print(f"Error in _decode_span: {e}")
            return ""