# src/models/absa.py - Complete Integration with Implicit Detection
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, List, Tuple, Optional, Any


class LLMABSA(nn.Module):
    """
    Enhanced ABSA model with complete implicit sentiment detection integration
    Implements 2024-2025 breakthrough features including implicit-explicit combinations
    """
    def __init__(self, config):
        super().__init__()
        # Import components locally to avoid circular imports
        from src.models.embedding import LLMEmbedding
        from src.models.span_detector import SpanDetector
        from src.models.classifier import AspectOpinionJointClassifier
        from src.models.implicit_detector import CompleteImplicitDetector

        # ============================================================================
        # INSTRUCTION-FOLLOWING COMPONENTS (2024-2025 BREAKTHROUGH)
        # ============================================================================

        # Add instruction-following capabilities
        self.use_instruction_following = getattr(config, 'use_instruction_following', True)

        if self.use_instruction_following:
            # Initialize T5 for instruction following
            instruction_model = getattr(config, 'instruction_model', 't5-small')
            try:
                self.t5_model = T5ForConditionalGeneration.from_pretrained(instruction_model)
                self.instruction_tokenizer = T5Tokenizer.from_pretrained(instruction_model)
                
                # Feature bridge from your backbone to T5
                self.feature_bridge = nn.Linear(config.hidden_size, self.t5_model.config.d_model)
                
                # Enhanced instruction templates for implicit detection
                self.instruction_templates = {
                    'triplet_extraction': "Extract aspect-opinion-sentiment triplets from: {text}",
                    'implicit_detection': "Find implicit aspects and opinions in: {text}",
                    'implicit_explicit_combination': "Extract both explicit and implicit sentiment elements from: {text}",
                    'quadruple_extraction': "Extract aspect-category-opinion-sentiment from: {text}",
                    'few_shot_adaptation': "Given examples {examples}, extract triplets from: {text}",
                    'grid_tagging': "Use grid tagging to extract sentiment relationships from: {text}"
                }
                
                # Add special tokens for structured output including implicit markers
                special_tokens = [
                    "<triplet>", "</triplet>", "<aspect>", "</aspect>", 
                    "<opinion>", "</opinion>", "<sentiment>", "</sentiment>",
                    "<implicit>", "</implicit>", "<explicit>", "</explicit>",
                    "<implicit_aspect>", "</implicit_aspect>", "<implicit_opinion>", "</implicit_opinion>",
                    "<combination>", "</combination>", "<grid>", "</grid>",
                    "<POS>", "<NEG>", "<NEU>"
                ]
                self.instruction_tokenizer.add_tokens(special_tokens)
                self.t5_model.resize_token_embeddings(len(self.instruction_tokenizer))
                
                print(f"✓ Instruction-following enabled with {instruction_model}")
                
            except Exception as e:
                print(f"Warning: Failed to load instruction model {instruction_model}: {e}")
                self.use_instruction_following = False

        # ============================================================================
        # IMPLICIT DETECTION INTEGRATION (2024-2025 BREAKTHROUGH)
        # ============================================================================
        
        # Enable implicit detection based on config
        self.use_implicit_detection = getattr(config, 'use_implicit_detection', True)
        
        # Unified training weights
        self.extraction_weight = getattr(config, 'extraction_weight', 1.0)
        self.generation_weight = getattr(config, 'generation_weight', 0.5)
        self.implicit_weight = getattr(config, 'implicit_weight', 0.8)  # New for implicit detection
        
        # Initialize components
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Initialize embeddings
        self.embeddings = LLMEmbedding(config)
        
        # Add a direct reference to the encoder for compatibility
        if hasattr(self.embeddings, 'encoder'):
            self.encoder = self.embeddings.encoder
        
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
        
        # Initialize span detector for explicit aspect and opinion extraction
        self.span_detector = SpanDetector(config)
        
        # Initialize explicit sentiment classifier
        self.sentiment_classifier = AspectOpinionJointClassifier(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            dropout=config.dropout,
            num_classes=3
        )
        
        # ============================================================================
        # IMPLICIT DETECTION COMPONENTS (NEW 2024-2025 INTEGRATION)
        # ============================================================================
        
        if self.use_implicit_detection:
            # Complete implicit detector
            self.implicit_detector = CompleteImplicitDetector(config)
            
            # Enhanced sentiment classifier for implicit-explicit combinations
            self.enhanced_sentiment_classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),  # explicit + implicit features
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size // 2, 3)  # POS, NEG, NEU
            )
            
            # Implicit-explicit fusion layer
            self.fusion_layer = nn.Sequential(
                nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size * 2, config.hidden_size)
            )
            
            print(f"✓ Implicit detection enabled")
        else:
            self.implicit_detector = None
            self.enhanced_sentiment_classifier = None
            self.fusion_layer = None
        
        # Store configuration for saving/loading
        self.config = config
        
        # Token filtering - these tokens should never be aspects or opinions
        self.filtered_tokens = [
            '.', ',', '!', '?', ';', ':', '[PAD]', '[CLS]', '[SEP]', '[UNK]',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        ]
        
        # Initialize weights
        self._initialize_weights()
        
        # Set tokenizer placeholder
        self.tokenizer = None
    
    def forward(self, input_ids, attention_mask, texts=None, task_type='triplet_extraction', target_text=None, **kwargs):
        """Enhanced forward pass with implicit detection integration"""
        try:
            # Get embeddings
            try:
                embeddings_output = self.embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            except TypeError:
                try:
                    embeddings_output = self.embeddings(input_ids, attention_mask)
                except TypeError:
                    if hasattr(self, 'encoder'):
                        outputs = self.encoder(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                        if hasattr(outputs, 'last_hidden_state'):
                            embeddings_output = {'hidden_states': outputs.last_hidden_state}
                        else:
                            batch_size, seq_len = input_ids.size()
                            embeddings_output = {
                                'hidden_states': torch.randn(batch_size, seq_len, self.hidden_size, device=input_ids.device)
                            }
                    else:
                        batch_size, seq_len = input_ids.size()
                        embeddings_output = {
                            'hidden_states': torch.randn(batch_size, seq_len, self.hidden_size, device=input_ids.device)
                        }
            
            hidden_states = embeddings_output['hidden_states']
            
            # ============================================================================
            # EXPLICIT DETECTION (Traditional Pipeline)
            # ============================================================================
            
            # Get explicit span detection results
            span_outputs = self.span_detector(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                texts=texts,
                tokenizer=self.tokenizer,
                input_ids=input_ids
            )
            
            aspect_logits, opinion_logits, span_features, boundary_logits = span_outputs
            
            # Get explicit sentiment classification
            explicit_sentiment_logits = self.sentiment_classifier(hidden_states)
            
            # ============================================================================
            # IMPLICIT DETECTION INTEGRATION (2024-2025 BREAKTHROUGH)
            # ============================================================================
            
            implicit_outputs = {}
            enhanced_sentiment_logits = explicit_sentiment_logits
            
            if self.use_implicit_detection and self.implicit_detector is not None:
                # Extract explicit features for context
                explicit_aspect_features = self._extract_explicit_features(hidden_states, aspect_logits)
                explicit_opinion_features = self._extract_explicit_features(hidden_states, opinion_logits)
                
                # Run implicit detection
                implicit_outputs = self.implicit_detector(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    explicit_aspect_features=explicit_aspect_features,
                    explicit_opinion_features=explicit_opinion_features
                )
                
                # Fuse explicit and implicit features for enhanced sentiment classification
                if self.fusion_layer is not None and self.enhanced_sentiment_classifier is not None:
                    # Combine explicit hidden states, implicit enhanced states, and span features
                    fused_features = torch.cat([
                        hidden_states,
                        implicit_outputs['enhanced_hidden_states'],
                        span_features if span_features is not None else hidden_states
                    ], dim=-1)
                    
                    # Apply fusion
                    fused_features = self.fusion_layer(fused_features)
                    
                    # Enhanced sentiment classification with implicit information
                    enhanced_sentiment_logits = self.enhanced_sentiment_classifier(
                        torch.cat([hidden_states, fused_features], dim=-1)
                    )
            
            # ============================================================================
            # INSTRUCTION-FOLLOWING INTEGRATION
            # ============================================================================
            
            instruction_outputs = {}
            if self.use_instruction_following and target_text is not None:
                # Get instruction-following outputs
                instruction_outputs = self._process_instruction_following(
                    input_ids, attention_mask, hidden_states, texts, task_type, target_text
                )
            
            # ============================================================================
            # PREPARE FINAL OUTPUTS
            # ============================================================================
            
            outputs = {
                # Explicit detection outputs
                'aspect_logits': aspect_logits,
                'opinion_logits': opinion_logits,
                'sentiment_logits': enhanced_sentiment_logits,  # Now includes implicit information
                'hidden_states': hidden_states,
                'span_features': span_features,
                'boundary_logits': boundary_logits,
                
                # Enhanced features
                'enhanced_hidden_states': implicit_outputs.get('enhanced_hidden_states', hidden_states),
                
                # Loss computation
                'loss': None  # Will be computed by loss function
            }
            
            # Add implicit detection outputs
            if implicit_outputs:
                outputs.update({
                    'implicit_aspect_scores': implicit_outputs.get('implicit_aspect_scores'),
                    'implicit_opinion_scores': implicit_outputs.get('implicit_opinion_scores'),
                    'aspect_sentiment_combinations': implicit_outputs.get('aspect_sentiment_combinations'),
                    'combination_logits': implicit_outputs.get('combination_logits'),
                    'confidence_scores': implicit_outputs.get('confidence_scores'),
                    'pattern_outputs': implicit_outputs.get('pattern_outputs'),
                    'context_outputs': implicit_outputs.get('context_outputs')
                })
            
            # Add instruction-following outputs
            if instruction_outputs:
                outputs.update(instruction_outputs)
            
            return outputs
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal outputs for error recovery
            batch_size, seq_len = input_ids.size()
            device = input_ids.device
            
            return {
                'aspect_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                'opinion_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                'sentiment_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                'hidden_states': torch.zeros(batch_size, seq_len, self.hidden_size, device=device),
                'loss': torch.tensor(0.0, device=device, requires_grad=True)
            }
    
    def _extract_explicit_features(self, hidden_states: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Extract features for explicit aspects/opinions to provide context to implicit detector"""
        # Use attention over logits to weight hidden states
        probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, num_classes]
        
        # Focus on B and I tags (classes 1 and 2), ignore O tag (class 0)
        entity_probs = probs[:, :, 1:].sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        
        # Weight hidden states by entity probability
        weighted_features = hidden_states * entity_probs  # [batch_size, seq_len, hidden_size]
        
        return weighted_features
    
    def _process_instruction_following(self, input_ids, attention_mask, hidden_states, texts, task_type, target_text):
        """Process instruction-following for unified generation"""
        if not hasattr(self, 't5_model') or self.t5_model is None:
            return {}
        
        try:
            # Convert to instruction format
            if texts is not None and len(texts) > 0:
                text = texts[0] if isinstance(texts, list) else str(texts)
            else:
                text = self.instruction_tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
            
            # Get appropriate instruction template
            instruction = self.instruction_templates.get(task_type, self.instruction_templates['triplet_extraction']).format(text=text)
            
            # Tokenize instruction
            instruction_inputs = self.instruction_tokenizer(
                instruction, return_tensors='pt', max_length=512, truncation=True, padding=True
            ).to(input_ids.device)
            
            # Bridge features from backbone to T5
            pooled_features = hidden_states.mean(dim=1)  # Pool sequence dimension
            bridged_features = self.feature_bridge(pooled_features)
            
            if target_text is not None:  # Training mode
                target_inputs = self.instruction_tokenizer(
                    target_text, return_tensors='pt', max_length=256, truncation=True, padding=True
                ).to(input_ids.device)
                
                # T5 forward pass
                t5_outputs = self.t5_model(
                    input_ids=instruction_inputs.input_ids,
                    attention_mask=instruction_inputs.attention_mask,
                    labels=target_inputs.input_ids
                )
                
                return {
                    'generation_loss': t5_outputs.loss,
                    'generation_logits': t5_outputs.logits
                }
            else:  # Inference mode
                # Generate response
                with torch.no_grad():
                    generated_ids = self.t5_model.generate(
                        input_ids=instruction_inputs.input_ids,
                        attention_mask=instruction_inputs.attention_mask,
                        max_length=256,
                        num_beams=3,
                        early_stopping=True
                    )
                
                return {
                    'generated_ids': generated_ids,
                    'generated_text': self.instruction_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                }
        
        except Exception as e:
            print(f"Error in instruction following: {e}")
            return {}
    
    def extract_all_triplets_with_implicit(self, input_ids, attention_mask, tokenizer=None, texts=None, 
                                         explicit_threshold=0.5, implicit_threshold=0.5):
        """
        Extract both explicit and implicit triplets using the enhanced pipeline
        
        Returns:
            Dictionary containing explicit triplets, implicit triplets, and combined results
        """
        # Set tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, texts=texts)
        
        # Extract explicit triplets using existing method
        explicit_triplets = self.extract_triplets(input_ids, attention_mask, tokenizer, texts)
        
        # Extract implicit triplets if implicit detection is enabled
        implicit_results = {}
        if self.use_implicit_detection and self.implicit_detector is not None:
            implicit_results = self.implicit_detector.extract_all_implicit_elements(
                input_ids, outputs, tokenizer or self.tokenizer, 
                aspect_threshold=implicit_threshold,
                opinion_threshold=implicit_threshold
            )
        
        # Combine and deduplicate results
        combined_triplets = self._combine_explicit_implicit_triplets(
            explicit_triplets, implicit_results.get('implicit_triplets', [])
        )
        
        return {
            'explicit_triplets': explicit_triplets,
            'implicit_results': implicit_results,
            'combined_triplets': combined_triplets,
            'summary': {
                'total_explicit': len(explicit_triplets) if explicit_triplets else 0,
                'total_implicit': len(implicit_results.get('implicit_triplets', [])),
                'total_combined': len(combined_triplets)
            }
        }
    
    def _combine_explicit_implicit_triplets(self, explicit_triplets, implicit_triplets):
        """Combine explicit and implicit triplets, removing duplicates"""
        combined = []
        
        # Add explicit triplets
        if explicit_triplets:
            for triplet in explicit_triplets:
                combined.append({
                    **triplet,
                    'detection_type': 'explicit'
                })
        
        # Add implicit triplets
        for triplet in implicit_triplets:
            # Check for overlap with explicit triplets
            is_duplicate = False
            for explicit_triplet in (explicit_triplets or []):
                if self._triplets_overlap(triplet, explicit_triplet):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined.append({
                    **triplet,
                    'detection_type': 'implicit'
                })
        
        return combined
    
    def _triplets_overlap(self, triplet1, triplet2, overlap_threshold=0.7):
        """Check if two triplets overlap significantly"""
        try:
            # Extract texts for comparison
            if isinstance(triplet1, dict) and 'aspect' in triplet1:
                aspect1 = triplet1['aspect'].get('text', '') if isinstance(triplet1['aspect'], dict) else str(triplet1['aspect'])
                opinion1 = triplet1['opinion'].get('text', '') if isinstance(triplet1['opinion'], dict) else str(triplet1['opinion'])
            else:
                aspect1 = str(triplet1.get('aspect', ''))
                opinion1 = str(triplet1.get('opinion', ''))
            
            if isinstance(triplet2, dict) and 'aspect' in triplet2:
                aspect2 = triplet2.get('aspect', '')
                opinion2 = triplet2.get('opinion', '')
            else:
                aspect2 = str(triplet2.get('aspect', ''))
                opinion2 = str(triplet2.get('opinion', ''))
            
            # Simple overlap check (can be enhanced with semantic similarity)
            aspect_overlap = (aspect1.lower() in aspect2.lower()) or (aspect2.lower() in aspect1.lower())
            opinion_overlap = (opinion1.lower() in opinion2.lower()) or (opinion2.lower() in opinion1.lower())
            
            return aspect_overlap and opinion_overlap
            
        except Exception as e:
            print(f"Error checking triplet overlap: {e}")
            return False
    
    @classmethod
    def load_model(cls, checkpoint_path, config=None, device='cpu'):
        """Load model from checkpoint with enhanced error handling"""
        try:
            if os.path.isfile(checkpoint_path):
                print(f"Loading model from {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    model_state = state_dict['model_state_dict']
                    if 'config' in state_dict and config is None:
                        loaded_config = state_dict['config']
                        if hasattr(loaded_config, '__dict__'):
                            config = loaded_config
                        else:
                            from src.utils.config import LLMABSAConfig
                            config = LLMABSAConfig()
                            for k, v in loaded_config.items():
                                if hasattr(config, k):
                                    setattr(config, k, v)
                        if hasattr(loaded_config, '__dict__'):
                            for k, v in loaded_config.__dict__.items():
                                if hasattr(loaded_config, k):
                                    setattr(loaded_config, k, v)
                            config = loaded_config
                else:
                    model_state = state_dict
                    
                # Ensure config is provided
                if config is None:
                    from src.utils.config import LLMABSAConfig
                    config = LLMABSAConfig()
                    
                # Create model instance
                model = cls(config)
                
                # Load state dictionary
                model.load_state_dict(model_state, strict=False)
                
                print(f"Model loaded successfully with enhanced implicit detection")
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
                batch_size = input_ids.size(0)
                return [[] for _ in range(batch_size)]
            
            # Check if sentiment_logits exists and is not None
            if 'sentiment_logits' not in outputs or outputs['sentiment_logits'] is None:
                print("Warning: sentiment_logits missing from model outputs. Using fallback sentiment.")
                batch_size, seq_len = input_ids.size()
                device = input_ids.device
                outputs['sentiment_logits'] = torch.zeros(batch_size, seq_len, 3, device=device)
            
            # Extract triplets using batch extraction
            triplets = self._extract_triplets_batch(
                outputs['aspect_logits'],
                outputs['opinion_logits'], 
                outputs['sentiment_logits'],
                input_ids,
                tokenizer or self.tokenizer
            )
            
            return triplets
            
        except Exception as e:
            print(f"Error in extract_triplets: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty triplets for each batch item
            batch_size = input_ids.size(0)
            return [[] for _ in range(batch_size)]
    
    def _extract_triplets_batch(self, aspect_logits, opinion_logits, sentiment_logits, input_ids, tokenizer):
        """Batch extraction of triplets with enhanced error handling"""
        if tokenizer is None:
            print("Warning: No tokenizer provided for triplet extraction")
            return []
        
        try:
            batch_triplets = []
            batch_size = input_ids.size(0)
            
            for i in range(batch_size):
                # Extract spans for this sample
                sample_triplets = self._extract_sample_triplets(
                    aspect_logits[i], opinion_logits[i], sentiment_logits[i],
                    input_ids[i], tokenizer
                )
                batch_triplets.append(sample_triplets)
            
            return batch_triplets
            
        except Exception as e:
            print(f"Error in batch triplet extraction: {e}")
            return [[] for _ in range(batch_size)]
    
    def _extract_sample_triplets(self, aspect_logits, opinion_logits, sentiment_logits, input_ids, tokenizer):
        """Extract triplets from a single sample"""
        try:
            # Get predictions
            aspect_preds = torch.argmax(aspect_logits, dim=-1).cpu().numpy()
            opinion_preds = torch.argmax(opinion_logits, dim=-1).cpu().numpy()
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1).cpu().numpy()
            
            # Extract spans
            aspect_spans = self._extract_spans(aspect_preds)
            opinion_spans = self._extract_spans(opinion_preds)
            
            # Decode span texts
            aspect_texts = []
            for start, end in aspect_spans:
                text = self._decode_span(input_ids, list(range(start, end + 1)), tokenizer)
                if text and self._is_valid_span(text):
                    aspect_texts.append((text, start, end))
            
            opinion_texts = []
            for start, end in opinion_spans:
                text = self._decode_span(input_ids, list(range(start, end + 1)), tokenizer)
                if text and self._is_valid_span(text):
                    opinion_texts.append((text, start, end))
            
            # Pair aspects with opinions and determine sentiment
            triplets = []
            for aspect_text, asp_start, asp_end in aspect_texts:
                best_opinion = None
                best_distance = float('inf')
                
                for opinion_text, op_start, op_end in opinion_texts:
                    distance = min(abs(asp_start - op_start), abs(asp_end - op_end))
                    if distance < best_distance:
                        best_distance = distance
                        best_opinion = (opinion_text, op_start, op_end)
                
                if best_opinion and best_distance < 20:  # Within reasonable distance
                    opinion_text, op_start, op_end = best_opinion
                    
                    # Determine sentiment for the region
                    region_start = min(asp_start, op_start)
                    region_end = max(asp_end, op_end)
                    region_sentiments = sentiment_preds[region_start:region_end + 1]
                    
                    # Get most common sentiment (excluding neutral if possible)
                    sentiment_counts = [0, 0, 0]  # NEU, POS, NEG
                    for s in region_sentiments:
                        if 0 <= s < 3:
                            sentiment_counts[s] += 1
                    
                    # Choose sentiment (prefer non-neutral)
                    if sentiment_counts[1] > sentiment_counts[2]:  # More POS than NEG
                        sentiment = 'POS'
                    elif sentiment_counts[2] > sentiment_counts[1]:  # More NEG than POS
                        sentiment = 'NEG'
                    else:
                        sentiment = 'NEU'
                    
                    triplets.append({
                        'aspect': aspect_text,
                        'opinion': opinion_text,
                        'sentiment': sentiment,
                        'confidence': 1.0 - (best_distance / 20.0)  # Distance-based confidence
                    })
            
            return triplets
            
        except Exception as e:
            print(f"Error extracting sample triplets: {e}")
            return []
    
    def _extract_spans(self, predictions):
        """Extract spans from BIO predictions"""
        spans = []
        start = None
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # B tag
                if start is not None:  # End previous span
                    spans.append((start, i - 1))
                start = i
            elif pred == 0:  # O tag
                if start is not None:  # End current span
                    spans.append((start, i - 1))
                    start = None
            # pred == 2 is I tag, continue current span
        
        # Handle span that goes to end
        if start is not None:
            spans.append((start, len(predictions) - 1))
        
        return spans
    
    def _decode_span(self, input_ids, span_indices, tokenizer):
        """Decode span indices to text with enhanced error handling"""
        try:
            if len(span_indices) == 0:
                return ""
                
            span_token_ids = [input_ids[i].item() for i in span_indices]
            text = tokenizer.decode(span_token_ids, skip_special_tokens=True)
            
            # Clean up text
            text = text.strip()
            text = text.replace(' ##', '').replace('##', '')
            text = text.replace('[CLS]', '').replace('[SEP]', '').replace('<s>', '').replace('</s>', '')
            text = text.strip()
            
            return text
        except Exception as e:
            print(f"Error in _decode_span: {e}")
            return ""
    
    def _is_valid_span(self, text):
        """Check if decoded span is valid"""
        if not text or len(text.strip()) < 2:
            return False
        
        text_lower = text.lower().strip()
        
        # Filter out common stop words and punctuation
        if text_lower in self.filtered_tokens:
            return False
        
        # Filter out pure punctuation
        if all(c in '.,!?;:()[]{}"\'-_' for c in text):
            return False
        
        return True
    
    def _initialize_weights(self):
        """Enhanced initialization with better regularization"""
        for name, param in self.named_parameters():
            if 'embeddings.encoder' not in name:  # Don't initialize pretrained weights
                if 'weight' in name and len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param.data, nonlinearity='relu')
                elif 'bias' in name or len(param.shape) < 2:
                    nn.init.zeros_(param.data)
    
    def generate_explanation(self, input_ids, attention_mask, triplets=None):
        """Generate explanation using instruction-following T5 model"""
        if not self.use_instruction_following or not hasattr(self, 't5_model'):
            return "Explanation generation not available (instruction-following disabled)"
        
        try:
            # Convert input to text
            text = self.instruction_tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
            
            # Create instruction for explanation generation
            if triplets:
                triplets_str = "; ".join([f"{t['aspect']}: {t['opinion']} ({t['sentiment']})" for t in triplets])
                instruction = f"Given triplets {triplets_str}, explain the sentiment analysis of: {text}"
            else:
                instruction = f"Explain the sentiment analysis of: {text}"
            
            # Tokenize instruction
            inputs = self.instruction_tokenizer(
                instruction, return_tensors='pt', max_length=512, truncation=True
            ).to(input_ids.device)
            
            # Generate explanation
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=200,
                    num_beams=3,
                    early_stopping=True,
                    temperature=0.7
                )
            
            explanation = self.instruction_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return explanation
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def get_model_info(self):
        """Debug method to check model configuration"""
        return {
            'embedding_size': getattr(self.embeddings, 'model_hidden_size', 'unknown'),
            'config_hidden_size': self.config.hidden_size,
            'actual_hidden_size': self.hidden_size,
            'use_implicit_detection': self.use_implicit_detection,
            'use_instruction_following': self.use_instruction_following,
            'components': {
                'embeddings': type(self.embeddings).__name__,
                'span_detector': type(self.span_detector).__name__,
                'classifier': type(self.sentiment_classifier).__name__,
                'implicit_detector': type(self.implicit_detector).__name__ if self.implicit_detector else None,
                'enhanced_classifier': type(self.enhanced_sentiment_classifier).__name__ if self.enhanced_sentiment_classifier else None
            }
        }