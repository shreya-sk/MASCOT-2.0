# src/data/preprocessor.py
from typing import List, Dict, Any
import torch
from transformers import PreTrainedTokenizer
import spacy
from spacy.tokens import Doc
import numpy as np

# Fix this import to include SpanLabel
from src.data.utils import SENTIMENT_MAP, convert_to_bio_labels, SpanLabel

class LLMABSAPreprocessor:
    """
    Enhanced preprocessor for ABSA data with consistent tensor shapes
    and robust error handling
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        use_syntax: bool = True,
        context_window: int = 2
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_syntax = use_syntax
        self.context_window = context_window
        
        # Load spaCy for syntax features if needed
        self.nlp = None
        if self.use_syntax:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ Loaded spaCy for syntax-aware features")
            except Exception as e:
                print(f"Could not load spaCy: {e}")
                print("Falling back to syntax-free processing")
                self.use_syntax = False

    def preprocess(self, text: str, span_labels: List[SpanLabel]) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single instance with enhanced features and consistent shapes
        
        Args:
            text: Input text string
            span_labels: List of SpanLabel objects
            
        Returns:
            Dictionary with preprocessed tensors
        """
        try:
            # Tokenize text
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            # FIXED: Ensure consistent tensor shapes
            # Always create tensors for at least 1 span, even if empty
            max_spans = max(1, len(span_labels))
            
            # Initialize label tensors with consistent shapes
            aspect_labels = torch.zeros(max_spans, self.max_length, dtype=torch.long)
            opinion_labels = torch.zeros(max_spans, self.max_length, dtype=torch.long)
            sentiment_labels = torch.zeros(max_spans, dtype=torch.long)
            
            # Process each span label
            for i, span_label in enumerate(span_labels):
                if i < max_spans:
                    # Convert indices to BIO labels
                    aspect_bio = convert_to_bio_labels(
                        span_label.aspect_indices, 
                        self.max_length
                    )
                    opinion_bio = convert_to_bio_labels(
                        span_label.opinion_indices,
                        self.max_length
                    )
                    
                    # Apply context enhancement if enabled
                    if self.context_window > 0:
                        aspect_bio = self._enhance_span_context(aspect_bio)
                        opinion_bio = self._enhance_span_context(opinion_bio)
                    
                    # Store in tensors
                    aspect_labels[i] = aspect_bio.long()
                    opinion_labels[i] = opinion_bio.long()
                    sentiment_labels[i] = SENTIMENT_MAP[span_label.sentiment]
            
            # Get syntax features if enabled
            syntax_features = None
            if self.use_syntax and self.nlp is not None:
                try:
                    syntax_features = self._extract_syntax_features(text, self.max_length)
                except Exception as e:
                    print(f"Warning: Failed to extract syntax features: {e}")
                    syntax_features = torch.zeros(self.max_length, 64)  # Fallback
            
            # Create the output dictionary with consistent shapes
            output = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'aspect_labels': aspect_labels,  # [max_spans, seq_len]
                'opinion_labels': opinion_labels,  # [max_spans, seq_len]
                'sentiment_labels': sentiment_labels,  # [max_spans]
                'num_spans': torch.tensor(len(span_labels), dtype=torch.long)
            }
            
            # Add syntax features if available
            if syntax_features is not None:
                output['syntax_features'] = syntax_features
            else:
                # Always provide syntax_features tensor for consistency
                output['syntax_features'] = torch.zeros(self.max_length, 64)
                
            return output
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Return safe fallback tensors
            return self._create_fallback_output()
    
    def _enhance_span_context(self, span_labels: torch.Tensor) -> torch.Tensor:
        """
        Enhance span labels by considering surrounding context
        
        Args:
            span_labels: BIO labels tensor [seq_len]
            
        Returns:
            Enhanced labels tensor [seq_len]
        """
        try:
            enhanced_labels = span_labels.clone().float()
            seq_length = span_labels.size(0)
            
            # Find B and I tags
            b_indices = (span_labels == 1).nonzero(as_tuple=True)[0]
            i_indices = (span_labels == 2).nonzero(as_tuple=True)[0]
            
            # Add context window around beginning tags with decreasing weights
            for idx in b_indices:
                for offset in range(1, self.context_window + 1):
                    # Context before span start
                    if idx - offset >= 0 and enhanced_labels[idx - offset] == 0:
                        enhanced_labels[idx - offset] = 0.3 / offset
                    
                    # Context after span start (if not already I tag)
                    if (idx + offset < seq_length and 
                        idx + offset not in i_indices and 
                        enhanced_labels[idx + offset] == 0):
                        enhanced_labels[idx + offset] = 0.2 / offset
            
            # Add context after ending tags
            for idx in i_indices:
                # Check if this is an ending tag (no I tag follows it)
                if idx + 1 >= seq_length or (idx + 1 not in i_indices and enhanced_labels[idx + 1] == 0):
                    for offset in range(1, self.context_window + 1):
                        if idx + offset < seq_length and enhanced_labels[idx + offset] == 0:
                            enhanced_labels[idx + offset] = 0.2 / offset
            
            return enhanced_labels
            
        except Exception as e:
            print(f"Error in context enhancement: {e}")
            return span_labels.float()
    
    def _extract_syntax_features(self, text: str, max_length: int) -> torch.Tensor:
        """
        Extract syntax features using spaCy
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Syntax features tensor [max_length, feature_dim]
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
            
            # Extract features
            pos_tags = [token.pos_ for token in doc]
            deps = [token.dep_ for token in doc]
            ner_tags = [token.ent_type_ if token.ent_type_ else "O" for token in doc]
            
            # Create vocabulary mappings
            pos_set = set(pos_tags + ["UNK"])
            dep_set = set(deps + ["UNK"])
            ner_set = set(ner_tags + ["UNK"])
            
            pos_map = {tag: i for i, tag in enumerate(sorted(pos_set))}
            dep_map = {dep: i for i, dep in enumerate(sorted(dep_set))}
            ner_map = {tag: i for i, tag in enumerate(sorted(ner_set))}
            
            # Calculate feature dimensions
            feature_dim = len(pos_set) + len(dep_set) + len(ner_set) + 1  # +1 for head distance
            
            # Initialize feature tensor
            features = torch.zeros(max_length, feature_dim)
            
            # Fill in features for actual tokens
            for i, token in enumerate(doc):
                if i >= max_length:
                    break
                    
                feature_idx = 0
                
                # POS features (one-hot)
                pos_idx = pos_map.get(token.pos_, pos_map["UNK"])
                features[i, feature_idx + pos_idx] = 1.0
                feature_idx += len(pos_set)
                
                # Dependency features (one-hot)
                dep_idx = dep_map.get(token.dep_, dep_map["UNK"])
                features[i, feature_idx + dep_idx] = 1.0
                feature_idx += len(dep_set)
                
                # NER features (one-hot)
                ner_idx = ner_map.get(token.ent_type_ if token.ent_type_ else "O", ner_map["UNK"])
                features[i, feature_idx + ner_idx] = 1.0
                feature_idx += len(ner_set)
                
                # Head distance (normalized)
                head_dist = abs(token.i - token.head.i) / len(doc) if len(doc) > 0 else 0.0
                features[i, feature_idx] = head_dist
            
            return features
            
        except Exception as e:
            print(f"Error extracting syntax features: {e}")
            # Return fallback features
            return torch.zeros(max_length, 64)
        
    def preprocess_for_inference(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess a text for inference without span labels
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with preprocessed tensors
        """
        try:
            # Tokenize text
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Get syntax features if enabled
            syntax_features = None
            if self.use_syntax and self.nlp is not None:
                try:
                    syntax_features = self._extract_syntax_features(text, self.max_length)
                    syntax_features = syntax_features.unsqueeze(0)  # Add batch dimension
                except Exception as e:
                    print(f"Warning: Failed to extract syntax features for inference: {e}")
                    syntax_features = torch.zeros(1, self.max_length, 64)
            
            # Create output dictionary
            output = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
            }
            
            # Add syntax features
            if syntax_features is not None:
                output['syntax_features'] = syntax_features
            else:
                output['syntax_features'] = torch.zeros(1, self.max_length, 64)
                
            return output
            
        except Exception as e:
            print(f"Error in inference preprocessing: {e}")
            # Return safe fallback
            batch_size = 1
            return {
                'input_ids': torch.zeros(batch_size, self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(batch_size, self.max_length, dtype=torch.long),
                'syntax_features': torch.zeros(batch_size, self.max_length, 64)
            }
    
    def _create_fallback_output(self) -> Dict[str, torch.Tensor]:
        """Create safe fallback output when preprocessing fails"""
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'aspect_labels': torch.zeros(1, self.max_length, dtype=torch.long),
            'opinion_labels': torch.zeros(1, self.max_length, dtype=torch.long),
            'sentiment_labels': torch.zeros(1, dtype=torch.long),
            'num_spans': torch.tensor(0, dtype=torch.long),
            'syntax_features': torch.zeros(self.max_length, 64)
        }