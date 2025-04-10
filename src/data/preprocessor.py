from typing import List, Dict, Any
import torch # type: ignore # type: ignore
from transformers import PreTrainedTokenizer
import spacy # type: ignore
from spacy.tokens import Doc # type: ignore # type: ignore
import numpy as np

# Fix this import to include SpanLabel
from src.data.utils import SENTIMENT_MAP, convert_to_bio_labels, SpanLabel

class LLMABSAPreprocessor:
    """
    Enhanced preprocessor for ABSA data using Stella tokenization
    with syntax-aware features and span context enhancement
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
                print("Loaded spaCy for syntax-aware features")
            except Exception as e:
                print(f"Could not load spaCy: {e}")
                print("Falling back to syntax-free processing")
                self.use_syntax = False

    def preprocess(self, text: str, span_labels: List[SpanLabel]) -> Dict[str, torch.Tensor]:
        """Preprocess a single instance with enhanced features"""
        
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
        
        # Create span labels
        aspect_labels_list = []
        opinion_labels_list = []
        sentiment_labels_list = []
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for span_label in span_labels:
            # Convert to BIO scheme
            aspect_labels = convert_to_bio_labels(
                span_label.aspect_indices, 
                self.max_length
            )
            opinion_labels = convert_to_bio_labels(
                span_label.opinion_indices,
                self.max_length
            )
            
            # Create context-enhanced span labels
            # This improves detection by considering surrounding context
            aspect_labels = self._enhance_span_context(aspect_labels)
            opinion_labels = self._enhance_span_context(opinion_labels)
            
            # Convert sentiment to label id
            sentiment_label = torch.tensor(SENTIMENT_MAP[span_label.sentiment])
            
            aspect_labels_list.append(aspect_labels)
            opinion_labels_list.append(opinion_labels)
            sentiment_labels_list.append(sentiment_label)
        
        # Get syntax features if enabled
        syntax_features = None
        if self.use_syntax and self.nlp is not None:
            syntax_features = self._extract_syntax_features(text, self.max_length)
        
        # Create the output dictionary
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'aspect_labels': torch.stack(aspect_labels_list) if aspect_labels_list else torch.zeros((0, self.max_length)),
            'opinion_labels': torch.stack(opinion_labels_list) if opinion_labels_list else torch.zeros((0, self.max_length)),
            'sentiment_labels': torch.stack(sentiment_labels_list) if sentiment_labels_list else torch.zeros(0),
            'num_spans': len(span_labels)
        }
        
        if syntax_features is not None:
            output['syntax_features'] = syntax_features
            
        return output
    
    def _enhance_span_context(self, span_labels: torch.Tensor) -> torch.Tensor:
        """
        Enhance span labels by considering surrounding context
        
        This creates soft boundary labels that give partial weight to
        tokens surrounding the actual spans, improving boundary detection.
        """
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
                    # Add soft label with decreasing weight
                    enhanced_labels[idx - offset] = 0.3 / offset
                
                # Context after span start
                if idx + offset < seq_length and idx + offset not in i_indices and enhanced_labels[idx + offset] == 0:
                    enhanced_labels[idx + offset] = 0.2 / offset
        
        # Add context after ending tags
        for idx in i_indices:
            # Check if this is an ending tag (no I tag follows it)
            if idx + 1 >= seq_length or idx + 1 not in i_indices:
                for offset in range(1, self.context_window + 1):
                    if idx + offset < seq_length and enhanced_labels[idx + offset] == 0:
                        enhanced_labels[idx + offset] = 0.2 / offset
        
        return enhanced_labels
    
    def _extract_syntax_features(self, text: str, max_length: int) -> torch.Tensor:
        """
        Extract syntax features using spaCy
        
        Returns a tensor of shape [max_length, feature_dim] containing:
        - POS tag embeddings
        - Dependency relation embeddings
        - Parse tree depth information
        - Named entity information
        """
        # Process text with spaCy
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        
        # Extract features
        pos_tags = [token.pos_ for token in doc]
        deps = [token.dep_ for token in doc]
        head_indices = [token.head.i for token in doc]
        ner_tags = [token.ent_type_ if token.ent_type_ else "O" for token in doc]
        
        # Convert to one-hot embeddings
        pos_set = set(pos_tags)
        dep_set = set(deps)
        ner_set = set(ner_tags)
        
        pos_map = {tag: i for i, tag in enumerate(pos_set)}
        dep_map = {dep: i for i, dep in enumerate(dep_set)}
        ner_map = {tag: i for i, tag in enumerate(ner_set)}
        
        # Initialize feature tensor
        feature_dim = len(pos_set) + len(dep_set) + 1 + len(ner_set)  # POS + DEP + HEAD_DIST + NER
        features = torch.zeros(len(tokens), feature_dim)
        
        # Fill in features
        for i, token in enumerate(doc):
            # POS features
            pos_idx = pos_map[token.pos_]
            features[i, pos_idx] = 1.0
            
            # Dependency features
            dep_idx = dep_map[token.dep_]
            features[i, len(pos_set) + dep_idx] = 1.0
            
            # Head distance (normalized)
            head_dist = abs(token.i - token.head.i) / len(tokens)
            features[i, len(pos_set) + len(dep_set)] = head_dist
            
            # NER features
            if token.ent_type_:
                ner_idx = ner_map[token.ent_type_]
                features[i, len(pos_set) + len(dep_set) + 1 + ner_idx] = 1.0
        
        # Resize to max_length
        if len(tokens) > max_length:
            # Truncate
            features = features[:max_length]
        elif len(tokens) < max_length:
            # Pad with zeros
            padding = torch.zeros(max_length - len(tokens), feature_dim)
            features = torch.cat([features, padding], dim=0)
        
        return features
        
    def preprocess_for_inference(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess a text for inference without span labels"""
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
            syntax_features = self._extract_syntax_features(text, self.max_length)
            syntax_features = syntax_features.unsqueeze(0)  # Add batch dimension
        
        # Create output dictionary
        output = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }
        
        if syntax_features is not None:
            output['syntax_features'] = syntax_features
            
        return output