#!/usr/bin/env python3
"""
GRADIENT Preprocessor
Advanced preprocessing for gradient reversal and domain adaptation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoTokenizer
import re

class ABSAPreprocessor:
    """
    GRADIENT ABSA Preprocessor with domain-aware features
    Handles text preprocessing for gradient reversal training
    """
    
    def __init__(self, config, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = getattr(config, 'max_seq_length', 128)
        
        # Domain mappings for gradient reversal
        self.domain_mappings = {
            'laptop14': 0,
            'rest14': 1,
            'rest15': 1,  # Same as rest14
            'rest16': 1,  # Same as rest14
            'hotel': 2,
            'electronics': 3
        }
        
        # Try to load spaCy if available
        self.use_spacy = False
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
            self.use_spacy = True
            print("✓ SpaCy loaded for advanced syntax features")
        except:
            print("⚠ SpaCy not available, using basic preprocessing")
            self.nlp = None
    
    def preprocess(self, text: str, labels: List[Dict], dataset_name: str = 'laptop14') -> Dict[str, torch.Tensor]:
        """
        Preprocess text and labels for GRADIENT training
        
        Args:
            text: Input text
            labels: List of aspect-opinion-sentiment triplets
            dataset_name: Dataset name for domain identification
            
        Returns:
            Dictionary of preprocessed tensors
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get basic tensors
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        # Create aspect and opinion labels
        aspect_labels, opinion_labels = self._create_sequence_labels(text, labels)
        result['aspect_labels'] = aspect_labels
        result['opinion_labels'] = opinion_labels
        
        # Create sentiment labels
        sentiment_labels = self._create_sentiment_labels(labels)
        result['sentiment_labels'] = sentiment_labels
        
        # Add domain information for gradient reversal
        domain_id = self.domain_mappings.get(dataset_name, 0)
        result['domain_id'] = torch.tensor(domain_id, dtype=torch.long)
        
        # Add syntax features if SpaCy available
        if self.use_spacy:
            syntax_features = self._extract_syntax_features(text)
            result['syntax_features'] = syntax_features
        else:
            # Create dummy syntax features
            result['syntax_features'] = torch.zeros(self.max_seq_length, 16)
        
        # Add span information
        result['num_spans'] = torch.tensor(len(labels), dtype=torch.long)
        
        return result
    
    def _create_sequence_labels(self, text: str, labels: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create BIO-style sequence labels for aspects and opinions"""
        tokens = self.tokenizer.tokenize(text)
        
        # Initialize labels
        aspect_labels = torch.zeros(2, self.max_seq_length, dtype=torch.long)  # [num_label_types, seq_len]
        opinion_labels = torch.zeros(2, self.max_seq_length, dtype=torch.long)
        
        # Simple approach: mark first few tokens for each aspect/opinion
        for i, label in enumerate(labels[:2]):  # Limit to first 2 for simplicity
            if 'aspect' in label:
                aspect_labels[i, :min(5, len(tokens))] = 1  # Mark first 5 tokens
            if 'opinion' in label:
                opinion_labels[i, :min(5, len(tokens))] = 1
        
        return aspect_labels, opinion_labels
    
    def _create_sentiment_labels(self, labels: List[Dict]) -> torch.Tensor:
        """Create sentiment labels"""
        sentiment_map = {'POS': 0, 'NEG': 1, 'NEU': 2, 'positive': 0, 'negative': 1, 'neutral': 2}
        
        sentiments = []
        for label in labels[:2]:  # Limit to first 2
            sentiment = label.get('sentiment', 'NEU')
            sentiments.append(sentiment_map.get(sentiment, 2))
        
        # Pad to length 2
        while len(sentiments) < 2:
            sentiments.append(2)  # Neutral
        
        return torch.tensor(sentiments[:2], dtype=torch.long)
    
    def _extract_syntax_features(self, text: str) -> torch.Tensor:
        """Extract syntax features using SpaCy"""
        if not self.use_spacy:
            return torch.zeros(self.max_seq_length, 16)
        
        doc = self.nlp(text)
        features = []
        
        for token in doc[:self.max_seq_length]:
            # Create feature vector for each token
            feature_vec = [
                1.0 if token.is_alpha else 0.0,      # is_alpha
                1.0 if token.is_digit else 0.0,      # is_digit
                1.0 if token.is_punct else 0.0,      # is_punct
                1.0 if token.is_stop else 0.0,       # is_stop
                float(token.pos_ == 'NOUN'),          # is_noun
                float(token.pos_ == 'ADJ'),           # is_adj
                float(token.pos_ == 'VERB'),          # is_verb
                float(token.pos_ == 'ADV'),           # is_adv
                float(token.dep_ == 'nsubj'),         # is_subject
                float(token.dep_ == 'dobj'),          # is_object
                float(token.dep_ == 'amod'),          # is_modifier
                float(token.dep_ == 'compound'),      # is_compound
                float(len(token.text)),               # token_length
                float(token.i),                       # position
                1.0 if token.ent_type_ else 0.0,     # has_entity
                float(token.sentiment),               # sentiment_score
            ]
            features.append(feature_vec)
        
        # Pad to max_seq_length
        while len(features) < self.max_seq_length:
            features.append([0.0] * 16)
        
        return torch.tensor(features[:self.max_seq_length], dtype=torch.float32)

class GRADIENTPreprocessor(ABSAPreprocessor):
    """Alias for GRADIENT branding"""
    pass

# Backward compatibility
preprocessor = ABSAPreprocessor
