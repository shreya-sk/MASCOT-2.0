# src/models/implicit_detector.py - Complete Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import re
import numpy as np


class ImplicitPatternDetector(nn.Module):
    """Detects implicit opinion patterns using linguistic rules and neural networks"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        # Pattern embedding layers
        self.pattern_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Attention for pattern focus
        self.pattern_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Pattern classification
        self.pattern_classifier = nn.Linear(hidden_size, 4)  # Direct, Indirect, Comparative, Implicit
        
        # Linguistic pattern rules
        self.implicit_patterns = {
            'comparative': ['better', 'worse', 'more', 'less', 'superior', 'inferior', 'compared to'],
            'expectation': ['expected', 'supposed to', 'should be', 'would be', 'disappointing', 'surprising'],
            'emotional': ['love', 'hate', 'enjoy', 'dislike', 'appreciate', 'regret', 'frustrated'],
            'evaluative': ['worth', 'value', 'quality', 'standard', 'level', 'grade', 'recommend']
        }
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Pattern encoding
        pattern_features = self.pattern_encoder(hidden_states)
        
        # Self-attention for pattern focus
        attended_features, attention_weights = self.pattern_attention(
            pattern_features, pattern_features, pattern_features,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Pattern classification
        pattern_logits = self.pattern_classifier(attended_features)
        
        return {
            'pattern_features': attended_features,
            'pattern_logits': pattern_logits,
            'attention_weights': attention_weights
        }


class ContextualOpinionScorer(nn.Module):
    """Scores contextual opinion expressions for implicit detection"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        # Context encoding
        self.context_encoder = nn.LSTM(
            hidden_size, hidden_size // 2, num_layers=2,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        
        # Opinion scoring layers
        self.opinion_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Context encoding
        context_features, _ = self.context_encoder(hidden_states)
        
        # Opinion scoring
        opinion_scores = self.opinion_scorer(context_features).squeeze(-1)
        
        # Apply attention mask
        opinion_scores = opinion_scores * attention_mask.float()
        
        return {
            'context_features': context_features,
            'opinion_scores': opinion_scores
        }


class ImplicitAspectDetector(nn.Module):
    """
    Complete implicit aspect detection following 2024-2025 breakthrough standards
    Implements Grid Tagging Matching (GM-GTM) and sentiment combination vectors
    """
    
    def __init__(self, hidden_size: int, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Grid tagging matrix for relationships
        self.grid_tagger = nn.Linear(hidden_size, num_classes * num_classes)
        
        # Sentiment combination vectors (4 fully connected layers as per EMNLP 2024)
        self.sentiment_combiner = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # Implicit-explicit combination classifier
        self.combination_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4)  # Explicit-Explicit, Explicit-Implicit, Implicit-Explicit, Implicit-Implicit
        )
        
        # Aspect candidate scorer
        self.aspect_scorer = nn.Linear(hidden_size, 1)
        
        # Implicit detection patterns
        self.implicit_indicators = {
            'implicit_aspects': [
                'it', 'this', 'that', 'place', 'restaurant', 'spot', 'experience',
                'meal', 'visit', 'time', 'everything', 'overall', 'general'
            ],
            'aspect_categories': {
                'food': ['taste', 'flavor', 'quality', 'freshness', 'temperature'],
                'service': ['staff', 'waiter', 'waitress', 'server', 'attention'],
                'ambiance': ['atmosphere', 'environment', 'setting', 'mood', 'vibe'],
                'price': ['cost', 'expensive', 'cheap', 'affordable', 'value'],
                'location': ['place', 'spot', 'location', 'convenient', 'accessible']
            }
        }
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                explicit_aspects: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for implicit aspect detection
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            explicit_aspects: Optional explicit aspect representations
            
        Returns:
            Dictionary with implicit detection results
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Grid tagging matrix computation
        grid_logits = self.grid_tagger(hidden_states)  # [batch_size, seq_len, num_classes^2]
        grid_logits = grid_logits.view(batch_size, seq_len, self.num_classes, self.num_classes)
        
        # 2. Sentiment combination vectors (following EMNLP 2024)
        sentiment_combinations = self.sentiment_combiner(hidden_states)  # [batch_size, seq_len, num_classes]
        
        # 3. Implicit aspect scoring
        implicit_scores = self.aspect_scorer(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        implicit_scores = torch.sigmoid(implicit_scores) * attention_mask.float()
        
        # 4. Implicit-explicit combination classification
        combination_logits = None
        if explicit_aspects is not None:
            # Combine implicit and explicit representations
            combined_repr = torch.cat([hidden_states, explicit_aspects], dim=-1)
            combination_logits = self.combination_classifier(combined_repr)
        
        # 5. Apply attention mask to all outputs
        sentiment_combinations = sentiment_combinations * attention_mask.unsqueeze(-1)
        grid_logits = grid_logits * attention_mask.unsqueeze(-1).unsqueeze(-1)
        
        return {
            'implicit_aspect_scores': implicit_scores,
            'sentiment_combinations': sentiment_combinations,
            'grid_logits': grid_logits,
            'combination_logits': combination_logits,
            'hidden_states': hidden_states
        }
    
    def extract_implicit_aspects(self,
                                input_ids: torch.Tensor,
                                hidden_states: torch.Tensor,
                                implicit_scores: torch.Tensor,
                                tokenizer,
                                threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Extract implicit aspects using scores and linguistic patterns"""
        
        implicit_aspects = []
        batch_size = input_ids.size(0)
        
        for batch_idx in range(batch_size):
            # Get tokens for this sample
            tokens = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
            scores = implicit_scores[batch_idx].cpu().numpy()
            
            # Find high-scoring implicit regions
            implicit_positions = np.where(scores > threshold)[0]
            
            if len(implicit_positions) == 0:
                continue
                
            # Group consecutive positions into spans
            spans = self._group_consecutive_positions(implicit_positions)
            
            for span_start, span_end in spans:
                # Extract span text
                span_tokens = tokens[span_start:span_end+1]
                span_text = tokenizer.convert_tokens_to_string(span_tokens).strip()
                
                # Check if it's a valid implicit aspect
                if self._is_valid_implicit_aspect(span_text, tokens):
                    implicit_aspects.append({
                        'text': span_text,
                        'start': span_start,
                        'end': span_end,
                        'confidence': float(scores[span_start:span_end+1].mean()),
                        'type': 'implicit_aspect',
                        'category': self._categorize_aspect(span_text)
                    })
        
        return implicit_aspects
    
    def _group_consecutive_positions(self, positions: np.ndarray) -> List[Tuple[int, int]]:
        """Group consecutive positions into spans"""
        if len(positions) == 0:
            return []
        
        spans = []
        start = positions[0]
        prev = positions[0]
        
        for pos in positions[1:]:
            if pos - prev > 1:  # Gap found
                spans.append((start, prev))
                start = pos
            prev = pos
        
        spans.append((start, prev))
        return spans
    
    def _is_valid_implicit_aspect(self, text: str, all_tokens: List[str]) -> bool:
        """Check if text is a valid implicit aspect"""
        text_lower = text.lower()
        
        # Check if it's in implicit indicators
        if any(indicator in text_lower for indicator in self.implicit_indicators['implicit_aspects']):
            return True
        
        # Check if it's too short or contains only punctuation
        if len(text.strip()) < 2 or text.strip() in ['[PAD]', '[CLS]', '[SEP]']:
            return False
        
        # Check if it's a pronoun or demonstrative that could refer to an aspect
        pronouns = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
        if text_lower in pronouns:
            return True
        
        return False
    
    def _categorize_aspect(self, text: str) -> str:
        """Categorize aspect into predefined categories"""
        text_lower = text.lower()
        
        for category, keywords in self.implicit_indicators['aspect_categories'].items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general'


class ImplicitOpinionDetector(nn.Module):
    """
    Complete implicit opinion detection with contrastive learning
    Implements span-level contextual interactions (SCI-Net) approach
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Bi-directional contextual interaction layers (SCI-Net)
        self.contextual_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Linear projection for task-oriented representations
        self.task_projection = nn.Linear(hidden_size, hidden_size)
        
        # Cross-task attention for aspect-opinion interaction
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Opinion classification with implicit detection
        self.opinion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # B-I-O for opinions
        )
        
        # Implicit opinion scorer
        self.implicit_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Pattern detector and context scorer
        self.pattern_detector = ImplicitPatternDetector(hidden_size, dropout)
        self.context_scorer = ContextualOpinionScorer(hidden_size, dropout)
        
        # Implicit opinion patterns
        self.implicit_opinion_patterns = {
            'positive_implicit': [
                'recommend', 'worth', 'appreciate', 'enjoy', 'love', 'amazing',
                'excellent', 'perfect', 'wonderful', 'fantastic', 'great'
            ],
            'negative_implicit': [
                'disappointed', 'regret', 'waste', 'avoid', 'terrible', 'awful',
                'horrible', 'disgusting', 'unacceptable', 'disappointing'
            ],
            'comparative': [
                'better than', 'worse than', 'compared to', 'unlike', 'different from',
                'similar to', 'as good as', 'not as good as'
            ]
        }
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                aspect_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for implicit opinion detection
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            aspect_features: Optional aspect representations for cross-attention
            
        Returns:
            Dictionary with implicit opinion detection results
        """
        
        # 1. Bi-directional contextual interactions (SCI-Net)
        contextualized_states = hidden_states
        for layer in self.contextual_layers:
            contextualized_states = layer(
                contextualized_states, 
                src_key_padding_mask=~attention_mask.bool()
            )
        
        # 2. Task-oriented projection
        task_features = self.task_projection(contextualized_states)
        
        # 3. Cross-task attention with aspects
        if aspect_features is not None:
            attended_features, cross_attention_weights = self.cross_attention(
                task_features, aspect_features, aspect_features,
                key_padding_mask=~attention_mask.bool()
            )
            # Residual connection
            task_features = task_features + attended_features
        else:
            cross_attention_weights = None
        
        # 4. Pattern detection
        pattern_outputs = self.pattern_detector(task_features, attention_mask)
        
        # 5. Contextual scoring
        context_outputs = self.context_scorer(task_features, attention_mask)
        
        # 6. Combine all features
        enhanced_features = (task_features + 
                           pattern_outputs['pattern_features'] + 
                           context_outputs['context_features'])
        
        # 7. Opinion classification
        opinion_logits = self.opinion_classifier(enhanced_features)
        
        # 8. Implicit opinion scoring
        implicit_scores = self.implicit_scorer(enhanced_features).squeeze(-1)
        implicit_scores = implicit_scores * attention_mask.float()
        
        return {
            'opinion_logits': opinion_logits,
            'implicit_opinion_scores': implicit_scores,
            'contextualized_features': enhanced_features,
            'pattern_outputs': pattern_outputs,
            'context_outputs': context_outputs,
            'cross_attention_weights': cross_attention_weights
        }
    
    def extract_implicit_opinions(self,
                                 input_ids: torch.Tensor,
                                 implicit_scores: torch.Tensor,
                                 pattern_outputs: Dict[str, torch.Tensor],
                                 tokenizer,
                                 threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Extract implicit opinions using scores and patterns"""
        
        implicit_opinions = []
        batch_size = input_ids.size(0)
        
        for batch_idx in range(batch_size):
            # Get tokens and scores for this sample
            tokens = tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
            scores = implicit_scores[batch_idx].cpu().numpy()
            pattern_logits = pattern_outputs['pattern_logits'][batch_idx].cpu().numpy()
            
            # Combine implicit scores with pattern scores
            combined_scores = scores + pattern_logits.max(axis=-1) * 0.3
            
            # Find high-scoring regions
            implicit_positions = np.where(combined_scores > threshold)[0]
            
            if len(implicit_positions) == 0:
                continue
            
            # Group into spans
            spans = self._group_consecutive_positions(implicit_positions)
            
            for span_start, span_end in spans:
                # Extract span text
                span_tokens = tokens[span_start:span_end+1]
                span_text = tokenizer.convert_tokens_to_string(span_tokens).strip()
                
                # Determine sentiment and pattern type
                pattern_type = self._classify_opinion_pattern(span_text)
                sentiment = self._infer_implicit_sentiment(span_text, pattern_type)
                
                if self._is_valid_implicit_opinion(span_text):
                    implicit_opinions.append({
                        'text': span_text,
                        'start': span_start,
                        'end': span_end,
                        'confidence': float(combined_scores[span_start:span_end+1].mean()),
                        'type': 'implicit_opinion',
                        'pattern_type': pattern_type,
                        'sentiment': sentiment
                    })
        
        return implicit_opinions
    
    def _group_consecutive_positions(self, positions: np.ndarray) -> List[Tuple[int, int]]:
        """Group consecutive positions into spans"""
        if len(positions) == 0:
            return []
        
        spans = []
        start = positions[0]
        prev = positions[0]
        
        for pos in positions[1:]:
            if pos - prev > 1:  # Gap found
                spans.append((start, prev))
                start = pos
            prev = pos
        
        spans.append((start, prev))
        return spans
    
    def _classify_opinion_pattern(self, text: str) -> str:
        """Classify the type of implicit opinion pattern"""
        text_lower = text.lower()
        
        for pattern_type, keywords in self.implicit_opinion_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return pattern_type
        
        return 'general'
    
    def _infer_implicit_sentiment(self, text: str, pattern_type: str) -> str:
        """Infer sentiment from implicit opinion text and pattern"""
        text_lower = text.lower()
        
        # Pattern-based sentiment inference
        if pattern_type == 'positive_implicit':
            return 'POS'
        elif pattern_type == 'negative_implicit':
            return 'NEG'
        elif pattern_type == 'comparative':
            # More sophisticated comparative analysis needed
            if any(word in text_lower for word in ['better', 'superior', 'prefer']):
                return 'POS'
            elif any(word in text_lower for word in ['worse', 'inferior', 'disappointing']):
                return 'NEG'
        
        return 'NEU'
    
    def _is_valid_implicit_opinion(self, text: str) -> bool:
        """Check if text represents a valid implicit opinion"""
        text_lower = text.lower()
        
        # Filter out very short spans
        if len(text.strip()) < 2:
            return False
        
        # Filter out punctuation and special tokens
        if text.strip() in ['[PAD]', '[CLS]', '[SEP]', '.', ',', '!', '?']:
            return False
        
        # Check for opinion indicators
        opinion_indicators = ['recommend', 'worth', 'love', 'hate', 'enjoy', 'appreciate', 
                            'regret', 'disappointed', 'amazing', 'terrible', 'great', 'awful']
        
        if any(indicator in text_lower for indicator in opinion_indicators):
            return True
        
        # Check for evaluative language
        evaluative = ['good', 'bad', 'excellent', 'poor', 'wonderful', 'horrible', 
                     'fantastic', 'disappointing', 'impressive', 'unacceptable']
        
        if any(eval_word in text_lower for eval_word in evaluative):
            return True
        
        return False


class CompleteImplicitDetector(nn.Module):
    """
    Complete implicit sentiment detection system integrating aspects and opinions
    Implements instruction tuning-based contrastive learning for implicit-explicit combinations
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Initialize component detectors
        self.implicit_aspect_detector = ImplicitAspectDetector(
            hidden_size=self.hidden_size,
            num_classes=3,
            dropout=config.dropout
        )
        
        self.implicit_opinion_detector = ImplicitOpinionDetector(
            hidden_size=self.hidden_size,
            num_heads=8,
            num_layers=2,
            dropout=config.dropout
        )
        
        # Unified implicit-explicit combiner
        self.implicit_explicit_combiner = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),  # aspect + opinion + context
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, 4)  # 4 combination types
        )
        
        # Confidence scorer for implicit detection
        self.confidence_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                explicit_aspect_features: Optional[torch.Tensor] = None,
                explicit_opinion_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for implicit detection
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            explicit_aspect_features: Optional explicit aspect representations
            explicit_opinion_features: Optional explicit opinion representations
            
        Returns:
            Complete implicit detection results
        """
        
        # 1. Implicit aspect detection
        aspect_outputs = self.implicit_aspect_detector(
            hidden_states, attention_mask, explicit_aspect_features
        )
        
        # 2. Implicit opinion detection (with aspect context)
        opinion_outputs = self.implicit_opinion_detector(
            hidden_states, attention_mask, aspect_outputs['hidden_states']
        )
        
        # 3. Implicit-explicit combination analysis
        combination_logits = None
        if explicit_aspect_features is not None and explicit_opinion_features is not None:
            # Combine all representations
            combined_repr = torch.cat([
                aspect_outputs['hidden_states'],
                opinion_outputs['contextualized_features'],
                (explicit_aspect_features + explicit_opinion_features) / 2
            ], dim=-1)
            
            combination_logits = self.implicit_explicit_combiner(combined_repr)
        
        # 4. Overall confidence scoring
        confidence_features = (aspect_outputs['hidden_states'] + 
                             opinion_outputs['contextualized_features']) / 2
        confidence_scores = self.confidence_scorer(confidence_features).squeeze(-1)
        confidence_scores = confidence_scores * attention_mask.float()
        
        return {
            # Aspect outputs
            'implicit_aspect_scores': aspect_outputs['implicit_aspect_scores'],
            'aspect_sentiment_combinations': aspect_outputs['sentiment_combinations'],
            'aspect_grid_logits': aspect_outputs['grid_logits'],
            
            # Opinion outputs  
            'implicit_opinion_scores': opinion_outputs['implicit_opinion_scores'],
            'opinion_logits': opinion_outputs['opinion_logits'],
            'opinion_contextualized_features': opinion_outputs['contextualized_features'],
            'pattern_outputs': opinion_outputs['pattern_outputs'],
            'context_outputs': opinion_outputs['context_outputs'],
            
            # Combined outputs
            'combination_logits': combination_logits,
            'confidence_scores': confidence_scores,
            'enhanced_hidden_states': opinion_outputs['contextualized_features']
        }
    
    def extract_all_implicit_elements(self,
                                    input_ids: torch.Tensor,
                                    outputs: Dict[str, torch.Tensor],
                                    tokenizer,
                                    aspect_threshold: float = 0.5,
                                    opinion_threshold: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all implicit elements (aspects and opinions) from the outputs
        """
        
        # Extract implicit aspects
        implicit_aspects = self.implicit_aspect_detector.extract_implicit_aspects(
            input_ids, 
            outputs['enhanced_hidden_states'],
            outputs['implicit_aspect_scores'],
            tokenizer,
            aspect_threshold
        )
        
        # Extract implicit opinions
        implicit_opinions = self.implicit_opinion_detector.extract_implicit_opinions(
            input_ids,
            outputs['implicit_opinion_scores'],
            outputs['pattern_outputs'],
            tokenizer,
            opinion_threshold
        )
        
        # Combine into triplets with confidence scores
        implicit_triplets = self._combine_implicit_triplets(
            implicit_aspects, implicit_opinions, outputs['confidence_scores'], input_ids
        )
        
        return {
            'implicit_aspects': implicit_aspects,
            'implicit_opinions': implicit_opinions,
            'implicit_triplets': implicit_triplets,
            'summary': {
                'num_implicit_aspects': len(implicit_aspects),
                'num_implicit_opinions': len(implicit_opinions),
                'num_implicit_triplets': len(implicit_triplets)
            }
        }
    
    def _combine_implicit_triplets(self,
                                 implicit_aspects: List[Dict[str, Any]],
                                 implicit_opinions: List[Dict[str, Any]],
                                 confidence_scores: torch.Tensor,
                                 input_ids: torch.Tensor) -> List[Dict[str, Any]]:
        """Combine implicit aspects and opinions into triplets"""
        
        triplets = []
        
        # Simple proximity-based pairing for now
        for aspect in implicit_aspects:
            best_opinion = None
            best_distance = float('inf')
            
            for opinion in implicit_opinions:
                # Calculate distance between aspect and opinion
                distance = abs(aspect['start'] - opinion['start'])
                
                if distance < best_distance and distance < 10:  # Within 10 tokens
                    best_distance = distance
                    best_opinion = opinion
            
            if best_opinion is not None:
                # Get confidence from the region
                region_start = min(aspect['start'], best_opinion['start'])
                region_end = max(aspect['end'], best_opinion['end'])
                region_confidence = confidence_scores[0][region_start:region_end+1].mean().item()
                
                triplets.append({
                    'aspect': aspect,
                    'opinion': best_opinion,
                    'sentiment': best_opinion.get('sentiment', 'NEU'),
                    'confidence': region_confidence,
                    'type': 'implicit',
                    'distance': best_distance
                })
        
        return triplets