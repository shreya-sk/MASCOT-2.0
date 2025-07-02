# src/models/complete_implicit_detector.py
"""
Complete Implicit Sentiment Detection System
Integrates implicit aspect/opinion detection with contrastive learning framework
Implements state-of-the-art EMNLP 2024 approaches for implicit sentiment analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class ImplicitAspectDetector(nn.Module):
    """
    Advanced implicit aspect detection with contextual span interactions
    Implements Grid-based Multi-Task Learning (GM-GTM) approach
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Multi-head attention for contextual modeling
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Bi-directional contextual interaction layers
        self.contextual_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # Grid-based tagging matrix (GM-GTM approach)
        self.grid_projector = nn.Linear(hidden_size, hidden_size)
        self.grid_classifier = nn.Linear(hidden_size, 4)  # O, B-ASP, I-ASP, implicit
        
        # Implicit aspect classification
        self.implicit_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # explicit, implicit, none
        )
        
        # Sentiment combination vectors (EMNLP 2024)
        self.sentiment_combiner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # pos, neu, neg sentiment combinations
        )
        
        # Confidence scoring
        self.confidence_scorer = nn.Linear(hidden_size, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Implicit aspect categories for better classification
        self.aspect_categories = {
            'service': ['staff', 'waiter', 'service', 'waitress', 'server', 'employee'],
            'food': ['dish', 'meal', 'cuisine', 'flavor', 'taste', 'portion'],
            'ambiance': ['atmosphere', 'vibe', 'mood', 'setting', 'environment'],
            'price': ['cost', 'price', 'value', 'expensive', 'cheap', 'affordable'],
            'quality': ['quality', 'standard', 'grade', 'level', 'condition']
        }
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                explicit_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for implicit aspect detection
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] 
            explicit_features: Optional explicit aspect features for combination
            
        Returns:
            Dictionary containing implicit aspect detection outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Contextual interaction modeling
        contextualized_states = hidden_states
        for layer in self.contextual_layers:
            contextualized_states = layer(contextualized_states, src_key_padding_mask=~attention_mask.bool())
        
        # 2. Cross-attention with explicit features if available
        if explicit_features is not None:
            attended_states, _ = self.context_attention(
                query=contextualized_states,
                key=explicit_features,
                value=explicit_features,
                key_padding_mask=~attention_mask.bool()
            )
            contextualized_states = contextualized_states + attended_states
        
        # Apply layer normalization
        contextualized_states = self.layer_norm(contextualized_states)
        
        # 3. Grid-based tagging matrix (GM-GTM)
        grid_features = self.grid_projector(contextualized_states)
        grid_logits = self.grid_classifier(grid_features)  # [batch, seq_len, 4]
        
        # 4. Implicit aspect classification  
        implicit_aspect_scores = self.implicit_classifier(contextualized_states)  # [batch, seq_len, 3]
        
        # 5. Sentiment combination vectors
        if explicit_features is not None:
            combined_features = torch.cat([contextualized_states, explicit_features], dim=-1)
            sentiment_combinations = self.sentiment_combiner(combined_features)
        else:
            # Use self-attention for sentiment combination when no explicit features
            repeated_states = contextualized_states.unsqueeze(2).repeat(1, 1, seq_len, 1)
            transposed_states = contextualized_states.unsqueeze(1).repeat(1, seq_len, 1, 1)
            pairwise_features = torch.cat([repeated_states, transposed_states], dim=-1)
            sentiment_combinations = self.sentiment_combiner(pairwise_features).mean(dim=2)
        
        # 6. Confidence scoring
        confidence_scores = self.confidence_scorer(contextualized_states).squeeze(-1)
        confidence_scores = confidence_scores * attention_mask.float()
        
        return {
            'hidden_states': contextualized_states,
            'grid_logits': grid_logits,
            'implicit_aspect_scores': implicit_aspect_scores,
            'sentiment_combinations': sentiment_combinations,
            'confidence_scores': confidence_scores
        }


class ImplicitOpinionDetector(nn.Module):
    """
    Advanced implicit opinion detection with pattern recognition
    Implements contrastive learning for implicit-explicit opinion alignment
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Pattern recognition networks
        self.pattern_networks = nn.ModuleDict({
            'comparative': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size)
            ),
            'temporal': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size)
            ),
            'conditional': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size)
            ),
            'evaluative': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        })
        
        # Contextual interaction layers (SCI-Net approach)
        self.contextual_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # Cross-task attention for aspect-opinion interaction
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Opinion classification with implicit detection
        self.opinion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 4)  # O, B-OPN, I-OPN, implicit
        )
        
        # Implicit opinion scoring
        self.implicit_opinion_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # explicit, implicit, none
        )
        
        # Pattern-based sentiment inference
        self.pattern_sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size * len(self.pattern_networks), hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3)  # pos, neu, neg
        )
        
        # Implicit opinion patterns for detection
        self.opinion_patterns = {
            'positive_implicit': ['recommend', 'worth', 'love', 'enjoy', 'appreciate', 'amazing', 'great', 'excellent'],
            'negative_implicit': ['regret', 'disappointed', 'terrible', 'awful', 'horrible', 'waste', 'avoid'],
            'comparative': ['better', 'worse', 'superior', 'inferior', 'prefer', 'rather', 'instead'],
            'temporal': ['used to', 'before', 'previously', 'now', 'currently', 'lately'],
            'conditional': ['if', 'unless', 'should', 'would', 'could', 'might']
        }
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                aspect_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for implicit opinion detection
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            aspect_context: Optional aspect context for cross-task interaction
            
        Returns:
            Dictionary containing implicit opinion detection outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. Contextual interaction modeling
        contextualized_states = hidden_states
        for layer in self.contextual_layers:
            contextualized_states = layer(contextualized_states, src_key_padding_mask=~attention_mask.bool())
        
        # 2. Cross-attention with aspect context if available
        if aspect_context is not None:
            cross_attended, _ = self.cross_attention(
                query=contextualized_states,
                key=aspect_context,
                value=aspect_context,
                key_padding_mask=~attention_mask.bool()
            )
            contextualized_states = contextualized_states + cross_attended
        
        # 3. Pattern recognition processing
        pattern_features = []
        for pattern_name, pattern_network in self.pattern_networks.items():
            pattern_output = pattern_network(contextualized_states)
            pattern_features.append(pattern_output)
        
        # Combine pattern features
        combined_pattern_features = torch.cat(pattern_features, dim=-1)
        pattern_outputs = self.pattern_sentiment_classifier(combined_pattern_features)
        
        # 4. Opinion classification
        opinion_logits = self.opinion_classifier(contextualized_states)
        
        # 5. Implicit opinion scoring
        implicit_opinion_scores = self.implicit_opinion_scorer(contextualized_states)
        
        return {
            'contextualized_features': contextualized_states,
            'opinion_logits': opinion_logits,
            'implicit_opinion_scores': implicit_opinion_scores,
            'pattern_outputs': pattern_outputs,
            'context_outputs': combined_pattern_features
        }


class CompleteImplicitDetector(nn.Module):
    """
    Complete implicit sentiment detection system
    Integrates aspect and opinion detection with contrastive learning
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Core components
        self.implicit_aspect_detector = ImplicitAspectDetector(
            hidden_size=self.hidden_size,
            num_heads=getattr(config, 'num_attention_heads', 8),
            num_layers=getattr(config, 'implicit_layers', 3),
            dropout=getattr(config, 'dropout', 0.1)
        )
        
        self.implicit_opinion_detector = ImplicitOpinionDetector(
            hidden_size=self.hidden_size,
            num_heads=getattr(config, 'num_attention_heads', 8),
            num_layers=getattr(config, 'implicit_layers', 2),
            dropout=getattr(config, 'dropout', 0.1)
        )
        
        # Implicit-explicit combination analysis
        self.implicit_explicit_combiner = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, 4)  # none, implicit_only, explicit_only, combined
        )
        
        # Overall confidence scoring
        self.confidence_scorer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Contrastive projection heads for alignment
        self.aspect_projector = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        self.opinion_projector = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                explicit_aspect_features: Optional[torch.Tensor] = None,
                explicit_opinion_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass for implicit sentiment detection
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            explicit_aspect_features: Optional explicit aspect features
            explicit_opinion_features: Optional explicit opinion features
            
        Returns:
            Dictionary containing all implicit detection outputs
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
        
        # 5. Contrastive projections for alignment
        aspect_projections = self.aspect_projector(aspect_outputs['hidden_states'])
        opinion_projections = self.opinion_projector(opinion_outputs['contextualized_features'])
        
        return {
            # Aspect outputs
            'implicit_aspect_scores': aspect_outputs['implicit_aspect_scores'],
            'aspect_sentiment_combinations': aspect_outputs['sentiment_combinations'],
            'aspect_grid_logits': aspect_outputs['grid_logits'],
            'aspect_projections': aspect_projections,
            
            # Opinion outputs  
            'implicit_opinion_scores': opinion_outputs['implicit_opinion_scores'],
            'opinion_logits': opinion_outputs['opinion_logits'],
            'opinion_contextualized_features': opinion_outputs['contextualized_features'],
            'pattern_outputs': opinion_outputs['pattern_outputs'],
            'context_outputs': opinion_outputs['context_outputs'],
            'opinion_projections': opinion_projections,
            
            # Combined outputs
            'combination_logits': combination_logits,
            'confidence_scores': confidence_scores,
            'enhanced_hidden_states': opinion_outputs['contextualized_features']
        }
    
    def extract_implicit_elements(self,
                                input_ids: torch.Tensor,
                                outputs: Dict[str, torch.Tensor],
                                tokenizer,
                                aspect_threshold: float = 0.5,
                                opinion_threshold: float = 0.5) -> Dict[str, List[Dict]]:
        """
        Extract implicit aspects and opinions from model outputs
        
        Args:
            input_ids: [batch_size, seq_len]
            outputs: Model outputs containing scores
            tokenizer: Tokenizer for text conversion
            aspect_threshold: Threshold for implicit aspect detection
            opinion_threshold: Threshold for implicit opinion detection
            
        Returns:
            Dictionary containing extracted implicit elements
        """
        batch_size = input_ids.size(0)
        results = {'implicit_aspects': [], 'implicit_opinions': []}
        
        # Extract implicit aspects
        aspect_scores = outputs['implicit_aspect_scores']  # [batch, seq_len, 3]
        implicit_aspect_probs = F.softmax(aspect_scores, dim=-1)[:, :, 1]  # implicit class
        
        # Extract implicit opinions
        opinion_scores = outputs['implicit_opinion_scores']  # [batch, seq_len, 3]
        implicit_opinion_probs = F.softmax(opinion_scores, dim=-1)[:, :, 1]  # implicit class
        
        for b in range(batch_size):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
            
            # Find implicit aspects
            aspect_positions = torch.where(implicit_aspect_probs[b] > aspect_threshold)[0]
            implicit_aspects = self._extract_spans(
                tokens, aspect_positions.cpu().numpy(), tokenizer, 'aspect'
            )
            results['implicit_aspects'].append(implicit_aspects)
            
            # Find implicit opinions
            opinion_positions = torch.where(implicit_opinion_probs[b] > opinion_threshold)[0]
            implicit_opinions = self._extract_spans(
                tokens, opinion_positions.cpu().numpy(), tokenizer, 'opinion'
            )
            results['implicit_opinions'].append(implicit_opinions)
        
        return results
    
    def _extract_spans(self, tokens: List[str], positions: np.ndarray, 
                      tokenizer, span_type: str) -> List[Dict[str, Any]]:
        """Extract and validate spans from token positions"""
        if len(positions) == 0:
            return []
        
        spans = []
        
        # Group consecutive positions
        grouped_spans = self._group_consecutive_positions(positions)
        
        for start, end in grouped_spans:
            # Extract span text
            span_tokens = tokens[start:end+1]
            span_text = tokenizer.convert_tokens_to_string(span_tokens).strip()
            
            # Validate span
            if self._is_valid_span(span_text, span_type):
                spans.append({
                    'text': span_text,
                    'start': int(start),
                    'end': int(end),
                    'type': f'implicit_{span_type}',
                    'tokens': span_tokens
                })
        
        return spans
    
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
    
    def _is_valid_span(self, text: str, span_type: str) -> bool:
        """Validate extracted spans"""
        text = text.strip()
        
        # Basic validation
        if len(text) < 2 or text in ['[PAD]', '[CLS]', '[SEP]', '.', ',', '!', '?']:
            return False
        
        # Type-specific validation
        if span_type == 'aspect':
            # Check for aspect indicators
            aspect_words = ['food', 'service', 'place', 'staff', 'atmosphere', 'price', 'quality']
            return any(word in text.lower() for word in aspect_words) or len(text.split()) <= 3
        
        elif span_type == 'opinion':
            # Check for opinion indicators
            opinion_words = ['good', 'bad', 'great', 'terrible', 'love', 'hate', 'recommend', 'avoid']
            return any(word in text.lower() for word in opinion_words) or len(text.split()) <= 4
        
        return True


def create_complete_implicit_detector(config) -> CompleteImplicitDetector:
    """Factory function to create complete implicit detector"""
    return CompleteImplicitDetector(config)