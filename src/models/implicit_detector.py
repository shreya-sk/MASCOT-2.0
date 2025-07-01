# src/models/implicit_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple, Optional

class GridTaggingMatcher(nn.Module):
    """
    Grid Tagging Matching (GM-GTM) for implicit sentiment detection
    
    2024-2025 breakthrough: Uses grid tagging matrices for relationship modeling
    between explicit and implicit aspects/opinions with causality constraints.
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_seq_len = getattr(config, 'max_seq_length', 128)
        self.num_relation_types = getattr(config, 'num_relation_types', 8)
        
        # Grid tagging matrices for different relation types
        self.aspect_aspect_grid = nn.Parameter(
            torch.zeros(self.max_seq_len, self.max_seq_len, self.num_relation_types)
        )
        self.aspect_opinion_grid = nn.Parameter(
            torch.zeros(self.max_seq_len, self.max_seq_len, self.num_relation_types)
        )
        self.opinion_opinion_grid = nn.Parameter(
            torch.zeros(self.max_seq_len, self.max_seq_len, self.num_relation_types)
        )
        
        # Relation type embeddings
        self.relation_embeddings = nn.Embedding(self.num_relation_types, self.hidden_size)
        
        # Grid tag classifiers
        self.aspect_grid_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),  # token1 + token2 + relation
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_relation_types)
        )
        
        self.opinion_grid_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.num_relation_types)
        )
        
        # Implicit detection networks
        self.implicit_aspect_detector = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),  # context + relation
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Implicit-B, Implicit-I, Explicit-B, Explicit-I
        )
        
        self.implicit_opinion_detector = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Implicit-B, Implicit-I, Explicit-B, Explicit-I
        )
        
        # Causality constraints
        self.causality_constraints = CausalityConstraints(config)
        
        # Initialize grids
        self._initialize_grids()
    
    def _initialize_grids(self):
        """Initialize grid matrices with reasonable values"""
        # Initialize with small random values
        nn.init.xavier_normal_(self.aspect_aspect_grid, gain=0.1)
        nn.init.xavier_normal_(self.aspect_opinion_grid, gain=0.1)
        nn.init.xavier_normal_(self.opinion_opinion_grid, gain=0.1)
        
        # Add identity bias for self-relations
        for i in range(min(self.max_seq_len, self.num_relation_types)):
            if i < self.max_seq_len:
                self.aspect_aspect_grid.data[i, i, 0] += 0.5  # Self-relation
                self.opinion_opinion_grid.data[i, i, 0] += 0.5
    
    def forward(self, hidden_states, attention_mask=None, explicit_aspects=None, explicit_opinions=None):
        """
        Forward pass for grid tagging matching
        
        Args:
            hidden_states: Token hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            explicit_aspects: Known explicit aspect positions [batch_size, seq_len]
            explicit_opinions: Known explicit opinion positions [batch_size, seq_len]
            
        Returns:
            Grid tagging predictions and implicit sentiment detections
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Ensure seq_len doesn't exceed max_seq_len
        seq_len = min(seq_len, self.max_seq_len)
        hidden_states = hidden_states[:, :seq_len, :]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            hidden_states = hidden_states * mask_expanded
        
        # ============================================================================
        # GRID TAGGING MATRIX COMPUTATION
        # ============================================================================
        
        # Compute pairwise relationships using grid matrices
        aspect_aspect_relations = self._compute_grid_relations(
            hidden_states, self.aspect_aspect_grid[:seq_len, :seq_len, :], 'aspect_aspect'
        )
        
        aspect_opinion_relations = self._compute_grid_relations(
            hidden_states, self.aspect_opinion_grid[:seq_len, :seq_len, :], 'aspect_opinion'
        )
        
        opinion_opinion_relations = self._compute_grid_relations(
            hidden_states, self.opinion_opinion_grid[:seq_len, :seq_len, :], 'opinion_opinion'
        )
        
        # ============================================================================
        # IMPLICIT ASPECT/OPINION DETECTION
        # ============================================================================
        
        # Detect implicit aspects
        implicit_aspect_logits = self._detect_implicit_elements(
            hidden_states, aspect_aspect_relations, self.implicit_aspect_detector, 'aspect'
        )
        
        # Detect implicit opinions
        implicit_opinion_logits = self._detect_implicit_elements(
            hidden_states, opinion_opinion_relations, self.implicit_opinion_detector, 'opinion'
        )
        
        # ============================================================================
        # CAUSALITY CONSTRAINT APPLICATION
        # ============================================================================
        
        # Apply causality constraints to ensure logical consistency
        constrained_aspects, constrained_opinions = self.causality_constraints(
            implicit_aspect_logits,
            implicit_opinion_logits,
            aspect_opinion_relations,
            explicit_aspects,
            explicit_opinions
        )
        
        return {
            'implicit_aspect_logits': constrained_aspects,
            'implicit_opinion_logits': constrained_opinions,
            'aspect_aspect_relations': aspect_aspect_relations,
            'aspect_opinion_relations': aspect_opinion_relations,
            'opinion_opinion_relations': opinion_opinion_relations,
            'grid_matrices': {
                'aspect_aspect': self.aspect_aspect_grid[:seq_len, :seq_len, :],
                'aspect_opinion': self.aspect_opinion_grid[:seq_len, :seq_len, :],
                'opinion_opinion': self.opinion_opinion_grid[:seq_len, :seq_len, :]
            }
        }
    
    def _compute_grid_relations(self, hidden_states, grid_matrix, relation_type):
        """Compute grid-based relations between tokens"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Grid matrix: [seq_len, seq_len, num_relation_types]
        relations = torch.zeros(batch_size, seq_len, seq_len, self.num_relation_types, device=hidden_states.device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Get token representations
                token_i = hidden_states[:, i, :]  # [batch_size, hidden_size]
                token_j = hidden_states[:, j, :]  # [batch_size, hidden_size]
                
                # Compute relation scores for each relation type
                for r in range(self.num_relation_types):
                    # Get relation embedding
                    relation_emb = self.relation_embeddings(torch.tensor(r, device=hidden_states.device))
                    
                    # Combine token representations with relation
                    combined_repr = torch.cat([token_i, token_j, relation_emb.unsqueeze(0).expand(batch_size, -1)], dim=-1)
                    
                    # Use appropriate classifier
                    if relation_type == 'aspect_aspect' or relation_type == 'aspect_opinion':
                        relation_score = self.aspect_grid_classifier(combined_repr)[:, r]
                    else:
                        relation_score = self.opinion_grid_classifier(combined_repr)[:, r]
                    
                    # Apply grid matrix bias
                    grid_bias = grid_matrix[i, j, r]
                    relations[:, i, j, r] = relation_score + grid_bias
        
        return relations
    
    def _detect_implicit_elements(self, hidden_states, relations, detector, element_type):
        """Detect implicit aspects or opinions using relational context"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Aggregate relational information for each token
        implicit_logits = torch.zeros(batch_size, seq_len, 4, device=hidden_states.device)
        
        for i in range(seq_len):
            token_repr = hidden_states[:, i, :]  # [batch_size, hidden_size]
            
            # Aggregate relations with other tokens
            if element_type == 'aspect':
                # Use aspect-aspect and aspect-opinion relations
                relation_context = relations[:, i, :, :].mean(dim=1)  # Average over all relations
            else:
                # Use opinion-opinion and aspect-opinion relations
                relation_context = relations[:, :, i, :].mean(dim=1)  # Average over all relations
            
            # Flatten relation context
            relation_flat = relation_context.view(batch_size, -1)
            
            # Ensure relation_flat has correct size
            expected_size = self.num_relation_types
            if relation_flat.size(1) != expected_size:
                # Pad or truncate to expected size
                if relation_flat.size(1) < expected_size:
                    padding = torch.zeros(batch_size, expected_size - relation_flat.size(1), device=hidden_states.device)
                    relation_flat = torch.cat([relation_flat, padding], dim=1)
                else:
                    relation_flat = relation_flat[:, :expected_size]
            
            # Expand relation_flat to match hidden_size for concatenation
            if relation_flat.size(1) < hidden_size:
                # Repeat to match hidden size
                repeat_factor = hidden_size // relation_flat.size(1) + 1
                relation_expanded = relation_flat.repeat(1, repeat_factor)[:, :hidden_size]
            else:
                relation_expanded = relation_flat[:, :hidden_size]
            
            # Combine token representation with relational context
            combined_input = torch.cat([token_repr, relation_expanded], dim=1)
            
            # Detect implicit elements
            implicit_logits[:, i, :] = detector(combined_input)
        
        return implicit_logits


class CausalityConstraints(nn.Module):
    """
    Causality-compliant constraints for logical consistency in implicit sentiment detection
    
    Ensures that detected implicit aspects and opinions follow logical causality rules.
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Constraint networks
        self.aspect_opinion_consistency = nn.Sequential(
            nn.Linear(8, 32),  # 4 (aspect) + 4 (opinion)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Logical rules for aspect-opinion pairs
        self.logical_rules = {
            'implicit_aspect_requires_explicit_opinion': True,
            'implicit_opinion_requires_explicit_aspect': True,
            'no_isolated_implicit_elements': True,
            'causality_ordering': True
        }
    
    def forward(self, aspect_logits, opinion_logits, aspect_opinion_relations, 
                explicit_aspects=None, explicit_opinions=None):
        """
        Apply causality constraints to implicit detections
        
        Args:
            aspect_logits: Implicit aspect predictions [batch_size, seq_len, 4]
            opinion_logits: Implicit opinion predictions [batch_size, seq_len, 4]
            aspect_opinion_relations: Relations between aspects and opinions
            explicit_aspects: Known explicit aspect positions
            explicit_opinions: Known explicit opinion positions
            
        Returns:
            Constrained aspect and opinion logits
        """
        batch_size, seq_len, _ = aspect_logits.shape
        device = aspect_logits.device
        
        # Initialize constrained logits
        constrained_aspects = aspect_logits.clone()
        constrained_opinions = opinion_logits.clone()
        
        # Apply constraints for each position
        for b in range(batch_size):
            for i in range(seq_len):
                # Get current predictions
                aspect_pred = aspect_logits[b, i, :]  # [4]
                opinion_pred = opinion_logits[b, i, :]  # [4]
                
                # Check consistency between aspect and opinion
                combined_pred = torch.cat([aspect_pred, opinion_pred])
                consistency_score = self.aspect_opinion_consistency(combined_pred)
                
                # Apply consistency penalty
                if consistency_score < 0.5:  # Low consistency
                    penalty = (0.5 - consistency_score) * 2.0  # Scale penalty
                    constrained_aspects[b, i, :] *= (1.0 - penalty)
                    constrained_opinions[b, i, :] *= (1.0 - penalty)
                
                # Rule 1: Implicit aspect requires explicit opinion nearby
                if self.logical_rules['implicit_aspect_requires_explicit_opinion']:
                    implicit_aspect_prob = F.softmax(aspect_pred, dim=0)[:2].sum()  # Implicit-B + Implicit-I
                    
                    if implicit_aspect_prob > 0.5:  # Likely implicit aspect
                        # Check for nearby explicit opinions
                        has_nearby_explicit_opinion = self._has_nearby_explicit_element(
                            b, i, seq_len, explicit_opinions, window=3
                        )
                        
                        if not has_nearby_explicit_opinion:
                            # Penalize implicit aspect prediction
                            constrained_aspects[b, i, :2] *= 0.3  # Reduce implicit predictions
                
                # Rule 2: Implicit opinion requires explicit aspect nearby
                if self.logical_rules['implicit_opinion_requires_explicit_aspect']:
                    implicit_opinion_prob = F.softmax(opinion_pred, dim=0)[:2].sum()
                    
                    if implicit_opinion_prob > 0.5:
                        has_nearby_explicit_aspect = self._has_nearby_explicit_element(
                            b, i, seq_len, explicit_aspects, window=3
                        )
                        
                        if not has_nearby_explicit_aspect:
                            constrained_opinions[b, i, :2] *= 0.3
                
                # Rule 3: No isolated implicit elements
                if self.logical_rules['no_isolated_implicit_elements']:
                    # Check if this token would be isolated
                    is_isolated = self._is_isolated_implicit(
                        b, i, seq_len, constrained_aspects, constrained_opinions
                    )
                    
                    if is_isolated:
                        constrained_aspects[b, i, :2] *= 0.1  # Heavily penalize
                        constrained_opinions[b, i, :2] *= 0.1
                
                # Rule 4: Causality ordering (aspects typically come before opinions)
                if self.logical_rules['causality_ordering']:
                    # Check relative positions in aspect-opinion relations
                    aspect_opinion_strength = aspect_opinion_relations[b, i, :, :].max()
                    
                    if aspect_opinion_strength > 0.5:  # Strong relation exists
                        # Find related opinion positions
                        related_opinions = (aspect_opinion_relations[b, i, :, :] > 0.5).any(dim=1)
                        opinion_positions = torch.where(related_opinions)[0]
                        
                        if len(opinion_positions) > 0:
                            # If this is an implicit aspect after an opinion, reduce confidence
                            if (opinion_positions < i).any():
                                causality_penalty = 0.7
                                constrained_aspects[b, i, :2] *= causality_penalty
        
        return constrained_aspects, constrained_opinions
    
    def _has_nearby_explicit_element(self, batch_idx, position, seq_len, explicit_elements, window=3):
        """Check if there are explicit elements nearby"""
        if explicit_elements is None:
            return False
        
        start = max(0, position - window)
        end = min(seq_len, position + window + 1)
        
        # Check if any position in window has explicit elements
        nearby_region = explicit_elements[batch_idx, start:end]
        return (nearby_region > 0).any().item()
    
    def _is_isolated_implicit(self, batch_idx, position, seq_len, aspect_logits, opinion_logits):
        """Check if an implicit element would be isolated"""
        window = 2
        start = max(0, position - window)
        end = min(seq_len, position + window + 1)
        
        # Check nearby positions for other implicit or explicit elements
        nearby_aspects = aspect_logits[batch_idx, start:end, :]
        nearby_opinions = opinion_logits[batch_idx, start:end, :]
        
        # Count non-zero predictions nearby (excluding current position)
        mask = torch.ones(end - start, dtype=torch.bool)
        if position - start < len(mask):
            mask[position - start] = False
        
        nearby_aspect_activity = (nearby_aspects.max(dim=-1)[0] > 0.5)[mask].any()
        nearby_opinion_activity = (nearby_opinions.max(dim=-1)[0] > 0.5)[mask].any()
        
        return not (nearby_aspect_activity or nearby_opinion_activity)


class SpanContextualInteraction(nn.Module):
    """
    Span-Level Contextual Interaction Network (SCI-Net)
    
    Implements bi-directional aspect-opinion interactions at span level
    for enhanced implicit sentiment detection.
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, 'sci_num_heads', 8)
        self.num_layers = getattr(config, 'sci_num_layers', 2)
        
        # Span representation networks
        self.aspect_span_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        self.opinion_span_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Cross-attention layers for aspect-opinion interaction
        self.aspect_opinion_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        self.opinion_aspect_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.aspect_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)
        ])
        
        self.opinion_layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.span_interaction_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
    def forward(self, hidden_states, aspect_spans, opinion_spans):
        """
        Perform span-level contextual interaction
        
        Args:
            hidden_states: Token representations [batch_size, seq_len, hidden_size]
            aspect_spans: Aspect span boundaries [batch_size, num_aspects, 2]
            opinion_spans: Opinion span boundaries [batch_size, num_opinions, 2]
            
        Returns:
            Enhanced representations with span-level interactions
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Extract span representations
        aspect_representations = self._extract_span_representations(
            hidden_states, aspect_spans, self.aspect_span_encoder
        )
        
        opinion_representations = self._extract_span_representations(
            hidden_states, opinion_spans, self.opinion_span_encoder
        )
        
        # Apply cross-attention layers
        enhanced_aspects = aspect_representations
        enhanced_opinions = opinion_representations
        
        for layer_idx in range(self.num_layers):
            # Aspect attending to opinions
            if enhanced_opinions.size(1) > 0:  # Check if opinions exist
                aspect_attended, _ = self.aspect_opinion_attention[layer_idx](
                    query=enhanced_aspects,
                    key=enhanced_opinions,
                    value=enhanced_opinions
                )
                enhanced_aspects = self.aspect_layer_norms[layer_idx](
                    enhanced_aspects + aspect_attended
                )
            
            # Opinion attending to aspects
            if enhanced_aspects.size(1) > 0:  # Check if aspects exist
                opinion_attended, _ = self.opinion_aspect_attention[layer_idx](
                    query=enhanced_opinions,
                    key=enhanced_aspects,
                    value=enhanced_aspects
                )
                enhanced_opinions = self.opinion_layer_norms[layer_idx](
                    enhanced_opinions + opinion_attended
                )
        
        # Project back to token level
        enhanced_token_representations = self._project_to_token_level(
            hidden_states, enhanced_aspects, enhanced_opinions, aspect_spans, opinion_spans
        )
        
        return enhanced_token_representations
    
    def _extract_span_representations(self, hidden_states, span_boundaries, span_encoder):
        """Extract span representations using LSTM encoding"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        max_spans = span_boundaries.size(1) if len(span_boundaries.shape) > 2 else 1
        
        span_representations = []
        
        for b in range(batch_size):
            batch_spans = []
            
            for s in range(max_spans):
                if len(span_boundaries.shape) == 3:
                    start, end = span_boundaries[b, s, 0].item(), span_boundaries[b, s, 1].item()
                else:
                    # Handle case where span_boundaries is 2D
                    if s < span_boundaries.size(1):
                        start, end = span_boundaries[b, s].item(), span_boundaries[b, s].item() + 1
                    else:
                        start, end = 0, 1  # Default span
                
                # Ensure valid span boundaries
                start = max(0, min(start, seq_len - 1))
                end = max(start + 1, min(end, seq_len))
                
                # Extract span tokens
                span_tokens = hidden_states[b, start:end, :].unsqueeze(0)  # [1, span_len, hidden_size]
                
                # Encode span
                span_output, _ = span_encoder(span_tokens)
                
                # Use last hidden state as span representation
                span_repr = span_output[0, -1, :]  # [hidden_size]
                batch_spans.append(span_repr)
            
            if batch_spans:
                span_representations.append(torch.stack(batch_spans))
            else:
                # Create dummy span if no spans
                dummy_span = torch.zeros(1, hidden_size, device=hidden_states.device)
                span_representations.append(dummy_span)
        
        # Pad to same length
        max_num_spans = max(span_repr.size(0) for span_repr in span_representations)
        padded_spans = []
        
        for span_repr in span_representations:
            if span_repr.size(0) < max_num_spans:
                padding = torch.zeros(
                    max_num_spans - span_repr.size(0),
                    span_repr.size(1),
                    device=span_repr.device
                )
                padded_span = torch.cat([span_repr, padding], dim=0)
            else:
                padded_span = span_repr[:max_num_spans]
            
            padded_spans.append(padded_span)
        
        return torch.stack(padded_spans)  # [batch_size, max_spans, hidden_size]
    
    def _project_to_token_level(self, original_hidden_states, enhanced_aspects, 
                               enhanced_opinions, aspect_spans, opinion_spans):
        """Project enhanced span representations back to token level"""
        batch_size, seq_len, hidden_size = original_hidden_states.shape
        enhanced_tokens = original_hidden_states.clone()
        
        for b in range(batch_size):
            # Project aspect representations
            for s in range(enhanced_aspects.size(1)):
                if len(aspect_spans.shape) == 3 and s < aspect_spans.size(1):
                    start = max(0, aspect_spans[b, s, 0].item())
                    end = min(seq_len, aspect_spans[b, s, 1].item())
                    
                    if start < end:
                        # Combine original and enhanced representations
                        span_repr = enhanced_aspects[b, s, :]
                        for t in range(start, end):
                            combined = torch.cat([enhanced_tokens[b, t, :], span_repr])
                            enhanced_tokens[b, t, :] = self.span_interaction_projection(combined)
            
            # Project opinion representations
            for s in range(enhanced_opinions.size(1)):
                if len(opinion_spans.shape) == 3 and s < opinion_spans.size(1):
                    start = max(0, opinion_spans[b, s, 0].item())
                    end = min(seq_len, opinion_spans[b, s, 1].item())
                    
                    if start < end:
                        span_repr = enhanced_opinions[b, s, :]
                        for t in range(start, end):
                            combined = torch.cat([enhanced_tokens[b, t, :], span_repr])
                            enhanced_tokens[b, t, :] = self.span_interaction_projection(combined)
        
        return enhanced_tokens


class ImplicitQuadrupleExtractor(nn.Module):
    """
    Complete implicit sentiment detector combining Grid Tagging and SCI-Net
    
    Handles aspect-opinion-sentiment-category quadruple extraction with
    implicit component detection.
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Core components
        self.grid_tagging_matcher = GridTaggingMatcher(config)
        self.sci_net = SpanContextualInteraction(config)
        
        # Quadruple classification heads
        self.aspect_category_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 12)  # 12 aspect categories (food, service, ambiance, etc.)
        )
        
        self.sentiment_intensity_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5)  # Very negative, negative, neutral, positive, very positive
        )
        
        # Implicit-explicit interaction module
        self.implicit_explicit_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
    def forward(self, hidden_states, attention_mask=None, explicit_aspects=None, 
                explicit_opinions=None, aspect_spans=None, opinion_spans=None):
        """
        Extract implicit sentiment quadruples
        
        Args:
            hidden_states: Token representations [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            explicit_aspects: Known explicit aspect positions
            explicit_opinions: Known explicit opinion positions
            aspect_spans: Aspect span boundaries for SCI-Net
            opinion_spans: Opinion span boundaries for SCI-Net
            
        Returns:
            Complete quadruple extraction results with implicit detection
        """
        # Apply Grid Tagging Matcher
        grid_results = self.grid_tagging_matcher(
            hidden_states, attention_mask, explicit_aspects, explicit_opinions
        )
        
        # Apply SCI-Net if span information available
        if aspect_spans is not None and opinion_spans is not None:
            enhanced_hidden_states = self.sci_net(hidden_states, aspect_spans, opinion_spans)
        else:
            enhanced_hidden_states = hidden_states
        
        # Fuse implicit and explicit information
        fused_representations = self.implicit_explicit_fusion(
            torch.cat([hidden_states, enhanced_hidden_states], dim=-1)
        )
        
        # Extract aspect categories
        aspect_categories = self.aspect_category_classifier(fused_representations)
        
        # Extract sentiment intensities
        sentiment_intensities = self.sentiment_intensity_classifier(fused_representations)
        
        # Combine all results
        return {
            'implicit_aspects': grid_results['implicit_aspect_logits'],
            'implicit_opinions': grid_results['implicit_opinion_logits'],
            'aspect_categories': aspect_categories,
            'sentiment_intensities': sentiment_intensities,
            'grid_relations': {
                'aspect_aspect': grid_results['aspect_aspect_relations'],
                'aspect_opinion': grid_results['aspect_opinion_relations'],
                'opinion_opinion': grid_results['opinion_opinion_relations']
            },
            'enhanced_representations': fused_representations,
            'grid_matrices': grid_results['grid_matrices']
        }
    
    def extract_implicit_quadruples(self, outputs, tokenizer, texts):
        """
        Extract implicit quadruples from model outputs
        
        Args:
            outputs: Model outputs from forward pass
            tokenizer: Tokenizer for text decoding
            texts: Original input texts
            
        Returns:
            List of extracted implicit quadruples
        """
        implicit_aspects = outputs['implicit_aspects'].argmax(dim=-1)  # [batch_size, seq_len]
        implicit_opinions = outputs['implicit_opinions'].argmax(dim=-1)
        aspect_categories = outputs['aspect_categories'].argmax(dim=-1)
        sentiment_intensities = outputs['sentiment_intensities'].argmax(dim=-1)
        
        batch_size, seq_len = implicit_aspects.shape
        quadruples = []
        
        for b in range(batch_size):
            text = texts[b] if b < len(texts) else ""
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            batch_quadruples = []
            
            # Extract implicit aspect spans
            implicit_aspect_spans = self._extract_implicit_spans(implicit_aspects[b])
            implicit_opinion_spans = self._extract_implicit_spans(implicit_opinions[b])
            
            # Create quadruples from detected spans
            for aspect_span in implicit_aspect_spans:
                for opinion_span in implicit_opinion_spans:
                    # Get category and sentiment for this span
                    aspect_category = self._get_span_category(aspect_categories[b], aspect_span)
                    sentiment_intensity = self._get_span_sentiment(sentiment_intensities[b], aspect_span, opinion_span)
                    
                    # Decode text spans
                    aspect_text = self._decode_span(tokens, aspect_span, tokenizer)
                    opinion_text = self._decode_span(tokens, opinion_span, tokenizer)
                    
                    quadruple = {
                        'aspect': aspect_text,
                        'opinion': opinion_text,
                        'category': aspect_category,
                        'sentiment': sentiment_intensity,
                        'type': 'implicit',
                        'aspect_span': aspect_span,
                        'opinion_span': opinion_span
                    }
                    
                    batch_quadruples.append(quadruple)
            
            quadruples.append(batch_quadruples)
        
        return quadruples
    
    def _extract_implicit_spans(self, predictions):
        """Extract spans from implicit predictions (Implicit-B, Implicit-I labels)"""
        spans = []
        current_span = []
        
        for i, pred in enumerate(predictions):
            if pred == 0:  # Implicit-B
                if current_span:
                    spans.append(current_span)
                current_span = [i]
            elif pred == 1 and current_span:  # Implicit-I
                current_span.append(i)
            else:  # End of span
                if current_span:
                    spans.append(current_span)
                    current_span = []
        
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def _get_span_category(self, categories, span):
        """Get most common category in span"""
        span_categories = categories[span]
        return span_categories.mode()[0].item()
    
    def _get_span_sentiment(self, sentiments, aspect_span, opinion_span):
        """Get sentiment intensity for aspect-opinion pair"""
        # Use opinion span for sentiment if available, otherwise aspect span
        if opinion_span:
            span_sentiments = sentiments[opinion_span]
        else:
            span_sentiments = sentiments[aspect_span]
        
        return span_sentiments.mode()[0].item()
    
    def _decode_span(self, tokens, span, tokenizer):
        """Decode token span to text"""
        if not span:
            return ""
        
        span_tokens = [tokens[i] for i in span if i < len(tokens)]
        return tokenizer.decode(span_tokens, skip_special_tokens=True)