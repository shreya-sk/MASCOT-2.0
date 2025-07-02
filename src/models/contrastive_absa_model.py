"""
Complete ABSA Model with Integrated Contrastive Learning
This is the main model that integrates all contrastive learning components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModel, AutoConfig

from .implicit_sentiment_detector import ContrastiveImplicitABSA
from ..training.contrastive_losses import SupervisedContrastiveLoss
from ..training.negative_sampling import NegativeSamplingManager


"""
Complete ABSA Model with Integrated Contrastive Learning
This is the main model that integrates all contrastive learning components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModel, AutoConfig

from .implicit_sentiment_detector import ContrastiveImplicitABSA
from ..training.contrastive_losses import SupervisedContrastiveLoss
from ..training.negative_sampling import NegativeSamplingManager


class ContrastiveABSAModel(nn.Module):
    """
    Complete ABSA model with contrastive learning integration
    Combines explicit/implicit detection with advanced contrastive training
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Core ABSA model with contrastive learning
        self.contrastive_absa = ContrastiveImplicitABSA(config)
        
        # Additional contrastive components
        self.contrastive_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, 256),
            nn.LayerNorm(256)
        )
        
        # Multi-granularity contrastive heads
        self.aspect_contrastive_head = nn.Linear(256, 128)
        self.opinion_contrastive_head = nn.Linear(256, 128)
        self.sentiment_contrastive_head = nn.Linear(256, 128)
        self.triplet_contrastive_head = nn.Linear(256, 128)
        
        # Cross-modal attention for aspect-opinion interaction
        self.aspect_opinion_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Sentiment combination with contrastive alignment
        self.sentiment_alignment_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, 3)  # POS, NEG, NEU
        )
        
        # Confidence estimation for implicit detection
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Advanced fusion mechanisms
        self.hierarchical_fusion = HierarchicalFeatureFusion(self.hidden_size, config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                aspect_labels: Optional[torch.Tensor] = None,
                opinion_labels: Optional[torch.Tensor] = None,
                sentiment_labels: Optional[torch.Tensor] = None,
                implicit_labels: Optional[torch.Tensor] = None,
                syntax_features: Optional[torch.Tensor] = None,
                return_contrastive_features: bool = False,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive learning
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            aspect_labels: [batch_size, seq_len] - aspect labels
            opinion_labels: [batch_size, seq_len] - opinion labels
            sentiment_labels: [batch_size, seq_len] - sentiment labels
            implicit_labels: [batch_size, seq_len] - implicit/explicit labels
            syntax_features: [batch_size, seq_len, syntax_dim] - syntax features
            return_contrastive_features: Whether to return contrastive features
            training: Whether in training mode
            
        Returns:
            Dictionary with predictions and features
        """
        # Get base ABSA outputs with contrastive learning
        absa_outputs = self.contrastive_absa(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aspect_labels=aspect_labels,
            opinion_labels=opinion_labels,
            sentiment_labels=sentiment_labels,
            implicit_labels=implicit_labels,
            syntax_features=syntax_features,
            training=training
        )
        
        # Extract key features
        fused_features = absa_outputs['fused_features']  # [batch_size, seq_len, hidden_size]
        
        # Hierarchical feature fusion
        hierarchical_outputs = self.hierarchical_fusion(
            aspect_features=absa_outputs.get('aspect_features', fused_features),
            opinion_features=absa_outputs.get('opinion_features', fused_features),
            sentiment_features=fused_features,
            attention_mask=attention_mask
        )
        
        # Aspect-opinion cross-attention
        aspect_features = hierarchical_outputs['aspect_features']
        opinion_features = hierarchical_outputs['opinion_features']
        
        cross_attended_aspects, _ = self.aspect_opinion_attention(
            aspect_features, opinion_features, opinion_features,
            key_padding_mask=~attention_mask.bool()
        )
        
        cross_attended_opinions, _ = self.aspect_opinion_attention(
            opinion_features, aspect_features, aspect_features,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Enhanced sentiment prediction with cross-attention
        combined_ao_features = torch.cat([cross_attended_aspects, cross_attended_opinions], dim=-1)
        enhanced_sentiment_logits = self.sentiment_alignment_layer(combined_ao_features)
        
        # Contrastive feature projection
        projected_features = self.contrastive_projector(fused_features)
        
        # Multi-granularity contrastive features
        aspect_contrastive = F.normalize(self.aspect_contrastive_head(projected_features), dim=-1)
        opinion_contrastive = F.normalize(self.opinion_contrastive_head(projected_features), dim=-1)
        sentiment_contrastive = F.normalize(self.sentiment_contrastive_head(projected_features), dim=-1)
        triplet_contrastive = F.normalize(self.triplet_contrastive_head(projected_features), dim=-1)
        
        # Confidence estimation for implicit predictions
        confidence_scores = self.confidence_estimator(fused_features)
        
        # Prepare outputs
        outputs = {
            # Base predictions
            'aspect_logits': absa_outputs.get('aspect_logits'),
            'opinion_logits': absa_outputs.get('opinion_logits'),
            'sentiment_logits': absa_outputs.get('sentiment_logits'),
            'enhanced_sentiment_logits': enhanced_sentiment_logits,
            
            # Implicit detection
            'implicit_aspect_logits': absa_outputs.get('implicit_aspect_logits'),
            'implicit_opinion_logits': absa_outputs.get('implicit_opinion_logits'),
            'implicit_sentiment_logits': absa_outputs.get('implicit_sentiment_logits'),
            'explicit_sentiment_logits': absa_outputs.get('explicit_sentiment_logits'),
            
            # Features
            'fused_features': fused_features,
            'projected_features': projected_features,
            'hierarchical_features': hierarchical_outputs,
            'cross_attended_aspects': cross_attended_aspects,
            'cross_attended_opinions': cross_attended_opinions,
            
            # Confidence
            'confidence_scores': confidence_scores,
            
            # Attention weights
            'attention_weights': absa_outputs.get('attention_weights')
        }
        
        # Add contrastive features if requested
        if return_contrastive_features or training:
            outputs.update({
                'aspect_contrastive': aspect_contrastive,
                'opinion_contrastive': opinion_contrastive,
                'sentiment_contrastive': sentiment_contrastive,
                'triplet_contrastive': triplet_contrastive
            })
        
        # Add contrastive losses if in training mode
        if training and aspect_labels is not None:
            contrastive_loss_outputs = self._compute_advanced_contrastive_losses(
                outputs, attention_mask, aspect_labels, opinion_labels, sentiment_labels
            )
            outputs.update(contrastive_loss_outputs)
        
        return outputs
    
    def _compute_advanced_contrastive_losses(self,
                                           outputs: Dict[str, torch.Tensor],
                                           attention_mask: torch.Tensor,
                                           aspect_labels: torch.Tensor,
                                           opinion_labels: Optional[torch.Tensor],
                                           sentiment_labels: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute advanced contrastive losses with multi-granularity
        """
        contrastive_losses = {}
        
        # Get valid tokens
        valid_mask = attention_mask.bool()
        
        # Multi-granularity contrastive learning
        granularities = ['aspect', 'opinion', 'sentiment', 'triplet']
        label_maps = {
            'aspect': aspect_labels,
            'opinion': opinion_labels if opinion_labels is not None else aspect_labels,
            'sentiment': sentiment_labels if sentiment_labels is not None else aspect_labels,
            'triplet': aspect_labels * 100 + (opinion_labels if opinion_labels is not None else 0) * 10 + 
                      (sentiment_labels if sentiment_labels is not None else 0)
        }
        
        for granularity in granularities:
            if f'{granularity}_contrastive' in outputs and granularity in label_maps:
                features = outputs[f'{granularity}_contrastive']
                labels = label_maps[granularity]
                
                # Flatten features and labels
                flat_features = features[valid_mask]
                flat_labels = labels[valid_mask]
                
                # Remove padding tokens
                non_pad_mask = flat_labels != -100
                if non_pad_mask.any():
                    clean_features = flat_features[non_pad_mask]
                    clean_labels = flat_labels[non_pad_mask]
                    
                    # Multi-view contrastive loss
                    if len(clean_features) > 1:
                        # Create augmented views
                        aug_features = F.dropout(clean_features, p=0.1, training=True)
                        multi_view_features = torch.stack([clean_features, aug_features], dim=1)
                        
                        # Supervised contrastive loss
                        supervised_loss = SupervisedContrastiveLoss(temperature=0.07)
                        loss_outputs = supervised_loss(
                            features=multi_view_features,
                            labels=clean_labels,
                            aspect_labels=clean_labels,
                            opinion_labels=clean_labels,
                            sentiment_labels=clean_labels
                        )
                        
                        contrastive_losses[f'{granularity}_contrastive_loss'] = loss_outputs['total']
        
        # Cross-granularity contrastive learning
        if ('aspect_contrastive' in outputs and 'opinion_contrastive' in outputs and 
            'sentiment_contrastive' in outputs):
            
            aspect_features = outputs['aspect_contrastive'][valid_mask]
            opinion_features = outputs['opinion_contrastive'][valid_mask]
            sentiment_features = outputs['sentiment_contrastive'][valid_mask]
            
            if aspect_labels is not None:
                flat_aspect_labels = aspect_labels[valid_mask]
                non_pad_mask = flat_aspect_labels != -100
                
                if non_pad_mask.any():
                    clean_aspect_features = aspect_features[non_pad_mask]
                    clean_opinion_features = opinion_features[non_pad_mask]
                    clean_sentiment_features = sentiment_features[non_pad_mask]
                    clean_labels = flat_aspect_labels[non_pad_mask]
                    
                    # Cross-granularity alignment loss
                    cross_alignment_loss = self._compute_cross_granularity_loss(
                        clean_aspect_features, clean_opinion_features, 
                        clean_sentiment_features, clean_labels
                    )
                    contrastive_losses['cross_granularity_loss'] = cross_alignment_loss
        
        # Implicit-explicit contrastive learning
        if ('implicit_aspect_logits' in outputs and 'confidence_scores' in outputs):
            implicit_contrast_loss = self._compute_implicit_explicit_contrast(
                outputs, valid_mask, aspect_labels
            )
            if implicit_contrast_loss is not None:
                contrastive_losses['implicit_explicit_loss'] = implicit_contrast_loss
        
        return contrastive_losses
    
    def _compute_cross_granularity_loss(self,
                                       aspect_features: torch.Tensor,
                                       opinion_features: torch.Tensor,
                                       sentiment_features: torch.Tensor,
                                       labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-granularity contrastive loss
        """
        # Ensure features have same granularity labels should be aligned
        similarity_ao = F.cosine_similarity(aspect_features, opinion_features, dim=-1)
        similarity_as = F.cosine_similarity(aspect_features, sentiment_features, dim=-1)
        similarity_os = F.cosine_similarity(opinion_features, sentiment_features, dim=-1)
        
        # For same labels, features should be similar
        same_label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Positive pairs should have high similarity
        positive_loss = torch.zeros(1, device=aspect_features.device)
        negative_loss = torch.zeros(1, device=aspect_features.device)
        
        if same_label_mask.any():
            # Encourage high similarity for same labels
            positive_similarities = similarity_ao[same_label_mask]
            if len(positive_similarities) > 0:
                positive_loss = (1 - positive_similarities).clamp(min=0).mean()
        
        # Negative pairs should have low similarity
        diff_label_mask = ~same_label_mask
        if diff_label_mask.any():
            negative_similarities = similarity_ao[diff_label_mask]
            if len(negative_similarities) > 0:
                negative_loss = (negative_similarities + 0.2).clamp(min=0).mean()
        
        return positive_loss + negative_loss
    
    def _compute_implicit_explicit_contrast(self,
                                          outputs: Dict[str, torch.Tensor],
                                          valid_mask: torch.Tensor,
                                          aspect_labels: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute contrastive loss between implicit and explicit features
        """
        if 'implicit_aspect_logits' not in outputs:
            return None
        
        implicit_probs = F.softmax(outputs['implicit_aspect_logits'], dim=-1)
        implicit_scores = implicit_probs[:, :, 1]  # Implicit class probability
        
        fused_features = outputs['fused_features']
        
        # Get valid features
        flat_implicit_scores = implicit_scores[valid_mask]
        flat_features = fused_features[valid_mask]
        flat_labels = aspect_labels[valid_mask]
        
        # Remove padding
        non_pad_mask = flat_labels != -100
        if not non_pad_mask.any():
            return None
        
        clean_implicit_scores = flat_implicit_scores[non_pad_mask]
        clean_features = flat_features[non_pad_mask]
        
        # Separate implicit and explicit features
        implicit_threshold = 0.5
        implicit_mask = clean_implicit_scores > implicit_threshold
        explicit_mask = ~implicit_mask
        
        if implicit_mask.any() and explicit_mask.any():
            implicit_features = clean_features[implicit_mask]
            explicit_features = clean_features[explicit_mask]
            
            # Contrastive loss: implicit and explicit should be distinguishable
            implicit_centroid = implicit_features.mean(dim=0, keepdim=True)
            explicit_centroid = explicit_features.mean(dim=0, keepdim=True)
            
            # Encourage separation between centroids
            centroid_similarity = F.cosine_similarity(implicit_centroid, explicit_centroid, dim=1)
            separation_loss = (centroid_similarity + 0.1).clamp(min=0)
            
            return separation_loss.mean()
        
        return None
    
    def extract_contrastive_triplets(self,
                                   input_ids: torch.Tensor,
                                   attention_mask: torch.Tensor,
                                   confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Extract triplets using contrastive learning confidence
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            confidence_threshold: Minimum confidence for extraction
            
        Returns:
            List of extracted triplets with confidence scores
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_contrastive_features=True,
                training=False
            )
            
            batch_size, seq_len = input_ids.shape
            extracted_triplets = []
            
            for b in range(batch_size):
                triplets = []
                valid_len = attention_mask[b].sum().item()
                
                # Get predictions
                aspect_preds = outputs['aspect_logits'][b, :valid_len].argmax(dim=-1)
                opinion_preds = outputs['opinion_logits'][b, :valid_len].argmax(dim=-1)
                sentiment_preds = outputs['enhanced_sentiment_logits'][b, :valid_len].argmax(dim=-1)
                confidence_scores = outputs['confidence_scores'][b, :valid_len, 0]
                
                # Extract high-confidence spans
                high_conf_mask = confidence_scores > confidence_threshold
                
                if high_conf_mask.any():
                    # Find contiguous spans
                    aspect_spans = self._extract_spans(aspect_preds, high_conf_mask)
                    opinion_spans = self._extract_spans(opinion_preds, high_conf_mask)
                    
                    # Match aspects with opinions and sentiments
                    for aspect_span in aspect_spans:
                        for opinion_span in opinion_spans:
                            # Check if spans are in reasonable proximity
                            if self._spans_are_related(aspect_span, opinion_span, max_distance=10):
                                # Get sentiment for this aspect-opinion pair
                                span_sentiment = self._get_span_sentiment(
                                    sentiment_preds, aspect_span, opinion_span
                                )
                                
                                # Get confidence for this triplet
                                triplet_confidence = self._get_triplet_confidence(
                                    confidence_scores, aspect_span, opinion_span
                                )
                                
                                triplets.append({
                                    'aspect_span': aspect_span,
                                    'opinion_span': opinion_span,
                                    'sentiment': span_sentiment,
                                    'confidence': triplet_confidence,
                                    'is_implicit': self._is_implicit_triplet(outputs, b, aspect_span, opinion_span)
                                })
                
                extracted_triplets.append(triplets)
            
            return extracted_triplets
    
    def _extract_spans(self, predictions: torch.Tensor, mask: torch.Tensor) -> List[Tuple[int, int]]:
        """Extract contiguous spans from predictions"""
        spans = []
        start = None
        
        for i in range(len(predictions)):
            if predictions[i] > 0 and mask[i]:  # Positive prediction with high confidence
                if start is None:
                    start = i
            else:
                if start is not None:
                    spans.append((start, i - 1))
                    start = None
        
        # Handle span at the end
        if start is not None:
            spans.append((start, len(predictions) - 1))
        
        return spans
    
    def _spans_are_related(self, span1: Tuple[int, int], span2: Tuple[int, int], max_distance: int = 10) -> bool:
        """Check if two spans are within reasonable distance"""
        distance = min(
            abs(span1[0] - span2[1]),
            abs(span1[1] - span2[0]),
            abs(span1[0] - span2[0]),
            abs(span1[1] - span2[1])
        )
        return distance <= max_distance
    
    def _get_span_sentiment(self, sentiment_preds: torch.Tensor, 
                           aspect_span: Tuple[int, int], 
                           opinion_span: Tuple[int, int]) -> int:
        """Get sentiment for aspect-opinion pair"""
        # Average sentiment predictions in the region
        start = min(aspect_span[0], opinion_span[0])
        end = max(aspect_span[1], opinion_span[1])
        
        region_sentiments = sentiment_preds[start:end+1]
        return region_sentiments.mode().values.item() if len(region_sentiments) > 0 else 1  # Default to neutral
    
    def _get_triplet_confidence(self, confidence_scores: torch.Tensor,
                               aspect_span: Tuple[int, int],
                               opinion_span: Tuple[int, int]) -> float:
        """Get confidence score for triplet"""
        # Average confidence in both spans
        aspect_conf = confidence_scores[aspect_span[0]:aspect_span[1]+1].mean()
        opinion_conf = confidence_scores[opinion_span[0]:opinion_span[1]+1].mean()
        return (aspect_conf + opinion_conf).item() / 2
    
    def _is_implicit_triplet(self, outputs: Dict[str, torch.Tensor], 
                            batch_idx: int, aspect_span: Tuple[int, int], 
                            opinion_span: Tuple[int, int]) -> bool:
        """Check if triplet contains implicit aspects or opinions"""
        if 'implicit_aspect_logits' not in outputs:
            return False
        
        implicit_aspect_probs = F.softmax(outputs['implicit_aspect_logits'], dim=-1)
        implicit_scores = implicit_aspect_probs[batch_idx, :, 1]  # Implicit class
        
        # Check if either span has high implicit probability
        aspect_implicit = implicit_scores[aspect_span[0]:aspect_span[1]+1].mean() > 0.5
        opinion_implicit = implicit_scores[opinion_span[0]:opinion_span[1]+1].mean() > 0.5
        
        return aspect_implicit or opinion_implicit


class HierarchicalFeatureFusion(nn.Module):
    """
    Hierarchical feature fusion for multi-level ABSA representations
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Level-specific encoders
        self.word_level_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )
        
        self.phrase_level_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )
        
        self.sentence_level_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )
        
        # Cross-level attention
        self.cross_level_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )
        
        # Output projections
        self.aspect_projection = nn.Linear(hidden_size, hidden_size)
        self.opinion_projection = nn.Linear(hidden_size, hidden_size)
        self.sentiment_projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self,
                aspect_features: torch.Tensor,
                opinion_features: torch.Tensor,
                sentiment_features: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Hierarchical feature fusion
        
        Args:
            aspect_features: [batch_size, seq_len, hidden_size]
            opinion_features: [batch_size, seq_len, hidden_size]
            sentiment_features: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Dictionary with fused features at different levels
        """
        # Multi-level encoding
        word_level = self.word_level_encoder(
            aspect_features,
            src_key_padding_mask=~attention_mask.bool()
        )
        
        phrase_level = self.phrase_level_encoder(
            opinion_features,
            src_key_padding_mask=~attention_mask.bool()
        )
        
        sentence_level = self.sentence_level_encoder(
            sentiment_features,
            src_key_padding_mask=~attention_mask.bool()
        )
        
        # Cross-level attention
        cross_attended, _ = self.cross_level_attention(
            word_level, phrase_level, sentence_level,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Hierarchical fusion
        concatenated = torch.cat([word_level, phrase_level, sentence_level], dim=-1)
        fused_features = self.fusion_layer(concatenated)
        
        # Add cross-attention
        fused_features = fused_features + cross_attended
        
        # Task-specific projections
        fused_aspect_features = self.aspect_projection(fused_features)
        fused_opinion_features = self.opinion_projection(fused_features)
        fused_sentiment_features = self.sentiment_projection(fused_features)
        
        return {
            'aspect_features': fused_aspect_features,
            'opinion_features': fused_opinion_features,
            'sentiment_features': fused_sentiment_features,
            'word_level': word_level,
            'phrase_level': phrase_level,
            'sentence_level': sentence_level,
            'fused_features': fused_features
        }