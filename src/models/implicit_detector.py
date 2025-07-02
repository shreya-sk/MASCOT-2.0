return {
            'implicit_aspect_spans': implicit_aspect_spans,
            'implicit_opinion_predictions': implicit_opinion_preds.cpu().tolist(),
            'implicit_sentiment_probs': implicit_sentiment_probs.cpu().tolist(),
            'explicit_sentiment_probs': explicit_sentiment_probs.cpu().tolist()
        }


class AdvancedImplicitExtractor:
    """
    Advanced extraction methods for implicit sentiment analysis results
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 consistency_threshold: float = 0.6,
                 min_span_length: int = 1,
                 max_span_length: int = 10):
        
        self.confidence_threshold = confidence_threshold
        self.consistency_threshold = consistency_threshold
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        
        # Implicit pattern types
        self.implicit_patterns = {
            'negation': ['not', 'never', 'no', 'nothing', 'nobody', 'nowhere'],
            'intensifiers': ['very', 'extremely', 'highly', 'quite', 'really'],
            'comparatives': ['better', 'worse', 'more', 'less', 'superior', 'inferior'],
            'temporal': ['used to', 'before', 'previously', 'formerly'],
            'conditional': ['if', 'unless', 'provided', 'assuming'],
            'expectation': ['should', 'ought', 'expected', 'supposed', 'meant']
        }
    
    def extract_implicit_triplets(self,
                                 model_outputs: Dict[str, torch.Tensor],
                                 tokenizer,
                                 input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Extract implicit triplets with advanced confidence and consistency checking
        
        Args:
            model_outputs: Dictionary of model outputs
            tokenizer: Tokenizer for text decoding
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            List of extracted implicit triplets with metadata
        """
        batch_size, seq_len = input_ids.shape
        all_triplets = []
        
        for b in range(batch_size):
            # Get valid sequence length
            valid_len = attention_mask[b].sum().item()
            
            # Extract tokens
            tokens = tokenizer.convert_ids_to_tokens(input_ids[b][:valid_len])
            text = tokenizer.decode(input_ids[b][:valid_len], skip_special_tokens=True)
            
            # Get predictions and confidence scores
            batch_triplets = self._extract_batch_triplets(
                model_outputs, b, valid_len, tokens, text
            )
            
            all_triplets.append(batch_triplets)
        
        return all_triplets
    
    def _extract_batch_triplets(self,
                               outputs: Dict[str, torch.Tensor],
                               batch_idx: int,
                               valid_len: int,
                               tokens: List[str],
                               text: str) -> List[Dict[str, Any]]:
        """Extract triplets for a single batch item"""
        
        # Extract predictions
        aspect_preds = outputs.get('aspect_logits', torch.zeros(1, valid_len, 2))
        opinion_preds = outputs.get('opinion_logits', torch.zeros(1, valid_len, 2))
        sentiment_preds = outputs.get('enhanced_sentiment_logits', torch.zeros(1, valid_len, 3))
        
        # Get implicit detection results
        implicit_aspect_probs = torch.zeros(valid_len)
        implicit_opinion_probs = torch.zeros(valid_len)
        if 'implicit_aspect_logits' in outputs:
            implicit_aspect_probs = F.softmax(outputs['implicit_aspect_logits'], dim=-1)[batch_idx, :valid_len, 1]
        if 'implicit_opinion_logits' in outputs:
            implicit_opinion_probs = F.softmax(outputs['implicit_opinion_logits'], dim=-1)[batch_idx, :valid_len, 0]
        
        # Get confidence scores
        confidence_scores = torch.ones(valid_len)
        if 'confidence_scores' in outputs:
            confidence_scores = outputs['confidence_scores'][batch_idx, :valid_len, 0]
        
        # Extract aspects
        aspect_spans = self._extract_confident_spans(
            aspect_preds[batch_idx, :valid_len].argmax(dim=-1),
            confidence_scores,
            implicit_aspect_probs,
            span_type='aspect'
        )
        
        # Extract opinions
        opinion_spans = self._extract_confident_spans(
            opinion_preds[batch_idx, :valid_len].argmax(dim=-1),
            confidence_scores,
            implicit_opinion_probs,
            span_type='opinion'
        )
        
        # Match aspects with opinions and extract triplets
        triplets = []
        for aspect_span in aspect_spans:
            for opinion_span in opinion_spans:
                if self._spans_are_compatible(aspect_span, opinion_span, valid_len):
                    
                    # Get sentiment for this aspect-opinion pair
                    sentiment_info = self._get_triplet_sentiment(
                        sentiment_preds[batch_idx, :valid_len],
                        aspect_span, opinion_span
                    )
                    
                    # Calculate overall confidence
                    triplet_confidence = self._calculate_triplet_confidence(
                        aspect_span, opinion_span, confidence_scores, 
                        implicit_aspect_probs, implicit_opinion_probs
                    )
                    
                    # Check if triplet meets quality thresholds
                    if triplet_confidence >= self.confidence_threshold:
                        
                        # Extract text spans
                        aspect_text = self._extract_span_text(tokens, aspect_span['indices'])
                        opinion_text = self._extract_span_text(tokens, opinion_span['indices'])
                        
                        # Detect implicit patterns
                        implicit_patterns = self._detect_implicit_patterns(
                            tokens, aspect_span['indices'], opinion_span['indices']
                        )
                        
                        triplet = {
                            'aspect': {
                                'text': aspect_text,
                                'span': aspect_span['indices'],
                                'confidence': aspect_span['confidence'],
                                'is_implicit': aspect_span['is_implicit']
                            },
                            'opinion': {
                                'text': opinion_text,
                                'span': opinion_span['indices'],
                                'confidence': opinion_span['confidence'],
                                'is_implicit': opinion_span['is_implicit']
                            },
                            'sentiment': sentiment_info['label'],
                            'sentiment_confidence': sentiment_info['confidence'],
                            'overall_confidence': triplet_confidence,
                            'implicit_patterns': implicit_patterns,
                            'extraction_method': 'contrastive_implicit',
                            'quality_score': self._calculate_quality_score(
                                aspect_span, opinion_span, sentiment_info, implicit_patterns
                            )
                        }
                        
                        triplets.append(triplet)
        
        # Post-process and filter triplets
        filtered_triplets = self._post_process_triplets(triplets, text)
        
        return filtered_triplets
    
    def _extract_confident_spans(self,
                                predictions: torch.Tensor,
                                confidence_scores: torch.Tensor,
                                implicit_probs: torch.Tensor,
                                span_type: str) -> List[Dict[str, Any]]:
        """Extract confident spans of given type"""
        
        spans = []
        seq_len = len(predictions)
        
        # Find contiguous positive predictions
        in_span = False
        span_start = None
        
        for i in range(seq_len):
            is_positive = predictions[i] > 0
            is_confident = confidence_scores[i] >= self.confidence_threshold
            is_implicit = implicit_probs[i] >= 0.5
            
            if is_positive and is_confident:
                if not in_span:
                    span_start = i
                    in_span = True
            else:
                if in_span and span_start is not None:
                    # End current span
                    span_end = i - 1
                    
                    # Check span length constraints
                    span_length = span_end - span_start + 1
                    if self.min_span_length <= span_length <= self.max_span_length:
                        
                        # Calculate span-level metrics
                        span_confidence = confidence_scores[span_start:span_end+1].mean().item()
                        span_implicit_prob = implicit_probs[span_start:span_end+1].mean().item()
                        
                        spans.append({
                            'indices': (span_start, span_end),
                            'confidence': span_confidence,
                            'is_implicit': span_implicit_prob >= 0.5,
                            'implicit_confidence': span_implicit_prob,
                            'type': span_type,
                            'length': span_length
                        })
                    
                    in_span = False
                    span_start = None
        
        # Handle span at end of sequence
        if in_span and span_start is not None:
            span_end = seq_len - 1
            span_length = span_end - span_start + 1
            
            if self.min_span_length <= span_length <= self.max_span_length:
                span_confidence = confidence_scores[span_start:span_end+1].mean().item()
                span_implicit_prob = implicit_probs[span_start:span_end+1].mean().item()
                
                spans.append({
                    'indices': (span_start, span_end),
                    'confidence': span_confidence,
                    'is_implicit': span_implicit_prob >= 0.5,
                    'implicit_confidence': span_implicit_prob,
                    'type': span_type,
                    'length': span_length
                })
        
        return spans
    
    def _spans_are_compatible(self, 
                             aspect_span: Dict[str, Any], 
                             opinion_span: Dict[str, Any], 
                             seq_len: int) -> bool:
        """Check if aspect and opinion spans are compatible for triplet formation"""
        
        # Distance constraint
        aspect_start, aspect_end = aspect_span['indices']
        opinion_start, opinion_end = opinion_span['indices']
        
        # Calculate minimum distance between spans
        if aspect_end < opinion_start:
            distance = opinion_start - aspect_end
        elif opinion_end < aspect_start:
            distance = aspect_start - opinion_end
        else:
            distance = 0  # Overlapping spans
        
        # Maximum distance constraint
        max_distance = min(15, seq_len // 3)  # Adaptive based on sequence length
        if distance > max_distance:
            return False
        
        # Confidence compatibility
        confidence_diff = abs(aspect_span['confidence'] - opinion_span['confidence'])
        if confidence_diff > 0.3:  # Large confidence gap
            return False
        
        # Implicit-explicit compatibility
        both_implicit = aspect_span['is_implicit'] and opinion_span['is_implicit']
        both_explicit = (not aspect_span['is_implicit']) and (not opinion_span['is_implicit'])
        mixed = aspect_span['is_implicit'] != opinion_span['is_implicit']
        
        # Allow all combinations but prefer consistent types
        return True
    
    def _get_triplet_sentiment(self,
                              sentiment_preds: torch.Tensor,
                              aspect_span: Dict[str, Any],
                              opinion_span: Dict[str, Any]) -> Dict[str, Any]:
        """Get sentiment for aspect-opinion pair"""
        
        aspect_start, aspect_end = aspect_span['indices']
        opinion_start, opinion_end = opinion_span['indices']
        
        # Define region for sentiment analysis
        region_start = min(aspect_start, opinion_start)
        region_end = max(aspect_end, opinion_end)
        
        # Get sentiment predictions in region
        region_sentiments = sentiment_preds[region_start:region_end+1]
        sentiment_probs = F.softmax(region_sentiments, dim=-1)
        
        # Aggregate sentiment (majority vote with confidence weighting)
        avg_sentiment_probs = sentiment_probs.mean(dim=0)
        predicted_sentiment = avg_sentiment_probs.argmax().item()
        sentiment_confidence = avg_sentiment_probs[predicted_sentiment].item()
        
        # Map to sentiment labels
        sentiment_labels = ['negative', 'neutral', 'positive']
        
        return {
            'label': sentiment_labels[predicted_sentiment],
            'confidence': sentiment_confidence,
            'probabilities': avg_sentiment_probs.tolist(),
            'region': (region_start, region_end)
        }
    
    def _calculate_triplet_confidence(self,
                                     aspect_span: Dict[str, Any],
                                     opinion_span: Dict[str, Any],
                                     confidence_scores: torch.Tensor,
                                     implicit_aspect_probs: torch.Tensor,
                                     implicit_opinion_probs: torch.Tensor) -> float:
        """Calculate overall confidence for triplet"""
        
        # Individual span confidences
        aspect_conf = aspect_span['confidence']
        opinion_conf = opinion_span['confidence']
        
        # Implicit detection consistency
        aspect_start, aspect_end = aspect_span['indices']
        opinion_start, opinion_end = opinion_span['indices']
        
        # Regional confidence
        region_start = min(aspect_start, opinion_start)
        region_end = max(aspect_end, opinion_end)
        region_confidence = confidence_scores[region_start:region_end+1].mean().item()
        
        # Implicit consistency score
        implicit_consistency = 1.0
        if aspect_span['is_implicit'] or opinion_span['is_implicit']:
            aspect_implicit_conf = implicit_aspect_probs[aspect_start:aspect_end+1].mean().item()
            opinion_implicit_conf = implicit_opinion_probs[opinion_start:opinion_end+1].mean().item()
            implicit_consistency = (aspect_implicit_conf + opinion_implicit_conf) / 2
        
        # Weighted combination
        overall_confidence = (
            0.3 * aspect_conf +
            0.3 * opinion_conf +
            0.2 * region_confidence +
            0.2 * implicit_consistency
        )
        
        return overall_confidence
    
    def _extract_span_text(self, tokens: List[str], span_indices: Tuple[int, int]) -> str:
        """Extract text for span indices"""
        start, end = span_indices
        span_tokens = tokens[start:end+1]
        
        # Handle subword tokens (starting with ##)
        cleaned_tokens = []
        for token in span_tokens:
            if token.startswith('##'):
                if cleaned_tokens:
                    cleaned_tokens[-1] += token[2:]
                else:
                    cleaned_tokens.append(token[2:])
            else:
                cleaned_tokens.append(token)
        
        return ' '.join(cleaned_tokens)
    
    def _detect_implicit_patterns(self,
                                 tokens: List[str],
                                 aspect_indices: Tuple[int, int],
                                 opinion_indices: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Detect implicit opinion patterns in the vicinity of spans"""
        
        patterns_found = []
        
        # Define search region
        aspect_start, aspect_end = aspect_indices
        opinion_start, opinion_end = opinion_indices
        search_start = max(0, min(aspect_start, opinion_start) - 5)
        search_end = min(len(tokens), max(aspect_end, opinion_end) + 5)
        
        search_tokens = [token.lower() for token in tokens[search_start:search_end]]
        
        # Check for each pattern type
        for pattern_type, pattern_words in self.implicit_patterns.items():
            for word in pattern_words:
                if word in search_tokens:
                    position = search_tokens.index(word) + search_start
                    patterns_found.append({
                        'type': pattern_type,
                        'word': word,
                        'position': position,
                        'distance_to_aspect': abs(position - (aspect_start + aspect_end) // 2),
                        'distance_to_opinion': abs(position - (opinion_start + opinion_end) // 2)
                    })
        
        return patterns_found
    
    def _calculate_quality_score(self,
                               aspect_span: Dict[str, Any],
                               opinion_span: Dict[str, Any],
                               sentiment_info: Dict[str, Any],
                               implicit_patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score for triplet"""
        
        # Base confidence score
        base_score = (aspect_span['confidence'] + opinion_span['confidence'] + 
                     sentiment_info['confidence']) / 3
        
        # Implicit pattern bonus
        pattern_bonus = min(0.1 * len(implicit_patterns), 0.3)
        
        # Span length penalty (very short or very long spans are less reliable)
        aspect_length_penalty = 0.0
        if aspect_span['length'] == 1:
            aspect_length_penalty = 0.05
        elif aspect_span['length'] > 7:
            aspect_length_penalty = 0.1
        
        opinion_length_penalty = 0.0
        if opinion_span['length'] == 1:
            opinion_length_penalty = 0.05
        elif opinion_span['length'] > 7:
            opinion_length_penalty = 0.1
        
        # Implicit consistency bonus
        implicit_bonus = 0.0
        if aspect_span['is_implicit'] or opinion_span['is_implicit']:
            implicit_bonus = 0.05
        
        quality_score = (base_score + pattern_bonus + implicit_bonus - 
                        aspect_length_penalty - opinion_length_penalty)
        
        return max(0.0, min(1.0, quality_score))
    
    def _post_process_triplets(self, 
                              triplets: List[Dict[str, Any]], 
                              text: str) -> List[Dict[str, Any]]:
        """Post-process and filter triplets"""
        
        # Remove duplicate or overlapping triplets
        filtered_triplets = []
        
        for triplet in triplets:
            is_duplicate = False
            
            for existing in filtered_triplets:
                if self._triplets_overlap(triplet, existing):
                    # Keep the one with higher quality score
                    if triplet['quality_score'] > existing['quality_score']:
                        filtered_triplets.remove(existing)
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_triplets.append(triplet)
        
        # Sort by quality score
        filtered_triplets.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return filtered_triplets
    
    def _triplets_overlap(self, triplet1: Dict[str, Any], triplet2: Dict[str, Any]) -> bool:
        """Check if two triplets have overlapping spans"""
        
        def spans_overlap(span1, span2):
            start1, end1 = span1
            start2, end2 = span2
            return not (end1 < start2 or end2 < start1)
        
        aspect_overlap = spans_overlap(
            triplet1['aspect']['span'], 
            triplet2['aspect']['span']
        )
        
        opinion_overlap = spans_overlap(
            triplet1['opinion']['span'], 
            triplet2['opinion']['span']
        )
        
        return aspect_overlap and opinion_overlap


class ImplicitEvaluationMetrics:
    """
    Specialized evaluation metrics for implicit sentiment detection
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_implicit_metrics(self,
                                predictions: Dict[str, torch.Tensor],
                                ground_truth: Dict[str, torch.Tensor],
                                attention_mask: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive metrics for implicit sentiment detection
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            attention_mask: Valid token mask
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Basic classification metrics
        basic_metrics = self._compute_basic_classification_metrics(
            predictions, ground_truth, attention_mask
        )
        metrics.update(basic_metrics)
        
        # Implicit-specific metrics
        implicit_metrics = self._compute_implicit_specific_metrics(
            predictions, ground_truth, attention_mask
        )
        metrics.update(implicit_metrics)
        
        # Contrastive learning metrics
        contrastive_metrics = self._compute_contrastive_metrics(
            predictions, ground_truth, attention_mask
        )
        metrics.update(contrastive_metrics)
        
        return metrics
    
    def _compute_basic_classification_metrics(self,
                                            predictions: Dict[str, torch.Tensor],
                                            ground_truth: Dict[str, torch.Tensor],
                                            attention_mask: torch.Tensor) -> Dict[str, float]:
        """Compute basic classification metrics"""
        
        metrics = {}
        
        # Aspect detection metrics
        if 'aspect_logits' in predictions and 'aspect_labels' in ground_truth:
            aspect_metrics = self._compute_span_metrics(
                predictions['aspect_logits'],
                ground_truth['aspect_labels'],
                attention_mask,
                prefix='aspect'
            )
            metrics.update(aspect_metrics)
        
        # Opinion detection metrics
        if 'opinion_logits' in predictions and 'opinion_labels' in ground_truth:
            opinion_metrics = self._compute_span_metrics(
                predictions['opinion_logits'],
                ground_truth['opinion_labels'],
                attention_mask,
                prefix='opinion'
            )
            metrics.update(opinion_metrics)
        
        # Sentiment classification metrics
        if 'sentiment_logits' in predictions and 'sentiment_labels' in ground_truth:
            sentiment_metrics = self._compute_multiclass_metrics(
                predictions['sentiment_logits'],
                ground_truth['sentiment_labels'],
                attention_mask,
                prefix='sentiment'
            )
            metrics.update(sentiment_metrics)
        
        return metrics
    
    def _compute_implicit_specific_metrics(self,
                                         predictions: Dict[str, torch.Tensor],
                                         ground_truth: Dict[str, torch.Tensor],
                                         attention_mask: torch.Tensor) -> Dict[str, float]:
        """Compute metrics specific to implicit detection"""
        
        metrics = {}
        
        # Implicit vs explicit classification
        if ('implicit_aspect_logits' in predictions and 
            'implicit_labels' in ground_truth):
            
            implicit_metrics = self._compute_span_metrics(
                predictions['implicit_aspect_logits'],
                ground_truth['implicit_labels'],
                attention_mask,
                prefix='implicit_aspect'
            )
            metrics.update(implicit_metrics)
        
        # Implicit-explicit sentiment alignment
        if ('implicit_sentiment_logits' in predictions and 
            'explicit_sentiment_logits' in predictions and
            'sentiment_labels' in ground_truth):
            
            alignment_metrics = self._compute_alignment_metrics(
                predictions['implicit_sentiment_logits'],
                predictions['explicit_sentiment_logits'],
                ground_truth['sentiment_labels'],
                attention_mask
            )
            metrics.update(alignment_metrics)
        
        return metrics
    
    def _compute_contrastive_metrics(self,
                                   predictions: Dict[str, torch.Tensor],
                                   ground_truth: Dict[str, torch.Tensor],
                                   attention_mask: torch.Tensor) -> Dict[str, float]:
        """Compute contrastive learning specific metrics"""
        
        metrics = {}
        
        # Contrastive feature quality (if available)
        if 'aspect_contrastive' in predictions:
            contrastive_quality = self._compute_contrastive_quality(
                predictions['aspect_contrastive'],
                ground_truth.get('aspect_labels'),
                attention_mask
            )
            metrics['contrastive_aspect_quality'] = contrastive_quality
        
        return metrics
    
    def _compute_span_metrics(self,
                            logits: torch.Tensor,
                            labels: torch.Tensor,
                            attention_mask: torch.Tensor,
                            prefix: str) -> Dict[str, float]:
        """Compute precision, recall, F1 for span detection"""
        
        # Get predictions
        preds = logits.argmax(dim=-1)
        
        # Flatten and remove padding
        valid_mask = attention_mask.bool()
        flat_preds = preds[valid_mask]
        flat_labels = labels[valid_mask]
        
        non_pad_mask = flat_labels != -100
        if not non_pad_mask.any():
            return {f'{prefix}_precision': 0.0, f'{prefix}_recall': 0.0, f'{prefix}_f1': 0.0}
        
        clean_preds = flat_preds[non_pad_mask]
        clean_labels = flat_labels[non_pad_mask]
        
        # Calculate metrics
        tp = ((clean_preds == 1) & (clean_labels == 1)).sum().item()
        fp = ((clean_preds == 1) & (clean_labels == 0)).sum().item()
        fn = ((clean_preds == 0) & (clean_labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            f'{prefix}_precision': precision,
            f'{prefix}_recall': recall,
            f'{prefix}_f1': f1
        }
    
    def _compute_multiclass_metrics(self,
                                   logits: torch.Tensor,
                                   labels: torch.Tensor,
                                   attention_mask: torch.Tensor,
                                   prefix: str) -> Dict[str, float]:
        """Compute accuracy for multiclass classification"""
        
        preds = logits.argmax(dim=-1)
        
        # Flatten and remove padding
        valid_mask = attention_mask.bool()
        flat_preds = preds[valid_mask]
        flat_labels = labels[valid_mask]
        
        non_pad_mask = flat_labels != -100
        if not non_pad_mask.any():
            return {f'{prefix}_accuracy': 0.0}
        
        clean_preds = flat_preds[non_pad_mask]
        clean_labels = flat_labels[non_pad_mask]
        
        accuracy = (clean_preds == clean_labels).float().mean().item()
        
        return {f'{prefix}_accuracy': accuracy}
    
    def _compute_alignment_metrics(self,
                                  implicit_logits: torch.Tensor,
                                  explicit_logits: torch.Tensor,
                                  ground_truth: torch.Tensor,
                                  attention_mask: torch.Tensor) -> Dict[str, float]:
        """Compute alignment between implicit and explicit predictions"""
        
        implicit_preds = implicit_logits.argmax(dim=-1)
        explicit_preds = explicit_logits.argmax(dim=-1)
        
        # Flatten
        valid_mask = attention_mask.bool()
        flat_implicit = implicit_preds[valid_mask]
        flat_explicit = explicit_preds[valid_mask]
        flat_truth = ground_truth[valid_mask]
        
        non_pad_mask = flat_truth != -100
        if not non_pad_mask.any():
            return {'alignment_consistency': 0.0}
        
        clean_implicit = flat_implicit[non_pad_mask]
        clean_explicit = flat_explicit[non_pad_mask]
        
        # Alignment consistency
        consistency = (clean_implicit == clean_explicit).float().mean().item()
        
        return {'alignment_consistency': consistency}
    
    def _compute_contrastive_quality(self,
                                   contrastive_features: torch.Tensor,
                                   labels: Optional[torch.Tensor],
                                   attention_mask: torch.Tensor) -> float:
        """Compute quality of contrastive features"""
        
        if labels is None:
            return 0.0
        
        # Flatten features and labels
        valid_mask = attention_mask.bool()
        flat_features = contrastive_features[valid_mask]
        flat_labels = labels[valid_mask]
        
        non_pad_mask = flat_labels != -100
        if not non_pad_mask.any():
            return 0.0
        
        clean_features = flat_features[non_pad_mask]
        clean_labels = flat_labels[non_pad_mask]
        
        # Compute intra-class vs inter-class similarity
        unique_labels = torch.unique(clean_labels)
        if len(unique_labels) < 2:
            return 0.0
        
        intra_class_sim = 0.0
        inter_class_sim = 0.0
        intra_count = 0
        inter_count = 0
        
        for i, label1 in enumerate(clean_labels):
            for j, label2 in enumerate(clean_labels):
                if i >= j:
                    continue
                
                sim = F.cosine_similarity(
                    clean_features[i:i+1], 
                    clean_features[j:j+1], 
                    dim=1
                ).item()
                
                if label1 == label2:
                    intra_class_sim += sim
                    intra_count += 1
                else:
                    inter_class_sim += sim
                    inter_count += 1
        
        if intra_count > 0 and inter_count > 0:
            avg_intra = intra_class_sim / intra_count
            avg_inter = inter_class_sim / inter_count
            quality = avg_intra - avg_inter  # Higher is better
            return max(0.0, quality)
        
        return 0.0


class AdvancedImplicitExtractor:
    """
    Advanced extraction methods for implicit sentiment analysis results
    """
    
    def __init__(self, confidence_threshold: float = """
Implicit Sentiment Detection Module with Contrastive Learning
Implements breakthrough techniques for implicit aspect/opinion detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoModel, AutoConfig


class ImplicitAspectDetector(nn.Module):
    """
    Detects implicit aspects using contrastive learning
    Based on EMNLP 2024 breakthrough for implicit sentiment detection
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_syntax: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_syntax = use_syntax
        
        # Feature extraction layers
        self.aspect_encoder = nn.LSTM(
            hidden_size, hidden_size // 2, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Implicit detection heads
        self.implicit_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # implicit vs explicit
        )
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 128)  # Projection dimension
        )
        
        # Syntax-aware enhancement
        if use_syntax:
            self.syntax_encoder = nn.Linear(50, hidden_size // 4)  # POS tags, dependencies
            self.syntax_fusion = nn.Linear(hidden_size + hidden_size // 4, hidden_size)
        
        # Context aggregation
        self.context_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                hidden_states: torch.Tensor,           # [batch_size, seq_len, hidden_size]
                attention_mask: torch.Tensor,          # [batch_size, seq_len]
                syntax_features: Optional[torch.Tensor] = None,  # [batch_size, seq_len, syntax_dim]
                return_contrastive: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for implicit aspect detection
        
        Args:
            hidden_states: Token-level hidden states from transformer
            attention_mask: Attention mask for valid tokens
            syntax_features: Optional syntax features (POS, dependencies)
            return_contrastive: Whether to return contrastive features
            
        Returns:
            Dictionary with predictions and features
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # LSTM encoding for sequence modeling
        aspect_features, _ = self.aspect_encoder(hidden_states)
        aspect_features = self.dropout(aspect_features)
        
        # Syntax integration
        if self.use_syntax and syntax_features is not None:
            syntax_emb = self.syntax_encoder(syntax_features)
            combined_features = torch.cat([aspect_features, syntax_emb], dim=-1)
            aspect_features = self.syntax_fusion(combined_features)
            aspect_features = F.relu(aspect_features)
        
        # Context-aware attention
        attended_features, attention_weights = self.context_attention(
            aspect_features, aspect_features, aspect_features,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Residual connection and normalization
        aspect_features = self.layer_norm(aspect_features + attended_features)
        
        # Implicit/explicit classification
        implicit_logits = self.implicit_classifier(aspect_features)
        
        # Token-level predictions
        implicit_probs = F.softmax(implicit_logits, dim=-1)
        implicit_predictions = implicit_logits.argmax(dim=-1)
        
        outputs = {
            'implicit_logits': implicit_logits,
            'implicit_probs': implicit_probs,
            'implicit_predictions': implicit_predictions,
            'attention_weights': attention_weights,
            'aspect_features': aspect_features
        }
        
        # Contrastive features for training
        if return_contrastive:
            # Project features for contrastive learning
            contrastive_features = self.projection_head(aspect_features)
            contrastive_features = F.normalize(contrastive_features, dim=-1)
            outputs['contrastive_features'] = contrastive_features
        
        return outputs
    
    def extract_implicit_spans(self, 
                              predictions: torch.Tensor,
                              attention_mask: torch.Tensor,
                              threshold: float = 0.5) -> List[List[Tuple[int, int]]]:
        """
        Extract implicit aspect spans from predictions
        
        Args:
            predictions: [batch_size, seq_len, 2] - implicit probabilities
            attention_mask: [batch_size, seq_len]
            threshold: Confidence threshold for implicit detection
            
        Returns:
            List of implicit spans for each batch item
        """
        batch_size, seq_len = predictions.shape[:2]
        implicit_spans = []
        
        for b in range(batch_size):
            spans = []
            valid_len = attention_mask[b].sum().item()
            
            # Get implicit probabilities
            implicit_probs = predictions[b, :valid_len, 1]  # Implicit class
            implicit_mask = implicit_probs > threshold
            
            # Find contiguous spans
            start = None
            for i in range(valid_len):
                if implicit_mask[i] and start is None:
                    start = i
                elif not implicit_mask[i] and start is not None:
                    spans.append((start, i-1))
                    start = None
            
            # Handle span at end
            if start is not None:
                spans.append((start, valid_len-1))
            
            implicit_spans.append(spans)
        
        return implicit_spans


class ImplicitOpinionDetector(nn.Module):
    """
    Detects implicit opinions using advanced contrastive techniques
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_emotion_classes: int = 8,
                 dropout: float = 0.1,
                 use_emotional_lexicon: bool = True):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_emotional_lexicon = use_emotional_lexicon
        self.num_emotion_classes = num_emotion_classes
        
        # Opinion-specific encoder
        self.opinion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Multi-level opinion detection
        self.opinion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # implicit, explicit, neutral
        )
        
        # Emotion classification for implicit opinions
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotion_classes)
        )
        
        # Contrastive learning components
        self.opinion_projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Emotional lexicon integration
        if use_emotional_lexicon:
            self.lexicon_encoder = nn.Embedding(5000, 64)  # Emotional word embeddings
            self.lexicon_fusion = nn.Linear(hidden_size + 64, hidden_size)
        
        # Cross-attention for aspect-opinion interaction
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Implicit opinion pattern detector
        self.pattern_detector = ImplicitPatternDetector(hidden_size, dropout)
        
        # Contextual opinion scorer
        self.context_scorer = ContextualOpinionScorer(hidden_size, dropout)
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                aspect_features: Optional[torch.Tensor] = None,
                lexicon_ids: Optional[torch.Tensor] = None,
                return_contrastive: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for implicit opinion detection
        """
        # Opinion encoding
        opinion_features = self.opinion_encoder(
            hidden_states,
            src_key_padding_mask=~attention_mask.bool()
        )
        
        # Lexicon integration
        if self.use_emotional_lexicon and lexicon_ids is not None:
            lexicon_emb = self.lexicon_encoder(lexicon_ids)
            combined_features = torch.cat([opinion_features, lexicon_emb], dim=-1)
            opinion_features = self.lexicon_fusion(combined_features)
            opinion_features = F.gelu(opinion_features)
        
        # Cross-attention with aspects if available
        if aspect_features is not None:
            attended_features, _ = self.cross_attention(
                opinion_features, aspect_features, aspect_features,
                key_padding_mask=~attention_mask.bool()
            )
            opinion_features = opinion_features + attended_features
        
        # Implicit pattern detection
        pattern_outputs = self.pattern_detector(opinion_features, attention_mask)
        
        # Contextual opinion scoring
        context_outputs = self.context_scorer(opinion_features, attention_mask)
        
        # Enhanced features combining patterns and context
        enhanced_features = opinion_features + pattern_outputs['pattern_features'] + context_outputs['context_features']
        
        # Opinion classification
        opinion_logits = self.opinion_classifier(enhanced_features)
        emotion_logits = self.emotion_classifier(enhanced_features)
        
        outputs = {
            'opinion_logits': opinion_logits,
            'emotion_logits': emotion_logits,
            'opinion_features': enhanced_features,
            'opinion_probs': F.softmax(opinion_logits, dim=-1),
            'emotion_probs': F.softmax(emotion_logits, dim=-1),
            'pattern_scores': pattern_outputs['pattern_scores'],
            'context_scores': context_outputs['context_scores'],
            'implicit_indicators': pattern_outputs['implicit_indicators']
        }
        
        if return_contrastive:
            contrastive_features = self.opinion_projection(enhanced_features)
            contrastive_features = F.normalize(contrastive_features, dim=-1)
            outputs['contrastive_features'] = contrastive_features
        
        return outputs
    
    def extract_implicit_opinions(self, 
                                 predictions: torch.Tensor,
                                 pattern_scores: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 threshold: float = 0.6) -> List[List[Tuple[int, int, str]]]:
        """
        Extract implicit opinion spans with types
        
        Args:
            predictions: [batch_size, seq_len, 3] - opinion predictions
            pattern_scores: [batch_size, seq_len] - implicit pattern scores
            attention_mask: [batch_size, seq_len]
            threshold: Confidence threshold
            
        Returns:
            List of implicit opinion spans with types for each batch
        """
        batch_size, seq_len = predictions.shape[:2]
        implicit_opinions = []
        
        for b in range(batch_size):
            opinions = []
            valid_len = attention_mask[b].sum().item()
            
            # Get implicit probabilities
            implicit_probs = predictions[b, :valid_len, 0]  # Implicit class
            pattern_conf = pattern_scores[b, :valid_len]
            
            # Combined confidence
            combined_conf = (implicit_probs + pattern_conf) / 2
            confident_mask = combined_conf > threshold
            
            # Find contiguous spans
            start = None
            for i in range(valid_len):
                if confident_mask[i] and start is None:
                    start = i
                elif not confident_mask[i] and start is not None:
                    # Determine opinion type based on pattern
                    span_pattern = pattern_conf[start:i].mean().item()
                    opinion_type = self._classify_opinion_type(span_pattern)
                    opinions.append((start, i-1, opinion_type))
                    start = None
            
            # Handle span at end
            if start is not None:
                span_pattern = pattern_conf[start:valid_len].mean().item()
                opinion_type = self._classify_opinion_type(span_pattern)
                opinions.append((start, valid_len-1, opinion_type))
            
            implicit_opinions.append(opinions)
        
        return implicit_opinions
    
    def _classify_opinion_type(self, pattern_score: float) -> str:
        """Classify implicit opinion type based on pattern score"""
        if pattern_score > 0.8:
            return 'strong_implicit'
        elif pattern_score > 0.6:
            return 'moderate_implicit'
        else:
            return 'weak_implicit'


class ImplicitPatternDetector(nn.Module):
    """
    Detects implicit opinion patterns using linguistic cues
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Pattern recognition layers
        self.pattern_cnn = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size // 2,
            kernel_size=3,
            padding=1
        )
        
        self.pattern_lstm = nn.LSTM(
            hidden_size // 2,
            hidden_size // 4,
            batch_first=True,
            bidirectional=True
        )
        
        # Implicit indicators detector
        self.indicator_detector = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # Different types of implicit indicators
        )
        
        # Pattern scorer
        self.pattern_scorer = nn.Sequential(
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                features: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect implicit opinion patterns
        
        Args:
            features: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Dictionary with pattern information
        """
        batch_size, seq_len, hidden_size = features.shape
        
        # CNN for local pattern detection
        # Transpose for conv1d: [batch_size, hidden_size, seq_len]
        features_transposed = features.transpose(1, 2)
        cnn_output = F.relu(self.pattern_cnn(features_transposed))
        cnn_output = cnn_output.transpose(1, 2)  # Back to [batch_size, seq_len, hidden_size//2]
        
        # LSTM for sequential pattern modeling
        lstm_output, _ = self.pattern_lstm(cnn_output)
        pattern_features = self.dropout(lstm_output)
        
        # Implicit indicators detection
        implicit_indicators = self.indicator_detector(pattern_features)
        
        # Pattern scoring
        pattern_scores = self.pattern_scorer(pattern_features).squeeze(-1)
        
        return {
            'pattern_features': pattern_features,
            'implicit_indicators': implicit_indicators,
            'pattern_scores': pattern_scores
        }


class ContextualOpinionScorer(nn.Module):
    """
    Scores opinions based on contextual information
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Context attention
        self.context_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )
        
        # Context scorer
        self.context_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Sentiment polarity detector for context
        self.polarity_detector = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # positive, negative, neutral context
        )
        
    def forward(self, 
                features: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Score opinions based on context
        
        Args:
            features: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Dictionary with context information
        """
        # Self-attention for context modeling
        context_attended, _ = self.context_attention(
            features, features, features,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Context encoding
        context_features = self.context_encoder(
            context_attended,
            src_key_padding_mask=~attention_mask.bool()
        )
        
        # Context scoring
        context_scores = self.context_scorer(context_features).squeeze(-1)
        
        # Polarity detection
        polarity_logits = self.polarity_detector(context_features)
        
        return {
            'context_features': context_features,
            'context_scores': context_scores,
            'polarity_logits': polarity_logits,
            'polarity_probs': F.softmax(polarity_logits, dim=-1)
        }


class SentimentCombinationVector(nn.Module):
    """
    Sentiment combination vectors processed through fully connected layers
    For handling implicit-explicit combinations (EMNLP 2024)
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_sentiment_classes: int = 3,
                 num_combination_layers: int = 4):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_sentiment_classes = num_sentiment_classes
        
        # Four fully connected layers as mentioned in the paper
        self.combination_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size if i == 0 else hidden_size // 2, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_size // 2)
            ) for i in range(num_combination_layers)
        ])
        
        # Sentiment combination classifier
        self.combination_classifier = nn.Linear(hidden_size // 2, num_sentiment_classes * 2)  # implicit + explicit
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Implicit-explicit interaction modeling
        self.interaction_modeler = ImplicitExplicitInteractionModel(hidden_size // 2)
        
        # Sentiment polarity refinement
        self.polarity_refiner = SentimentPolarityRefiner(hidden_size // 2, num_sentiment_classes)
        
    def forward(self, 
                aspect_features: torch.Tensor,
                opinion_features: torch.Tensor,
                interaction_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process sentiment combination vectors
        
        Args:
            aspect_features: [batch_size, seq_len, hidden_size]
            opinion_features: [batch_size, seq_len, hidden_size]
            interaction_features: [batch_size, seq_len, hidden_size]
            
        Returns:
            Dictionary with combination predictions
        """
        # Combine all features
        combined_features = aspect_features + opinion_features + interaction_features
        
        # Process through four FC layers
        x = combined_features
        layer_outputs = []
        for layer in self.combination_layers:
            x = layer(x)
            layer_outputs.append(x)
        
        # Interaction modeling between implicit and explicit
        interaction_outputs = self.interaction_modeler(x, layer_outputs)
        
        # Polarity refinement
        polarity_outputs = self.polarity_refiner(x, interaction_outputs['interaction_features'])
        
        # Final predictions
        combination_logits = self.combination_classifier(x)
        confidence_scores = self.confidence_estimator(x)
        
        # Split into implicit and explicit components
        batch_size, seq_len, _ = combination_logits.shape
        implicit_logits = combination_logits[:, :, :self.num_sentiment_classes]
        explicit_logits = combination_logits[:, :, self.num_sentiment_classes:]
        
        return {
            'implicit_sentiment_logits': implicit_logits,
            'explicit_sentiment_logits': explicit_logits,
            'combination_confidence': confidence_scores,
            'combined_features': x,
            'implicit_sentiment_probs': F.softmax(implicit_logits, dim=-1),
            'explicit_sentiment_probs': F.softmax(explicit_logits, dim=-1),
            'layer_outputs': layer_outputs,
            'interaction_scores': interaction_outputs['interaction_scores'],
            'polarity_alignment': polarity_outputs['alignment_scores'],
            'refined_sentiment_logits': polarity_outputs['refined_logits']
        }


class ImplicitExplicitInteractionModel(nn.Module):
    """
    Models the interaction between implicit and explicit sentiment components
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Interaction attention
        self.interaction_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True
        )
        
        # Implicit-explicit separator
        self.ie_separator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 2)  # implicit vs explicit probability
        )
        
        # Interaction scorer
        self.interaction_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Cross-component encoder
        self.cross_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size,
                batch_first=True
            ),
            num_layers=1
        )
        
    def forward(self, 
                features: torch.Tensor, 
                layer_outputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Model implicit-explicit interactions
        
        Args:
            features: [batch_size, seq_len, hidden_size]
            layer_outputs: List of intermediate layer outputs
            
        Returns:
            Dictionary with interaction information
        """
        # Separate implicit and explicit components
        ie_probs = F.softmax(self.ie_separator(features), dim=-1)
        implicit_prob = ie_probs[:, :, 0:1]  # [batch_size, seq_len, 1]
        explicit_prob = ie_probs[:, :, 1:2]  # [batch_size, seq_len, 1]
        
        # Weight features by implicit/explicit probabilities
        implicit_features = features * implicit_prob
        explicit_features = features * explicit_prob
        
        # Cross-attention between implicit and explicit
        cross_attended, _ = self.interaction_attention(
            implicit_features, explicit_features, explicit_features
        )
        
        # Interaction scoring
        interaction_input = torch.cat([implicit_features, explicit_features], dim=-1)
        interaction_scores = self.interaction_scorer(interaction_input)
        
        # Cross-component encoding
        stacked_features = torch.stack([implicit_features, explicit_features], dim=1)
        batch_size, num_components, seq_len, hidden_size = stacked_features.shape
        stacked_features = stacked_features.view(batch_size * num_components, seq_len, hidden_size)
        
        encoded_features = self.cross_encoder(stacked_features)
        encoded_features = encoded_features.view(batch_size, num_components, seq_len, hidden_size)
        
        # Aggregate cross-component features
        interaction_features = encoded_features.mean(dim=1)  # Average over components
        
        return {
            'interaction_features': interaction_features,
            'interaction_scores': interaction_scores,
            'implicit_features': implicit_features,
            'explicit_features': explicit_features,
            'implicit_prob': implicit_prob,
            'explicit_prob': explicit_prob,
            'cross_attended': cross_attended
        }


class SentimentPolarityRefiner(nn.Module):
    """
    Refines sentiment polarity predictions using implicit-explicit alignment
    """
    
    def __init__(self, hidden_size: int, num_sentiment_classes: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_sentiment_classes = num_sentiment_classes
        
        # Polarity alignment scorer
        self.alignment_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_sentiment_classes),
            nn.Softmax(dim=-1)
        )
        
        # Polarity consistency checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Refined sentiment predictor
        self.refined_predictor = nn.Sequential(
            nn.Linear(hidden_size + num_sentiment_classes, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_sentiment_classes)
        )
        
    def forward(self, 
                features: torch.Tensor, 
                interaction_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Refine sentiment polarity predictions
        
        Args:
            features: [batch_size, seq_len, hidden_size]
            interaction_features: [batch_size, seq_len, hidden_size]
            
        Returns:
            Dictionary with refined predictions
        """
        # Alignment scoring
        alignment_input = torch.cat([features, interaction_features], dim=-1)
        alignment_scores = self.alignment_scorer(alignment_input)
        
        # Consistency checking
        consistency_scores = self.consistency_checker(features)
        
        # Refined prediction
        refined_input = torch.cat([features, alignment_scores], dim=-1)
        refined_logits = self.refined_predictor(refined_input)
        
        # Apply consistency weighting
        weighted_refined_logits = refined_logits * consistency_scores
        
        return {
            'alignment_scores': alignment_scores,
            'consistency_scores': consistency_scores,
            'refined_logits': refined_logits,
            'weighted_refined_logits': weighted_refined_logits
        }


class ContrastiveImplicitABSA(nn.Module):
    """
    Main module combining all implicit detection components with contrastive learning
    Implements the full ITSCL framework from EMNLP 2024
    """
    
    def __init__(self, 
                 config,
                 backbone_model: Optional[nn.Module] = None):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Backbone transformer (shared)
        if backbone_model is None:
            self.backbone = AutoModel.from_pretrained(config.model_name)
        else:
            self.backbone = backbone_model
        
        # Implicit detection modules
        self.implicit_aspect_detector = ImplicitAspectDetector(
            hidden_size=self.hidden_size,
            dropout=config.dropout,
            use_syntax=getattr(config, 'use_syntax_features', True)
        )
        
        self.implicit_opinion_detector = ImplicitOpinionDetector(
            hidden_size=self.hidden_size,
            dropout=config.dropout,
            use_emotional_lexicon=getattr(config, 'use_emotional_lexicon', True)
        )
        
        self.sentiment_combiner = SentimentCombinationVector(
            hidden_size=self.hidden_size,
            num_sentiment_classes=3
        )
        
        # Contrastive learning loss
        from .contrastive_losses import SupervisedContrastiveLoss, InfoNCELoss, EnhancedTripletLoss
        
        self.contrastive_loss = SupervisedContrastiveLoss(
            temperature=getattr(config, 'contrastive_temperature', 0.07)
        )
        
        self.infonce_loss = InfoNCELoss(
            temperature=getattr(config, 'infonce_temperature', 0.07)
        )
        
        self.triplet_loss = EnhancedTripletLoss(
            margin=getattr(config, 'triplet_margin', 0.3),
            mining_strategy=getattr(config, 'triplet_mining', 'hard')
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Final prediction heads
        self.final_aspect_head = nn.Linear(self.hidden_size, 2)  # aspect/no-aspect
        self.final_opinion_head = nn.Linear(self.hidden_size, 2)  # opinion/no-opinion
        self.final_sentiment_head = nn.Linear(self.hidden_size, 3)  # pos/neg/neu
        
    def create_contrastive_pairs(self,
                                features: torch.Tensor,
                                labels: Dict[str, torch.Tensor],
                                num_augmentations: int = 2) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Create contrastive pairs with data augmentation
        """
        batch_size, seq_len, hidden_size = features.shape
        
        # Create augmented views (simple dropout augmentation)
        augmented_features = []
        augmented_labels = {k: [] for k in labels.keys()}
        
        for _ in range(num_augmentations):
            # Apply dropout as augmentation
            noise = torch.rand_like(features) > 0.1  # 10% dropout
            aug_features = features * noise.float()
            augmented_features.append(aug_features)
            
            # Labels remain the same
            for k, v in labels.items():
                augmented_labels[k].append(v)
        
        # Stack augmentations
        all_features = torch.stack([features] + augmented_features, dim=1)  # [batch, views, seq, hidden]
        
        for k in augmented_labels:
            augmented_labels[k] = torch.stack([labels[k]] + augmented_labels[k], dim=1)
        
        return all_features, augmented_labels
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                aspect_labels: Optional[torch.Tensor] = None,
                opinion_labels: Optional[torch.Tensor] = None,
                sentiment_labels: Optional[torch.Tensor] = None,
                implicit_labels: Optional[torch.Tensor] = None,
                syntax_features: Optional[torch.Tensor] = None,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive learning
        """
        # Get backbone features
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = backbone_outputs.last_hidden_state
        
        # Implicit aspect detection
        aspect_outputs = self.implicit_aspect_detector(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            syntax_features=syntax_features,
            return_contrastive=training
        )
        
        # Implicit opinion detection
        opinion_outputs = self.implicit_opinion_detector(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            aspect_features=aspect_outputs['aspect_features'],
            return_contrastive=training
        )
        
        # Sentiment combination
        combination_outputs = self.sentiment_combiner(
            aspect_features=aspect_outputs['aspect_features'],
            opinion_features=opinion_outputs['opinion_features'],
            interaction_features=hidden_states
        )
        
        # Feature fusion
        fused_features = self.feature_fusion(torch.cat([
            aspect_outputs['aspect_features'],
            opinion_outputs['opinion_features'],
            combination_outputs['combined_features']
        ], dim=-1))
        
        # Final predictions
        final_aspect_logits = self.final_aspect_head(fused_features)
        final_opinion_logits = self.final_opinion_head(fused_features)
        final_sentiment_logits = self.final_sentiment_head(fused_features)
        
        outputs = {
            'aspect_logits': final_aspect_logits,
            'opinion_logits': final_opinion_logits,
            'sentiment_logits': final_sentiment_logits,
            'implicit_aspect_logits': aspect_outputs['implicit_logits'],
            'implicit_opinion_logits': opinion_outputs['opinion_logits'],
            'implicit_sentiment_logits': combination_outputs['implicit_sentiment_logits'],
            'explicit_sentiment_logits': combination_outputs['explicit_sentiment_logits'],
            'attention_weights': aspect_outputs['attention_weights'],
            'fused_features': fused_features
        }
        
        # Contrastive learning during training
        if training and aspect_labels is not None:
            contrastive_losses = self.compute_contrastive_losses(
                aspect_features=aspect_outputs.get('contrastive_features'),
                opinion_features=opinion_outputs.get('contrastive_features'),
                aspect_labels=aspect_labels,
                opinion_labels=opinion_labels,
                sentiment_labels=sentiment_labels,
                attention_mask=attention_mask
            )
            outputs.update(contrastive_losses)
        
        return outputs
    
    def compute_contrastive_losses(self,
                                 aspect_features: torch.Tensor,
                                 opinion_features: torch.Tensor,
                                 aspect_labels: torch.Tensor,
                                 opinion_labels: torch.Tensor,
                                 sentiment_labels: torch.Tensor,
                                 attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all contrastive losses
        """
        losses = {}
        
        if aspect_features is not None and opinion_features is not None:
            # Create combined labels for triplet-level contrastive learning
            batch_size, seq_len = aspect_labels.shape
            
            # Combine aspect, opinion, sentiment into triplet labels
            triplet_labels = (aspect_labels * 100 + 
                            opinion_labels * 10 + 
                            sentiment_labels)
            
            # Create contrastive pairs
            combined_features = torch.cat([aspect_features, opinion_features], dim=-1)
            augmented_features, augmented_labels = self.create_contrastive_pairs(
                combined_features,
                {
                    'triplet': triplet_labels,
                    'aspect': aspect_labels,
                    'opinion': opinion_labels,
                    'sentiment': sentiment_labels
                }
            )
            
            # Supervised contrastive loss
            contrastive_outputs = self.contrastive_loss(
                features=augmented_features,
                labels=augmented_labels['triplet'][:, 0],  # Use original labels
                aspect_labels=augmented_labels['aspect'][:, 0],
                opinion_labels=augmented_labels['opinion'][:, 0],
                sentiment_labels=augmented_labels['sentiment'][:, 0]
            )
            
            for k, v in contrastive_outputs.items():
                losses[f'contrastive_{k}'] = v
            
            # InfoNCE loss for implicit detection
            # Create positive/negative samples for implicit vs explicit
            valid_mask = attention_mask.bool()
            
            # Flatten features and labels for InfoNCE
            flat_aspect_features = aspect_features[valid_mask]
            flat_aspect_labels = aspect_labels[valid_mask]
            
            if len(flat_aspect_features) > 1:
                # Create positive and negative samples
                unique_labels = torch.unique(flat_aspect_labels)
                
                for label in unique_labels:
                    if label == -100:  # Skip padding
                        continue
                        
                    # Positive samples (same label)
                    pos_mask = flat_aspect_labels == label
                    if pos_mask.sum() < 2:
                        continue
                        
                    pos_features = flat_aspect_features[pos_mask]
                    
                    # Negative samples (different labels)
                    neg_mask = flat_aspect_labels != label
                    if neg_mask.sum() == 0:
                        continue
                        
                    neg_features = flat_aspect_features[neg_mask]
                    
                    # Sample for InfoNCE
                    if len(pos_features) >= 2 and len(neg_features) >= 1:
                        query = pos_features[:1]
                        positives = pos_features[1:2].unsqueeze(1)
                        negatives = neg_features[:5].unsqueeze(1)  # Sample 5 negatives
                        
                        infonce_loss = self.infonce_loss(query, positives, negatives)
                        losses[f'infonce_aspect_{label.item()}'] = infonce_loss
            
            # Enhanced triplet loss
            if len(flat_aspect_features) > 2:
                triplet_outputs = self.triplet_loss(flat_aspect_features, flat_aspect_labels)
                losses['enhanced_triplet'] = triplet_outputs['loss']
                losses['triplet_stats'] = {
                    'num_triplets': triplet_outputs['num_triplets'],
                    'valid_triplets': triplet_outputs['valid_triplets']
                }
        
        return losses
    
    def extract_implicit_predictions(self,
                                   outputs: Dict[str, torch.Tensor],
                                   attention_mask: torch.Tensor,
                                   confidence_threshold: float = 0.7) -> Dict[str, List]:
        """
        Extract high-confidence implicit predictions
        """
        # Get implicit aspect spans
        implicit_aspect_spans = self.implicit_aspect_detector.extract_implicit_spans(
            outputs['implicit_aspect_logits'],
            attention_mask,
            threshold=confidence_threshold
        )
        
        # Get implicit opinion predictions
        implicit_opinion_probs = F.softmax(outputs['implicit_opinion_logits'], dim=-1)
        implicit_opinion_preds = implicit_opinion_probs.argmax(dim=-1)
        
        # Get sentiment combinations
        implicit_sentiment_probs = F.softmax(outputs['implicit_sentiment_logits'], dim=-1)
        explicit_sentiment_probs = F.softmax(outputs['explicit_sentiment_logits'], dim=-1)
        
        return {
            'implicit_aspect_spans': implicit_aspect_spans,
            'implicit_opinion_predictions': implicit_opinion_preds.cpu().tolist(),
            'implicit_sentiment_probs': implicit_sentiment_probs.cpu().tolist(),
            'explicit_sentiment_probs': explicit_sentiment_probs.cpu().tolist()
        }