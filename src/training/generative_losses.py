# src/training/generative_losses.py
"""
Specialized Loss Functions for Generative ABSA
Implements generation-specific losses for training sequence-to-sequence ABSA models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from collections import defaultdict
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import logging

logger = logging.getLogger(__name__)


class GenerativeLoss(nn.Module):
    """
    Main loss function for generative ABSA training
    Combines multiple generation-specific losses
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Loss weights
        self.generation_weight = getattr(config, 'generation_loss_weight', 1.0)
        self.bleu_weight = getattr(config, 'bleu_loss_weight', 0.1)
        self.rouge_weight = getattr(config, 'rouge_loss_weight', 0.1)
        self.triplet_recovery_weight = getattr(config, 'triplet_recovery_weight', 0.2)
        self.consistency_weight = getattr(config, 'consistency_loss_weight', 0.05)
        self.coverage_weight = getattr(config, 'coverage_loss_weight', 0.1)
        
        # Initialize sub-losses
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.bleu_loss = BLEULoss()
        self.rouge_loss = ROUGELoss()
        self.triplet_recovery_loss = TripletRecoveryLoss(config)
        self.consistency_loss = ConsistencyLoss(config)
        self.coverage_loss = CoverageLoss(config)
        
        # Label smoothing
        self.label_smoothing = getattr(config, 'label_smoothing', 0.1)
        if self.label_smoothing > 0:
            self.cross_entropy_loss = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=self.label_smoothing
            )
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute combined generative loss
        
        Args:
            predictions: Model predictions containing logits, generated_text, etc.
            targets: Target data containing target_ids, original_triplets, etc.
        
        Returns:
            Dictionary of losses
        """
        
        losses = {}
        total_loss = 0.0
        
        # 1. Standard generation loss (cross-entropy)
        if 'logits' in predictions and 'target_ids' in targets:
            logits = predictions['logits']
            target_ids = targets['target_ids']
            
            # Reshape for loss computation
            vocab_size = logits.size(-1)
            generation_loss = self.cross_entropy_loss(
                logits.view(-1, vocab_size),
                target_ids.view(-1)
            )
            
            losses['generation_loss'] = generation_loss
            total_loss += self.generation_weight * generation_loss
        
        # 2. BLEU-based loss
        if 'generated_text' in predictions and 'target_text' in targets:
            bleu_loss = self.bleu_loss(
                predictions['generated_text'],
                targets['target_text']
            )
            losses['bleu_loss'] = bleu_loss
            total_loss += self.bleu_weight * bleu_loss
        
        # 3. ROUGE-based loss
        if 'generated_text' in predictions and 'target_text' in targets:
            rouge_loss = self.rouge_loss(
                predictions['generated_text'],
                targets['target_text']
            )
            losses['rouge_loss'] = rouge_loss
            total_loss += self.rouge_weight * rouge_loss
        
        # 4. Triplet recovery loss
        if 'generated_triplets' in predictions and 'original_triplets' in targets:
            triplet_loss = self.triplet_recovery_loss(
                predictions['generated_triplets'],
                targets['original_triplets']
            )
            losses['triplet_recovery_loss'] = triplet_loss
            total_loss += self.triplet_recovery_weight * triplet_loss
        
        # 5. Consistency loss (between classification and generation)
        if 'classification_outputs' in predictions:
            consistency_loss = self.consistency_loss(
                predictions['classification_outputs'],
                predictions.get('generated_triplets', [])
            )
            losses['consistency_loss'] = consistency_loss
            total_loss += self.consistency_weight * consistency_loss
        
        # 6. Coverage loss (ensure all aspects/opinions are generated)
        if 'attention_weights' in predictions and 'original_triplets' in targets:
            coverage_loss = self.coverage_loss(
                predictions['attention_weights'],
                targets['original_triplets']
            )
            losses['coverage_loss'] = coverage_loss
            total_loss += self.coverage_weight * coverage_loss
        
        losses['total_loss'] = total_loss
        return losses


class BLEULoss(nn.Module):
    """BLEU score based loss for generation quality"""
    
    def __init__(self, n_gram: int = 4):
        super().__init__()
        self.n_gram = n_gram
        self.smoothing = SmoothingFunction()
    
    def forward(self, 
                generated_texts: List[str], 
                target_texts: List[str]) -> torch.Tensor:
        """Compute BLEU-based loss"""
        
        if len(generated_texts) != len(target_texts):
            logger.warning(f"BLEU loss: Length mismatch - generated: {len(generated_texts)}, targets: {len(target_texts)}")
            return torch.tensor(0.0, requires_grad=True)
        
        bleu_scores = []
        
        for gen_text, target_text in zip(generated_texts, target_texts):
            # Tokenize
            gen_tokens = gen_text.lower().split()
            target_tokens = target_text.lower().split()
            
            # Compute BLEU score
            try:
                bleu_score = sentence_bleu(
                    [target_tokens], 
                    gen_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),  # 4-gram BLEU
                    smoothing_function=self.smoothing.method1
                )
                bleu_scores.append(bleu_score)
            except:
                bleu_scores.append(0.0)
        
        # Convert to loss (1 - BLEU)
        avg_bleu = np.mean(bleu_scores)
        bleu_loss = 1.0 - avg_bleu
        
        return torch.tensor(bleu_loss, requires_grad=True)


class ROUGELoss(nn.Module):
    """ROUGE score based loss for generation quality"""
    
    def __init__(self, rouge_type: str = 'rouge1'):
        super().__init__()
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    
    def forward(self, 
                generated_texts: List[str], 
                target_texts: List[str]) -> torch.Tensor:
        """Compute ROUGE-based loss"""
        
        if len(generated_texts) != len(target_texts):
            logger.warning(f"ROUGE loss: Length mismatch - generated: {len(generated_texts)}, targets: {len(target_texts)}")
            return torch.tensor(0.0, requires_grad=True)
        
        rouge_scores = []
        
        for gen_text, target_text in zip(generated_texts, target_texts):
            try:
                scores = self.scorer.score(target_text, gen_text)
                rouge_score = scores[self.rouge_type].fmeasure
                rouge_scores.append(rouge_score)
            except:
                rouge_scores.append(0.0)
        
        # Convert to loss (1 - ROUGE)
        avg_rouge = np.mean(rouge_scores)
        rouge_loss = 1.0 - avg_rouge
        
        return torch.tensor(rouge_loss, requires_grad=True)


class TripletRecoveryLoss(nn.Module):
    """
    Novel loss function for ABSA: Triplet Recovery Score (TRS)
    Measures how well original triplets can be recovered from generated text
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Matching thresholds
        self.exact_match_weight = 1.0
        self.partial_match_weight = 0.5
        self.sentiment_weight = 0.8
        
        # Penalties
        self.missing_penalty = 1.0
        self.extra_penalty = 0.5
    
    def forward(self, 
                generated_triplets: List[List[Dict[str, str]]], 
                original_triplets: List[List[Dict[str, str]]]) -> torch.Tensor:
        """
        Compute triplet recovery loss
        
        Args:
            generated_triplets: List of triplet lists for each example
            original_triplets: List of original triplet lists for each example
        
        Returns:
            Triplet recovery loss
        """
        
        if len(generated_triplets) != len(original_triplets):
            logger.warning(f"TRS loss: Length mismatch - generated: {len(generated_triplets)}, original: {len(original_triplets)}")
            return torch.tensor(0.0, requires_grad=True)
        
        total_recovery_score = 0.0
        total_examples = len(original_triplets)
        
        for gen_triplets, orig_triplets in zip(generated_triplets, original_triplets):
            if not orig_triplets:  # Skip if no original triplets
                continue
            
            recovery_score = self._compute_triplet_recovery_score(gen_triplets, orig_triplets)
            total_recovery_score += recovery_score
        
        # Average recovery score
        avg_recovery_score = total_recovery_score / max(total_examples, 1)
        
        # Convert to loss (1 - recovery_score)
        recovery_loss = 1.0 - avg_recovery_score
        
        return torch.tensor(recovery_loss, requires_grad=True)
    
    def _compute_triplet_recovery_score(self, 
                                       generated_triplets: List[Dict[str, str]], 
                                       original_triplets: List[Dict[str, str]]) -> float:
        """Compute recovery score for single example"""
        
        if not original_triplets:
            return 1.0 if not generated_triplets else 0.0
        
        total_score = 0.0
        
        for orig_triplet in original_triplets:
            best_match_score = 0.0
            
            for gen_triplet in generated_triplets:
                match_score = self._compute_triplet_match_score(gen_triplet, orig_triplet)
                best_match_score = max(best_match_score, match_score)
            
            total_score += best_match_score
        
        # Penalty for extra generated triplets
        extra_triplets = max(0, len(generated_triplets) - len(original_triplets))
        penalty = extra_triplets * self.extra_penalty
        
        # Normalize by number of original triplets
        recovery_score = (total_score - penalty) / len(original_triplets)
        return max(0.0, min(1.0, recovery_score))
    
    def _compute_triplet_match_score(self, 
                                    generated: Dict[str, str], 
                                    original: Dict[str, str]) -> float:
        """Compute match score between two triplets"""
        
        scores = []
        
        # Aspect matching
        if 'aspect' in generated and 'aspect' in original:
            aspect_score = self._compute_text_similarity(
                generated['aspect'], original['aspect']
            )
            scores.append(aspect_score)
        
        # Opinion matching
        if 'opinion' in generated and 'opinion' in original:
            opinion_score = self._compute_text_similarity(
                generated['opinion'], original['opinion']
            )
            scores.append(opinion_score)
        
        # Sentiment matching
        if 'sentiment' in generated and 'sentiment' in original:
            sentiment_score = self._compute_sentiment_similarity(
                generated['sentiment'], original['sentiment']
            )
            scores.append(sentiment_score * self.sentiment_weight)
        
        # Return average score
        return np.mean(scores) if scores else 0.0
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings"""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return self.exact_match_weight
        
        # Partial match (substring)
        if text1 in text2 or text2 in text1:
            return self.partial_match_weight
        
        # Token overlap
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if tokens1 and tokens2:
            overlap = len(tokens1.intersection(tokens2))
            total = len(tokens1.union(tokens2))
            return (overlap / total) * self.partial_match_weight
        
        return 0.0
    
    def _compute_sentiment_similarity(self,