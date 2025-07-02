"""
Advanced Negative Sampling Strategies for ABSA Contrastive Learning
Implements multiple sophisticated sampling techniques for 2024-2025 breakthroughs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import random


class AdaptiveNegativeSampler:
    """
    Adaptive negative sampling with curriculum learning
    Gradually increases difficulty of negative samples during training
    """
    
    def __init__(self, 
                 initial_strategy: str = 'random',
                 final_strategy: str = 'hard',
                 curriculum_steps: int = 1000,
                 temperature: float = 1.0):
        self.initial_strategy = initial_strategy
        self.final_strategy = final_strategy
        self.curriculum_steps = curriculum_steps
        self.temperature = temperature
        self.current_step = 0
        
        # Strategy weights for interpolation
        self.strategies = ['random', 'semi_hard', 'hard', 'focal']
        self.strategy_weights = {s: 0.25 for s in self.strategies}
        
    def update_curriculum(self, step: int):
        """Update sampling strategy based on training progress"""
        self.current_step = step
        progress = min(step / self.curriculum_steps, 1.0)
        
        # Interpolate between initial and final strategies
        if self.initial_strategy == 'random' and self.final_strategy == 'hard':
            # Gradually shift from random to hard
            self.strategy_weights = {
                'random': (1 - progress) * 0.8 + 0.1,
                'semi_hard': 0.3,
                'hard': progress * 0.6 + 0.1,
                'focal': 0.1
            }
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v/total_weight for k, v in self.strategy_weights.items()}
    
    def sample(self, 
               anchor_features: torch.Tensor,
               candidate_features: torch.Tensor,
               anchor_labels: torch.Tensor,
               candidate_labels: torch.Tensor,
               num_negatives: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negatives using current curriculum strategy
        """
        # Choose strategy based on weights
        strategy = np.random.choice(
            list(self.strategy_weights.keys()),
            p=list(self.strategy_weights.values())
        )
        
        if strategy == 'random':
            return self._random_sampling(anchor_labels, candidate_features, candidate_labels, num_negatives)
        elif strategy == 'semi_hard':
            return self._semi_hard_sampling(anchor_features, anchor_labels, candidate_features, candidate_labels, num_negatives)
        elif strategy == 'hard':
            return self._hard_sampling(anchor_features, anchor_labels, candidate_features, candidate_labels, num_negatives)
        elif strategy == 'focal':
            return self._focal_sampling(anchor_features, anchor_labels, candidate_features, candidate_labels, num_negatives)
    
    def _random_sampling(self, anchor_labels, candidate_features, candidate_labels, num_negatives):
        """Random negative sampling"""
        negatives = []
        neg_labels = []
        
        for anchor_label in anchor_labels:
            neg_mask = candidate_labels != anchor_label
            neg_candidates = candidate_features[neg_mask]
            neg_cand_labels = candidate_labels[neg_mask]
            
            if len(neg_candidates) == 0:
                continue
            
            num_samples = min(num_negatives, len(neg_candidates))
            indices = torch.randperm(len(neg_candidates))[:num_samples]
            
            negatives.append(neg_candidates[indices])
            neg_labels.append(neg_cand_labels[indices])
        
        if not negatives:
            return torch.empty(0, candidate_features.size(-1)), torch.empty(0, dtype=torch.long)
        
        return torch.cat(negatives, dim=0), torch.cat(neg_labels, dim=0)
    
    def _semi_hard_sampling(self, anchor_features, anchor_labels, candidate_features, candidate_labels, num_negatives):
        """Semi-hard negative sampling"""
        negatives = []
        neg_labels = []
        
        for i, anchor_label in enumerate(anchor_labels):
            anchor_emb = anchor_features[i:i+1]
            
            # Find positive samples for reference
            pos_mask = candidate_labels == anchor_label
            if not pos_mask.any():
                continue
                
            pos_candidates = candidate_features[pos_mask]
            pos_dist = torch.cdist(anchor_emb, pos_candidates, p=2).min().item()
            
            # Find negative samples
            neg_mask = candidate_labels != anchor_label
            neg_candidates = candidate_features[neg_mask]
            neg_cand_labels = candidate_labels[neg_mask]
            
            if len(neg_candidates) == 0:
                continue
            
            # Semi-hard: negatives closer than hardest positive but still negative
            neg_distances = torch.cdist(anchor_emb, neg_candidates, p=2).squeeze(0)
            semi_hard_mask = neg_distances > pos_dist
            
            if semi_hard_mask.any():
                semi_hard_candidates = neg_candidates[semi_hard_mask]
                semi_hard_labels = neg_cand_labels[semi_hard_mask]
                semi_hard_distances = neg_distances[semi_hard_mask]
                
                # Select closest semi-hard negatives
                num_samples = min(num_negatives, len(semi_hard_candidates))
                _, indices = semi_hard_distances.topk(num_samples, largest=False)
                
                negatives.append(semi_hard_candidates[indices])
                neg_labels.append(semi_hard_labels[indices])
            else:
                # Fallback to random if no semi-hard negatives
                num_samples = min(num_negatives, len(neg_candidates))
                indices = torch.randperm(len(neg_candidates))[:num_samples]
                negatives.append(neg_candidates[indices])
                neg_labels.append(neg_cand_labels[indices])
        
        if not negatives:
            return torch.empty(0, candidate_features.size(-1)), torch.empty(0, dtype=torch.long)
        
        return torch.cat(negatives, dim=0), torch.cat(neg_labels, dim=0)
    
    def _hard_sampling(self, anchor_features, anchor_labels, candidate_features, candidate_labels, num_negatives):
        """Hard negative sampling (closest negatives)"""
        negatives = []
        neg_labels = []
        
        for i, anchor_label in enumerate(anchor_labels):
            anchor_emb = anchor_features[i:i+1]
            
            neg_mask = candidate_labels != anchor_label
            neg_candidates = candidate_features[neg_mask]
            neg_cand_labels = candidate_labels[neg_mask]
            
            if len(neg_candidates) == 0:
                continue
            
            distances = torch.cdist(anchor_emb, neg_candidates, p=2).squeeze(0)
            num_samples = min(num_negatives, len(neg_candidates))
            _, hard_indices = distances.topk(num_samples, largest=False)
            
            negatives.append(neg_candidates[hard_indices])
            neg_labels.append(neg_cand_labels[hard_indices])
        
        if not negatives:
            return torch.empty(0, candidate_features.size(-1)), torch.empty(0, dtype=torch.long)
        
        return torch.cat(negatives, dim=0), torch.cat(neg_labels, dim=0)
    
    def _focal_sampling(self, anchor_features, anchor_labels, candidate_features, candidate_labels, num_negatives):
        """Focal sampling with temperature-scaled probabilities"""
        negatives = []
        neg_labels = []
        
        for i, anchor_label in enumerate(anchor_labels):
            anchor_emb = anchor_features[i:i+1]
            
            neg_mask = candidate_labels != anchor_label
            neg_candidates = candidate_features[neg_mask]
            neg_cand_labels = candidate_labels[neg_mask]
            
            if len(neg_candidates) == 0:
                continue
            
            # Compute similarities (higher = harder)
            similarities = F.cosine_similarity(anchor_emb, neg_candidates, dim=1)
            
            # Temperature-scaled probabilities
            probs = F.softmax(similarities / self.temperature, dim=0)
            
            num_samples = min(num_negatives, len(neg_candidates))
            try:
                indices = torch.multinomial(probs, num_samples, replacement=False)
            except RuntimeError:  # Fallback if multinomial fails
                indices = torch.randperm(len(neg_candidates))[:num_samples]
            
            negatives.append(neg_candidates[indices])
            neg_labels.append(neg_cand_labels[indices])
        
        if not negatives:
            return torch.empty(0, candidate_features.size(-1)), torch.empty(0, dtype=torch.long)
        
        return torch.cat(negatives, dim=0), torch.cat(neg_labels, dim=0)


class HierarchicalNegativeSampler:
    """
    Hierarchical negative sampling for aspect-opinion-sentiment relationships
    Samples negatives at different granularity levels
    """
    
    def __init__(self,
                 sampling_ratios: Dict[str, float] = None,
                 max_negatives_per_level: int = 3):
        
        # Default sampling ratios for different levels
        self.sampling_ratios = sampling_ratios or {
            'sentiment': 0.4,    # Different sentiment, same aspect/opinion
            'aspect': 0.3,       # Different aspect, same sentiment
            'opinion': 0.2,      # Different opinion, same aspect/sentiment
            'random': 0.1        # Completely random
        }
        
        self.max_negatives_per_level = max_negatives_per_level
        
    def sample_hierarchical_negatives(self,
                                    anchor_triplets: List[Tuple[int, int, int]],  # (aspect, opinion, sentiment)
                                    candidate_triplets: List[Tuple[int, int, int]],
                                    candidate_features: torch.Tensor,
                                    total_negatives: int = 10) -> Tuple[torch.Tensor, List[str]]:
        """
        Sample negatives at different hierarchy levels
        
        Args:
            anchor_triplets: List of (aspect_id, opinion_id, sentiment_id) for anchors
            candidate_triplets: List of (aspect_id, opinion_id, sentiment_id) for candidates
            candidate_features: [num_candidates, hidden_dim]
            total_negatives: Total number of negatives to sample
            
        Returns:
            Sampled negative features and their types
        """
        all_negatives = []
        negative_types = []
        
        # Calculate number of negatives per level
        num_sentiment = int(total_negatives * self.sampling_ratios['sentiment'])
        num_aspect = int(total_negatives * self.sampling_ratios['aspect'])
        num_opinion = int(total_negatives * self.sampling_ratios['opinion'])
        num_random = total_negatives - num_sentiment - num_aspect - num_opinion
        
        for anchor_triplet in anchor_triplets:
            anchor_aspect, anchor_opinion, anchor_sentiment = anchor_triplet
            
            # Level 1: Different sentiment, same aspect/opinion
            sentiment_negatives = []
            for i, (c_aspect, c_opinion, c_sentiment) in enumerate(candidate_triplets):
                if (c_aspect == anchor_aspect and 
                    c_opinion == anchor_opinion and 
                    c_sentiment != anchor_sentiment):
                    sentiment_negatives.append(i)
            
            # Level 2: Different aspect, same sentiment
            aspect_negatives = []
            for i, (c_aspect, c_opinion, c_sentiment) in enumerate(candidate_triplets):
                if (c_aspect != anchor_aspect and 
                    c_sentiment == anchor_sentiment):
                    aspect_negatives.append(i)
            
            # Level 3: Different opinion, same aspect/sentiment
            opinion_negatives = []
            for i, (c_aspect, c_opinion, c_sentiment) in enumerate(candidate_triplets):
                if (c_aspect == anchor_aspect and 
                    c_opinion != anchor_opinion and 
                    c_sentiment == anchor_sentiment):
                    opinion_negatives.append(i)
            
            # Level 4: Completely random negatives
            random_negatives = []
            for i, candidate_triplet in enumerate(candidate_triplets):
                if candidate_triplet != anchor_triplet:
                    random_negatives.append(i)
            
            # Sample from each level
            sampled_indices = []
            sampled_types = []
            
            # Sample sentiment negatives
            if sentiment_negatives and num_sentiment > 0:
                selected = random.sample(sentiment_negatives, 
                                       min(num_sentiment, len(sentiment_negatives)))
                sampled_indices.extend(selected)
                sampled_types.extend(['sentiment'] * len(selected))
            
            # Sample aspect negatives
            if aspect_negatives and num_aspect > 0:
                selected = random.sample(aspect_negatives,
                                       min(num_aspect, len(aspect_negatives)))
                sampled_indices.extend(selected)
                sampled_types.extend(['aspect'] * len(selected))
            
            # Sample opinion negatives
            if opinion_negatives and num_opinion > 0:
                selected = random.sample(opinion_negatives,
                                       min(num_opinion, len(opinion_negatives)))
                sampled_indices.extend(selected)
                sampled_types.extend(['opinion'] * len(selected))
            
            # Sample random negatives to fill remaining slots
            remaining_slots = total_negatives - len(sampled_indices)
            if random_negatives and remaining_slots > 0:
                # Exclude already selected indices
                available_random = [idx for idx in random_negatives if idx not in sampled_indices]
                if available_random:
                    selected = random.sample(available_random,
                                           min(remaining_slots, len(available_random)))
                    sampled_indices.extend(selected)
                    sampled_types.extend(['random'] * len(selected))
            
            # Get features for sampled indices
            if sampled_indices:
                sampled_features = candidate_features[sampled_indices]
                all_negatives.append(sampled_features)
                negative_types.extend(sampled_types)
        
        if all_negatives:
            return torch.cat(all_negatives, dim=0), negative_types
        else:
            return torch.empty(0, candidate_features.size(-1)), []


class DynamicNegativeSampler:
    """
    Dynamic negative sampling that adapts based on model performance
    Uses online hard example mining with momentum updates
    """
    
    def __init__(self,
                 memory_size: int = 1000,
                 momentum: float = 0.9,
                 update_frequency: int = 100,
                 difficulty_threshold: float = 0.7):
        
        self.memory_size = memory_size
        self.momentum = momentum
        self.update_frequency = update_frequency
        self.difficulty_threshold = difficulty_threshold
        
        # Memory banks for hard negatives
        self.hard_negatives_memory = {
            'features': [],
            'labels': [],
            'difficulties': []
        }
        
        self.update_counter = 0
        
    def update_memory(self,
                     negative_features: torch.Tensor,
                     negative_labels: torch.Tensor,
                     negative_difficulties: torch.Tensor):
        """
        Update memory bank with hard negatives
        
        Args:
            negative_features: [num_negatives, hidden_dim]
            negative_labels: [num_negatives]
            negative_difficulties: [num_negatives] - loss values or prediction confidences
        """
        self.update_counter += 1
        
        # Add to memory
        self.hard_negatives_memory['features'].append(negative_features.detach().cpu())
        self.hard_negatives_memory['labels'].append(negative_labels.detach().cpu())
        self.hard_negatives_memory['difficulties'].append(negative_difficulties.detach().cpu())
        
        # Maintain memory size
        if len(self.hard_negatives_memory['features']) > self.memory_size:
            # Remove oldest entries
            self.hard_negatives_memory['features'].pop(0)
            self.hard_negatives_memory['labels'].pop(0)
            self.hard_negatives_memory['difficulties'].pop(0)
        
        # Periodic cleanup - keep only hardest examples
        if self.update_counter % self.update_frequency == 0:
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Keep only the hardest examples in memory"""
        if not self.hard_negatives_memory['features']:
            return
        
        # Concatenate all memory
        all_features = torch.cat(self.hard_negatives_memory['features'], dim=0)
        all_labels = torch.cat(self.hard_negatives_memory['labels'], dim=0)
        all_difficulties = torch.cat(self.hard_negatives_memory['difficulties'], dim=0)
        
        # Keep top difficult examples
        keep_size = min(self.memory_size // 2, len(all_features))
        if keep_size > 0:
            _, top_indices = all_difficulties.topk(keep_size, largest=True)
            
            self.hard_negatives_memory['features'] = [all_features[top_indices]]
            self.hard_negatives_memory['labels'] = [all_labels[top_indices]]
            self.hard_negatives_memory['difficulties'] = [all_difficulties[top_indices]]
    
    def sample_from_memory(self, 
                          num_samples: int,
                          anchor_labels: torch.Tensor,
                          device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hard negatives from memory
        
        Args:
            num_samples: Number of samples to draw
            anchor_labels: Labels to avoid (positive samples)
            device: Target device
            
        Returns:
            Sampled features and labels
        """
        if not self.hard_negatives_memory['features']:
            return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
        
        # Concatenate memory
        all_features = torch.cat(self.hard_negatives_memory['features'], dim=0)
        all_labels = torch.cat(self.hard_negatives_memory['labels'], dim=0)
        all_difficulties = torch.cat(self.hard_negatives_memory['difficulties'], dim=0)
        
        # Filter out positive samples
        valid_mask = torch.ones(len(all_labels), dtype=torch.bool)
        for anchor_label in anchor_labels:
            valid_mask &= (all_labels != anchor_label.item())
        
        if not valid_mask.any():
            return torch.empty(0, all_features.size(-1)), torch.empty(0, dtype=torch.long)
        
        valid_features = all_features[valid_mask]
        valid_labels = all_labels[valid_mask]
        valid_difficulties = all_difficulties[valid_mask]
        
        # Sample based on difficulty scores
        num_samples = min(num_samples, len(valid_features))
        if num_samples == 0:
            return torch.empty(0, all_features.size(-1)), torch.empty(0, dtype=torch.long)
        
        # Probability-based sampling weighted by difficulty
        probs = F.softmax(valid_difficulties, dim=0)
        try:
            indices = torch.multinomial(probs, num_samples, replacement=False)
        except RuntimeError:
            indices = torch.randperm(len(valid_features))[:num_samples]
        
        return valid_features[indices].to(device), valid_labels[indices].to(device)


class MultiViewNegativeSampler:
    """
    Multi-view negative sampling for different ABSA perspectives
    Generates negatives from syntactic, semantic, and pragmatic views
    """
    
    def __init__(self,
                 view_weights: Dict[str, float] = None,
                 num_views: int = 3):
        
        self.view_weights = view_weights or {
            'syntactic': 0.3,    # Based on POS tags, dependencies
            'semantic': 0.4,     # Based on word embeddings, semantics
            'pragmatic': 0.3     # Based on context, discourse
        }
        
        self.num_views = num_views
        
    def generate_syntactic_negatives(self,
                                   anchor_features: torch.Tensor,
                                   anchor_syntax: torch.Tensor,
                                   candidate_features: torch.Tensor,
                                   candidate_syntax: torch.Tensor,
                                   num_negatives: int) -> torch.Tensor:
        """
        Generate syntactic negatives based on POS tags and dependencies
        """
        negatives = []
        
        for i, (anchor_feat, anchor_syn) in enumerate(zip(anchor_features, anchor_syntax)):
            # Find candidates with different syntactic patterns
            syntax_distances = torch.cdist(anchor_syn.unsqueeze(0), candidate_syntax, p=1)
            
            # Select candidates with most different syntax
            _, syntax_indices = syntax_distances.topk(num_negatives, largest=True, dim=1)
            syntax_indices = syntax_indices.squeeze(0)
            
            selected_features = candidate_features[syntax_indices]
            negatives.append(selected_features)
        
        if negatives:
            return torch.cat(negatives, dim=0)
        else:
            return torch.empty(0, anchor_features.size(-1))
    
    def generate_semantic_negatives(self,
                                  anchor_features: torch.Tensor,
                                  candidate_features: torch.Tensor,
                                  semantic_embeddings: torch.Tensor,
                                  num_negatives: int) -> torch.Tensor:
        """
        Generate semantic negatives based on word embeddings
        """
        negatives = []
        
        for i, anchor_feat in enumerate(anchor_features):
            # Compute semantic similarities
            semantic_sims = F.cosine_similarity(
                anchor_feat.unsqueeze(0), 
                semantic_embeddings, 
                dim=1
            )
            
            # Select semantically distant candidates
            _, semantic_indices = semantic_sims.topk(num_negatives, largest=False)
            
            selected_features = candidate_features[semantic_indices]
            negatives.append(selected_features)
        
        if negatives:
            return torch.cat(negatives, dim=0)
        else:
            return torch.empty(0, anchor_features.size(-1))
    
    def generate_pragmatic_negatives(self,
                                   anchor_features: torch.Tensor,
                                   anchor_context: torch.Tensor,
                                   candidate_features: torch.Tensor,
                                   candidate_context: torch.Tensor,
                                   num_negatives: int) -> torch.Tensor:
        """
        Generate pragmatic negatives based on context and discourse
        """
        negatives = []
        
        for i, (anchor_feat, anchor_ctx) in enumerate(zip(anchor_features, anchor_context)):
            # Compute context similarities
            context_sims = F.cosine_similarity(
                anchor_ctx.unsqueeze(0),
                candidate_context,
                dim=1
            )
            
            # Select candidates with different contexts
            _, context_indices = context_sims.topk(num_negatives, largest=False)
            
            selected_features = candidate_features[context_indices]
            negatives.append(selected_features)
        
        if negatives:
            return torch.cat(negatives, dim=0)
        else:
            return torch.empty(0, anchor_features.size(-1))
    
    def sample_multi_view_negatives(self,
                                  anchor_features: torch.Tensor,
                                  candidate_features: torch.Tensor,
                                  auxiliary_data: Dict[str, torch.Tensor],
                                  total_negatives: int) -> Tuple[torch.Tensor, List[str]]:
        """
        Sample negatives from multiple views
        
        Args:
            anchor_features: [batch_size, hidden_dim]
            candidate_features: [num_candidates, hidden_dim]
            auxiliary_data: Dictionary with 'syntax', 'semantic', 'context' data
            total_negatives: Total number of negatives to sample
            
        Returns:
            Sampled negative features and their view types
        """
        # Calculate negatives per view
        num_syntactic = int(total_negatives * self.view_weights['syntactic'])
        num_semantic = int(total_negatives * self.view_weights['semantic'])
        num_pragmatic = total_negatives - num_syntactic - num_semantic
        
        all_negatives = []
        negative_types = []
        
        # Syntactic negatives
        if num_syntactic > 0 and 'syntax' in auxiliary_data:
            syntactic_negs = self.generate_syntactic_negatives(
                anchor_features,
                auxiliary_data['syntax']['anchor'],
                candidate_features,
                auxiliary_data['syntax']['candidates'],
                num_syntactic
            )
            if len(syntactic_negs) > 0:
                all_negatives.append(syntactic_negs)
                negative_types.extend(['syntactic'] * len(syntactic_negs))
        
        # Semantic negatives
        if num_semantic > 0 and 'semantic' in auxiliary_data:
            semantic_negs = self.generate_semantic_negatives(
                anchor_features,
                candidate_features,
                auxiliary_data['semantic'],
                num_semantic
            )
            if len(semantic_negs) > 0:
                all_negatives.append(semantic_negs)
                negative_types.extend(['semantic'] * len(semantic_negs))
        
        # Pragmatic negatives
        if num_pragmatic > 0 and 'context' in auxiliary_data:
            pragmatic_negs = self.generate_pragmatic_negatives(
                anchor_features,
                auxiliary_data['context']['anchor'],
                candidate_features,
                auxiliary_data['context']['candidates'],
                num_pragmatic
            )
            if len(pragmatic_negs) > 0:
                all_negatives.append(pragmatic_negs)
                negative_types.extend(['pragmatic'] * len(pragmatic_negs))
        
        if all_negatives:
            return torch.cat(all_negatives, dim=0), negative_types
        else:
            return torch.empty(0, candidate_features.size(-1)), []


class NegativeSamplingManager:
    """
    Main manager class that coordinates different negative sampling strategies
    Implements the complete negative sampling pipeline for ABSA contrastive learning
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize all samplers
        self.adaptive_sampler = AdaptiveNegativeSampler(
            initial_strategy=getattr(config, 'initial_neg_strategy', 'random'),
            final_strategy=getattr(config, 'final_neg_strategy', 'hard'),
            curriculum_steps=getattr(config, 'curriculum_steps', 1000)
        )
        
        self.hierarchical_sampler = HierarchicalNegativeSampler(
            max_negatives_per_level=getattr(config, 'max_negs_per_level', 3)
        )
        
        self.dynamic_sampler = DynamicNegativeSampler(
            memory_size=getattr(config, 'neg_memory_size', 1000),
            momentum=getattr(config, 'neg_momentum', 0.9)
        )
        
        self.multiview_sampler = MultiViewNegativeSampler(
            num_views=getattr(config, 'num_neg_views', 3)
        )
        
        # Strategy selection weights
        self.strategy_weights = getattr(config, 'neg_strategy_weights', {
            'adaptive': 0.4,
            'hierarchical': 0.3,
            'dynamic': 0.2,
            'multiview': 0.1
        })
        
        self.current_step = 0
    
    def update_step(self, step: int):
        """Update all samplers with current training step"""
        self.current_step = step
        self.adaptive_sampler.update_curriculum(step)
    
    def update_hard_negatives(self,
                            negative_features: torch.Tensor,
                            negative_labels: torch.Tensor,
                            negative_losses: torch.Tensor):
        """Update dynamic sampler with hard negatives"""
        self.dynamic_sampler.update_memory(
            negative_features, negative_labels, negative_losses
        )
    
    def sample_negatives(self,
                        anchor_features: torch.Tensor,
                        anchor_labels: torch.Tensor,
                        candidate_features: torch.Tensor,
                        candidate_labels: torch.Tensor,
                        triplet_data: Optional[Dict] = None,
                        auxiliary_data: Optional[Dict] = None,
                        num_negatives: int = 10,
                        strategy: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Main negative sampling method
        
        Args:
            anchor_features: [batch_size, hidden_dim]
            anchor_labels: [batch_size]
            candidate_features: [num_candidates, hidden_dim]
            candidate_labels: [num_candidates]
            triplet_data: Optional triplet information for hierarchical sampling
            auxiliary_data: Optional auxiliary data for multi-view sampling
            num_negatives: Number of negatives to sample
            strategy: Specific strategy to use (optional)
            
        Returns:
            Dictionary with sampled negatives and metadata
        """
        if strategy is None:
            # Choose strategy based on weights
            strategy = np.random.choice(
                list(self.strategy_weights.keys()),
                p=list(self.strategy_weights.values())
            )
        
        device = anchor_features.device
        
        if strategy == 'adaptive':
            neg_features, neg_labels = self.adaptive_sampler.sample(
                anchor_features, candidate_features, anchor_labels, candidate_labels, num_negatives
            )
            negative_types = ['adaptive'] * len(neg_features) if len(neg_features) > 0 else []
            
        elif strategy == 'hierarchical' and triplet_data is not None:
            neg_features, negative_types = self.hierarchical_sampler.sample_hierarchical_negatives(
                triplet_data['anchor_triplets'],
                triplet_data['candidate_triplets'],
                candidate_features,
                num_negatives
            )
            neg_labels = torch.zeros(len(neg_features), dtype=torch.long, device=device)
            
        elif strategy == 'dynamic':
            # Mix dynamic memory samples with fresh samples
            memory_samples = min(num_negatives // 2, 5)
            fresh_samples = num_negatives - memory_samples
            
            # Sample from memory
            mem_features, mem_labels = self.dynamic_sampler.sample_from_memory(
                memory_samples, anchor_labels, device
            )
            
            # Sample fresh negatives
            fresh_features, fresh_labels = self.adaptive_sampler.sample(
                anchor_features, candidate_features, anchor_labels, candidate_labels, fresh_samples
            )
            
            # Combine
            if len(mem_features) > 0 and len(fresh_features) > 0:
                neg_features = torch.cat([mem_features, fresh_features], dim=0)
                neg_labels = torch.cat([mem_labels, fresh_labels], dim=0)
            elif len(mem_features) > 0:
                neg_features, neg_labels = mem_features, mem_labels
            else:
                neg_features, neg_labels = fresh_features, fresh_labels
            
            negative_types = (['memory'] * len(mem_features) + 
                            ['fresh'] * len(fresh_features))
            
        elif strategy == 'multiview' and auxiliary_data is not None:
            neg_features, negative_types = self.multiview_sampler.sample_multi_view_negatives(
                anchor_features, candidate_features, auxiliary_data, num_negatives
            )
            neg_labels = torch.zeros(len(neg_features), dtype=torch.long, device=device)
            
        else:
            # Fallback to adaptive sampling
            neg_features, neg_labels = self.adaptive_sampler.sample(
                anchor_features, candidate_features, anchor_labels, candidate_labels, num_negatives
            )
            negative_types = ['fallback'] * len(neg_features) if len(neg_features) > 0 else []
        
        return {
            'negative_features': neg_features.to(device) if len(neg_features) > 0 else torch.empty(0, anchor_features.size(-1), device=device),
            'negative_labels': neg_labels.to(device) if len(neg_labels) > 0 else torch.empty(0, dtype=torch.long, device=device),
            'negative_types': negative_types,
            'strategy_used': strategy,
            'num_sampled': len(neg_features) if len(neg_features) > 0 else 0
        }