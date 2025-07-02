# src/models/few_shot_learner.py
"""
Complete Few-Shot Learning Implementation for ABSA
2024-2025 Breakthrough: Meta-learning, Domain Adaptation, and Cross-Domain Transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

class DualRelationsPropagation(nn.Module):
    """
    Dual Relations Propagation (DRP) Network
    
    2024-2025 breakthrough: Metric-free approach for few-shot ABSA
    Models associated relations among aspects via similarity and diversity analysis
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.num_relations = getattr(config, 'num_relations', 16)
        self.propagation_steps = getattr(config, 'propagation_steps', 3)
        self.temperature = getattr(config, 'few_shot_temperature', 0.1)
        
        # Relation encoders
        self.similarity_encoder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_relations),
            nn.Softmax(dim=-1)
        )
        
        self.diversity_encoder = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_relations),
            nn.Softmax(dim=-1)
        )
        
        # Relation propagation networks
        self.relation_propagator = nn.ModuleList([
            RelationPropagationLayer(self.hidden_size, self.num_relations)
            for _ in range(self.propagation_steps)
        ])
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 3)  # Sentiment classes
        )
    
    def forward(self, support_features, support_labels, query_features):
        """
        Forward pass for few-shot learning
        
        Args:
            support_features: Features from support set [num_support, hidden_size]
            support_labels: Labels for support set [num_support]
            query_features: Features from query set [num_query, hidden_size]
        
        Returns:
            Query predictions and relation weights
        """
        # Build relation graphs
        similarity_relations = self._build_similarity_relations(support_features, query_features)
        diversity_relations = self._build_diversity_relations(support_features, query_features)
        
        # Initialize node features
        all_features = torch.cat([support_features, query_features], dim=0)
        node_features = all_features
        
        # Propagate through relation networks
        for layer in self.relation_propagator:
            node_features = layer(node_features, similarity_relations, diversity_relations)
        
        # Extract query features and classify
        num_support = support_features.size(0)
        query_node_features = node_features[num_support:]
        
        predictions = self.classifier(query_node_features)
        
        return {
            'predictions': predictions,
            'similarity_relations': similarity_relations,
            'diversity_relations': diversity_relations,
            'node_features': node_features
        }
    
    def _build_similarity_relations(self, support_features, query_features):
        """Build similarity-based relation matrix"""
        all_features = torch.cat([support_features, query_features], dim=0)
        num_nodes = all_features.size(0)
        
        # Pairwise similarities
        similarity_matrix = torch.zeros(num_nodes, num_nodes, self.num_relations, device=all_features.device)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Concatenate features for relation encoding
                pair_features = torch.cat([all_features[i], all_features[j]], dim=0)
                relation_weights = self.similarity_encoder(pair_features.unsqueeze(0))
                
                similarity_matrix[i, j] = relation_weights.squeeze(0)
                similarity_matrix[j, i] = relation_weights.squeeze(0)  # Symmetric
        
        return similarity_matrix
    
    def _build_diversity_relations(self, support_features, query_features):
        """Build diversity-based relation matrix"""
        all_features = torch.cat([support_features, query_features], dim=0)
        num_nodes = all_features.size(0)
        
        # Pairwise diversity (complementary to similarity)
        diversity_matrix = torch.zeros(num_nodes, num_nodes, self.num_relations, device=all_features.device)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Use difference features for diversity
                diff_features = torch.abs(all_features[i] - all_features[j])
                pair_features = torch.cat([all_features[i], diff_features], dim=0)
                relation_weights = self.diversity_encoder(pair_features.unsqueeze(0))
                
                diversity_matrix[i, j] = relation_weights.squeeze(0)
                diversity_matrix[j, i] = relation_weights.squeeze(0)  # Symmetric
        
        return diversity_matrix


class RelationPropagationLayer(nn.Module):
    """Single layer of relation propagation"""
    
    def __init__(self, hidden_size, num_relations):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        
        # Relation-specific transformation matrices
        self.relation_transforms = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_relations)
        ])
        
        # Aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Residual connection
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, node_features, similarity_relations, diversity_relations):
        """Propagate features through relations"""
        num_nodes = node_features.size(0)
        device = node_features.device
        
        # Initialize aggregated features
        aggregated_features = torch.zeros_like(node_features)
        
        for i in range(num_nodes):
            node_messages = []
            
            for j in range(num_nodes):
                if i == j:
                    continue
                
                # Combine similarity and diversity relations
                combined_relations = (similarity_relations[i, j] + diversity_relations[i, j]) / 2
                
                # Apply relation-specific transformations
                neighbor_message = torch.zeros_like(node_features[j])
                for r in range(self.num_relations):
                    relation_weight = combined_relations[r]
                    transformed_feature = self.relation_transforms[r](node_features[j])
                    neighbor_message += relation_weight * transformed_feature
                
                node_messages.append(neighbor_message)
            
            if node_messages:
                # Aggregate messages
                stacked_messages = torch.stack(node_messages, dim=0)
                aggregated_message = stacked_messages.mean(dim=0)
                
                # Combine with original features
                combined = torch.cat([node_features[i], aggregated_message], dim=0)
                aggregated_features[i] = self.aggregator(combined.unsqueeze(0)).squeeze(0)
            else:
                aggregated_features[i] = node_features[i]
        
        # Residual connection and layer norm
        output = self.layer_norm(aggregated_features + node_features)
        
        return output


class AspectFocusedMetaLearning(nn.Module):
    """
    Aspect-Focused Meta-Learning (AFML)
    
    2024-2025 breakthrough: Constructs aspect-aware and aspect-contrastive
    representations using external knowledge for few-shot aspect category
    sentiment analysis
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.num_aspects = getattr(config, 'num_aspects', 50)  # Maximum number of aspects
        self.meta_learning_rate = getattr(config, 'meta_learning_rate', 0.01)
        
        # Aspect-aware encoders
        self.aspect_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # External knowledge integration
        self.knowledge_projector = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Meta-learning components
        self.meta_optimizer = MetaOptimizer(self.hidden_size, self.meta_learning_rate)
        
        # Contrastive sentence generator (simulated)
        self.contrastive_generator = ContrastiveSentenceGenerator(config)
        
        # Fast adaptation network
        self.adaptation_network = nn.ModuleDict({
            'aspect_adapter': nn.Linear(self.hidden_size, self.hidden_size),
            'sentiment_adapter': nn.Linear(self.hidden_size, 3)
        })
    
    def forward(self, support_data, query_data, external_knowledge=None):
        """
        Meta-learning forward pass
        
        Args:
            support_data: Support set with aspects and sentiments
            query_data: Query set for prediction
            external_knowledge: Optional external knowledge embeddings
        """
        # Extract aspect-aware representations
        support_aspects = self._extract_aspect_representations(support_data)
        query_aspects = self._extract_aspect_representations(query_data)
        
        # Integrate external knowledge if available
        if external_knowledge is not None:
            support_aspects = self._integrate_external_knowledge(support_aspects, external_knowledge)
            query_aspects = self._integrate_external_knowledge(query_aspects, external_knowledge)
        
        # Generate contrastive examples
        contrastive_data = self.contrastive_generator(support_data, query_data)
        
        # Meta-learning adaptation
        adapted_params = self.meta_optimizer.adapt(
            support_aspects, support_data['labels'], 
            contrastive_data
        )
        
        # Apply adapted parameters for query prediction
        query_predictions = self._predict_with_adapted_params(
            query_aspects, adapted_params
        )
        
        return {
            'predictions': query_predictions,
            'adapted_params': adapted_params,
            'aspect_representations': query_aspects,
            'contrastive_data': contrastive_data
        }
    
    def _extract_aspect_representations(self, data):
        """Extract aspect-focused representations"""
        # Encode text features
        text_features = data['features']  # Assuming pre-encoded
        
        # Apply aspect-aware encoding
        aspect_features = self.aspect_encoder(text_features)
        
        return aspect_features
    
    def _integrate_external_knowledge(self, features, external_knowledge):
        """Integrate external knowledge into representations"""
        # Combine features with external knowledge
        combined = torch.cat([features, external_knowledge], dim=-1)
        enhanced_features = self.knowledge_projector(combined)
        
        return enhanced_features
    
    def _predict_with_adapted_params(self, query_features, adapted_params):
        """Make predictions using adapted parameters"""
        # Apply adapted transformations
        adapted_features = query_features
        
        for param_name, param_value in adapted_params.items():
            if 'aspect' in param_name:
                adapted_features = F.linear(adapted_features, param_value)
            elif 'sentiment' in param_name:
                predictions = F.linear(adapted_features, param_value)
        
        return predictions


class MetaOptimizer(nn.Module):
    """Meta-optimizer for fast adaptation"""
    
    def __init__(self, hidden_size, learning_rate):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Learnable learning rate
        self.adaptive_lr = nn.Parameter(torch.tensor(learning_rate))
        
        # Meta-gradient computation network
        self.meta_network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def adapt(self, support_features, support_labels, contrastive_data):
        """Perform fast adaptation using meta-learning"""
        batch_size = support_features.size(0)
        
        # Initialize adaptation parameters
        adapted_params = {
            'aspect_weight': torch.randn(self.hidden_size, self.hidden_size, requires_grad=True),
            'sentiment_weight': torch.randn(3, self.hidden_size, requires_grad=True)
        }
        
        # Compute meta-gradients
        meta_loss = self._compute_meta_loss(support_features, support_labels, adapted_params)
        
        # Update parameters using meta-gradients
        meta_grads = torch.autograd.grad(meta_loss, list(adapted_params.values()), create_graph=True)
        
        updated_params = {}
        for (param_name, param_value), grad in zip(adapted_params.items(), meta_grads):
            updated_params[param_name] = param_value - self.adaptive_lr * grad
        
        return updated_params
    
    def _compute_meta_loss(self, features, labels, params):
        """Compute meta-learning loss"""
        # Apply current parameters
        transformed = F.linear(features, params['aspect_weight'])
        predictions = F.linear(transformed, params['sentiment_weight'])
        
        # Cross-entropy loss
        loss = F.cross_entropy(predictions, labels)
        
        return loss


class ContrastiveSentenceGenerator(nn.Module):
    """
    Generate contrastive sentences for aspect-focused learning
    
    2024-2025 innovation: Creates auxiliary contrastive sentences
    with external knowledge incorporation for better few-shot learning
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.vocab_size = getattr(config, 'vocab_size', 30000)
        
        # Contrastive generation components
        self.aspect_modifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.sentiment_flipper = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Knowledge integration for contrastive examples
        self.knowledge_integrator = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True
        )
    
    def forward(self, support_data, query_data):
        """Generate contrastive examples"""
        support_features = support_data['features']
        
        # Generate aspect-modified examples
        aspect_modified = self.aspect_modifier(support_features)
        
        # Generate sentiment-flipped examples
        sentiment_flipped = self.sentiment_flipper(support_features)
        
        # Create contrastive labels
        original_labels = support_data['labels']
        flipped_labels = self._flip_sentiment_labels(original_labels)
        
        contrastive_data = {
            'aspect_modified': aspect_modified,
            'sentiment_flipped': sentiment_flipped,
            'original_labels': original_labels,
            'flipped_labels': flipped_labels,
            'original_features': support_features
        }
        
        return contrastive_data
    
    def _flip_sentiment_labels(self, labels):
        """Flip sentiment labels for contrastive learning"""
        # 0: NEG, 1: NEU, 2: POS
        flipped = labels.clone()
        flipped[labels == 0] = 2  # NEG -> POS
        flipped[labels == 2] = 0  # POS -> NEG
        # NEU stays the same
        
        return flipped


class CrossDomainAspectLabelPropagation(nn.Module):
    """
    Cross-Domain Aspect Label Propagation (CD-ALPHN)
    
    2024-2025 breakthrough: Overcomes traditional two-stage transfer learning
    limitations through unified learning approaches addressing inconsistency
    between source and target domains
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.num_domains = getattr(config, 'num_domains', 5)
        self.propagation_alpha = getattr(config, 'propagation_alpha', 0.8)
        
        # Domain-specific encoders
        self.domain_encoders = nn.ModuleDict({
            f'domain_{i}': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
            for i in range(self.num_domains)
        })
        
        # Domain-invariant encoder
        self.invariant_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Label propagation matrix
        self.propagation_matrix = nn.Parameter(
            torch.eye(self.num_domains) * self.propagation_alpha + 
            torch.ones(self.num_domains, self.num_domains) * (1 - self.propagation_alpha) / (self.num_domains - 1)
        )
        
        # Domain adversarial discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, self.num_domains)
        )
        
        # Aspect classifier
        self.aspect_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),  # domain + invariant
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 3)  # Aspect labels
        )
    
    def forward(self, features, domain_ids, labels=None, training=True):
        """
        Cross-domain forward pass with label propagation
        
        Args:
            features: Input features [batch_size, hidden_size]
            domain_ids: Domain identifiers [batch_size]
            labels: Ground truth labels for training
            training: Whether in training mode
        """
        batch_size = features.size(0)
        device = features.device
        
        # Extract domain-specific and invariant features
        domain_features = []
        invariant_features = self.invariant_encoder(features)
        
        for i, domain_id in enumerate(domain_ids):
            domain_encoder = self.domain_encoders[f'domain_{domain_id.item()}']
            domain_feat = domain_encoder(features[i].unsqueeze(0))
            domain_features.append(domain_feat)
        
        domain_features = torch.cat(domain_features, dim=0)
        
        # Combine domain-specific and invariant features
        combined_features = torch.cat([domain_features, invariant_features], dim=-1)
        
        # Aspect classification
        aspect_predictions = self.aspect_classifier(combined_features)
        
        results = {
            'aspect_predictions': aspect_predictions,
            'domain_features': domain_features,
            'invariant_features': invariant_features,
            'combined_features': combined_features
        }
        
        if training:
            # Domain adversarial loss
            domain_predictions = self.domain_discriminator(invariant_features)
            
            # Label propagation loss
            if labels is not None:
                propagation_loss = self._compute_propagation_loss(
                    aspect_predictions, labels, domain_ids
                )
                results['propagation_loss'] = propagation_loss
            
            # Domain adversarial loss (maximize domain confusion for invariant features)
            domain_labels = domain_ids
            domain_loss = F.cross_entropy(domain_predictions, domain_labels)
            results['domain_loss'] = domain_loss
            results['domain_predictions'] = domain_predictions
        
        return results
    
    def _compute_propagation_loss(self, predictions, labels, domain_ids):
        """Compute cross-domain label propagation loss"""
        device = predictions.device
        batch_size = predictions.size(0)
        
        # Group predictions by domain
        domain_predictions = {}
        domain_labels = {}
        
        for i, domain_id in enumerate(domain_ids):
            domain_key = domain_id.item()
            if domain_key not in domain_predictions:
                domain_predictions[domain_key] = []
                domain_labels[domain_key] = []
            
            domain_predictions[domain_key].append(predictions[i])
            domain_labels[domain_key].append(labels[i])
        
        # Compute propagation loss
        propagation_loss = 0.0
        num_domains_present = len(domain_predictions)
        
        for source_domain in domain_predictions:
            for target_domain in domain_predictions:
                if source_domain != target_domain:
                    # Get propagation weight
                    prop_weight = self.propagation_matrix[source_domain, target_domain]
                    
                    # Compute cross-domain consistency loss
                    source_preds = torch.stack(domain_predictions[source_domain])
                    target_preds = torch.stack(domain_predictions[target_domain])
                    
                    # KL divergence for soft propagation
                    source_probs = F.softmax(source_preds, dim=-1)
                    target_log_probs = F.log_softmax(target_preds, dim=-1)
                    
                    # Average over available samples
                    min_samples = min(len(source_probs), len(target_log_probs))
                    if min_samples > 0:
                        kl_loss = F.kl_div(
                            target_log_probs[:min_samples], 
                            source_probs[:min_samples], 
                            reduction='mean'
                        )
                        propagation_loss += prop_weight * kl_loss
        
        return propagation_loss / max(num_domains_present * (num_domains_present - 1), 1)
    
    def adapt_to_target_domain(self, source_features, source_labels, source_domain_id,
                             target_features, target_domain_id, adaptation_steps=5):
        """
        Adapt model to target domain using few-shot samples
        
        Args:
            source_features: Source domain features
            source_labels: Source domain labels
            source_domain_id: Source domain identifier
            target_features: Target domain features (few-shot)
            target_domain_id: Target domain identifier
            adaptation_steps: Number of adaptation steps
        """
        # Initialize target domain encoder if not exists
        target_key = f'domain_{target_domain_id}'
        if target_key not in self.domain_encoders:
            # Initialize from source domain
            source_key = f'domain_{source_domain_id}'
            self.domain_encoders[target_key] = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
            
            # Copy weights from source domain as initialization
            self.domain_encoders[target_key].load_state_dict(
                self.domain_encoders[source_key].state_dict()
            )
        
        # Fine-tune on target domain samples
        optimizer = torch.optim.Adam(
            self.domain_encoders[target_key].parameters(), 
            lr=0.001
        )
        
        for step in range(adaptation_steps):
            # Forward pass on target samples
            target_domain_ids = torch.full((target_features.size(0),), target_domain_id)
            
            outputs = self.forward(
                target_features, target_domain_ids, 
                training=True
            )
            
            # Compute adaptation loss (unsupervised domain adaptation)
            # Use domain consistency and feature similarity
            invariant_features = outputs['invariant_features']
            
            # Minimize feature distance between source and target
            if source_features.size(0) > 0:
                source_invariant = self.invariant_encoder(source_features)
                adaptation_loss = F.mse_loss(
                    invariant_features.mean(dim=0), 
                    source_invariant.mean(dim=0)
                )
                
                adaptation_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        return self.domain_encoders[target_key]


class FewShotABSAEvaluator:
    """
    Comprehensive evaluator for few-shot ABSA methods
    
    Implements evaluation protocols for cross-domain transfer,
    few-shot learning, and meta-learning approaches
    """
    
    def __init__(self, config):
        self.config = config
        self.k_shots = getattr(config, 'few_shot_k', 5)
        self.num_episodes = getattr(config, 'num_episodes', 100)
        
    def evaluate_few_shot_performance(self, model, datasets, domains):
        """
        Evaluate few-shot performance across domains
        
        Args:
            model: Few-shot ABSA model
            datasets: Dictionary of domain datasets
            domains: List of domain names
        
        Returns:
            Comprehensive evaluation results
        """
        results = {}
        
        for target_domain in domains:
            domain_results = []
            
            for episode in range(self.num_episodes):
                # Sample support and query sets
                support_data, query_data = self._sample_episode(
                    datasets[target_domain], self.k_shots
                )
                
                # Few-shot prediction
                predictions = model(support_data, query_data)
                
                # Calculate metrics
                episode_metrics = self._calculate_episode_metrics(
                    predictions, query_data['labels']
                )
                
                domain_results.append(episode_metrics)
            
            # Aggregate results for domain
            results[target_domain] = self._aggregate_episode_results(domain_results)
        
        return results
    
    def evaluate_cross_domain_transfer(self, model, source_domains, target_domains, datasets):
        """
        Evaluate cross-domain transfer performance
        
        Tests model's ability to transfer knowledge from source domains
        to target domains with minimal target domain data
        """
        transfer_results = {}
        
        for source_domain in source_domains:
            for target_domain in target_domains:
                if source_domain == target_domain:
                    continue
                
                # Train on source domain
                source_data = datasets[source_domain]['train']
                
                # Few-shot adaptation to target domain
                target_support, target_query = self._sample_episode(
                    datasets[target_domain], self.k_shots
                )
                
                # Perform transfer
                if hasattr(model, 'adapt_to_target_domain'):
                    model.adapt_to_target_domain(
                        source_data['features'], source_data['labels'], 
                        source_domains.index(source_domain),
                        target_support['features'], 
                        target_domains.index(target_domain)
                    )
                
                # Evaluate on target domain
                predictions = model(target_support, target_query)
                metrics = self._calculate_episode_metrics(
                    predictions, target_query['labels']
                )
                
                transfer_key = f"{source_domain}_to_{target_domain}"
                transfer_results[transfer_key] = metrics
        
        return transfer_results
    
    def _sample_episode(self, dataset, k_shots):
        """Sample support and query sets for an episode"""
        # Get unique classes
        unique_labels = torch.unique(dataset['labels'])
        
        support_indices = []
        query_indices = []
        
        for label in unique_labels:
            label_indices = torch.where(dataset['labels'] == label)[0]
            
            # Sample k-shot support examples
            if len(label_indices) >= k_shots:
                support_idx = torch.randperm(len(label_indices))[:k_shots]
                support_indices.extend(label_indices[support_idx].tolist())
                
                # Use remaining as query
                remaining_idx = torch.randperm(len(label_indices))[k_shots:]
                if len(remaining_idx) > 0:
                    query_indices.extend(label_indices[remaining_idx].tolist())
        
        # Create support and query sets
        support_data = {
            'features': dataset['features'][support_indices],
            'labels': dataset['labels'][support_indices]
        }
        
        query_data = {
            'features': dataset['features'][query_indices],
            'labels': dataset['labels'][query_indices]
        }
        
        return support_data, query_data
    
    def _calculate_episode_metrics(self, predictions, true_labels):
        """Calculate metrics for a single episode"""
        if isinstance(predictions, dict):
            predictions = predictions['predictions']
        
        # Convert to class predictions
        if predictions.dim() > 1:
            pred_labels = predictions.argmax(dim=-1)
        else:
            pred_labels = predictions
        
        # Calculate accuracy
        accuracy = (pred_labels == true_labels).float().mean().item()
        
        # Calculate F1 score (macro average)
        f1_scores = []
        for class_id in torch.unique(true_labels):
            true_pos = ((pred_labels == class_id) & (true_labels == class_id)).sum().item()
            false_pos = ((pred_labels == class_id) & (true_labels != class_id)).sum().item()
            false_neg = ((pred_labels != class_id) & (true_labels == class_id)).sum().item()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': f1_scores
        }
    
    def _aggregate_episode_results(self, episode_results):
        """Aggregate results across episodes"""
        if not episode_results:
            return {'accuracy': 0.0, 'macro_f1': 0.0}
        
        accuracies = [r['accuracy'] for r in episode_results]
        f1_scores = [r['macro_f1'] for r in episode_results]
        
        return {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'accuracy_ci': self._compute_confidence_interval(accuracies),
            'f1_ci': self._compute_confidence_interval(f1_scores)
        }
    
    def _compute_confidence_interval(self, values, confidence=0.95):
        """Compute confidence interval for values"""
        if len(values) == 0:
            return (0.0, 0.0)
        
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        
        # t-distribution critical value (approximation)
        t_crit = 1.96  # for 95% confidence, large n
        margin_error = t_crit * std / np.sqrt(n)
        
        return (mean - margin_error, mean + margin_error)


# Integration class for complete few-shot learning pipeline
class CompleteFewShotABSA(nn.Module):
    """
    Complete few-shot ABSA system combining all breakthrough methods
    
    Integrates DRP, AFML, and CD-ALPHN for comprehensive few-shot learning
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.drp = DualRelationsPropagation(config)
        self.afml = AspectFocusedMetaLearning(config)
        self.cd_alphn = CrossDomainAspectLabelPropagation(config)
        
        # Method selection
        self.use_drp = getattr(config, 'use_drp', True)
        self.use_afml = getattr(config, 'use_afml', True)
        self.use_cd_alphn = getattr(config, 'use_cd_alphn', True)
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        
    def forward(self, support_data, query_data, domain_ids=None, external_knowledge=None):
        """
        Complete few-shot forward pass combining all methods
        """
        predictions = []
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # DRP predictions
        if self.use_drp:
            drp_outputs = self.drp(
                support_data['features'], support_data['labels'], 
                query_data['features']
            )
            predictions.append(drp_outputs['predictions'])
        
        # AFML predictions
        if self.use_afml:
            afml_outputs = self.afml(
                support_data, query_data, external_knowledge
            )
            predictions.append(afml_outputs['predictions'])
        
        # CD-ALPHN predictions
        if self.use_cd_alphn and domain_ids is not None:
            cd_outputs = self.cd_alphn(
                query_data['features'], domain_ids, training=False
            )
            predictions.append(cd_outputs['aspect_predictions'])
        
        # Ensemble predictions
        if predictions:
            # Weighted ensemble
            ensemble_pred = torch.zeros_like(predictions[0])
            total_weight = 0.0
            
            for i, pred in enumerate(predictions):
                if i < len(weights):
                    ensemble_pred += weights[i] * pred
                    total_weight += weights[i]
                else:
                    ensemble_pred += pred
                    total_weight += 1.0
            
            ensemble_pred = ensemble_pred / total_weight
        else:
            # Fallback
            ensemble_pred = torch.zeros(query_data['features'].size(0), 3)
        
        return {
            'predictions': ensemble_pred,
            'individual_predictions': predictions,
            'ensemble_weights': weights.detach().cpu().numpy()
        }