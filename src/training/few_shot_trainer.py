# src/training/few_shot_trainer.py
"""
Few-Shot Learning Training Integration for ABSA
Integrates few-shot learning components into main training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict

from ..models.few_shot_learner import (
    CompleteFewShotABSA, 
    FewShotEvaluator,
    DualRelationsPropagation,
    AspectFocusedMetaLearning,
    CrossDomainAspectLabelPropagation,
    InstructionPromptFewShot
)


class FewShotABSATrainer:
    """
    Complete trainer for few-shot ABSA with all breakthrough methods
    
    Handles episodic training, domain adaptation, and meta-learning
    """
    
    def __init__(self, config, model, tokenizer, device='cuda'):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize few-shot components
        self.few_shot_model = CompleteFewShotABSA(config).to(device)
        self.few_shot_evaluator = FewShotEvaluator(config)
        
        # Training parameters
        self.episodes_per_epoch = getattr(config, 'episodes_per_epoch', 100)
        self.few_shot_k = getattr(config, 'few_shot_k', 5)
        self.adaptation_steps = getattr(config, 'adaptation_steps', 5)
        
        # Optimizer for few-shot components
        self.few_shot_optimizer = torch.optim.Adam(
            self.few_shot_model.parameters(),
            lr=getattr(config, 'meta_learning_rate', 0.01),
            weight_decay=getattr(config, 'weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.few_shot_scheduler = torch.optim.lr_scheduler.StepLR(
            self.few_shot_optimizer,
            step_size=10,
            gamma=0.9
        )
        
        # External knowledge (simulated)
        self.external_knowledge = self._initialize_external_knowledge()
        
        # Instruction templates
        self.instruction_templates = self._initialize_instruction_templates()
        
        # Performance tracking
        self.training_metrics = defaultdict(list)
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def train_few_shot_epoch(self, train_dataset, val_dataset=None):
        """
        Train one epoch with episodic few-shot learning
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
        """
        self.few_shot_model.train()
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        # Progress bar
        pbar = tqdm(range(self.episodes_per_epoch), desc="Few-shot episodes")
        
        for episode in pbar:
            # Sample episode
            support_data, query_data, domain_ids = self._sample_episode(train_dataset)
            
            # Zero gradients
            self.few_shot_optimizer.zero_grad()
            
            # Forward pass with loss computation
            total_loss, loss_components = self.few_shot_model.compute_few_shot_loss(
                support_data=support_data,
                query_data=query_data,
                domain_ids=domain_ids,
                external_knowledge=self.external_knowledge,
                instruction_templates=self.instruction_templates
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.few_shot_model.parameters(), max_norm=1.0)
            
            # Update
            self.few_shot_optimizer.step()
            
            # Track metrics
            epoch_losses.append(total_loss.item())
            for loss_name, loss_value in loss_components.items():
                epoch_metrics[loss_name].append(loss_value)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Avg_Loss': f"{np.mean(epoch_losses):.4f}"
            })
        
        # Update scheduler
        self.few_shot_scheduler.step()
        
        # Validation
        val_metrics = None
        if val_dataset is not None:
            val_metrics = self.evaluate_few_shot(val_dataset)
        
        # Log epoch results
        epoch_summary = {
            'train_loss': np.mean(epoch_losses),
            'train_loss_std': np.std(epoch_losses),
            'loss_components': {k: np.mean(v) for k, v in epoch_metrics.items()},
            'val_metrics': val_metrics
        }
        
        self.logger.info(f"Few-shot epoch completed: {epoch_summary}")
        
        return epoch_summary
    
    def evaluate_few_shot(self, dataset, num_episodes=50):
        """
        Evaluate few-shot performance
        
        Args:
            dataset: Dataset to evaluate on
            num_episodes: Number of episodes for evaluation
        """
        self.few_shot_model.eval()
        episode_results = []
        
        with torch.no_grad():
            for episode in tqdm(range(num_episodes), desc="Evaluating"):
                # Sample episode
                support_data, query_data, domain_ids = self._sample_episode(dataset)
                
                # Forward pass
                outputs = self.few_shot_model(
                    support_data=support_data,
                    query_data=query_data,
                    domain_ids=domain_ids,
                    external_knowledge=self.external_knowledge,
                    instruction_templates=self.instruction_templates
                )
                
                # Evaluate episode
                episode_metrics = self._evaluate_episode(
                    outputs['predictions'], 
                    query_data['labels']
                )
                
                episode_results.append(episode_metrics)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(episode_results)
        
        self.logger.info(f"Few-shot evaluation: {aggregated_results}")
        
        return aggregated_results
    
    def _sample_episode(self, dataset):
        """
        Sample support and query sets for one episode
        
        Returns:
            support_data: Support set data
            query_data: Query set data  
            domain_ids: Domain identifiers
        """
        # Get available data
        features = dataset.features if hasattr(dataset, 'features') else dataset.data
        labels = dataset.labels if hasattr(dataset, 'labels') else dataset.targets
        
        # Simulate domain IDs (in real implementation, these would come from dataset)
        num_samples = features.size(0)
        domain_ids = torch.randint(0, 3, (num_samples,), device=self.device)
        
        # Get unique classes
        unique_classes = torch.unique(labels)
        
        support_indices = []
        query_indices = []
        
        # Sample k examples per class for support, rest for query
        for class_id in unique_classes:
            class_indices = torch.where(labels == class_id)[0]
            
            if len(class_indices) >= self.few_shot_k + 5:
                # Sample k for support
                perm = torch.randperm(len(class_indices))
                support_class_indices = class_indices[perm[:self.few_shot_k]]
                query_class_indices = class_indices[perm[self.few_shot_k:self.few_shot_k+5]]
                
                support_indices.extend(support_class_indices.tolist())
                query_indices.extend(query_class_indices.tolist())
        
        # Create support and query data
        support_data = {
            'features': features[support_indices].to(self.device),
            'labels': labels[support_indices].to(self.device)
        }
        
        query_data = {
            'features': features[query_indices].to(self.device),
            'labels': labels[query_indices].to(self.device)
        }
        
        # Domain IDs for sampled data
        episode_domain_ids = domain_ids[query_indices].to(self.device)
        
        return support_data, query_data, episode_domain_ids
    
    def _evaluate_episode(self, predictions, true_labels):
        """Evaluate single episode performance"""
        # Convert to class predictions
        if predictions.dim() > 1:
            pred_labels = predictions.argmax(dim=-1)
        else:
            pred_labels = predictions
        
        # Calculate accuracy
        accuracy = (pred_labels == true_labels).float().mean().item()
        
        # Calculate F1 score
        f1_scores = []
        unique_labels = torch.unique(true_labels)
        
        for class_id in unique_labels:
            tp = ((pred_labels == class_id) & (true_labels == class_id)).sum().item()
            fp = ((pred_labels == class_id) & (true_labels != class_id)).sum().item()
            fn = ((pred_labels != class_id) & (true_labels == class_id)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': f1_scores
        }
    
    def _aggregate_results(self, episode_results):
        """Aggregate results across episodes"""
        if not episode_results:
            return {'accuracy': 0.0, 'macro_f1': 0.0}
        
        accuracies = [r['accuracy'] for r in episode_results]
        f1_scores = [r['macro_f1'] for r in episode_results]
        
        return {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_ci': self._compute_confidence_interval(accuracies),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'f1_ci': self._compute_confidence_interval(f1_scores),
            'num_episodes': len(episode_results)
        }
    
    def _compute_confidence_interval(self, values, confidence=0.95):
        """Compute confidence interval"""
        if len(values) == 0:
            return (0.0, 0.0)
        
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        
        t_crit = 1.96  # 95% confidence
        margin_error = t_crit * std / np.sqrt(n)
        
        return (mean - margin_error, mean + margin_error)
    
    def _initialize_external_knowledge(self):
        """Initialize external knowledge embeddings (simulated)"""
        # In real implementation, this would load from knowledge bases
        return torch.randn(100, self.config.hidden_size, device=self.device)
    
    def _initialize_instruction_templates(self):
        """Initialize instruction templates for IPT"""
        templates = {}
        
        # Template A: Aspect-focused
        templates['ipt_a'] = torch.randn(1, self.config.hidden_size, device=self.device)
        
        # Template B: Opinion-focused  
        templates['ipt_b'] = torch.randn(1, self.config.hidden_size, device=self.device)
        
        # Template C: Sentiment-focused
        templates['ipt_c'] = torch.randn(1, self.config.hidden_size, device=self.device)
        
        return templates
    
    def train_with_domain_adaptation(self, source_dataset, target_datasets, epochs=10):
        """
        Train with cross-domain adaptation using CD-ALPHN
        
        Args:
            source_dataset: Source domain dataset
            target_datasets: List of target domain datasets
            epochs: Number of training epochs
        """
        self.logger.info("Starting domain adaptation training...")
        
        for epoch in range(epochs):
            self.logger.info(f"Domain adaptation epoch {epoch+1}/{epochs}")
            
            # Train on source domain
            source_metrics = self.train_few_shot_epoch(source_dataset)
            
            # Adapt to each target domain
            for i, target_dataset in enumerate(target_datasets):
                self.logger.info(f"Adapting to target domain {i+1}")
                
                # Sample target domain support data
                support_data, _, _ = self._sample_episode(target_dataset)
                
                # Perform domain adaptation
                self.few_shot_model.adapt_to_new_domain(
                    target_support_data=support_data,
                    target_domain_id=i+1,  # Domain ID
                    adaptation_steps=self.adaptation_steps
                )
                
                # Evaluate on target domain
                target_metrics = self.evaluate_few_shot(target_dataset, num_episodes=20)
                
                self.logger.info(f"Target domain {i+1} metrics: {target_metrics}")
        
        return self.few_shot_model
    
    def save_few_shot_model(self, save_path):
        """Save few-shot model components"""
        checkpoint = {
            'few_shot_model_state_dict': self.few_shot_model.state_dict(),
            'optimizer_state_dict': self.few_shot_optimizer.state_dict(),
            'scheduler_state_dict': self.few_shot_scheduler.state_dict(),
            'config': self.config,
            'training_metrics': dict(self.training_metrics),
            'external_knowledge': self.external_knowledge,
            'instruction_templates': self.instruction_templates
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Few-shot model saved to {save_path}")
    
    def load_few_shot_model(self, load_path):
        """Load few-shot model components"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.few_shot_model.load_state_dict(checkpoint['few_shot_model_state_dict'])
        self.few_shot_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.few_shot_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_metrics = defaultdict(list, checkpoint['training_metrics'])
        self.external_knowledge = checkpoint['external_knowledge']
        self.instruction_templates = checkpoint['instruction_templates']
        
        self.logger.info(f"Few-shot model loaded from {load_path}")
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        return self.few_shot_model.get_performance_summary()


class FewShotDatasetAdapter:
    """
    Adapter to convert standard ABSA datasets for few-shot learning
    """
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.hidden_size = getattr(config, 'hidden_size', 768)
        
    def convert_to_few_shot_format(self):
        """Convert dataset to few-shot format"""
        # Extract features and labels
        if hasattr(self.dataset, 'features'):
            features = self.dataset.features
            labels = self.dataset.labels
        else:
            # Convert from text data if needed
            features, labels = self._extract_features_from_text()
        
        # Create few-shot compatible dataset
        few_shot_dataset = FewShotDataset(features, labels, self.config)
        
        return few_shot_dataset
    
    def _extract_features_from_text(self):
        """Extract features from text data (placeholder)"""
        # In real implementation, this would use the embedding model
        # For now, create random features
        num_samples = len(self.dataset)
        features = torch.randn(num_samples, self.hidden_size)
        
        # Extract labels (assuming dataset has sentiment labels)
        if hasattr(self.dataset, 'labels'):
            labels = torch.tensor(self.dataset.labels)
        else:
            # Create dummy labels
            labels = torch.randint(0, 3, (num_samples,))
        
        return features, labels


class FewShotDataset:
    """
    Dataset class for few-shot learning episodes
    """
    
    def __init__(self, features, labels, config):
        self.features = features
        self.labels = labels
        self.config = config
        
        # Organize by class for efficient sampling
        self.class_indices = {}
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            self.class_indices[label.item()] = torch.where(labels == label)[0]
    
    def sample_episode(self, k_shot=5, num_query=5):
        """Sample a few-shot episode"""
        support_data = {'features': [], 'labels': []}
        query_data = {'features': [], 'labels': []}
        
        for class_id, indices in self.class_indices.items():
            if len(indices) >= k_shot + num_query:
                # Random permutation
                perm = torch.randperm(len(indices))
                
                # Support samples
                support_indices = indices[perm[:k_shot]]
                support_data['features'].append(self.features[support_indices])
                support_data['labels'].append(self.labels[support_indices])
                
                # Query samples
                query_indices = indices[perm[k_shot:k_shot+num_query]]
                query_data['features'].append(self.features[query_indices])
                query_data['labels'].append(self.labels[query_indices])
        
        # Concatenate
        support_data['features'] = torch.cat(support_data['features'], dim=0)
        support_data['labels'] = torch.cat(support_data['labels'], dim=0)
        query_data['features'] = torch.cat(query_data['features'], dim=0)
        query_data['labels'] = torch.cat(query_data['labels'], dim=0)
        
        return support_data, query_data