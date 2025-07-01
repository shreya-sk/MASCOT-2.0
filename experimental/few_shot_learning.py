# src/models/few_shot_learning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

class DualRelationsPropagation(nn.Module):
    """
    Dual Relations Propagation Network for Few-Shot ABSA
    Based on metric-free approaches from the report
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_relations = getattr(config, 'num_relations', 8)
        
        # Similarity and diversity analysis components
        self.similarity_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.diversity_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Relation propagation layers
        self.relation_propagation = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) 
            for _ in range(self.num_relations)
        ])
        
        # Aspect-aware representation
        self.aspect_projector = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def compute_similarity_matrix(self, embeddings):
        """Compute pairwise similarity matrix"""
        batch_size = embeddings.size(0)
        similarity_matrix = torch.zeros(batch_size, batch_size, device=embeddings.device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    combined = torch.cat([embeddings[i], embeddings[j]], dim=0)
                    similarity_matrix[i, j] = self.similarity_encoder(combined)
                    
        return similarity_matrix
    
    def compute_diversity_matrix(self, embeddings):
        """Compute diversity matrix to address overlapping distributions"""
        batch_size = embeddings.size(0)
        diversity_matrix = torch.zeros(batch_size, batch_size, device=embeddings.device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    combined = torch.cat([embeddings[i], embeddings[j]], dim=0)
                    diversity_matrix[i, j] = self.diversity_encoder(combined)
                    
        return diversity_matrix
    
    def propagate_relations(self, embeddings, support_labels):
        """Propagate relations among aspects"""
        # Apply multiple relation propagations
        propagated_embeds = []
        
        for relation_layer in self.relation_propagation:
            prop_embed = relation_layer(embeddings)
            propagated_embeds.append(prop_embed)
            
        # Combine propagated embeddings
        combined_embed = torch.stack(propagated_embeds, dim=1).mean(dim=1)
        
        return combined_embed
    
    def forward(self, support_embeddings, support_labels, query_embeddings):
        """
        Forward pass for few-shot learning
        
        Args:
            support_embeddings: [num_support, hidden_dim]
            support_labels: [num_support]
            query_embeddings: [num_query, hidden_dim]
        """
        # Compute similarity and diversity matrices
        all_embeddings = torch.cat([support_embeddings, query_embeddings], dim=0)
        similarity_matrix = self.compute_similarity_matrix(all_embeddings)
        diversity_matrix = self.compute_diversity_matrix(all_embeddings)
        
        # Propagate relations
        propagated_embeds = self.propagate_relations(all_embeddings, support_labels)
        
        # Split back to support and query
        num_support = support_embeddings.size(0)
        prop_support = propagated_embeds[:num_support]
        prop_query = propagated_embeds[num_support:]
        
        return prop_support, prop_query, similarity_matrix, diversity_matrix

class AspectFocusedMetaLearning(nn.Module):
    """
    Aspect-Focused Meta-Learning for Few-Shot ABSA
    Constructs aspect-aware representations with external knowledge
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.aspect_categories = getattr(config, 'aspect_categories', ['food', 'service', 'ambiance', 'price'])
        
        # Aspect-aware encoder
        self.aspect_aware_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Aspect-contrastive encoder
        self.aspect_contrastive_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # External knowledge integration
        self.knowledge_integrator = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Category classifiers
        self.category_classifier = nn.Linear(self.hidden_dim, len(self.aspect_categories))
        
    def integrate_external_knowledge(self, embeddings, external_knowledge):
        """Integrate external knowledge with embeddings"""
        combined = torch.cat([embeddings, external_knowledge], dim=-1)
        integrated = self.knowledge_integrator(combined)
        return integrated
    
    def forward(self, embeddings, external_knowledge=None):
        """
        Forward pass for aspect-focused meta-learning
        
        Args:
            embeddings: [batch_size, hidden_dim]
            external_knowledge: [batch_size, hidden_dim] - optional external knowledge
        """
        # Aspect-aware representation
        aspect_aware = self.aspect_aware_encoder(embeddings)
        
        # Aspect-contrastive representation
        aspect_contrastive = self.aspect_contrastive_encoder(embeddings)
        
        # Integrate external knowledge if provided
        if external_knowledge is not None:
            aspect_aware = self.integrate_external_knowledge(aspect_aware, external_knowledge)
            aspect_contrastive = self.integrate_external_knowledge(aspect_contrastive, external_knowledge)
        
        # Combine representations
        combined_repr = aspect_aware + aspect_contrastive
        
        # Category classification
        category_logits = self.category_classifier(combined_repr)
        
        return combined_repr, category_logits

class InstructionPromptFewShot(nn.Module):
    """
    Instruction prompt-based few-shot learning
    Achieves 80% of fully supervised performance with 1/10 data
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        
        # Three instruction prompt templates (IPT-a, IPT-b, IPT-c)
        self.instruction_templates = {
            'IPT-a': "Extract aspects and their sentiments from: {text}",
            'IPT-b': "Given the text: {text}, identify (aspect, opinion, sentiment) triplets",
            'IPT-c': "Analyze sentiment for each aspect in: {text}. Format: aspect -> sentiment"
        }
        
        # Template encoders
        self.template_encoders = nn.ModuleDict({
            template_name: nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            for template_name in self.instruction_templates.keys()
        })
        
        # Template fusion
        self.template_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def encode_with_template(self, embeddings, template_name):
        """Encode embeddings with specific instruction template"""
        encoder = self.template_encoders[template_name]
        return encoder(embeddings)
    
    def forward(self, embeddings, template_weights=None):
        """
        Forward pass with instruction prompts
        
        Args:
            embeddings: [batch_size, hidden_dim]
            template_weights: Optional weights for template combination
        """
        # Encode with each template
        template_outputs = []
        for template_name in self.instruction_templates.keys():
            template_output = self.encode_with_template(embeddings, template_name)
            template_outputs.append(template_output)
        
        # Combine template outputs
        combined_templates = torch.cat(template_outputs, dim=-1)
        fused_output = self.template_fusion(combined_templates)
        
        return fused_output, template_outputs

class FewShotABSAFramework(nn.Module):
    """
    Unified few-shot learning framework combining all approaches
    """
    
    def __init__(self, config):
        super().__init__()
        self.drp = DualRelationsPropagation(config)
        self.afml = AspectFocusedMetaLearning(config)
        self.instruction_prompt = InstructionPromptFewShot(config)
        
        # Combination weights
        self.drp_weight = getattr(config, 'drp_weight', 0.4)
        self.afml_weight = getattr(config, 'afml_weight', 0.3)
        self.instruction_weight = getattr(config, 'instruction_weight', 0.3)
        
        # Final classifier
        self.final_classifier = nn.Linear(config.hidden_size, 3)  # POS/NEU/NEG
        
    def forward(self, support_embeddings, support_labels, query_embeddings, 
                external_knowledge=None, k_shot=1):
        """
        Unified few-shot learning forward pass
        
        Args:
            support_embeddings: [num_support, hidden_dim]
            support_labels: [num_support]
            query_embeddings: [num_query, hidden_dim]
            external_knowledge: Optional external knowledge
            k_shot: Number of examples per class
        """
        # DRP processing
        drp_support, drp_query, sim_matrix, div_matrix = self.drp(
            support_embeddings, support_labels, query_embeddings
        )
        
        # AFML processing
        afml_support, _ = self.afml(support_embeddings, external_knowledge)
        afml_query, query_categories = self.afml(query_embeddings, external_knowledge)
        
        # Instruction prompt processing
        instruction_support, _ = self.instruction_prompt(support_embeddings)
        instruction_query, _ = self.instruction_prompt(query_embeddings)
        
        # Combine approaches
        combined_query = (
            self.drp_weight * drp_query +
            self.afml_weight * afml_query +
            self.instruction_weight * instruction_query
        )
        
        # Final classification
        query_logits = self.final_classifier(combined_query)
        
        return {
            'query_logits': query_logits,
            'similarity_matrix': sim_matrix,
            'diversity_matrix': div_matrix,
            'category_logits': query_categories
        }
    
    def few_shot_evaluation(self, support_data, query_data, k_shot=1, n_way=3):
        """
        Evaluate few-shot performance
        
        Args:
            support_data: Support set data
            query_data: Query set data  
            k_shot: Number of examples per class
            n_way: Number of classes
        """
        self.eval()
        
        with torch.no_grad():
            # Sample support and query sets
            support_embeddings, support_labels = self._sample_support_set(
                support_data, k_shot, n_way
            )
            query_embeddings, query_labels = self._sample_query_set(query_data)
            
            # Forward pass
            outputs = self.forward(support_embeddings, support_labels, query_embeddings)
            
            # Compute accuracy
            predictions = outputs['query_logits'].argmax(dim=-1)
            accuracy = (predictions == query_labels).float().mean()
            
            return {
                'accuracy': accuracy.item(),
                'predictions': predictions,
                'query_labels': query_labels
            }
    
    def _sample_support_set(self, data, k_shot, n_way):
        """Sample k-shot n-way support set"""
        # Implementation for sampling support set
        # This would sample k examples per class for n classes
        pass
    
    def _sample_query_set(self, data):
        """Sample query set"""
        # Implementation for sampling query set
        pass