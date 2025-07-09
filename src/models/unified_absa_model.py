# src/models/unified_absa_model.py
"""
Clean, Unified ABSA Model implementing 2024-2025 breakthroughs
This replaces the complex integration with a clean, working implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration
import json

class ImplicitDetectionModule(nn.Module):
    """Simplified but complete implicit detection based on ABSA 2024-2025 report"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Grid Tagging Matrix (GM-GTM) for implicit aspects
        self.aspect_grid_size = 64  # Configurable grid size
        self.aspect_grid_projection = nn.Linear(self.hidden_size, self.aspect_grid_size)
        self.aspect_grid_classifier = nn.Linear(self.aspect_grid_size, 3)  # implicit/explicit/none
        
        # SCI-Net for implicit opinions
        self.opinion_contextual_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.opinion_bi_attention = nn.MultiheadAttention(
            self.hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Pattern-based sentiment inference
        self.sentiment_pattern_encoder = nn.LSTM(
            self.hidden_size, self.hidden_size // 2, bidirectional=True, batch_first=True
        )
        self.sentiment_pattern_classifier = nn.Linear(self.hidden_size, 3)
        
        # Contrastive alignment for implicit-explicit
        self.contrastive_projector = nn.Linear(self.hidden_size, 256)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 1. Grid Tagging for implicit aspects
        grid_features = self.aspect_grid_projection(hidden_states)  # [B, L, grid_size]
        implicit_aspect_logits = self.aspect_grid_classifier(grid_features)  # [B, L, 3]
        
        # 2. Bi-directional contextual interaction for opinions
        opinion_projected = self.opinion_contextual_proj(hidden_states)
        # Convert attention mask to proper format
        key_padding_mask = None
        if attention_mask is not None:
            # Convert to bool and invert (True = padded positions)
            key_padding_mask = ~attention_mask.bool()

        opinion_attended, _ = self.opinion_bi_attention(
            opinion_projected, opinion_projected, opinion_projected,
            key_padding_mask=key_padding_mask
        )
        implicit_opinion_logits = self.aspect_grid_classifier(
            self.aspect_grid_projection(opinion_attended)
        )
        
        # 3. Pattern-based sentiment inference
        pattern_output, _ = self.sentiment_pattern_encoder(hidden_states)
        implicit_sentiment_logits = self.sentiment_pattern_classifier(pattern_output)
        
        # 4. Contrastive features for alignment
        contrastive_features = self.contrastive_projector(hidden_states)
        
        return {
            'implicit_aspect_logits': implicit_aspect_logits,
            'implicit_opinion_logits': implicit_opinion_logits,
            'implicit_sentiment_logits': implicit_sentiment_logits,
            'contrastive_features': contrastive_features,
            'grid_features': grid_features
        }


class FewShotLearningModule(nn.Module):
    """Simplified few-shot learning with DRP and AFML concepts"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.few_shot_k = getattr(config, 'few_shot_k', 5)
        
        # Dual Relations Propagation (DRP) components
        self.aspect_similarity_projector = nn.Linear(self.hidden_size, 128)
        self.aspect_diversity_projector = nn.Linear(self.hidden_size, 128)
        
        # Aspect-Focused Meta-Learning (AFML) components
        self.meta_aspect_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        self.meta_contrastive_head = nn.Linear(self.hidden_size, 256)
        
        # Support set memory
        self.register_buffer('support_features', torch.zeros(100, self.hidden_size))
        self.register_buffer('support_labels', torch.zeros(100, dtype=torch.long))
        self.support_ptr = 0
        
    def update_support_set(self, features, labels):
        """Update support set for few-shot learning"""
        batch_size = features.size(0)
        end_ptr = (self.support_ptr + batch_size) % 100
        
        if end_ptr > self.support_ptr:
            self.support_features[self.support_ptr:end_ptr] = features.detach()
            self.support_labels[self.support_ptr:end_ptr] = labels.detach()
        else:
            self.support_features[self.support_ptr:] = features.detach()[:100-self.support_ptr]
            self.support_features[:end_ptr] = features.detach()[100-self.support_ptr:]
            self.support_labels[self.support_ptr:] = labels.detach()[:100-self.support_ptr]
            self.support_labels[:end_ptr] = labels.detach()[100-self.support_ptr:]
        
        self.support_ptr = end_ptr
    
    def forward(self, hidden_states, labels=None):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # DRP: Similarity and diversity analysis
        similarity_features = self.aspect_similarity_projector(hidden_states.mean(dim=1))
        diversity_features = self.aspect_diversity_projector(hidden_states.mean(dim=1))
        
        # AFML: Meta-learning representations
        meta_features = self.meta_aspect_encoder(hidden_states)
        meta_contrastive = self.meta_contrastive_head(meta_features.mean(dim=1))
        
        # Update support set if training
        if self.training and labels is not None:
            sentence_features = hidden_states.mean(dim=1)
            # Handle labels as dictionary
        if isinstance(labels, dict):
            # Use sentiment labels for few-shot learning
            if 'sentiment_labels' in labels:
                sentiment_labels = labels['sentiment_labels']
                sentence_labels = sentiment_labels[:, 0] if len(sentiment_labels.shape) > 1 else sentiment_labels
            else:
                # Fallback to first available label
                first_key = list(labels.keys())[0]
                first_labels = labels[first_key]
                sentence_labels = first_labels[:, 0] if len(first_labels.shape) > 1 else first_labels
        else:
            # Handle as tensor
            sentence_labels = labels[:, 0] if len(labels.shape) > 1 else labels
            self.update_support_set(sentence_features, sentence_labels)
        
        # Few-shot predictions based on support set similarity
        if self.support_features.norm() > 0:
            query_features = hidden_states.mean(dim=1)  # [B, H]
            similarities = torch.mm(query_features, self.support_features.T)  # [B, 100]
            top_k_similarities, top_k_indices = similarities.topk(self.few_shot_k, dim=1)
            
            # Weight by similarity
            weights = F.softmax(top_k_similarities, dim=1)  # [B, k]
            support_labels_selected = self.support_labels[top_k_indices]  # [B, k]
            
            # Simple voting mechanism
            few_shot_predictions = torch.zeros(batch_size, seq_len, 3, device=hidden_states.device)
            for b in range(batch_size):
                for i in range(self.few_shot_k):
                    label = support_labels_selected[b, i].item()
                    weight = weights[b, i].item()
                    if 0 <= label < 3:
                        few_shot_predictions[b, :, label] += weight
        else:
            few_shot_predictions = torch.zeros(batch_size, seq_len, 3, device=hidden_states.device)
        
        return {
            'few_shot_predictions': few_shot_predictions,
            'similarity_features': similarity_features,
            'diversity_features': diversity_features,
            'meta_contrastive': meta_contrastive
        }


class GenerativeABSAModule(nn.Module):
    """Simplified generative framework based on InstructABSA and T5"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Use smaller T5 model for efficiency
        model_name = getattr(config, 'generative_model_name', 't5-small')
        try:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.available = True
            print(f"✅ Generative module initialized with {model_name}")
        except Exception as e:
            print(f"⚠️ Could not load T5 model: {e}")
            self.available = False
            return
        
        # Feature bridge from ABSA features to T5
        self.feature_bridge = nn.Linear(config.hidden_size, self.t5_model.config.d_model)
        
        # Task-specific prompt templates
        self.task_templates = {
            'triplet_extraction': "Extract aspect-opinion-sentiment triplets from: {text}",
            'aspect_extraction': "Extract aspects from: {text}",
            'sentiment_classification': "Classify sentiment for {aspect} in: {text}",
            'explanation': "Explain sentiment analysis for: {text}"
        }
    
    def forward(self, hidden_states, input_ids, task_type='triplet_extraction', target_text=None):
        if not self.available:
            return {'available': False, 'loss': torch.tensor(0.0, device=hidden_states.device)}
        
        batch_size = hidden_states.size(0)
        
        # Bridge features to T5 dimension
        bridged_features = self.feature_bridge(hidden_states.mean(dim=1))  # [B, t5_dim]
        
        # Create input text (simplified)
        input_text = ["Extract sentiment triplets from the text."] * batch_size
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=128
        ).to(hidden_states.device)
        
        # Get T5 encoder outputs
        encoder_outputs = self.t5_model.encoder(**inputs)
        
        # Modify encoder outputs with ABSA features
        modified_hidden_states = encoder_outputs.last_hidden_state + bridged_features.unsqueeze(1)
        
        if target_text is not None and self.training:
            # Training mode - compute generation loss
            target_inputs = self.tokenizer(
                target_text if isinstance(target_text, list) else [target_text] * batch_size,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=64
            ).to(hidden_states.device)
            
            # Prepare decoder inputs
            decoder_input_ids = target_inputs.input_ids
            decoder_input_ids = torch.cat([
                torch.full((batch_size, 1), self.tokenizer.pad_token_id, device=hidden_states.device),
                decoder_input_ids[:, :-1]
            ], dim=1)
            
            # Forward through decoder
            decoder_outputs = self.t5_model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=modified_hidden_states,
                encoder_attention_mask=inputs.attention_mask
            )
            
            # Compute generation loss
            lm_logits = self.t5_model.lm_head(decoder_outputs.last_hidden_state)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            generation_loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                target_inputs.input_ids.view(-1)
            )
            
            return {
                'available': True,
                'loss': generation_loss,
                'logits': lm_logits,
                'hidden_states': decoder_outputs.last_hidden_state
            }
        else:
            # Inference mode - generate text
            generated_ids = self.t5_model.generate(
                encoder_outputs=type('', (), {
                    'last_hidden_state': modified_hidden_states,
                    'hidden_states': None,
                    'attentions': None
                })(),
                attention_mask=inputs.attention_mask,
                max_length=64,
                num_beams=2,
                do_sample=False
            )
            
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            return {
                'available': True,
                'generated_text': generated_text,
                'generated_ids': generated_ids
            }


class UnifiedABSAModel(nn.Module):
    """
    Clean, unified ABSA model implementing 2024-2025 breakthroughs
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base language model
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.backbone.config.hidden_size
        config.hidden_size = self.hidden_size
        
        # Core components
        self.implicit_detector = ImplicitDetectionModule(config)
        self.few_shot_learner = FewShotLearningModule(config)
        
        # Generative module (optional)
        if getattr(config, 'use_generative_framework', False):
            self.generative_module = GenerativeABSAModule(config)
            self.has_generative = True
        else:
            self.has_generative = False
        
        # Main prediction heads
        self.aspect_classifier = nn.Linear(self.hidden_size, 3)  # B-I-O
        self.opinion_classifier = nn.Linear(self.hidden_size, 3)  # B-I-O
        self.sentiment_classifier = nn.Linear(self.hidden_size, 3)  # POS-NEG-NEU
        
        # Contrastive learning
        self.contrastive_projector = nn.Linear(self.hidden_size, 256)
        
        # Feature fusion
        # Feature fusion - calculate correct size
        contrastive_size = 256  # From contrastive_projector output
        fusion_size = self.hidden_size + contrastive_size  # 768 + 256 = 1024
        self.feature_fusion = nn.Linear(fusion_size, self.hidden_size)
                
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None, labels=None, task_type='triplet_extraction', target_text=None):
        batch_size, seq_len = input_ids.shape
        
        # 1. Get base representations
        backbone_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = backbone_outputs.last_hidden_state
        
        # 2. Implicit detection
        implicit_outputs = self.implicit_detector(hidden_states, attention_mask)
        
        # 3. Few-shot learning
        few_shot_outputs = self.few_shot_learner(hidden_states, labels)
        
        # 4. Feature fusion
        fused_features = torch.cat([
            hidden_states, 
            implicit_outputs['contrastive_features']
        ], dim=-1)
        fused_features = self.feature_fusion(fused_features)
        fused_features = self.dropout(fused_features)
        
        # 5. Main predictions
        aspect_logits = self.aspect_classifier(fused_features)
        opinion_logits = self.opinion_classifier(fused_features)
        sentiment_logits = self.sentiment_classifier(fused_features)
        
        # 6. Generative module (if available)
        generative_outputs = {}
        if self.has_generative:
            generative_outputs = self.generative_module(
                fused_features, input_ids, task_type, target_text
            )
        
        # 7. Prepare outputs
        outputs = {
            'aspect_logits': aspect_logits,
            'opinion_logits': opinion_logits,
            'sentiment_logits': sentiment_logits,
            'hidden_states': fused_features,
            'implicit_outputs': implicit_outputs,
            'few_shot_outputs': few_shot_outputs,
            'generative_outputs': generative_outputs,
            'contrastive_features': self.contrastive_projector(fused_features)
        }
        
        # 8. Compute losses if training
        if labels is not None:
            outputs['losses'] = self._compute_losses(outputs, labels)
        
        return outputs
    
    def _compute_losses(self, outputs, labels):
        """Compute all losses"""
        losses = {}
        total_loss = 0.0
        
        # Main classification losses
        if 'aspect_labels' in labels:
            aspect_loss = F.cross_entropy(
                outputs['aspect_logits'].view(-1, 3),
                labels['aspect_labels'].view(-1),
                ignore_index=-100
            )
            losses['aspect_loss'] = aspect_loss
            total_loss += aspect_loss
        
        if 'opinion_labels' in labels:
            opinion_loss = F.cross_entropy(
                outputs['opinion_logits'].view(-1, 3),
                labels['opinion_labels'].view(-1),
                ignore_index=-100
            )
            losses['opinion_loss'] = opinion_loss
            total_loss += opinion_loss
        
        if 'sentiment_labels' in labels:
            sentiment_loss = F.cross_entropy(
                outputs['sentiment_logits'].view(-1, 3),
                labels['sentiment_labels'].view(-1),
                ignore_index=-100
            )
            losses['sentiment_loss'] = sentiment_loss
            total_loss += sentiment_loss
        
        # Implicit detection losses
        implicit_outputs = outputs['implicit_outputs']
        if 'implicit_aspect_labels' in labels:
            implicit_aspect_loss = F.cross_entropy(
                implicit_outputs['implicit_aspect_logits'].view(-1, 3),
                labels['implicit_aspect_labels'].view(-1),
                ignore_index=-100
            )
            losses['implicit_aspect_loss'] = implicit_aspect_loss
            total_loss += 0.5 * implicit_aspect_loss
        
        # Contrastive loss (simplified)
        contrastive_features = outputs['contrastive_features']
        if contrastive_features.size(0) > 1:
            # Simple contrastive loss
            contrastive_loss = self._compute_simple_contrastive_loss(contrastive_features, labels)
            losses['contrastive_loss'] = contrastive_loss
            total_loss += 0.3 * contrastive_loss
        
        # Generative loss
        if self.has_generative and outputs['generative_outputs'].get('available', False):
            generative_loss = outputs['generative_outputs'].get('loss', 0.0)
            if isinstance(generative_loss, torch.Tensor):
                losses['generative_loss'] = generative_loss
                total_loss += 0.2 * generative_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_simple_contrastive_loss(self, features, labels):
        """Simple contrastive loss implementation"""
        # Get sentence-level features
        sentence_features = features.mean(dim=1)  # [B, 256]
        
        # Get sentence-level labels (simplified)
        if 'sentiment_labels' in labels:
            sentence_labels = labels['sentiment_labels'][:, 0]  # Use first token label
        else:
            return torch.tensor(0.0, device=features.device)
        
        # Compute pairwise similarities
        similarities = torch.mm(sentence_features, sentence_features.T)
        
        # Create positive and negative masks
        labels_expanded = sentence_labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        negative_mask = (labels_expanded != labels_expanded.T).float()
        
        # Remove diagonal
        positive_mask.fill_diagonal_(0)
        negative_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        positive_similarities = similarities * positive_mask
        negative_similarities = similarities * negative_mask
        
        positive_loss = -torch.log(torch.clamp(positive_similarities, min=1e-8)).sum()
        negative_loss = torch.log(torch.clamp(1 - negative_similarities, min=1e-8)).sum()
        
        total_pairs = positive_mask.sum() + negative_mask.sum()
        if total_pairs > 0:
            contrastive_loss = (positive_loss + negative_loss) / total_pairs
        else:
            contrastive_loss = torch.tensor(0.0, device=features.device)
        
        return contrastive_loss
    
    def predict_triplets(self, input_ids, attention_mask=None):
        """Extract triplets from predictions"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            # Get predictions
            aspect_preds = torch.argmax(outputs['aspect_logits'], dim=-1)
            opinion_preds = torch.argmax(outputs['opinion_logits'], dim=-1)
            sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1)
            
            # Simple triplet extraction (can be enhanced)
            triplets = []
            batch_size = input_ids.size(0)
            
            for b in range(batch_size):
                batch_triplets = []
                seq_len = attention_mask[b].sum().item() if attention_mask is not None else input_ids.size(1)
                
                # Find aspects and opinions
                aspects = self._extract_spans(aspect_preds[b][:seq_len])
                opinions = self._extract_spans(opinion_preds[b][:seq_len])
                
                # Pair aspects with opinions and assign sentiment
                for aspect_span in aspects:
                    for opinion_span in opinions:
                        # Get sentiment for this aspect-opinion pair
                        sentiment_scores = outputs['sentiment_logits'][b][aspect_span[0]:aspect_span[1]+1].mean(dim=0)
                        sentiment_idx = torch.argmax(sentiment_scores).item()
                        sentiment_label = ['NEG', 'NEU', 'POS'][sentiment_idx]
                        
                        batch_triplets.append({
                            'aspect_span': aspect_span,
                            'opinion_span': opinion_span,
                            'sentiment': sentiment_label
                        })
                
                triplets.append(batch_triplets)
            
            return triplets
    
    def _extract_spans(self, predictions):
        """Extract spans from BIO predictions"""
        spans = []
        start = None
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # B (Beginning)
                if start is not None:
                    spans.append((start, i-1))
                start = i
            elif pred == 0:  # O (Outside)
                if start is not None:
                    spans.append((start, i-1))
                    start = None
            # pred == 2 is I (Inside), continue current span
        
        # Handle span at end of sequence
        if start is not None:
            spans.append((start, len(predictions)-1))
        
        return spans
    
    def save(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else self.config
        }, path)
    
    @classmethod
    def load(cls, path, config=None, device='cuda'):
        """Load model"""
        checkpoint = torch.load(path, map_location=device)
        
        if config is None:
            # Try to load config from checkpoint
            config_dict = checkpoint.get('config', {})
            from ..utils.config import LLMABSAConfig
            config = LLMABSAConfig()
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model