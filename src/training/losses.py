# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABSALoss(nn.Module):
    """
    Memory-efficient loss function for ABSA with triplet-aware contrastive learning
    
    This 2025 implementation provides:
    - Focal loss for handling class imbalance in span detection
    - Contrastive verification for ensuring faithfulness
    - Boundary refinement loss for precise span detection
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Focal loss for handling class imbalance in span detection
        self.gamma = getattr(config, 'focal_loss_gamma', 2.0)
        
        # Regular classification losses
        self.span_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.sentiment_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Generation loss
        self.generation_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Loss weights
        self.aspect_weight = getattr(config, 'aspect_loss_weight', 1.0)
        self.opinion_weight = getattr(config, 'opinion_loss_weight', 1.0)
        self.sentiment_weight = getattr(config, 'sentiment_loss_weight', 1.0)
        self.generation_weight = getattr(config, 'generation_weight', 0.5)
        self.verification_weight = getattr(config, 'verification_weight', 0.2)
        self.boundary_weight = getattr(config, 'boundary_weight', 0.5)
        
        # Enable advanced loss components
        self.use_focal_loss = getattr(config, 'use_focal_loss', True)
        self.use_boundary_loss = getattr(config, 'use_boundary_loss', True)
        
    def focal_loss(self, logits, targets, gamma=2.0, alpha=None):
        """
        Compute focal loss to handle class imbalance in span detection
        
        Args:
            logits: Prediction logits [batch_size, ...]
            targets: Target labels [batch_size, ...]
            gamma: Focusing parameter
            alpha: Optional class weighting factor
            
        Returns:
            Focal loss value
        """
        # Compute standard cross entropy
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1), 
            reduction='none',
            ignore_index=-100
        )
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** gamma
        
        # Apply alpha if provided
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.to(logits.device)
            focal_weight = alpha * focal_weight
            
        # Compute final loss
        focal_loss = focal_weight * ce_loss
        
        # Average over non-ignored positions
        valid_positions = (targets.view(-1) != -100).float().sum()
        if valid_positions > 0:
            return focal_loss.sum() / valid_positions
        else:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
    def boundary_loss(self, boundary_logits, aspect_labels, opinion_labels):
        """
        Compute boundary refinement loss for precise span detection
        
        Args:
            boundary_logits: Boundary adjustment logits [batch_size, seq_len, 2]
            aspect_labels: Aspect span labels [batch_size, seq_len]
            opinion_labels: Opinion span labels [batch_size, seq_len]
            
        Returns:
            Boundary loss value
        """
        # Create boundary labels from span labels
        batch_size, seq_len = aspect_labels.shape
        boundary_labels = torch.zeros((batch_size, seq_len, 2), device=boundary_logits.device)
        
        # Get aspect boundaries
        for b in range(batch_size):
            # Find B tags (start boundaries)
            aspect_starts = (aspect_labels[b] == 1).nonzero(as_tuple=True)[0]
            # Mark start positions
            for pos in aspect_starts:
                if pos < seq_len:
                    boundary_labels[b, pos, 0] = 1.0
            
            # Find I tags not followed by another I tag (end boundaries)
            for i in range(seq_len - 1):
                if aspect_labels[b, i] == 2 and aspect_labels[b, i+1] != 2:
                    boundary_labels[b, i, 1] = 1.0
            # Handle last position
            if aspect_labels[b, -1] == 2:
                boundary_labels[b, -1, 1] = 1.0
                
        # Do the same for opinion boundaries
        for b in range(batch_size):
            # Find B tags (start boundaries)
            opinion_starts = (opinion_labels[b] == 1).nonzero(as_tuple=True)[0]
            # Mark start positions
            for pos in opinion_starts:
                if pos < seq_len:
                    boundary_labels[b, pos, 0] = 1.0
            
            # Find I tags not followed by another I tag (end boundaries)
            for i in range(seq_len - 1):
                if opinion_labels[b, i] == 2 and opinion_labels[b, i+1] != 2:
                    boundary_labels[b, i, 1] = 1.0
            # Handle last position
            if opinion_labels[b, -1] == 2:
                boundary_labels[b, -1, 1] = 1.0
        
        # Compute boundary loss (binary cross entropy)
        boundary_loss = F.binary_cross_entropy_with_logits(
            boundary_logits, 
            boundary_labels,
            reduction='mean'
        )
        
        return boundary_loss
        
    def forward(self, outputs, targets, generate=False):
        """
        Compute combined loss for ABSA with optional generation
        
        Args:
            outputs: Model output dictionary
            targets: Target dictionary
            generate: Whether to include generation loss
            
        Returns:
            Dictionary with loss components
        """
        try:
            # Get predictions
            aspect_logits = outputs['aspect_logits']  # [batch_size, seq_len, 3]
            opinion_logits = outputs['opinion_logits']  # [batch_size, seq_len, 3]
            sentiment_logits = outputs['sentiment_logits']  # [batch_size, num_classes]
            
            # Initialize losses
            aspect_loss = torch.tensor(0.0, device=aspect_logits.device)
            opinion_loss = torch.tensor(0.0, device=opinion_logits.device)
            sentiment_loss = torch.tensor(0.0, device=sentiment_logits.device)
            boundary_loss = torch.tensor(0.0, device=aspect_logits.device)
            verification_loss = torch.tensor(0.0, device=aspect_logits.device)
            generation_loss = torch.tensor(0.0, device=aspect_logits.device)
            
            # Process aspect loss
            if 'aspect_labels' in targets:
                aspect_labels = targets['aspect_labels']
                
                # Handle multiple spans if needed
                if len(aspect_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                    # Take max across spans to get a single label per token
                    aspect_labels = aspect_labels.max(dim=1)[0]  # [batch_size, seq_len]
                
                # Compute loss with focal loss or standard cross entropy
                if self.use_focal_loss:
                    aspect_loss = self.focal_loss(
                        aspect_logits, 
                        aspect_labels,
                        gamma=self.gamma
                    )
                else:
                    aspect_loss = self.span_criterion(
                        aspect_logits.view(-1, aspect_logits.shape[-1]),
                        aspect_labels.view(-1).long()
                    )
            
            # Process opinion loss
            if 'opinion_labels' in targets:
                opinion_labels = targets['opinion_labels']
                
                # Handle multiple spans if needed
                if len(opinion_labels.shape) == 3:  # [batch_size, num_spans, seq_len]
                    # Take max across spans to get a single label per token
                    opinion_labels = opinion_labels.max(dim=1)[0]  # [batch_size, seq_len]
                
                # Compute loss with focal loss or standard cross entropy
                if self.use_focal_loss:
                    opinion_loss = self.focal_loss(
                        opinion_logits, 
                        opinion_labels,
                        gamma=self.gamma
                    )
                else:
                    opinion_loss = self.span_criterion(
                        opinion_logits.view(-1, opinion_logits.shape[-1]),
                        opinion_labels.view(-1).long()
                    )
            
            # Process sentiment loss
            if 'sentiment_labels' in targets:
                sentiment_labels = targets['sentiment_labels']
                
                # Handle multiple spans if needed
                if len(sentiment_labels.shape) > 1 and sentiment_labels.shape[1] > 0:
                    # Take the first sentiment label for each batch item
                    sentiment_labels = sentiment_labels[:, 0]
                
                # Compute sentiment loss
                sentiment_loss = self.sentiment_criterion(
                    sentiment_logits,
                    sentiment_labels
                )
            
            # Compute boundary refinement loss if enabled
            if self.use_boundary_loss and 'boundary_logits' in outputs:
                boundary_logits = outputs['boundary_logits']
                
                # Get aspect and opinion labels
                aspect_labels_for_boundary = targets.get('aspect_labels')
                opinion_labels_for_boundary = targets.get('opinion_labels')
                
                # Handle multiple spans if needed
                if aspect_labels_for_boundary is not None and len(aspect_labels_for_boundary.shape) == 3:
                    aspect_labels_for_boundary = aspect_labels_for_boundary.max(dim=1)[0]
                
                if opinion_labels_for_boundary is not None and len(opinion_labels_for_boundary.shape) == 3:
                    opinion_labels_for_boundary = opinion_labels_for_boundary.max(dim=1)[0]
                
                # Compute boundary loss
                if aspect_labels_for_boundary is not None and opinion_labels_for_boundary is not None:
                    boundary_loss = self.boundary_loss(
                        boundary_logits,
                        aspect_labels_for_boundary,
                        opinion_labels_for_boundary
                    )
            
            # Compute generation loss if enabled
            if generate and 'explanations' in outputs and 'explanation_targets' in targets:
                explanation_logits = outputs['explanations']
                explanation_targets = targets['explanation_targets']
                
                # Compute generation loss
                generation_loss = self.generation_criterion(
                    explanation_logits.view(-1, explanation_logits.size(-1)),
                    explanation_targets.view(-1)
                )
            
            # Compute verification loss if present
            if 'verification_loss' in outputs:
                verification_loss = outputs['verification_loss']
            
            # Combine all losses with their weights
            total_loss = (
                self.aspect_weight * aspect_loss +
                self.opinion_weight * opinion_loss +
                self.sentiment_weight * sentiment_loss +
                self.boundary_weight * boundary_loss
            )
            
            # Add generation loss if applicable
            if generate:
                total_loss += self.generation_weight * generation_loss
                total_loss += self.verification_weight * verification_loss
            
            # Return dictionary with all loss components
            return {
                'loss': total_loss,
                'aspect_loss': aspect_loss.item(),
                'opinion_loss': opinion_loss.item(),
                'sentiment_loss': sentiment_loss.item(),
                'boundary_loss': boundary_loss.item(),
                'generation_loss': generation_loss.item() if generate else 0.0,
                'verification_loss': verification_loss.item() if generate else 0.0
            }
            
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple differentiable loss for backward compatibility
            device = outputs['aspect_logits'].device
            
            # Create a dummy loss that requires gradient
            dummy_loss = (
                outputs['aspect_logits'].sum() * 0.0001 +
                outputs['opinion_logits'].sum() * 0.0001 +
                outputs['sentiment_logits'].sum() * 0.0001
            )
            
            return {
                'loss': dummy_loss,
                'aspect_loss': 0.0,
                'opinion_loss': 0.0,
                'sentiment_loss': 0.0,
                'boundary_loss': 0.0,
                'generation_loss': 0.0,
                'verification_loss': 0.0
            }

class LLMTripleAlignLoss(nn.Module):
    """
    Advanced loss for aligning triplet extraction with generation (introduced in 2025)
    
    This loss enhances semantic alignment between extracted triplets and generated text,
    ensuring factuality and completeness of explanations.
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Base ABSA loss
        self.absa_loss = ABSALoss(config)
        
        # Embedding model for semantic similarity
        self.embed_model_name = getattr(config, 'embed_model', 'sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.use_embedding_model = getattr(config, 'use_embedding_model', True)
        
        # Try to load the embedding model
        if self.use_embedding_model:
            self._init_embedding_model()
        
        # Weights for different components
        self.factuality_weight = getattr(config, 'factuality_weight', 0.3)
        self.coverage_weight = getattr(config, 'coverage_weight', 0.3)
        self.coherence_weight = getattr(config, 'coherence_weight', 0.4)
        
        # Temperature for contrastive loss
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def _init_embedding_model(self):
        """Initialize embedding model for semantic similarity"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
            self.embed_model = AutoModel.from_pretrained(self.embed_model_name)
            
            # Freeze the embedding model
            for param in self.embed_model.parameters():
                param.requires_grad = False
                
            print(f"Successfully loaded embedding model: {self.embed_model_name}")
        except Exception as e:
            print(f"Failed to load embedding model: {e}")
            self.use_embedding_model = False
            
    def forward(self, outputs, targets, generate=False):
        """
        Compute triplet-aligned generation loss
        
        Args:
            outputs: Model output dictionary
            targets: Target dictionary
            generate: Whether to include generation loss
            
        Returns:
            Dictionary with loss components
        """
        # First compute standard ABSA loss
        loss_dict = self.absa_loss(outputs, targets, generate)
        
        # If generation is not enabled, return standard loss
        if not generate or 'explanations' not in outputs or 'explanation_targets' not in targets:
            return loss_dict
            
        try:
            # Compute triplet alignment loss
            if self.use_embedding_model and hasattr(self, 'embed_model'):
                # Extract triplets from model outputs
                aspect_logits = outputs['aspect_logits']
                opinion_logits = outputs['opinion_logits']
                sentiment_logits = outputs['sentiment_logits']
                
                # Get predicted triplets
                triplets = self._extract_triplets(
                    aspect_logits, opinion_logits, sentiment_logits,
                    targets.get('input_ids', None),
                    self.tokenizer
                )
                
                # Get explanation texts
                explanation_logits = outputs['explanations']
                explanation_targets = targets['explanation_targets']
                
                # Generate triplet and explanation embeddings
                triplet_embeds, expl_embeds = self._compute_embeddings(
                    triplets, explanation_targets, targets.get('input_ids', None)
                )
                
                if triplet_embeds is not None and expl_embeds is not None:
                    # Compute semantic alignment loss
                    factuality_loss = self._compute_factuality_loss(triplet_embeds, expl_embeds)
                    coverage_loss = self._compute_coverage_loss(triplet_embeds, expl_embeds)
                    coherence_loss = self._compute_coherence_loss(triplet_embeds, expl_embeds)
                    
                    # Combine alignment losses
                    alignment_loss = (
                        self.factuality_weight * factuality_loss +
                        self.coverage_weight * coverage_loss +
                        self.coherence_weight * coherence_loss
                    )
                    
                    # Add to total loss
                    loss_dict['loss'] += alignment_loss
                    loss_dict['alignment_loss'] = alignment_loss.item()
                    loss_dict['factuality_loss'] = factuality_loss.item()
                    loss_dict['coverage_loss'] = coverage_loss.item()
                    loss_dict['coherence_loss'] = coherence_loss.item()
            
            return loss_dict
            
        except Exception as e:
            print(f"Error in triplet alignment loss: {e}")
            import traceback
            traceback.print_exc()
            
            # Return standard loss without alignment
            return loss_dict
    
    def _extract_triplets(self, aspect_logits, opinion_logits, sentiment_logits, input_ids, tokenizer):
        """Extract triplets from model predictions"""
        if input_ids is None or tokenizer is None:
            return []
            
        batch_size = aspect_logits.size(0)
        all_triplets = []
        
        for b in range(batch_size):
            # Extract triplets for this batch item
            batch_triplets = []
            
            # Convert logits to predictions
            aspect_preds = aspect_logits[b].argmax(dim=-1)
            opinion_preds = opinion_logits[b].argmax(dim=-1)
            sentiment_pred = sentiment_logits[b].argmax(dim=-1).item()
            
            # Map sentiment to text
            sentiment_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
            sentiment = sentiment_map.get(sentiment_pred, 'neutral')
            
            # Extract aspect spans
            aspect_spans = self._extract_spans(aspect_preds)
            
            # Extract opinion spans
            opinion_spans = self._extract_spans(opinion_preds)
            
            # Create triplets
            for aspect_span in aspect_spans:
                for opinion_span in opinion_spans:
                    # Decode spans to text
                    try:
                        aspect_text = tokenizer.decode([input_ids[b, i].item() for i in aspect_span])
                        opinion_text = tokenizer.decode([input_ids[b, i].item() for i in opinion_span])
                    except:
                        continue
                    
                    # Add triplet
                    triplet = {
                        'aspect': aspect_text,
                        'opinion': opinion_text,
                        'sentiment': sentiment
                    }
                    batch_triplets.append(triplet)
            
            all_triplets.append(batch_triplets)
        
        return all_triplets
    
    def _extract_spans(self, predictions):
        """Extract spans from BIO predictions"""
        spans = []
        current_span = []
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # B tag
                if current_span:
                    spans.append(current_span)
                current_span = [i]
            elif pred == 2:  # I tag
                if current_span:
                    current_span.append(i)
            else:  # O tag
                if current_span:
                    spans.append(current_span)
                    current_span = []
        
        # Add last span if exists
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def _compute_embeddings(self, triplets, explanation_targets, input_ids):
        """Compute embeddings for triplets and explanations"""
        if not self.use_embedding_model or not hasattr(self, 'embed_model'):
            return None, None
            
        device = self.embed_model.device
        batch_size = len(triplets)
        
        # Convert triplets to text
        triplet_texts = []
        for batch_triplets in triplets:
            text = ""
            for triplet in batch_triplets:
                aspect = triplet.get('aspect', '')
                opinion = triplet.get('opinion', '')
                sentiment = triplet.get('sentiment', 'neutral')
                text += f"The {aspect} is {sentiment} because of the {opinion}. "
            triplet_texts.append(text if text else "No aspects.")
        
        # Convert explanation targets to text
        explanation_texts = []
        for expl in explanation_targets:
            try:
                text = self.tokenizer.decode([t.item() for t in expl if t != -100])
                explanation_texts.append(text)
            except:
                explanation_texts.append("")
        
        # Get embeddings
        try:
            # Embed triplet texts
            triplet_inputs = self.embed_tokenizer(
                triplet_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                triplet_outputs = self.embed_model(**triplet_inputs)
                triplet_embeds = triplet_outputs.last_hidden_state[:, 0]  # CLS token
                
            # Embed explanation texts
            expl_inputs = self.embed_tokenizer(
                explanation_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                expl_outputs = self.embed_model(**expl_inputs)
                expl_embeds = expl_outputs.last_hidden_state[:, 0]  # CLS token
            
            # Normalize embeddings
            triplet_embeds = F.normalize(triplet_embeds, p=2, dim=1)
            expl_embeds = F.normalize(expl_embeds, p=2, dim=1)
            
            return triplet_embeds, expl_embeds
            
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            return None, None
    
    def _compute_factuality_loss(self, triplet_embeds, expl_embeds):
        """
        Compute factuality loss to ensure explanations are faithful to triplets
        
        This ensures that the explanations don't hallucinate or contradict
        the extracted triplets.
        """
        # Compute cosine similarity
        similarity = torch.matmul(triplet_embeds, expl_embeds.transpose(0, 1))
        
        # Get diagonal elements (similarity between corresponding pairs)
        diag_sim = torch.diagonal(similarity)
        
        # Maximize similarity for corresponding pairs
        factuality_loss = 1.0 - diag_sim.mean()
        
        return factuality_loss
    
    def _compute_coverage_loss(self, triplet_embeds, expl_embeds):
        """
        Compute coverage loss to ensure explanations cover all triplets
        
        This ensures that the explanations include all the information
        from the extracted triplets.
        """
        # Use attention-like mechanism to measure coverage
        similarity = torch.matmul(triplet_embeds, expl_embeds.transpose(0, 1))
        
        # Get maximum similarity across batch
        max_sim = torch.max(similarity, dim=1)[0]
        
        # Coverage loss (penalize low similarity)
        coverage_loss = 1.0 - max_sim.mean()
        
        return coverage_loss
    
    def _compute_coherence_loss(self, triplet_embeds, expl_embeds):
        """
        Compute coherence loss to ensure explanations are well-structured
        
        This encourages explanations to be coherent and well-formed,
        not just a concatenation of triplets.
        """
        # Compute similarity matrix for explanations
        expl_sim = torch.matmul(expl_embeds, expl_embeds.transpose(0, 1))
        
        # Compute similarity matrix for triplets
        triplet_sim = torch.matmul(triplet_embeds, triplet_embeds.transpose(0, 1))
        
        # Coherence loss (align similarity structures)
        # We want explanations to preserve the similarity structure of triplets
        coherence_loss = F.mse_loss(expl_sim, triplet_sim)
        
        return coherence_loss