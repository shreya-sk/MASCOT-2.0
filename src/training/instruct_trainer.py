"""
Training pipeline for instruction-following ABSA
"""
import torch
from transformers import get_linear_schedule_with_warmup

class InstructABSATrainer:
    """
    Trainer for the minimal InstructABSA model
    Extends your existing training with instruction-following capabilities
    """
    def __init__(self, model, config, tokenizer):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # Separate optimizers for backbone and T5
        backbone_params = list(self.model.absa_backbone.parameters())
        t5_params = list(self.model.t5_model.parameters()) + list(self.model.feature_bridge.parameters())
        
        self.backbone_optimizer = torch.optim.AdamW(backbone_params, lr=config.learning_rate)
        self.t5_optimizer = torch.optim.AdamW(t5_params, lr=config.learning_rate * 0.5)  # Lower LR for T5
        
        # Learning rate schedulers
        self.backbone_scheduler = get_linear_schedule_with_warmup(
            self.backbone_optimizer, 
            num_warmup_steps=100,
            num_training_steps=1000
        )
        self.t5_scheduler = get_linear_schedule_with_warmup(
            self.t5_optimizer,
            num_warmup_steps=100, 
            num_training_steps=1000
        )
    
    def train_step(self, batch):
        """
        Single training step with instruction following
        """
        self.model.train()
        
        # Extract data from batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Generate target text from existing labels
        target_text = self._create_target_text(batch)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_type='triplet_extraction',
            target_text=target_text
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Update both optimizers
        self.backbone_optimizer.step()
        self.t5_optimizer.step()
        
        self.backbone_scheduler.step()
        self.t5_scheduler.step()
        
        self.backbone_optimizer.zero_grad()
        self.t5_optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'extraction_loss': outputs['extraction_outputs'].get('loss', 0),
            'generation_loss': outputs['generation_outputs'].loss.item()
        }
    
    def _create_target_text(self, batch):
        """
        Create target text from existing labels for instruction training
        """
        # This is a simplified version - you'd want to make this more sophisticated
        target_texts = []
        
        batch_size = batch['input_ids'].size(0)
        for b in range(batch_size):
            # Extract labels for this sample
            aspect_labels = batch['aspect_labels'][b] if 'aspect_labels' in batch else None
            opinion_labels = batch['opinion_labels'][b] if 'opinion_labels' in batch else None
            sentiment_labels = batch['sentiment_labels'][b] if 'sentiment_labels' in batch else None
            
            if aspect_labels is not None and opinion_labels is not None:
                # Convert labels to target text format
                target_text = self._labels_to_target_text(aspect_labels, opinion_labels, sentiment_labels)
                target_texts.append(target_text)
            else:
                target_texts.append("No triplets found.")
        
        return target_texts[0] if len(target_texts) == 1 else target_texts
    
    def _labels_to_target_text(self, aspect_labels, opinion_labels, sentiment_labels):
        """
        Convert BIO labels to structured target text
        """
        # Extract spans from labels
        aspect_spans = self._extract_spans_from_labels(aspect_labels)
        opinion_spans = self._extract_spans_from_labels(opinion_labels)
        
        # Map sentiment
        sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
        sentiment = sentiment_map.get(sentiment_labels.item() if hasattr(sentiment_labels, 'item') else 0, 'NEU')
        
        # Create structured output
        triplets = []
        for asp_span in aspect_spans:
            for op_span in opinion_spans:
                triplet_text = f"<triplet><aspect>span_{asp_span[0]}_{asp_span[-1]}</aspect><opinion>span_{op_span[0]}_{op_span[-1]}</opinion><sentiment>{sentiment}</sentiment></triplet>"
                triplets.append(triplet_text)
        
        return " ".join(triplets) if triplets else "No triplets found."
    
    def _extract_spans_from_labels(self, labels):
        """Extract spans from BIO label tensor"""
        if len(labels.shape) > 1:
            labels = labels[0]  # Take first span if multi-span
        
        spans = []
        current_span = []
        
        for i, label in enumerate(labels):
            if label == 1:  # B tag
                if current_span:
                    spans.append(current_span)
                current_span = [i]
            elif label == 2 and current_span:  # I tag
                current_span.append(i)
            else:  # O tag
                if current_span:
                    spans.append(current_span)
                    current_span = []
        
        if current_span:
            spans.append(current_span)
        
        return spans