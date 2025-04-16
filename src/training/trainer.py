# src/training/trainer.py
import torch # type: ignore # type: ignore
import gc
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler # type: ignore

class ABSATrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        config
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=getattr(config, 'use_fp16', True))
        
        # Initialize best metrics
        self.best_f1 = 0
        
        # Gradient accumulation steps
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        # Implement scheduled dropout to reduce overfitting gradually
        base_dropout = self.config.dropout
        if epoch > self.config.num_epochs // 2:
            # Increase dropout rate in later epochs to combat overfitting
            adjusted_dropout = min(base_dropout * 1.5, 0.5)
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = adjusted_dropout

        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(train_iterator):
            # Extract text if available (for online embeddings)
            texts = batch.pop('text') if 'text' in batch else None
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_fp16):
                # Pass texts to the model for online embeddings
                outputs = self.model(
                    **batch,
                    texts=texts  # Add this parameter
                )
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict['loss']
            
                # Normalize loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients or at the end of the epoch
            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                # Unscale gradients for gradient clipping
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer and scheduler steps
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                step += 1
                
            # Log actual loss (not normalized by gradient accumulation)
            full_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += full_loss
            
            # Update progress bar
            train_iterator.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}", 
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Clean up memory periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_loss:.4f}")
        return avg_loss
        

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        # Initialize prediction and label trackers
        aspect_preds, opinion_preds, sentiment_preds = [], [], []
        aspect_labels, opinion_labels, sentiment_labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Extract text if available
                texts = batch.pop('text') if 'text' in batch else None
                
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with autocast(enabled=getattr(self.config, 'use_fp16', True)):
                    outputs = self.model(**batch, texts=texts)
    
                    loss_dict = self.loss_fn(outputs, batch)
                
                total_loss += loss_dict['loss'].item()
                
                # Get predictions
                batch_aspect_preds = outputs['aspect_logits'].argmax(dim=-1).cpu()
                batch_opinion_preds = outputs['opinion_logits'].argmax(dim=-1).cpu()
                batch_sentiment_preds = outputs['sentiment_logits'].argmax(dim=-1).cpu()
                
                # Get labels
                batch_aspect_labels = batch['aspect_labels'].cpu()
                batch_opinion_labels = batch['opinion_labels'].cpu()
                batch_sentiment_labels = batch['sentiment_labels'].cpu()
                
                # Add to lists
                aspect_preds.append(batch_aspect_preds)
                opinion_preds.append(batch_opinion_preds)
                sentiment_preds.append(batch_sentiment_preds)
                
                aspect_labels.append(batch_aspect_labels)
                opinion_labels.append(batch_opinion_labels)
                sentiment_labels.append(batch_sentiment_labels)
                
                # Free memory
                del outputs, batch
                torch.cuda.empty_cache()
                gc.collect()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            aspect_preds, opinion_preds, sentiment_preds,
            aspect_labels, opinion_labels, sentiment_labels
        )
        
        # Add loss to metrics
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def _calculate_metrics(self, aspect_preds, opinion_preds, sentiment_preds,
                         aspect_labels, opinion_labels, sentiment_labels):
        """Calculate metrics for all tasks"""
        metrics = {}
        
        # Calculate aspect metrics
        aspect_precision, aspect_recall, aspect_f1 = self._calculate_span_metrics(
            aspect_preds, aspect_labels, 'aspect'
        )
        
        # Calculate opinion metrics
        opinion_precision, opinion_recall, opinion_f1 = self._calculate_span_metrics(
            opinion_preds, opinion_labels, 'opinion'
        )
        
        # Calculate sentiment metrics
        sentiment_accuracy = self._calculate_sentiment_metrics(
            sentiment_preds, sentiment_labels
        )
        
        # Add all metrics to dict
        metrics.update({
            'aspect_precision': aspect_precision,
            'aspect_recall': aspect_recall,
            'aspect_f1': aspect_f1,
            'opinion_precision': opinion_precision,
            'opinion_recall': opinion_recall,
            'opinion_f1': opinion_f1,
            'sentiment_accuracy': sentiment_accuracy,
            'sentiment_f1': sentiment_accuracy  # For consistency
        })
        
        # Calculate overall F1
        metrics['f1'] = (aspect_f1 + opinion_f1 + sentiment_accuracy) / 3
        
        return metrics
    
    def _calculate_span_metrics(self, preds, labels, prefix):
        """Calculate precision, recall, F1 for span detection"""
        # Simple implementation - can be enhanced with more sophisticated metrics
        tp, fp, fn = 0, 0, 0
        
        for batch_preds, batch_labels in zip(preds, labels):
            # Only consider valid tokens (not padding)
            valid_mask = batch_labels != -100
            
            # Get predictions and labels for valid tokens
            batch_preds = batch_preds[valid_mask]
            batch_labels = batch_labels[valid_mask]
            
            # Calculate true positives, false positives, false negatives
            tp += ((batch_preds > 0) & (batch_labels > 0)).sum().item()
            fp += ((batch_preds > 0) & (batch_labels == 0)).sum().item()
            fn += ((batch_preds == 0) & (batch_labels > 0)).sum().item()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def _calculate_sentiment_metrics(self, preds, labels):
        """Calculate accuracy for sentiment classification"""
        correct = 0
        total = 0
        
        for batch_preds, batch_labels in zip(preds, labels):
            # Only consider valid labels (not padding)
            valid_mask = batch_labels != -100
            
            # Get predictions and labels for valid items
            batch_preds = batch_preds[valid_mask]
            batch_labels = batch_labels[valid_mask]
            
            # Calculate correct predictions
            correct += (batch_preds == batch_labels).sum().item()
            total += batch_labels.numel()
        
        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def get_attention_weights(self, val_loader):
        """Extract attention weights for visualization"""
        self.model.eval()
        
        # Check if model has the method
        if not hasattr(self.model, 'get_attention_weights'):
            return None
            
        # Get a sample batch
        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                attention_weights = self.model.get_attention_weights(**batch)
                return attention_weights
                
        return None