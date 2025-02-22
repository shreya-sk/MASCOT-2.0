# src/training/trainer.py
import torch
from tqdm import tqdm
from .metrics import ABSAMetrics
from torch.cuda.amp import autocast, GradScaler

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
        self.scaler = GradScaler()
        
        # Initialize best metrics
        self.best_f1 = 0
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch in train_iterator:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(**batch)
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict['loss']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.optimizer.zero_grad()
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Update progress bar
            total_loss += loss.item()
            train_iterator.set_postfix({
                'loss': f"{total_loss/(train_iterator.n+1):.4f}"
            })
            
        return total_loss / len(train_loader)
        
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss_dict = self.loss_fn(outputs, batch)
                
                total_loss += loss_dict['loss'].item()
                
                # Collect predictions and labels
                predictions = self._get_predictions(outputs)
                labels = self._get_labels(batch)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
        
        # Calculate metrics
        metrics = compute_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def _get_predictions(self, outputs):
        """Convert model outputs to predictions"""
        aspect_preds = outputs['aspect_logits'].argmax(dim=-1)
        opinion_preds = outputs['opinion_logits'].argmax(dim=-1)
        sentiment_preds = outputs['sentiment_logits'].argmax(dim=-1)
        
        return list(zip(aspect_preds, opinion_preds, sentiment_preds))
    
    def _get_labels(self, batch):
        """Get ground truth labels"""
        return list(zip(
            batch['aspect_labels'],
            batch['opinion_labels'],
            batch['sentiment_labels']
        ))