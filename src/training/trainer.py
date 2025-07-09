# src/training/clean_trainer.py
"""
Clean, unified training pipeline
Replaces complex training with a working implementation
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup
import os
import json
import time
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging

from src.data.dataset import load_datasets, create_dataloaders
from src.models.unified_absa_model import UnifiedABSAModel

class ABSATrainer:
    """Clean ABSA trainer implementation"""
    
    def __init__(self, model, config, train_dataloader, eval_dataloader=None):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_f1 = 0.0
        self.training_history = []
        
        # Output directory
        self.output_dir = config.get_experiment_dir()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting training...")
        print(f"   Model: {self.config.model_name}")
        print(f"   Epochs: {self.config.num_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Features enabled:")
        print(f"     - Implicit detection: {self.config.use_implicit_detection}")
        print(f"     - Few-shot learning: {self.config.use_few_shot_learning}")
        print(f"     - Generative framework: {self.config.use_generative_framework}")
        print(f"     - Contrastive learning: {self.config.use_contrastive_learning}")
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\n📚 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = self._train_epoch()
            
            # Evaluation
            if self.eval_dataloader and (epoch + 1) % 2 == 0:
                eval_results = self._evaluate()
                print(f"📊 Epoch {epoch + 1} Results:")
                print(f"   Train Loss: {epoch_loss:.4f}")
                print(f"   Eval F1: {eval_results.get('f1', 0):.4f}")
                print(f"   Eval Accuracy: {eval_results.get('accuracy', 0):.4f}")
                
                # Save best model
                if eval_results.get('f1', 0) > self.best_f1:
                    self.best_f1 = eval_results.get('f1', 0)
                    self._save_model('best_model.pt')
                    print(f"✅ New best F1: {self.best_f1:.4f}")
        
        # Save final model
        self._save_model('final_model.pt')
        
        # Save training history
        self._save_training_history()
        
        print(f"\n🎉 Training completed!")
        print(f"   Best F1: {self.best_f1:.4f}")
        print(f"   Models saved to: {self.output_dir}")
        
        return {
            'best_f1': self.best_f1,
            'training_history': self.training_history,
            'output_dir': self.output_dir
        }
    
    def _train_epoch(self):
        """Train one epoch"""
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Compute loss
            if 'losses' in outputs:
                loss = outputs['losses']['total_loss']
            else:
                # Fallback loss computation
                loss = self._compute_fallback_loss(outputs, batch['labels'])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log intermediate results
            if self.global_step % self.config.eval_interval == 0:
                self._log_step(loss.item())
        
        return total_loss / num_batches
    
    def _evaluate(self):
        """Evaluate model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # Compute loss
                if 'losses' in outputs:
                    loss = outputs['losses']['total_loss']
                    total_loss += loss.item()
                
                # Get predictions
                predictions = self._extract_predictions(outputs, batch)
                targets = self._extract_targets(batch)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics['eval_loss'] = total_loss / len(self.eval_dataloader)
        
        self.model.train()
        return metrics
    
    def _extract_predictions(self, outputs, batch):
        """Extract predictions from model outputs"""
        predictions = []
        
        # Get logits
        aspect_logits = outputs.get('aspect_logits')
        opinion_logits = outputs.get('opinion_logits')
        sentiment_logits = outputs.get('sentiment_logits')
        
        if aspect_logits is not None:
            aspect_preds = torch.argmax(aspect_logits, dim=-1)
            opinion_preds = torch.argmax(opinion_logits, dim=-1)
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1)
            
            # Convert to triplets (simplified)
            batch_size = aspect_preds.size(0)
            for b in range(batch_size):
                pred_triplets = self._convert_predictions_to_triplets(
                    aspect_preds[b], opinion_preds[b], sentiment_preds[b],
                    batch['attention_mask'][b]
                )
                predictions.append(pred_triplets)
        
        return predictions
    
    def _extract_targets(self, batch):
        """Extract target triplets from batch"""
        return batch.get('triplets', [])
    
    def _convert_predictions_to_triplets(self, aspect_preds, opinion_preds, sentiment_preds, attention_mask):
        """Convert predictions to triplet format"""
        triplets = []
        seq_len = attention_mask.sum().item()
        
        # Extract aspect spans
        aspect_spans = self._extract_spans(aspect_preds[:seq_len])
        opinion_spans = self._extract_spans(opinion_preds[:seq_len])
        
        # Create triplets
        for aspect_span in aspect_spans:
            for opinion_span in opinion_spans:
                # Get sentiment for this pair
                sentiment_scores = sentiment_preds[aspect_span[0]:aspect_span[1]+1]
                sentiment_idx = sentiment_scores.mode().values.item()
                sentiment_label = ['NEG', 'NEU', 'POS'][sentiment_idx]
                
                triplets.append({
                    'aspect': f"aspect_{aspect_span[0]}_{aspect_span[1]}",
                    'opinion': f"opinion_{opinion_span[0]}_{opinion_span[1]}",
                    'sentiment': sentiment_label
                })
        
        return triplets
    
    def _extract_spans(self, predictions):
        """Extract spans from BIO predictions"""
        spans = []
        start = None
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # B
                if start is not None:
                    spans.append((start, i-1))
                start = i
            elif pred == 0:  # O
                if start is not None:
                    spans.append((start, i-1))
                    start = None
        
        if start is not None:
            spans.append((start, len(predictions)-1))
        
        return spans
    
    def _compute_metrics(self, predictions, targets):
        """Compute evaluation metrics"""
        if not predictions or not targets:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Simplified metrics computation
        total_pred = sum(len(pred) for pred in predictions)
        total_target = sum(len(target) for target in targets)
        total_correct = 0
        
        for pred_list, target_list in zip(predictions, targets):
            # Convert to sets for comparison
            pred_set = set()
            target_set = set()
            
            for triplet in pred_list:
                if isinstance(triplet, dict):
                    pred_set.add((
                        str(triplet.get('aspect', '')),
                        str(triplet.get('opinion', '')),
                        str(triplet.get('sentiment', ''))
                    ))
            
            for triplet in target_list:
                if isinstance(triplet, (list, tuple)) and len(triplet) >= 3:
                    # Convert to strings if they're lists
                    aspect = str(triplet[0]) if not isinstance(triplet[0], str) else triplet[0]
                    opinion = str(triplet[1]) if not isinstance(triplet[1], str) else triplet[1]
                    sentiment = str(triplet[2]) if not isinstance(triplet[2], str) else triplet[2]
                    target_set.add((aspect, opinion, sentiment))
                elif isinstance(triplet, dict):
                    target_set.add((
                        str(triplet.get('aspect', '')),
                        str(triplet.get('opinion', '')),
                        str(triplet.get('sentiment', ''))
                    ))
                        
            total_correct += len(pred_set & target_set)
        
        # Compute metrics
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_target if total_target > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = total_correct / max(total_pred, total_target) if max(total_pred, total_target) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_predictions': total_pred,
            'total_targets': total_target,
            'total_correct': total_correct
        }
    
    def _compute_fallback_loss(self, outputs, labels):
        """Compute fallback loss if model doesn't provide losses"""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        if 'aspect_logits' in outputs and 'aspect_labels' in labels:
            aspect_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                outputs['aspect_logits'].view(-1, 3),
                labels['aspect_labels'].view(-1)
            )
            loss += aspect_loss
        
        if 'opinion_logits' in outputs and 'opinion_labels' in labels:
            opinion_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                outputs['opinion_logits'].view(-1, 3),
                labels['opinion_labels'].view(-1)
            )
            loss += opinion_loss
        
        if 'sentiment_logits' in outputs and 'sentiment_labels' in labels:
            sentiment_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                outputs['sentiment_logits'].view(-1, 3),
                labels['sentiment_labels'].view(-1)
            )
            loss += sentiment_loss
        
        return loss
    
    def _move_batch_to_device(self, batch):
        """Move batch to device"""
        device = next(self.model.parameters()).device
        
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device)
            elif isinstance(value, dict):
                moved_batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in value.items()}
            else:
                moved_batch[key] = value
        
        return moved_batch
    
    def _log_step(self, loss):
        """Log training step"""
        self.logger.info(f"Step {self.global_step}: loss={loss:.4f}")
    
    def _save_model(self, filename):
        """Save model checkpoint"""
        model_path = os.path.join(self.output_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_f1': self.best_f1
        }
        
        torch.save(checkpoint, model_path)
        self.logger.info(f"Model saved: {model_path}")
    
    def _save_training_history(self):
        """Save training history"""
        history_path = os.path.join(self.output_dir, 'training_history.json')
        
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Training history saved: {history_path}")


def train_absa_model(config):
    """Main training function"""
    print("🚀 Setting up ABSA training...")
    
    # Load datasets
    
    datasets = load_datasets(config)
    dataloaders = create_dataloaders(datasets, config)
    
    # Use first available dataset
    dataset_name = list(datasets.keys())[0]
    train_dataloader = dataloaders[dataset_name]['train']
    eval_dataloader = dataloaders[dataset_name].get('dev') or dataloaders[dataset_name].get('test')
    
    print(f"📊 Training on: {dataset_name}")
    print(f"   Train batches: {len(train_dataloader)}")
    print(f"   Eval batches: {len(eval_dataloader) if eval_dataloader else 0}")
    
    # Initialize model
    
    model = UnifiedABSAModel(config)
    model.to(config.device)
    
    print(f"🔧 Model initialized on {config.device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = ABSATrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )
    
    # Train model
    results = trainer.train()
    
    return results, model, trainer