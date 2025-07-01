#!/usr/bin/env python
# train.py - Complete training script for ABSA model
import os
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import traceback
import gc
from datetime import datetime

# Import modules
from src.utils.config import LLMABSAConfig
from src.data.dataset import ABSADataset, custom_collate_fn
from src.data.preprocessor import LLMABSAPreprocessor
from src.models.absa import LLMABSA
from src.training.losses import ABSALoss
from src.utils.logger import WandbLogger

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif (self.mode == 'max' and val_score <= self.best_score + self.min_delta) or \
             (self.mode == 'min' and val_score >= self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, config, dataset_name, phase, epoch=None, best=True):
    """Save model checkpoint with proper metadata"""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if best:
        file_path = f"{checkpoint_dir}/{config.experiment_name}_{dataset_name}_{phase}_best.pt"
    else:
        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
        file_path = f"{checkpoint_dir}/{config.experiment_name}_{dataset_name}_{phase}{epoch_str}_{timestamp}.pt"
    
    # Save model using the model's save method
    try:
        model.save(file_path)
        print(f"✓ Model saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving model: {e}")
        # Fallback to torch.save
        torch.save(model.state_dict(), file_path)
        return file_path

def create_data_loaders(config, tokenizer, dataset_name, domain_id=None):
    """Create train and validation data loaders"""
    try:
        # Create preprocessor
        preprocessor = LLMABSAPreprocessor(
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            use_syntax=config.use_syntax
        )
        
        # Get data directory
        data_dir = config.dataset_paths.get(dataset_name, f"Datasets/aste/{dataset_name}")
        
        # Create train dataset
        train_dataset = ABSADataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            split='train',
            dataset_name=dataset_name,
            max_length=config.max_seq_length,
            domain_id=domain_id
        )
        
        # Create validation dataset
        val_dataset = ABSADataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            split='dev',
            dataset_name=dataset_name,
            max_length=config.max_seq_length,
            domain_id=domain_id
        )
        
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Val dataset: {len(val_dataset)} samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for debugging
            collate_fn=custom_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        traceback.print_exc()
        raise

def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, config, 
               logger, global_step, epoch, dataset_name, phase):
    """Train for one epoch with improved error handling"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} ({phase})")
    
    # Gradient accumulation counter
    accumulation_counter = 0
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move batch to device (only tensors)
            batch_on_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                else:
                    batch_on_device[k] = v  # Keep non-tensors as is
            
            # Ensure input_ids are long tensors
            if 'input_ids' in batch_on_device:
                batch_on_device['input_ids'] = batch_on_device['input_ids'].long()
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.use_fp16):
                outputs = model(**batch_on_device)
                
                # Calculate loss
                loss_dict = loss_fn(outputs, batch_on_device)
                loss = loss_dict['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            if config.use_fp16:
                from torch.cuda.amp import GradScaler
                if not hasattr(train_epoch, 'scaler'):
                    train_epoch.scaler = GradScaler()
                train_epoch.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update accumulation counter
            accumulation_counter += 1
            
            # Update weights after accumulation steps
            if accumulation_counter >= config.gradient_accumulation_steps:
                # Gradient clipping
                if hasattr(config, 'max_grad_norm') and config.max_grad_norm > 0:
                    if config.use_fp16:
                        train_epoch.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Optimizer step
                if config.use_fp16:
                    train_epoch.scaler.step(optimizer)
                    train_epoch.scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                accumulation_counter = 0
            
            # Update metrics
            total_loss += loss.item() * config.gradient_accumulation_steps  # Unscale for logging
            num_batches += 1
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log metrics periodically
            if batch_idx % config.log_interval == 0:
                step = global_step + batch_idx
                log_metrics = {
                    'dataset': dataset_name,
                    'phase': phase,
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'train_batch_loss': loss.item() * config.gradient_accumulation_steps,
                    'train_aspect_loss': loss_dict.get('aspect_loss', 0.0),
                    'train_opinion_loss': loss_dict.get('opinion_loss', 0.0),
                    'train_sentiment_loss': loss_dict.get('sentiment_loss', 0.0),
                    'learning_rate': current_lr
                }
                logger.log_metrics(log_metrics, step)
            
            # Memory cleanup
            del outputs, loss_dict, loss
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠ OOM at batch {batch_idx}, skipping...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                print(f"Error in batch {batch_idx}: {e}")
                traceback.print_exc()
                continue
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Final gradient update if needed
    if accumulation_counter > 0:
        if config.use_fp16:
            train_epoch.scaler.step(optimizer)
            train_epoch.scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / max(1, num_batches)

def evaluate(model, val_loader, loss_fn, device):
    """Evaluate model on validation data"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Metrics tracking
    aspect_preds, opinion_preds, sentiment_preds = [], [], []
    aspect_labels, opinion_labels, sentiment_labels = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            try:
                # Move batch to device
                batch_on_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_on_device[k] = v.to(device)
                    else:
                        batch_on_device[k] = v
                
                # Forward pass
                outputs = model(**batch_on_device)
                
                # Calculate loss
                loss_dict = loss_fn(outputs, batch_on_device)
                total_loss += loss_dict['loss'].item()
                num_batches += 1
                
                # Collect predictions for metrics
                aspect_pred = outputs['aspect_logits'].argmax(dim=-1).cpu()
                opinion_pred = outputs['opinion_logits'].argmax(dim=-1).cpu()
                sentiment_pred = outputs['sentiment_logits'].argmax(dim=-1).cpu()
                
                # Get labels (handle multi-span case)
                aspect_label = batch['aspect_labels'].cpu()
                if len(aspect_label.shape) == 3:  # Multi-span case
                    aspect_label = aspect_label[:, 0]  # Take first span
                
                opinion_label = batch['opinion_labels'].cpu()
                if len(opinion_label.shape) == 3:
                    opinion_label = opinion_label[:, 0]
                
                sentiment_label = batch['sentiment_labels'].cpu()
                if len(sentiment_label.shape) > 1:
                    sentiment_label = sentiment_label[:, 0]
                
                # Store for metrics calculation
                aspect_preds.append(aspect_pred)
                opinion_preds.append(opinion_pred)
                sentiment_preds.append(sentiment_pred)
                
                aspect_labels.append(aspect_label)
                opinion_labels.append(opinion_label)
                sentiment_labels.append(sentiment_label)
                
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    # Calculate metrics
    metrics = calculate_metrics(
        aspect_preds, opinion_preds, sentiment_preds,
        aspect_labels, opinion_labels, sentiment_labels
    )
    
    # Add loss
    metrics['loss'] = total_loss / max(1, num_batches)
    
    return metrics

def calculate_metrics(aspect_preds, opinion_preds, sentiment_preds,
                     aspect_labels, opinion_labels, sentiment_labels):
    """Calculate evaluation metrics"""
    try:
        # Calculate span metrics (precision, recall, F1)
        aspect_metrics = calculate_span_metrics(aspect_preds, aspect_labels)
        opinion_metrics = calculate_span_metrics(opinion_preds, opinion_labels)
        sentiment_metrics = calculate_sentiment_metrics(sentiment_preds, sentiment_labels)
        
        # Combine metrics
        metrics = {
            'aspect_precision': aspect_metrics[0],
            'aspect_recall': aspect_metrics[1],
            'aspect_f1': aspect_metrics[2],
            'opinion_precision': opinion_metrics[0],
            'opinion_recall': opinion_metrics[1],
            'opinion_f1': opinion_metrics[2],
            'sentiment_accuracy': sentiment_metrics,
            'sentiment_f1': sentiment_metrics,  # Use accuracy as F1 for sentiment
        }
        
        # Calculate overall F1
        metrics['overall_f1'] = (metrics['aspect_f1'] + metrics['opinion_f1'] + metrics['sentiment_f1']) / 3
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Return default metrics
        return {
            'aspect_precision': 0.0, 'aspect_recall': 0.0, 'aspect_f1': 0.0,
            'opinion_precision': 0.0, 'opinion_recall': 0.0, 'opinion_f1': 0.0,
            'sentiment_accuracy': 0.0, 'sentiment_f1': 0.0, 'overall_f1': 0.0
        }

def calculate_span_metrics(preds, labels):
    """Calculate precision, recall, F1 for span detection"""
    tp, fp, fn = 0, 0, 0
    
    for batch_preds, batch_labels in zip(preds, labels):
        # Create mask for valid positions (not padding)
        valid_mask = batch_labels != -100
        
        if valid_mask.sum() == 0:
            continue
            
        # Get valid predictions and labels
        valid_preds = batch_preds[valid_mask]
        valid_labels = batch_labels[valid_mask]
        
        # Count B and I tags as positive
        pred_positive = (valid_preds > 0)
        label_positive = (valid_labels > 0)
        
        tp += (pred_positive & label_positive).sum().item()
        fp += (pred_positive & ~label_positive).sum().item()
        fn += (~pred_positive & label_positive).sum().item()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_sentiment_metrics(preds, labels):
    """Calculate accuracy for sentiment classification"""
    correct = 0
    total = 0
    
    for batch_preds, batch_labels in zip(preds, labels):
        # Create mask for valid labels
        valid_mask = batch_labels != -100
        
        if valid_mask.sum() == 0:
            continue
            
        # Get valid predictions and labels
        valid_preds = batch_preds[valid_mask]
        valid_labels = batch_labels[valid_mask]
        
        # Count correct predictions
        correct += (valid_preds == valid_labels).sum().item()
        total += valid_labels.numel()
    
    return correct / total if total > 0 else 0.0

def train_dataset(config, tokenizer, logger, dataset_name, device):
    """Train model on a specific dataset"""
    print(f"\n{'='*50}")
    print(f"Training on dataset: {dataset_name}")
    print(f"{'='*50}")
    
    try:
        # Get domain ID if using domain adaptation
        domain_id = config.domain_mapping.get(dataset_name, 0) if config.domain_adaptation else None
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(config, tokenizer, dataset_name, domain_id)
        
        # Initialize model
        print(f"Initializing LLMABSA model...")
        model = LLMABSA(config).to(device)
        
        # Print model info
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model parameters: {param_count:,} (trainable: {trainable_params:,})")
        
        # Set tokenizer on model
        model.tokenizer = tokenizer
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        # Initialize scheduler
        num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        print(f"✓ Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")
        
        # Initialize loss function
        loss_fn = ABSALoss(config)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode='max')
        
        # Training loop
        best_f1 = 0.0
        global_step = 0
        
        print(f"Starting training for {config.num_epochs} epochs...")
        
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            
            # Training
            train_loss = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                device=device,
                config=config,
                logger=logger,
                global_step=global_step,
                epoch=epoch,
                dataset_name=dataset_name,
                phase="training"
            )
            
            global_step += len(train_loader)
            
            # Evaluation
            print("Evaluating...")
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val F1 - Aspect: {val_metrics['aspect_f1']:.4f}, Opinion: {val_metrics['opinion_f1']:.4f}, Sentiment: {val_metrics['sentiment_f1']:.4f}")
            print(f"Overall F1: {val_metrics['overall_f1']:.4f}")
            
            # Log metrics
            log_metrics = {
                'dataset': dataset_name,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_overall_f1': val_metrics['overall_f1'],
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            logger.log_metrics(log_metrics, global_step)
            
            # Check for best model
            current_f1 = val_metrics['overall_f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                save_model(model, config, dataset_name, "training", best=True)
                print(f"✓ New best model saved! F1: {best_f1:.4f}")
            
            # Early stopping check
            early_stopping(current_f1)
            if early_stopping.early_stop:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                save_model(model, config, dataset_name, "training", epoch=epoch+1, best=False)
        
        # Save final model
        save_model(model, config, dataset_name, "training", epoch=config.num_epochs, best=False)
        
        print(f"\nTraining completed! Best F1: {best_f1:.4f}")
        return best_f1
        
    except Exception as e:
        print(f"Error training on {dataset_name}: {e}")
        traceback.print_exc()
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='Train ABSA Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, help='Specific dataset to train on')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--hidden_size', type=int, help='Hidden size')
    parser.add_argument('--model', type=str, help='Model name to use')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = LLMABSAConfig()
    
    # Override config with command line arguments
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.hidden_size:
        config.hidden_size = args.hidden_size
    if args.model:
        config.model_name = args.model
    if args.gradient_accumulation_steps:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_grad_norm:
        config.max_grad_norm = args.max_grad_norm
    
    # Debug mode settings
    if args.debug:
        config.num_epochs = 2
        config.batch_size = min(4, config.batch_size)
        config.log_interval = 1
        print("🐛 Debug mode enabled")
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ Using CPU (training will be slow)")
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize logger
    logger = WandbLogger(config, use_wandb=not args.no_wandb)
    
    # Initialize tokenizer
    print(f"Loading tokenizer: {config.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        
        # Add special tokens if needed
        special_tokens = {}
        if tokenizer.pad_token is None:
            special_tokens["pad_token"] = "[PAD]"
        if tokenizer.eos_token is None:
            special_tokens["eos_token"] = "</s>"
        if tokenizer.bos_token is None:
            special_tokens["bos_token"] = "<s>"
        if tokenizer.unk_token is None:
            special_tokens["unk_token"] = "[UNK]"
        
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
            
        print(f"✓ Tokenizer loaded successfully")
        
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Train on datasets
    datasets = [args.dataset] if args.dataset else config.datasets
    results = {}
    
    print(f"\nTraining on datasets: {datasets}")
    
    for dataset_name in datasets:
        try:
            best_f1 = train_dataset(config, tokenizer, logger, dataset_name, device)
            results[dataset_name] = best_f1
        except Exception as e:
            print(f"Failed to train on {dataset_name}: {e}")
            results[dataset_name] = 0.0
    
    # Print final results
    print(f"\n{'='*50}")
    print("TRAINING COMPLETED")
    print(f"{'='*50}")
    for dataset, f1 in results.items():
        print(f"{dataset}: Best F1 = {f1:.4f}")
    
    avg_f1 = sum(results.values()) / len(results) if results else 0.0
    print(f"Average F1: {avg_f1:.4f}")
    
    # End W&B run
    logger.finish()
    
    print("✓ Training finished successfully!")

if __name__ == '__main__':
    main()