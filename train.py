#!/usr/bin/env python3
"""
Complete Training Script for Enhanced ABSA with 2024-2025 Breakthrough Features
Supports instruction-following, contrastive learning, and few-shot adaptation
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import traceback
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import LLMABSAConfig
from src.models.absa import LLMABSA
from src.data.dataset import ABSADataset
from src.data.preprocessor import ABSAPreprocessor
from src.training.losses import ABSALoss
from src.training.metrics import calculate_comprehensive_metrics, save_metrics_to_file
from src.utils.logger import setup_logger

def setup_directories():
    """Create necessary directories"""
    dirs = ['checkpoints', 'logs', 'results', 'visualizations']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced ABSA Training')
    
    # Dataset and model arguments
    parser.add_argument('--dataset', type=str, default='rest15', 
                       choices=['laptop14', 'rest14', 'rest15', 'rest16'],
                       help='Dataset to use for training')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                       help='Pre-trained model name')
    parser.add_argument('--data_dir', type=str, default='Dataset/aste',
                       help='Data directory')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    
    # 2024-2025 Breakthrough Features
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                       help='Enable contrastive learning')
    parser.add_argument('--use_few_shot', action='store_true', default=True,
                       help='Enable few-shot learning')
    parser.add_argument('--use_implicit', action='store_true', default=True,
                       help='Enable implicit detection')
    parser.add_argument('--use_instruction', action='store_true', default=True,
                       help='Enable instruction following')
    
    # Advanced training options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (fewer epochs, smaller data)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    
    # Hardware optimization
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory')
    
    return parser.parse_args()

def create_config_from_args(args):
    """Create configuration from arguments"""
    config = LLMABSAConfig()
    
    # Update config with arguments
    config.model_name = args.model_name
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.epochs = args.epochs
    config.max_length = args.max_length
    
    # 2024-2025 breakthrough features
    config.use_contrastive_learning = args.use_contrastive
    config.use_few_shot_learning = args.use_few_shot
    config.use_implicit_detection = args.use_implicit
    config.use_instruction_following = args.use_instruction
    
    # Hardware optimization
    config.mixed_precision = args.mixed_precision
    config.gradient_checkpointing = args.gradient_checkpointing
    
    # Debug mode adjustments
    if args.debug:
        config.epochs = min(2, config.epochs)
        config.debug_mode = True
        print("🐛 Debug mode enabled - reduced epochs and data")
    
    return config

def load_datasets(config, tokenizer, args):
    """Load training, validation, and test datasets"""
    print(f"📊 Loading {args.dataset} dataset...")
    
    # Initialize preprocessor
    preprocessor = ABSAPreprocessor(
        tokenizer=tokenizer,
        max_length=config.max_length,
        use_instruction_following=config.use_instruction_following
    )
    
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'dev', 'test']:
        try:
            dataset = ABSADataset(
                data_dir=args.data_dir,
                tokenizer=tokenizer,
                preprocessor=preprocessor,
                split=split,
                dataset_name=args.dataset,
                max_length=config.max_length
            )
            
            # Debug mode: limit dataset size
            if args.debug and len(dataset) > 100:
                dataset.data = dataset.data[:100]
                print(f"🐛 Limited {split} dataset to 100 samples for debug")
            
            datasets[split] = dataset
            
            # Create dataloader
            shuffle = (split == 'train')
            batch_size = config.batch_size if split == 'train' else min(config.batch_size * 2, 16)
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=preprocessor.collate_fn,
                num_workers=2,
                pin_memory=True
            )
            dataloaders[split] = dataloader
            
            print(f"✓ {split}: {len(dataset)} samples, {len(dataloader)} batches")
            
        except Exception as e:
            print(f"❌ Failed to load {split} dataset: {e}")
            if split == 'train':
                raise
            datasets[split] = None
            dataloaders[split] = None
    
    return datasets, dataloaders

def initialize_model_and_optimizer(config, args):
    """Initialize model, optimizer, and scheduler"""
    print("🏗️ Initializing model and optimizer...")
    
    # Initialize model
    model = LLMABSA(config)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize loss function
    loss_fn = ABSALoss(config)
    
    # Prepare optimizer with different learning rates for different components
    backbone_params = []
    instruction_params = []
    
    for name, param in model.named_parameters():
        if 't5_model' in name or 'instruction' in name or 'feature_bridge' in name:
            instruction_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.learning_rate},
        {'params': instruction_params, 'lr': config.learning_rate * 0.5}  # Lower LR for T5
    ], weight_decay=0.01)
    
    # Calculate total training steps
    # Note: This is an approximation since we don't know exact dataset size yet
    estimated_steps = (1000 // config.batch_size) * config.epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * estimated_steps),
        num_training_steps=estimated_steps
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Device: {device}")
    print(f"✓ Mixed precision: {scaler is not None}")
    
    return model, loss_fn, optimizer, scheduler, scaler, device

def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, scaler, device, config, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    component_losses = {}
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        texts=batch.get('texts', None),
                        task_type='triplet_extraction',
                        target_text=batch.get('target_text', None)
                    )
                    
                    if outputs is None:
                        print(f"⚠ Skipping batch {batch_idx}: model returned None")
                        continue
                    
                    # Calculate loss
                    loss_dict = loss_fn(outputs, batch)
                    loss = loss_dict['loss']
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    texts=batch.get('texts', None),
                    task_type='triplet_extraction',
                    target_text=batch.get('target_text', None)
                )
                
                if outputs is None:
                    print(f"⚠ Skipping batch {batch_idx}: model returned None")
                    continue
                
                # Calculate loss
                loss_dict = loss_fn(outputs, batch)
                loss = loss_dict['loss']
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k != 'loss':
                    component_losses[k] = component_losses.get(k, 0) + v
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{current_lr:.2e}"
            })
            
            # Debug mode: break early
            if config.debug_mode and batch_idx >= 10:
                print("🐛 Debug mode: stopping epoch early")
                break
                
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            if config.debug_mode:
                traceback.print_exc()
            continue
    
    # Calculate average losses
    avg_loss = total_loss / num_batches
    avg_component_losses = {k: v / num_batches for k, v in component_losses.items()}
    
    return {
        'train_loss': avg_loss,
        **{f'train_{k}': v for k, v in avg_component_losses.items()}
    }

def evaluate_model(model, dataloader, loss_fn, device, config):
    """Evaluate model on validation/test set"""
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            try:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    texts=batch.get('texts', None),
                    task_type='triplet_extraction'
                )
                
                if outputs is None:
                    continue
                
                # Calculate loss
                loss_dict = loss_fn(outputs, batch)
                total_loss += loss_dict['loss'].item()
                
                # Extract predictions and targets for metrics
                predictions = model.extract_triplets(
                    outputs=outputs,
                    input_ids=batch['input_ids'],
                    texts=batch.get('texts', batch.get('original_texts', []))
                )
                
                targets = batch.get('triplets', [])
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                # Debug mode: break early
                if config.debug_mode and batch_idx >= 5:
                    break
                    
            except Exception as e:
                print(f"❌ Error in evaluation batch {batch_idx}: {e}")
                continue
    
    # Calculate comprehensive metrics
    try:
        metrics = calculate_comprehensive_metrics(all_predictions, all_targets, config)
        metrics['val_loss'] = total_loss / max(num_batches, 1)
    except Exception as e:
        print(f"⚠ Warning: Failed to calculate metrics: {e}")
        metrics = {
            'val_loss': total_loss / max(num_batches, 1),
            'val_f1': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0
        }
    
    return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_path):
    """Save model checkpoint"""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")

def main():
    """Main training function"""
    print("🚀 Starting Enhanced ABSA Training with 2024-2025 Breakthrough Features")
    print("=" * 80)
    
    # Setup
    setup_directories()
    args = parse_arguments()
    config = create_config_from_args(args)
    
    # Setup logging
    logger = setup_logger('training', 'logs/training.log')
    logger.info(f"Training started with config: {vars(args)}")
    
    # Initialize tokenizer
    print(f"🔤 Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    datasets, dataloaders = load_datasets(config, tokenizer, args)
    
    # Update config with actual dataset size for scheduler
    train_steps_per_epoch = len(dataloaders['train'])
    config.total_training_steps = train_steps_per_epoch * config.epochs
    
    # Initialize model
    model, loss_fn, optimizer, scheduler, scaler, device = initialize_model_and_optimizer(config, args)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_f1 = 0.0
    training_history = []
    
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('metrics', {}).get('val_f1', 0.0)
        print(f"✅ Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    # Training loop
    print(f"\n🏃 Starting training for {config.epochs} epochs...")
    print(f"📊 Dataset: {args.dataset} | Batch size: {config.batch_size} | Device: {device}")
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.time()
        
        print(f"\n📅 Epoch {epoch + 1}/{config.epochs}")
        print("-" * 50)
        
        # Training
        train_metrics = train_epoch(
            model, dataloaders['train'], loss_fn, optimizer, 
            scheduler, scaler, device, config, epoch + 1
        )
        
        # Evaluation
        val_metrics = {}
        if dataloaders['dev'] and (epoch + 1) % args.eval_every == 0:
            print("📊 Evaluating on validation set...")
            val_metrics = evaluate_model(model, dataloaders['dev'], loss_fn, device, config)
        
        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        epoch_metrics['epoch'] = epoch + 1
        epoch_metrics['epoch_time'] = time.time() - epoch_start_time
        
        training_history.append(epoch_metrics)
        
        # Print epoch summary
        print(f"\n📈 Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics.get('train_loss', 0):.4f}")
        if val_metrics:
            print(f"  Val Loss: {val_metrics.get('val_loss', 0):.4f}")
            print(f"  Val F1: {val_metrics.get('val_f1', 0):.4f}")
            print(f"  Val Precision: {val_metrics.get('val_precision', 0):.4f}")
            print(f"  Val Recall: {val_metrics.get('val_recall', 0):.4f}")
        print(f"  Time: {epoch_metrics['epoch_time']:.1f}s")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            current_f1 = val_metrics.get('val_f1', 0.0)
            
            # Save regular checkpoint
            checkpoint_path = f"checkpoints/{args.dataset}_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, checkpoint_path)
            
            # Save best model
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_checkpoint_path = f"checkpoints/{args.dataset}_best.pt"
                save_checkpoint(model, optimizer, scheduler, epoch, epoch_metrics, best_checkpoint_path)
                print(f"🏆 New best F1 score: {best_f1:.4f}")
        
        # Log to file
        logger.info(f"Epoch {epoch + 1}: {epoch_metrics}")
    
    # Final evaluation on test set
    if dataloaders['test']:
        print("\n🎯 Final evaluation on test set...")
        test_metrics = evaluate_model(model, dataloaders['test'], loss_fn, device, config)
        print(f"📊 Test Results:")
        print(f"  Test Loss: {test_metrics.get('val_loss', 0):.4f}")
        print(f"  Test F1: {test_metrics.get('val_f1', 0):.4f}")
        print(f"  Test Precision: {test_metrics.get('val_precision', 0):.4f}")
        print(f"  Test Recall: {test_metrics.get('val_recall', 0):.4f}")
        
        # Save test results
        results_path = f"results/{args.dataset}_test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"✅ Test results saved: {results_path}")
    
    # Save training history
    history_path = f"results/{args.dataset}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"✅ Training history saved: {history_path}")
    
    print("\n🎉 Training completed successfully!")
    print(f"🏆 Best validation F1: {best_f1:.4f}")
    print(f"💾 Best model saved: checkpoints/{args.dataset}_best.pt")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        traceback.print_exc()