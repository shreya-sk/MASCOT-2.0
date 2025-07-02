#!/usr/bin/env python
"""
Complete Training Script for ABSA Model
Aligned with 2024-2025 Breakthrough Report
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
import wandb
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import your project modules
from src.utils.config import LLMABSAConfig
from src.models.absa import LLMABSA
from src.data.dataset import ABSADataset
from src.data.preprocessor import LLMABSAPreprocessor
from src.training.trainer import ABSATrainer
from src.training.losses import ABSALoss
from src.training.metrics import ABSAMetrics
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train ABSA Model')
    parser.add_argument('--dataset', type=str, default='rest15', 
                       choices=['laptop14', 'rest14', 'rest15', 'rest16'],
                       help='Dataset to train on')
    parser.add_argument('--config', type=str, default='default',
                       choices=['memory_constrained', 'default', 'high_performance'],
                       help='Configuration preset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (1 epoch, small data)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--instruction_following', action='store_true', 
                       help='Enable instruction-following training (2024-2025 feature)')
    parser.add_argument('--contrastive_learning', action='store_true',
                       help='Enable contrastive learning (2024-2025 feature)')
    parser.add_argument('--few_shot', action='store_true',
                       help='Enable few-shot learning (2024-2025 feature)')
    
    return parser.parse_args()

def setup_config(args):
    """Setup configuration based on arguments"""
    config = LLMABSAConfig()
    
    # Apply configuration presets
    if args.config == 'memory_constrained':
        config.batch_size = 4
        config.hidden_size = 256
        config.num_attention_heads = 4
        config.use_gradient_checkpointing = True
        print("🐏 Memory-constrained configuration loaded")
        
    elif args.config == 'high_performance':
        config.batch_size = 32
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.use_gradient_checkpointing = False
        print("🚀 High-performance configuration loaded")
    
    # Override with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Enable 2024-2025 breakthrough features
    if args.instruction_following:
        config.use_instruction_following = True
        print("📝 Instruction-following enabled (InstructABSA paradigm)")
    
    if args.contrastive_learning:
        config.use_contrastive_learning = True
        print("🔄 Contrastive learning enabled")
    
    if args.few_shot:
        config.use_few_shot_learning = True
        print("🎯 Few-shot learning enabled")
    
    # Debug mode adjustments
    if args.debug:
        config.batch_size = min(config.batch_size, 4)
        config.max_samples_debug = 100
        args.epochs = 1
        print("🐛 Debug mode: small batch, 1 epoch, limited data")
    
    return config

def create_data_loaders(config, args, tokenizer, preprocessor):
    """Create data loaders for training and validation"""
    
    # Training dataset
    train_dataset = ABSADataset(
        data_dir='',  # Will use project root
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='train',
        dataset_name=args.dataset,
        max_length=config.max_length
    )
    
    # Validation dataset  
    val_dataset = ABSADataset(
        data_dir='',
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='dev',
        dataset_name=args.dataset,
        max_length=config.max_length
    )
    
    # Debug mode: limit data
    if args.debug and hasattr(config, 'max_samples_debug'):
        train_dataset.data = train_dataset.data[:config.max_samples_debug]
        val_dataset.data = val_dataset.data[:min(20, len(val_dataset.data))]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Training batches: {len(train_loader)}")
    
    return train_loader, val_loader

def setup_model_and_training(config, args, device):
    """Setup model, optimizer, scheduler, and loss function"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Initialize preprocessor
    preprocessor = LLMABSAPreprocessor(
        tokenizer=tokenizer,
        max_length=config.max_length,
        use_syntax=config.use_syntax_features
    )
    
    # Initialize model
    model = LLMABSA(config).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Setup scheduler
    num_training_steps = args.epochs * 100  # Estimate, will update later
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Setup loss function
    loss_fn = ABSALoss(config)
    
    # Setup metrics
    metrics = ABSAMetrics()
    
    return model, tokenizer, preprocessor, optimizer, scheduler, loss_fn, metrics

def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = loss_fn(outputs, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Print progress
        if batch_idx % 10 == 0 or args.debug:
            progress = (batch_idx + 1) / num_batches * 100
            print(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches} "
                  f"({progress:.1f}%), Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, val_loader, loss_fn, metrics, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = loss_fn(outputs, batch)
            
            total_loss += loss.item()
            
            # Collect predictions and labels for metrics
            all_predictions.append(outputs)
            all_labels.append(batch)
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    metric_results = metrics.compute_metrics(all_predictions, all_labels)
    
    return avg_loss, metric_results

def save_checkpoint(model, optimizer, scheduler, epoch, loss, args, config):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config.__dict__,
        'args': vars(args)
    }
    
    # Save checkpoint
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'model_{args.dataset}_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    best_path = os.path.join(checkpoint_dir, f'best_model_{args.dataset}.pt')
    torch.save(model.state_dict(), best_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def main():
    """Main training function"""
    args = parse_args()
    
    # Setup
    config = setup_config(args)
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Setup logging
    logger = setup_logger('training', 'logs/training.log')
    
    # Initialize Weights & Biases (optional)
    if args.wandb:
        wandb.init(
            project="absa-2024-2025",
            config=vars(args),
            name=f"{args.dataset}_{args.config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Setup model and training components
    model, tokenizer, preprocessor, optimizer, scheduler, loss_fn, metrics = setup_model_and_training(
        config, args, device
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, args, tokenizer, preprocessor)
    
    # Update scheduler with actual number of steps
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n📅 Epoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, 
                               loss_fn, device, epoch+1, args)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, loss_fn, metrics, device)
        
        # Print results
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        for metric_name, metric_value in val_metrics.items():
            print(f"   {metric_name}: {metric_value:.4f}")
        
        # Log to wandb
        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **val_metrics
            })
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, 
                                           epoch, val_loss, args, config)
            print(f"🌟 New best model saved!")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, args, config)
    
    print("\n🎉 Training completed!")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()