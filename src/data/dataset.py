#!/usr/bin/env python
# train_stella.py
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
import logging
import random
import numpy as np

from src.data.dataset import ABSADataset
from src.data.stella_preprocessor import StellaABSAPreprocessor
from src.models.stella_absa import StellaABSA
from src.utils.stella_config import StellaABSAConfig
from src.training.losses import ABSALoss
from src.utils.logger import WandbLogger
from src.training.metrics import ABSAMetrics
from src.inference.stella_predictor import StellaABSAPredictor

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_dataset(config, tokenizer, logger, dataset_name, device):
    """Train model on a specific dataset"""
    
    print(f"\nTraining on dataset: {dataset_name}")
    
    # Create preprocessor
    preprocessor = StellaABSAPreprocessor(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        use_syntax=config.use_syntax
    )
    
    # Create datasets with domain id if using domain adaptation
    domain_id = config.domain_mapping.get(dataset_name, 0) if config.domain_adaptation else None
    
    train_dataset = ABSADataset(
        data_dir=config.dataset_paths[dataset_name],
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='train',
        dataset_name=dataset_name,
        max_length=config.max_seq_length,
        domain_id=domain_id
    )
    
    val_dataset = ABSADataset(
        data_dir=config.dataset_paths[dataset_name],
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='dev',
        dataset_name=dataset_name,
        max_length=config.max_seq_length,
        domain_id=domain_id
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Initialize or load model
    if os.path.exists(f"checkpoints/{config.experiment_name}_{dataset_name}_last.pt"):
        print(f"Loading existing model from checkpoints/{config.experiment_name}_{dataset_name}_last.pt")
        model = StellaABSA(config)
        model.load_state_dict(torch.load(f"checkpoints/{config.experiment_name}_{dataset_name}_last.pt"))
        model = model.to(device)
    else:
        print("Initializing new model")
        model = StellaABSA(config).to(device)
    
    # Initialize optimizer
    # Use different learning rates for base model and new components
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "embeddings" in n],
            "lr": config.learning_rate / 10.0,  # Lower learning rate for pretrained parts
        },
        {
            "params": [p for n, p in model.named_parameters() if "embeddings" not in n],
            "lr": config.learning_rate,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize loss function
    loss_fn = ABSALoss(config)
    
    # Initialize metrics tracker
    metrics = ABSAMetrics()
    
    # Set up mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler() if config.use_fp16 else None
    
    # Training loop
    best_f1 = 0.0
    global_step = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} (Train)")
        
        for batch in train_iterator:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass with mixed precision if enabled
            if config.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss_dict = loss_fn(outputs, batch)
                    loss = loss_dict['loss']
                    
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Update weights with scaling
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(**batch)
                loss_dict = loss_fn(outputs, batch)
                loss = loss_dict['loss']
                
                loss.backward()
                
                # Gradient clipping
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Update scheduler
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            train_steps += 1
            global_step += 1
            
            # Update progress bar
            train_iterator.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'lr': f"{scheduler.get_last_lr()[0]:.7f}"
            })
            
            # Log training metrics
            if global_step % config.log_interval == 0:
                logger.log_metrics({
                    'dataset': dataset_name,
                    'epoch': epoch + 1,
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                }, global_step)
            
            # Evaluate model periodically
            if global_step % config.eval_interval == 0:
                val_metrics = evaluate(model, val_loader, loss_fn, device, metrics)
                
                # Log validation metrics
                val_metrics['dataset'] = dataset_name
                val_metrics['epoch'] = epoch + 1
                logger.log_metrics(val_metrics, global_step)
                
                # Check if this is the best model
                if val_metrics.get('overall_f1', 0) > best_f1:
                    best_f1 = val_metrics.get('overall_f1', 0)
                    
                    # Save best model
                    os.makedirs('checkpoints', exist_ok=True)
                    torch.save(
                        model.state_dict(), 
                        f"checkpoints/{config.experiment_name}_{dataset_name}_best.pt"
                    )
                    
                    logger.log_model(model, {
                        'dataset': dataset_name,
                        **val_metrics
                    })
                    
                    print(f"New best model saved with F1 = {best_f1:.4f}")
                
                # Back to training mode
                model.train()
            
            # Save model periodically
            if global_step % config.save_interval == 0:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(
                    model.state_dict(), 
                    f"checkpoints/{config.experiment_name}_{dataset_name}_last.pt"
                )
        
    def __len__(self) -> int:
        return len(self.data)
    
    
    def __getitem__(self, idx):
        text, span_labels = self.data[idx]
        
        # Regular preprocessing
        tokenized = self.preprocessor.preprocess(text, span_labels)
        
        # Add original text
        tokenized['text'] = text
        
        return tokenized
