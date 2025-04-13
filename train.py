#!/usr/bin/env python
# train_absa.py - Training script for Generative ABSA model
import os
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.utils.config import LLMABSAConfig
from transformers import get_linear_schedule_with_warmup
import traceback

# Import modules
from src.data.dataset import ABSADataset, custom_collate_fn
from src.data.preprocessor import LLMABSAPreprocessor
from src.models.absa import GenerativeLLMABSA
from src.training.losses import ABSALoss
from src.utils.config import LLMABSAConfig as con
from src.utils.logger import WandbLogger

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_dataset(config, tokenizer, logger, dataset_name, device, two_phase=True):
    """
    Train model on a specific dataset with optional two-phase training
    
    Args:
        config: Model configuration
        tokenizer: Tokenizer for text encoding
        logger: Logger for metrics
        dataset_name: Name of the dataset to train on
        device: Device to use for training
        two_phase: Whether to use two-phase training (extraction then generation)
    """
    print(f"\nTraining on dataset: {dataset_name}")
    
    # Create preprocessor
    preprocessor = LLMABSAPreprocessor(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        use_syntax=config.use_syntax
    )
    
    # Create datasets with domain id if using domain adaptation
    domain_id = config.domain_mapping.get(dataset_name, 0) if config.domain_adaptation else None
    
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
    
    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for debugging
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    print(f"Initializing GenerativeLLMABSA model")
    model = GenerativeLLMABSA(config).to(device)
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Set up gradient accumulation steps
    grad_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
    print(f"Using gradient accumulation with {grad_accum_steps} steps")
    
    # Two-phase training approach
    best_overall_f1 = 0.0
    
    # Phase 1: Train extraction only if two_phase is True
    if two_phase:
        print("\n===== Phase 1: Training extraction components =====")
        
        # Save original generation flag and disable generation for phase 1
        original_gen_flag = config.generate_explanations
        config.generate_explanations = False
        
        # Phase 1 training
        best_extraction_f1 = train_phase(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            logger=logger,
            device=device,
            dataset_name=dataset_name,
            phase="extraction",
            generate=False,
            epochs=config.num_epochs // 2  # Half of total epochs
        )
        
        print(f"\nPhase 1 complete. Best extraction F1: {best_extraction_f1:.4f}")
        
        # Restore original generation flag
        config.generate_explanations = original_gen_flag
        
        # Load best extraction model for phase 2
        checkpoint_path = f"checkpoints/{config.experiment_name}_{dataset_name}_extraction_best.pt"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded best extraction model from {checkpoint_path}")
    
    # Phase 2: Train with generation (or single phase if two_phase is False)
    phase_name = "generation" if two_phase else "single"
    epochs = config.num_epochs // 2 if two_phase else config.num_epochs
    
    print(f"\n===== {'Phase 2: Training with generation' if two_phase else 'Training full model'} =====")
    
    # Train the model
    best_overall_f1 = train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        device=device,
        dataset_name=dataset_name,
        phase=phase_name,
        generate=config.generate_explanations,
        epochs=epochs
    )
    
    print(f"\nTraining complete. Best overall F1: {best_overall_f1:.4f}")
    return best_overall_f1

def train_phase(model, train_loader, val_loader, config, logger, device, dataset_name, phase, generate, epochs):
    """Train model for a specific phase (extraction or generation)"""
    # Initialize optimizer with appropriate learning rate
    lr = config.learning_rate
    if phase == "generation":
        lr = lr / 2.0  # Lower learning rate for fine-tuning
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.weight_decay,
    )
    
    # Initialize scheduler
    grad_accum_steps = getattr(config, 'gradient_accumulation_steps', 1)
    num_training_steps = len(train_loader) // grad_accum_steps * epochs
    num_warmup_steps = int(num_training_steps * getattr(config, 'warmup_ratio', 0.1))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize loss function
    loss_fn = ABSALoss(config)
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_f1 = 0.0
    global_step = 0
    
    for epoch in range(epochs):
        # Training epoch
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
            phase=phase,
            generate=generate
        )
        global_step += len(train_loader)
        
        # Evaluation
        val_metrics = evaluate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            generate=generate
        )
        
        # Print evaluation results
        print(f"\nEpoch {epoch+1}/{epochs} Evaluation ({phase}):")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
            
        # Log metrics
        val_metrics['dataset'] = dataset_name
        val_metrics['epoch'] = epoch + 1
        val_metrics['train_loss'] = train_loss
        val_metrics['phase'] = phase
        logger.log_metrics(val_metrics, global_step)
        
        # Check for best model
        current_f1 = val_metrics.get('overall_f1', 0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            
            # Save best model
            checkpoint_path = f"checkpoints/{config.experiment_name}_{dataset_name}_{phase}_best.pt"
            torch.save(
                model.state_dict(),
                checkpoint_path
            )
            
            print(f"New best {phase} model saved with F1 = {best_f1:.4f}")
    
    # Save final model
    torch.save(
        model.state_dict(),
        f"checkpoints/{config.experiment_name}_{dataset_name}_{phase}_final.pt"
    )
    
    return best_f1

def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, config, 
               logger, global_step, epoch, dataset_name, phase, generate):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} ({phase})")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move batch to device - only tensor values
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch_on_device, generate=generate)
            
            # Calculate loss
            loss_dict = loss_fn(outputs, batch_on_device, generate=generate)
            loss = loss_dict['loss']
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if hasattr(config, 'max_grad_norm') and config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.7f}"
            })
            
            # Log metrics periodically
            if batch_idx % config.log_interval == 0:
                step = global_step + batch_idx
                logger.log_metrics({
                    'dataset': dataset_name,
                    'phase': phase,
                    'epoch': epoch + 1,
                    'batch': batch_idx,
                    'train_batch_loss': loss.item(),
                    'train_aspect_loss': loss_dict.get('aspect_loss', 0.0),
                    'train_opinion_loss': loss_dict.get('opinion_loss', 0.0),
                    'train_sentiment_loss': loss_dict.get('sentiment_loss', 0.0),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'generate': generate
                }, step)
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            traceback.print_exc()
            continue
    
    # Return average loss
    return total_loss / max(1, num_batches)

def evaluate(model, val_loader, loss_fn, device, generate=False):
    """Evaluate model on validation data"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Metrics tracking
    aspect_preds, opinion_preds, sentiment_preds = [], [], []
    aspect_labels, opinion_labels, sentiment_labels = [], [], []
    
    with torch.cuda.amp.autocast(enabled=config.use_fp16):
        for batch in tqdm(val_loader, desc="Evaluating"):
            try:
                # Move batch to device - only tensor values
                batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                
   
                # Forward pass
                outputs = model(**batch_on_device, generate=generate)
                
                # Calculate loss
                loss_dict = loss_fn(outputs, batch_on_device, generate=generate)
                
                # Update metrics
                total_loss += loss_dict['loss'].item()
                num_batches += 1
                
                # Get predictions and labels for metrics
                # Flatten aspect and opinion logits to get the predictions
                aspect_pred = outputs['aspect_logits'].argmax(dim=-1).cpu()
                opinion_pred = outputs['opinion_logits'].argmax(dim=-1).cpu()
                sentiment_pred = outputs['sentiment_logits'].argmax(dim=-1).cpu()
                
                # Get labels - handle multi-span case
                aspect_label = batch['aspect_labels'].cpu()
                if len(aspect_label.shape) == 3 and aspect_label.size(1) > 0:
                    aspect_label = aspect_label[:, 0]  # Use first span
                    
                opinion_label = batch['opinion_labels'].cpu()
                if len(opinion_label.shape) == 3 and opinion_label.size(1) > 0:
                    opinion_label = opinion_label[:, 0]  # Use first span
                    
                sentiment_label = batch['sentiment_labels'].cpu()
                if len(sentiment_label.shape) > 1 and sentiment_label.size(1) > 0:
                    sentiment_label = sentiment_label[:, 0]  # Use first span
                
                # Store predictions and labels
                aspect_preds.append(aspect_pred)
                opinion_preds.append(opinion_pred)
                sentiment_preds.append(sentiment_pred)
                
                aspect_labels.append(aspect_label)
                opinion_labels.append(opinion_label)
                sentiment_labels.append(sentiment_label)
                
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                traceback.print_exc()
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
    """Calculate metrics for evaluation"""
    try:
        metrics = {}
        
        # Calculate aspect metrics
        aspect_precision, aspect_recall, aspect_f1 = calculate_span_metrics(
            aspect_preds, aspect_labels, 'aspect'
        )
        
        # Calculate opinion metrics
        opinion_precision, opinion_recall, opinion_f1 = calculate_span_metrics(
            opinion_preds, opinion_labels, 'opinion'
        )
        
        # Calculate sentiment metrics
        sentiment_accuracy = calculate_sentiment_metrics(
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
        metrics['overall_f1'] = (aspect_f1 + opinion_f1 + sentiment_accuracy) / 3
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        traceback.print_exc()
        # Return default metrics
        return {
            'aspect_precision': 0.0,
            'aspect_recall': 0.0,
            'aspect_f1': 0.0,
            'opinion_precision': 0.0,
            'opinion_recall': 0.0,
            'opinion_f1': 0.0,
            'sentiment_accuracy': 0.0,
            'sentiment_f1': 0.0,
            'overall_f1': 0.0
        }

def calculate_span_metrics(preds, labels, prefix):
    """Calculate precision, recall, F1 for span detection"""
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

def calculate_sentiment_metrics(preds, labels):
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

def main():
    parser = argparse.ArgumentParser(description='Train Generative ABSA Model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset to train on')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=None, help='Hidden size')
    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--single_phase', action='store_true', help='Use single phase training (no separate extraction phase)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with fewer samples')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = LLMABSAConfig()
    
    # Override config with command line arguments
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.model is not None:
        config.model_name = args.model
    
    # Debug mode settings
    if args.debug:
        config.num_epochs = 1
        config.batch_size = min(4, config.batch_size)
        config.log_interval = 1
    
    # Create directories for outputs
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize W&B logger
    logger = WandbLogger(config, use_wandb=not args.no_wandb)
    
    # Initialize tokenizer
    print(f"Loading tokenizer from: {config.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True
        )
        print(f"Successfully loaded tokenizer from {config.model_name}")
    except Exception as e:
        print(f"Error loading tokenizer from {config.model_name}: {e}")
        print("Falling back to bert-base-uncased tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
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
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU for training (this will be slow)")
    
    # Train on each dataset or a specific one
    datasets = [args.dataset] if args.dataset else config.datasets
    results = {}
    
    for dataset_name in datasets:
        try:
            print(f"\nTraining on dataset: {dataset_name}")
            best_f1 = train_dataset(
                config=config, 
                tokenizer=tokenizer, 
                logger=logger, 
                dataset_name=dataset_name, 
                device=device,
                two_phase=not args.single_phase
            )
            results[dataset_name] = best_f1
        except Exception as e:
            print(f"Error training on {dataset_name}: {e}")
            traceback.print_exc()
            results[dataset_name] = "Failed"
    
    # Print final results
    print("\nFinal Results:")
    for dataset, f1 in results.items():
        print(f"{dataset}: Best F1 = {f1}")
    
    # End W&B run
    logger.finish()

if __name__ == '__main__':
    main()