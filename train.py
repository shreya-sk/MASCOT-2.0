# train.py
import os
import argparse
import torch # type: ignore # type: ignore
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader # type: ignore
from transformers import get_linear_schedule_with_warmup

# Import your custom modules
from src.data.dataset import ABSADataset
from src.data.preprocessor import StellaABSAPreprocessor
from src.models.absa import StellaABSA # You need this file
from src.training.losses import ABSALoss
from src.training.trainer import ABSATrainer
from src.training.metrics import ABSAMetrics
from src.utils.config import StellaABSAConfig
from src.utils.logger import WandbLogger
from src.utils.visualisation import AttentionVisualizer
from src.inference.predictor import StellaABSAPredictor

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
    
    # Initialize model
    print(f"Initializing StellaABSAConfig model for dataset: {dataset_name}")
    model = StellaABSA(config).to(device)
    
    # Set up gradient accumulation steps
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    
    # Initialize optimizer
    # Separate learning rates for base model and new components
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
    
    # Calculate steps with gradient accumulation
    num_training_steps = len(train_loader) // gradient_accumulation_steps * config.num_epochs
    num_warmup_steps = int(num_training_steps * getattr(config, 'warmup_ratio', 0.1))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize loss function
    loss_fn = ABSALoss(config)
    
    # Initialize metrics
    metrics = ABSAMetrics()
    
    # Create trainer with gradient accumulation
    trainer = ABSATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=config
    )
    
    # Initialize visualizer
    visualizer = AttentionVisualizer(tokenizer)
    
    # Set up mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler() if config.use_fp16 else None
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_f1 = 0.0
    global_step = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} (Train)")
        
        for batch_idx, batch in enumerate(train_iterator):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass with mixed precision if enabled
            if config.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss_dict = loss_fn(outputs, batch)
                    loss = loss_dict['loss'] / gradient_accumulation_steps
                    
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if hasattr(config, 'max_grad_norm') and config.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            else:
                # Standard forward and backward pass
                outputs = model(**batch)
                loss_dict = loss_fn(outputs, batch)
                loss = loss_dict['loss'] / gradient_accumulation_steps
                
                loss.backward()
                
                # Only step optimizer after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if hasattr(config, 'max_grad_norm') and config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            
            # Update metrics
            train_loss += loss.item() * gradient_accumulation_steps
            train_steps += 1
            
            # Update progress bar
            train_iterator.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'lr': f"{scheduler.get_last_lr()[0]:.7f}"
            })
            
            # Log training metrics
            if global_step > 0 and hasattr(config, 'log_interval') and global_step % config.log_interval == 0:
                logger.log_metrics({
                    'dataset': dataset_name,
                    'epoch': epoch + 1,
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                }, global_step)
            
            # Evaluate model periodically
            if global_step > 0 and hasattr(config, 'eval_interval') and global_step % config.eval_interval == 0:
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
            if global_step > 0 and hasattr(config, 'save_interval') and global_step % config.save_interval == 0:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(
                    model.state_dict(), 
                    f"checkpoints/{config.experiment_name}_{dataset_name}_last.pt"
                )
        
        # End of epoch evaluation
        val_metrics = evaluate(model, val_loader, loss_fn, device, metrics)
        
        # Log end of epoch metrics
        val_metrics['dataset'] = dataset_name
        val_metrics['epoch'] = epoch + 1
        val_metrics['train_loss'] = train_loss / train_steps
        logger.log_metrics(val_metrics, global_step)
        
        # Visualize attention if needed and available
        if epoch % config.viz_interval == 0:
            try:
                attention_weights = trainer.get_attention_weights(val_loader)
                if attention_weights is not None:
                    # Create visualizations directory if it doesn't exist
                    os.makedirs('visualizations', exist_ok=True)
                    
                    visualizer.plot_attention(
                        attention_weights,
                        tokens=tokenizer.convert_ids_to_tokens(val_loader.dataset[0]['input_ids']),
                        save_path=f'visualizations/{dataset_name}_attention_epoch_{epoch}.png'
                    )
            except Exception as e:
                print(f"Warning: Could not visualize attention: {e}")
        
        # Print end of epoch summary
        print(f"Epoch {epoch+1}/{config.num_epochs}:")
        print(f"  Train Loss: {train_loss/train_steps:.4f}")
        print(f"  Val Loss: {val_metrics.get('loss', 0):.4f}")
        print(f"  Val F1: {val_metrics.get('overall_f1', 0):.4f}")
        print(f"  Best F1: {best_f1:.4f}")
        
        # Save model at the end of each epoch
        torch.save(
            model.state_dict(), 
            f"checkpoints/{config.experiment_name}_{dataset_name}_last.pt"
        )
    
    return best_f1

def evaluate(model, dataloader, loss_fn, device, metrics):
    """Evaluate model on a dataloader"""
    model.eval()
    metrics.reset()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            loss_dict = loss_fn(outputs, batch)
            
            # Update metrics
            metrics.update(outputs, batch)
            total_loss += loss_dict['loss'].item()
            num_batches += 1
    
    # Compute metrics
    eval_metrics = metrics.compute()
    eval_metrics['loss'] = total_loss / num_batches
    
    return eval_metrics

def main():
    parser = argparse.ArgumentParser(description='Train LLAMA ABSA model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset to train on')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=None, help='Hidden size')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = StellaABSAConfig()
    
    # Override config with command line arguments
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.dropout is not None:
        config.dropout = args.dropout
    
    # Initialize W&B logger
    logger = WandbLogger(config)
    
    # Initialize tokenizer
    print(f"Loading tokenizer from: {config.model_name}")
    try:
        # First try loading from local model
        if config.use_local:
            model_path = config.model_name
            print(f"Loading tokenizer from local path: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                local_files_only=True
            )
        else:
            # Try loading from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                use_fast=True
            )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to default tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            use_fast=True
        )
    
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
    
    # Add task-specific tokens
    task_tokens = {
        "additional_special_tokens": ["[AT]", "[OT]", "[AC]", "[SP]"]
    }
    tokenizer.add_special_tokens(task_tokens)
    
    # Set device with mixed precision
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Print available GPU memory
        if hasattr(torch.cuda, 'mem_get_info'):
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            print(f"GPU Memory: {free_memory/1024**3:.2f}GB free / {total_memory/1024**3:.2f}GB total")
    else:
        device = torch.device('cpu')
        print("Using CPU - training will be very slow")
    
    # Train on each dataset or a specific one
    datasets = [args.dataset] if args.dataset else config.datasets
    results = {}
    
    for dataset_name in datasets:
        try:
            best_f1 = train_dataset(config, tokenizer, logger, dataset_name, device)
            results[dataset_name] = best_f1
        except Exception as e:
            print(f"Error training on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = "Failed"
    
    # Print final results
    print("\nFinal Results:")
    for dataset, f1 in results.items():
        print(f"{dataset}: Best F1 = {f1}")
    
    # Test on specific examples
    test_examples = [
        "The food was delicious but the service was terrible.",
        "Great atmosphere and friendly staff!",
        "The battery life is poor but the screen quality is excellent."
    ]
    
    # Initialize predictor with the best model
    if datasets and any(isinstance(f1, (int, float)) for f1 in results.values()):
        # Find best dataset with numerical F1 value
        valid_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        if valid_results:
            best_dataset = max(valid_results.items(), key=lambda x: x[1])[0]
            try:
                predictor = StellaABSAPredictor(
                    model_path=f"checkpoints/{config.experiment_name}_{best_dataset}_best.pt",
                    config=config,
                    device=device,
                    tokenizer_path=config.model_name
                )
                
                print("\nTest Examples:")
                for example in test_examples:
                    predictions = predictor.predict(example)
                    print(f"\nInput: {example}")
                    print("Predictions:")
                    for triplet in predictions['triplets']:
                        print(f"  Aspect: {triplet['aspect']}, Opinion: {triplet['opinion']}, "
                              f"Sentiment: {triplet['sentiment']} (Confidence: {triplet['confidence']:.2f})")
            except Exception as e:
                print(f"Error running predictions: {e}")
    
    # End W&B run
    logger.finish()

if __name__ == '__main__':
    main()