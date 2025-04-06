# train.py
import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from src.data import ABSADataset
from src.models.model import LlamaABSA
from src.training.losses import ABSALoss
from src.training.trainer import ABSATrainer
from src.utils.config import LlamaABSAConfig
from src.utils.logger import WandbLogger
from src.utils.visualisation import AttentionVisualizer

def train_dataset(config, tokenizer, logger, dataset_name, device):
    """Train model on a specific dataset"""
    
    print(f"\nTraining on dataset: {dataset_name}")
    
    # Create datasets
    train_dataset = ABSADataset(
        data_dir="",  # Path handled in dataset class
        tokenizer=tokenizer,
        split='train',
        dataset_name=dataset_name,
        max_length=config.max_span_length
    )
    
    val_dataset = ABSADataset(
        data_dir="",
        tokenizer=tokenizer,
        split='dev', 
        dataset_name=dataset_name,
        max_length=config.max_span_length
    )
    
    # Create dataloaders with smaller batch size for Llama
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Initialize model
    print(f"Initializing LlamaABSA model for dataset: {dataset_name}")
    model = LlamaABSA(config).to(device)
    
    # Set up gradient accumulation steps
    gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Calculate steps with gradient accumulation
    num_training_steps = len(train_loader) // gradient_accumulation_steps * config.num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    loss_fn = ABSALoss(config)
    
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
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_f1 = 0

    
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Log metrics with dataset name
        logger.log_metrics({
            'dataset': dataset_name,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics.get('f1', 0)
        }, epoch)
        
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
        
        # Print metrics
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val F1: {val_metrics.get('f1', 0):.4f}")
        
        # Save checkpoint every epoch
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            },
            f'checkpoints/checkpoint_{dataset_name}_epoch_{epoch}.pt'
        )
        
        # Save best model for this dataset
        current_f1 = val_metrics.get('f1', 0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), f'checkpoints/best_model_{dataset_name}.pt')
            logger.log_model(model, {
                'dataset': dataset_name,
                **val_metrics
            })
    
    return best_f1

def main():
    # Load config
   
    config = LlamaABSAConfig(use_online_model=True)
    
    # Initialize W&B logger
    logger = WandbLogger(config.__dict__)
    
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
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Print available GPU memory
        free_memory, total_memory = torch.cuda.mem_get_info(0)
        print(f"GPU Memory: {free_memory/1024**3:.2f}GB free / {total_memory/1024**3:.2f}GB total")
    else:
        device = torch.device('cpu')
        print("Using CPU - training will be very slow")
    
    # Train on each dataset
    results = {}
    for dataset_name in config.datasets:
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
    
    # End W&B run
    logger.finish()

if __name__ == '__main__':
    main()