# train.py
import torch
from transformers import LlamaTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from src.data import ABSADataset
from src.models.model import LlamaABSA 
from src.models.embedding import LlamaEmbedding 
from src.training.losses import ABSALoss
from src.training.trainer import ABSATrainer
from src.utils.config import LlamaABSAConfig
from src.utils.logger import WandbLogger
from src.utils.visualisation import AttentionVisualizer
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    
    # Create dataloaders
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
    
    # Initialize model, optimizer, scheduler and loss
    model = LlamaABSA(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    loss_fn = ABSALoss(config)
    
    # Create trainer
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
            'val_f1': val_metrics['f1']
        }, epoch)
        
        # Visualize attention if needed
        if epoch % config.viz_interval == 0:
            attention_weights = trainer.get_attention_weights(val_loader)
            visualizer.plot_attention(
                attention_weights,
                tokens=tokenizer.convert_ids_to_tokens(val_loader.dataset[0]['input_ids']),
                save_path=f'visualizations/{dataset_name}_attention_epoch_{epoch}.png'
            )
        
        # Print metrics
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model for this dataset
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            model_path = f'checkpoints/best_model_{dataset_name}.pt'
            torch.save(model.state_dict(), model_path)
            logger.log_model(model, {
                'dataset': dataset_name,
                **val_metrics
            })
    
    return best_f1

def main():
    # Load config
    config = LlamaABSAConfig()
    
    # Initialize W&B logger
    logger = WandbLogger(config)
    
    # Initialize tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(
        config.model_name,
        padding_side="right",
        truncation_side="right",
        model_max_length=config.max_span_length,
        trust_remote_code=True
    )
    
    special_tokens = {
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train on each dataset
    results = {}
    for dataset_name in ["laptop14", "rest14", "rest15", "rest16"]:
        best_f1 = train_dataset(config, tokenizer, logger, dataset_name, device)
        results[dataset_name] = best_f1
    
    # Print final results
    print("\nFinal Results:")
    for dataset, f1 in results.items():
        print(f"{dataset}: Best F1 = {f1:.4f}")
    
    # End W&B run
    logger.finish()

if __name__ == '__main__':
    main()