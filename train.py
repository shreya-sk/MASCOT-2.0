# train.py (in root directory)
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from src.data import ABSADataset
from src.models.model import LlamaABSA  # Explicit import from model.py
from src.models.embedding import LlamaEmbedding  # Explicit import from embedding.py
from src.training.losses import ABSALoss
from src.training.trainer import ABSATrainer
from src.utils.config import LlamaABSAConfig  # Use LlamaABSAConfig instead of ABSAConfig 
from src.utils.logger import WandbLogger
from src.utils.visualisation import AttentionVisualizer  # Correct spelling of visualization

def main():
    # Load config
    config = LlamaABSAConfig()
    # Initialize W&B logger
    logger = WandbLogger(config)
    # Initialize tokenizer and visualizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    visualizer = AttentionVisualizer(tokenizer)    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets and dataloaders
    train_dataset = ABSADataset(config.train_path, split='train')
    val_dataset = ABSADataset(config.val_path, split='val')
    
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
    
    # Create model
    model = LlamaABSA(config).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Create loss function
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
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Log metrics to W&B
        logger.log_metrics({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/aspect_f1': train_metrics['aspect_f1'],
            'train/opinion_f1': train_metrics['opinion_f1'],
            'train/sentiment_f1': train_metrics['sentiment_f1'],
            'val/loss': val_metrics['loss'],
            'val/aspect_f1': val_metrics['aspect_f1'],
            'val/opinion_f1': val_metrics['opinion_f1'],
            'val/sentiment_f1': val_metrics['sentiment_f1'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }, epoch)

        # Optionally visualize attention patterns
        if epoch % config.viz_interval == 0:
            attention_weights = trainer.get_attention_weights(val_loader)
            visualizer.plot_attention(
                attention_weights,
                tokens=tokenizer.convert_ids_to_tokens(val_batch['input_ids'][0]),
                save_path=f'visualizations/attention_epoch_{epoch}.png'
            )
        

        # Log metrics
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > trainer.best_f1:
            trainer.best_f1 = val_metrics['f1']
            model_path = f'checkpoints/best_model.pt'
            torch.save(model.state_dict(), model_path)
            logger.log_model(model, val_metrics)
        
    # End W&B run
    logger.finish()
if __name__ == '__main__':
    main()