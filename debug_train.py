#!/usr/bin/env python
# debug_train.py - Lightweight version of train.py for quick debugging with your current model
import os
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from src.data.dataset import custom_collate_fn

# Import your custom modules
from src.data.dataset import ABSADataset
from src.data.preprocessor import LLMABSAPreprocessor
from src.models.absa import LLMABSA 
from src.training.losses import ABSALoss
from src.utils.config import LLMABSAConfig
from src.utils.logger import WandbLogger

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Add the missing _extract_hidden_states method to LLMABSA
def patch_absa_model():
    """Patch the LLMABSA model to add any missing methods"""
    if not hasattr(LLMABSA, '_extract_hidden_states'):
        print("Adding missing _extract_hidden_states method to LLMABSA")
        def _extract_hidden_states(self, embeddings_output):
            """Extract hidden states from embeddings output"""
            if isinstance(embeddings_output, dict):
                if 'hidden_states' in embeddings_output:
                    return embeddings_output['hidden_states']
                elif 'last_hidden_state' in embeddings_output:
                    return embeddings_output['last_hidden_state']
                else:
                    return list(embeddings_output.values())[0]
            else:
                return embeddings_output
                
        # Add the method to the class
        setattr(LLMABSA, '_extract_hidden_states', _extract_hidden_states)

def debug_train_dataset(config, tokenizer, logger, dataset_name, device, debug_samples=20):
    """Train model on a specific dataset in debug mode"""
    
    print(f"\nDebug training on dataset: {dataset_name} with {debug_samples} samples")
    
    # Patch the model first
    patch_absa_model()
    
    # Create preprocessor
    preprocessor = LLMABSAPreprocessor(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        use_syntax=config.use_syntax
    )
    
    # Create datasets with domain id if using domain adaptation
    domain_id = config.domain_mapping.get(dataset_name, 0) if config.domain_adaptation else None
    
    # Create datasets
    train_dataset = ABSADataset(
        data_dir=config.dataset_paths[dataset_name],
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='train',
        dataset_name=dataset_name,
        max_length=config.max_seq_length,
        domain_id=domain_id
    )
    
    # Limit dataset size for debugging
    print(f"Original training set size: {len(train_dataset)}")
    train_dataset.data = train_dataset.data[:debug_samples]
    print(f"Debug training set size: {len(train_dataset)}")
    
    val_dataset = ABSADataset(
        data_dir=config.dataset_paths[dataset_name],
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='dev',
        dataset_name=dataset_name,
        max_length=config.max_seq_length,
        domain_id=domain_id
    )
    
    # Limit validation size too
    val_dataset.data = val_dataset.data[:min(10, len(val_dataset.data))]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(4, len(train_dataset)),  # Small batch size
        shuffle=True,
        num_workers=0,  # No multiprocessing for debug
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(4, len(val_dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model - using your actual model architecture
    print(f"Initializing LLMABSA model for dataset: {dataset_name}")
    model = LLMABSA(config).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )
    
    # Initialize loss function
    loss_fn = ABSALoss(config)
    
    # Training loop - just do one epoch for debugging
    print("Starting debug training loop")
    model.train()
    
    # Simplified training loop
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nProcessing batch {batch_idx+1}/{len(train_loader)}")
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Forward pass
        print("Running forward pass...")
        try:
            outputs = model(**batch)
            
            # Print output shapes
            print("Model outputs:")
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape {v.shape}, values min={v.min().item():.4f}, max={v.max().item():.4f}")
                    
            # Calculate loss
            print("Calculating loss...")
            try:
                loss_dict = loss_fn(outputs, batch)
                loss = loss_dict['loss']
                print(f"Loss components: {', '.join([f'{k}: {v:.6f}' for k, v in loss_dict.items() if not isinstance(v, torch.Tensor)])}")
            except Exception as e:
                print(f"Error in loss calculation: {e}")
                # Create a dummy loss for testing gradient flow
                loss = outputs['aspect_logits'].mean()
                print(f"Using dummy loss: {loss.item():.6f}")
            
            # Backward pass
            print("Running backward pass...")
            loss.backward()
            
            # Check gradient flow
            print("Gradient information (sample of parameters):")
            has_grads = False
            for name, param in list(model.named_parameters())[:10]:  # Just show first 10 params
                if param.requires_grad:
                    if param.grad is not None:
                        has_grads = True
                        grad_norm = param.grad.norm().item()
                        print(f"  {name}: grad norm {grad_norm:.6f}")
                    else:
                        print(f"  {name}: No gradient")
            
            if not has_grads:
                print("WARNING: No gradients are flowing! Check your model architecture.")
                
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
        except Exception as e:
            print(f"Error during forward/backward pass: {e}")
            import traceback
            traceback.print_exc()
        
        # Only process a few batches
        if batch_idx >= 2:
            break
    
    # Do one validation batch for testing
    print("\nRunning validation batch...")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch)
                print("Validation outputs:")
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape {v.shape}")
            except Exception as e:
                print(f"Error during validation: {e}")
            break  # Just one batch
    
    print("\nDebug training completed!")
    return 0.0  # Dummy F1 score

def main():
    parser = argparse.ArgumentParser(description='Debug Train ABSA Model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default="rest15", help='Specific dataset to train on')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--samples', type=int, default=20, help='Number of samples for debugging')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load config
    config = LLMABSAConfig()
    
    # Override config with command line arguments
    if args.model:
        config.model_name = args.model
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize logger without wandb for faster debugging
    logger = WandbLogger(config, use_wandb=False)
    
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
        
    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU for debugging")
    
    # Run debug training
    dataset_name = args.dataset if args.dataset else "rest15"
    try:
        debug_train_dataset(config, tokenizer, logger, dataset_name, device, debug_samples=args.samples)
    except Exception as e:
        print(f"Error during debug training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDebug script completed.")

if __name__ == '__main__':
    main()




#     src/models/absa.py - Rename LLMABSA to LLMABSA
# src/data/preprocessor.py - Rename LLMABSAPreprocessor to LLMABSAPreprocessor
# src/utils/config.py - Rename LLMABSAConfig to LLMABSAConfig