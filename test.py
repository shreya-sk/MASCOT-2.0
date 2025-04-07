#!/usr/bin/env python
# test_stella.py
import os
import argparse
import torch # type: ignore # type: ignore
from tqdm import tqdm
from torch.utils.data import DataLoader # type: ignore # type: ignore
import wandb

from src.data.dataset import ABSADataset
from src.data.preprocessor import StellaABSAPreprocessor
from src.models.absa import StellaABSA
from src.utils.config import StellaABSAConfig
from src.training.metrics import ABSAMetrics
from transformers import AutoTokenizer
from src.inference.predictor import StellaABSAPredictor

def test_model(model, data_loader, device):
    """Test model on a data loader and return metrics"""
    model.eval()
    metrics = ABSAMetrics()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            # Forward pass
            outputs = model(**batch)
            
            # Update metrics
            metrics.update(outputs, batch)
    
    # Compute metrics
    return metrics.compute()

def main():
    parser = argparse.ArgumentParser(description='Test Stella ABSA model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset to test on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    parser.add_argument('--log_wandb', action='store_true', help='Log results to W&B')
    args = parser.parse_args()
    
    # Load config
    config = StellaABSAConfig()
    
    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.log_wandb:
        wandb.init(project="stella-absa-test", name=f"test_{os.path.basename(args.model)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True
    )
    
    # Initialize preprocessor
    preprocessor = StellaABSAPreprocessor(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        use_syntax=config.use_syntax
    )
    
    # Load model
    model = StellaABSA(config)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    # Test on each dataset or a specific one
    datasets = [args.dataset] if args.dataset else config.datasets
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\nTesting on {dataset_name} dataset")
        
        # Get domain ID if using domain adaptation
        domain_id = config.domain_mapping.get(dataset_name, 0) if config.domain_adaptation else None
        
        # Create test dataset
        test_dataset = ABSADataset(
            data_dir=config.dataset_paths[dataset_name],
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            split='test',
            dataset_name=dataset_name,
            max_length=config.max_seq_length,
            domain_id=domain_id
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        # Test model
        test_metrics = test_model(model, test_loader, device)
        
        # Store results
        all_results[dataset_name] = test_metrics
        
        # Print metrics
        print(f"Results for {dataset_name}:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Log to wandb if requested
        if args.log_wandb:
            wandb.log({f"{dataset_name}_{k}": v for k, v in test_metrics.items()})
    
    # Print overall results
    if len(datasets) > 1:
        print("\nAverage results across all datasets:")
        avg_metrics = {}
        for metric in list(all_results.values())[0].keys():
            avg_metrics[metric] = sum(results[metric] for results in all_results.values()) / len(all_results)
            print(f"  {metric}: {avg_metrics[metric]:.4f}")
        
        if args.log_wandb:
            wandb.log({f"avg_{k}": v for k, v in avg_metrics.items()})
    
    if args.log_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()