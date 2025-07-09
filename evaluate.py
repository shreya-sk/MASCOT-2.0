#!/usr/bin/env python
"""
Clean evaluation script for ABSA
"""

import torch
import sys
from pathlib import Path

# Add src to path  
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.clean_config import ABSAConfig
from data.clean_dataset import load_datasets, create_dataloaders
from models.unified_absa_model import UnifiedABSAModel

def evaluate_model(model_path, dataset_name="laptop14"):
    """Evaluate trained model"""
    print(f"📊 Evaluating model: {model_path}")
    
    # Load config and model
    config = ABSAConfig()
    config.datasets = [dataset_name]
    
    model = UnifiedABSAModel.load(model_path, config)
    model.eval()
    
    # Load test data
    datasets = load_datasets(config)
    dataloaders = create_dataloaders(datasets, config)
    
    test_dataloader = dataloaders[dataset_name]['test']
    
    # Evaluate
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            # Get predictions
            predictions = model.predict_triplets(input_ids, attention_mask)
            targets = batch['triplets']
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Compute metrics (simplified)
    total_pred = sum(len(pred) for pred in all_predictions)
    total_target = sum(len(target) for target in all_targets)
    
    print(f"Results on {dataset_name}:")
    print(f"  Predictions: {total_pred}")
    print(f"  Targets: {total_target}")
    
    return all_predictions, all_targets

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--dataset", default="laptop14", help="Dataset to evaluate")
    args = parser.parse_args()
    
    evaluate_model(args.model, args.dataset)
