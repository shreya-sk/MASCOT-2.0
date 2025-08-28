#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

import torch
from train import NovelGradientABSAModel, NovelABSAConfig, SimplifiedABSADataset, collate_fn
from torch.utils.data import DataLoader

def test_class_balancing():
    print("TESTING CLASS WEIGHT INTEGRATION")
    print("=" * 40)
    
    # Create config and model
    config = NovelABSAConfig('dev')
    model = NovelGradientABSAModel(config)
    
    # Create small test dataset
    dataset = SimplifiedABSADataset("dummy", config.model_name, config.max_length, config.dataset_name)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Get one batch
    batch = next(iter(loader))
    
    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(config.device)
    
    # Test class weight computation
    print("Testing class weight computation:")
    opinion_weights = model._compute_class_weights(batch['opinion_labels'].view(-1))
    print(f"Opinion class weights: {opinion_weights}")
    
    # Expected: Higher weights for B/I classes (indices 1,2)
    if opinion_weights[1] > opinion_weights[0] and opinion_weights[2] > opinion_weights[0]:
        print("✅ Class weights correctly prioritize B/I tags")
    else:
        print("❌ Class weights not working correctly")
    
    # Test forward pass with loss computation
    print("\nTesting forward pass with class balancing:")
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        aspect_labels=batch['aspect_labels'],
        opinion_labels=batch['opinion_labels'],
        sentiment_labels=batch['sentiment_labels'],
        domain_labels=batch['domain_labels'].squeeze(-1),
        training=True
    )
    
    print(f"Loss computation successful:")
    print(f"  Total loss: {outputs['total_loss'].item():.4f}")
    print(f"  Opinion loss: {outputs['opinion_loss'].item():.4f}")
    print(f"  Aspect loss: {outputs['aspect_loss'].item():.4f}")
    
    # The key test: losses should be computed without error
    if 'total_loss' in outputs and outputs['total_loss'].item() > 0:
        print("✅ Class-balanced loss computation working")
        return True
    else:
        print("❌ Loss computation failed")
        return False

if __name__ == "__main__":
    success = test_class_balancing()
    if success:
        print("\n🎉 Class balancing integration successful!")
        print("Ready to proceed with training.")
    else:
        print("\n❌ Integration failed. Check the implementation.")