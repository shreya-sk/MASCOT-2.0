#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import torch
from train import NovelGradientABSAModel, NovelABSAConfig

def test_integration():
    print("Testing class weight integration...")
    
    config = NovelABSAConfig('dev')
    model = NovelGradientABSAModel(config)
    
    # Test class weight method exists
    if hasattr(model, '_compute_class_weights'):
        print("✅ _compute_class_weights method added")
    else:
        print("❌ _compute_class_weights method missing")
        return False
    
    # Test with sample data
    labels = torch.tensor([0, 0, 0, 0, 1, 0, 0, 2, 0, 0])  # Mostly O-tags
    weights = model._compute_class_weights(labels)
    
    print(f"Class weights: {weights}")
    if weights[1] > weights[0] and weights[2] > weights[0]:
        print("✅ B/I tags have higher weights than O tags")
        return True
    else:
        print("❌ Class weights not working correctly")
        return False

if __name__ == "__main__":
    success = test_integration()
    print("Ready for training!" if success else "Fix needed before training")