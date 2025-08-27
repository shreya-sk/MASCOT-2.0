#!/usr/bin/env python3
"""
ALIGNMENT FIX TEST - Verify the tensor alignment bug is resolved
Save this as alignment_test.py and run it
"""

import sys
sys.path.append('.')

from train import SimplifiedABSADataset, collate_fn
from torch.utils.data import DataLoader
import torch

def test_alignment_fix():
    """Test the specific alignment bug causing 4 B-tags to become 1 B-tag"""
    
    print("ALIGNMENT FIX TEST - Verifying tensor alignment")
    print("=" * 50)
    
    # Create dataset with the problematic sample
    dataset = SimplifiedABSADataset(
        "Datasets/aste/laptop14/train.txt", 
        "bert-base-uncased", 
        128, 
        "laptop14"
    )
    
    # Test the specific problematic item (index 1 has 4 opinions)
    print("Testing sample with 4 opinions...")
    
    # Get the raw item directly
    item = dataset.data[1]
    print(f"Raw opinions: {item.get('opinions', [])}")
    
    # Test __getitem__ method
    processed_item = dataset.__getitem__(1)
    
    # Check the final tensor
    opinion_labels = processed_item['opinion_labels']
    print(f"Opinion labels tensor shape: {opinion_labels.shape}")
    print(f"Opinion labels (first 15): {opinion_labels[:15].tolist()}")
    
    # Count B-tags (label == 1)
    b_count = sum(1 for x in opinion_labels if x == 1)
    print(f"Final B-tag count in tensor: {b_count}")
    
    # Expected: Should be 4 B-tags, not 1
    if b_count == 4:
        print("✅ ALIGNMENT FIX SUCCESS: 4 B-tags preserved!")
    elif b_count == 1:
        print("❌ ALIGNMENT BUG PERSISTS: Only 1 B-tag found")
    else:
        print(f"⚠️ UNEXPECTED: Found {b_count} B-tags")
    
    print("\nTesting DataLoader batching...")
    
    # Test with DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    # Check batch processing
    for i in range(min(2, len(batch['opinion_labels']))):
        opinion_labels = batch['opinion_labels'][i]
        valid_mask = opinion_labels != -100
        valid_opinions = opinion_labels[valid_mask]
        
        b_count = sum(valid_opinions == 1)
        i_count = sum(valid_opinions == 2)
        
        print(f"Batch item {i}: B={b_count}, I={i_count}")
        
        # Check if this is the 4-opinion sample
        if i == 1:  # Second item should be the 4-opinion sample
            if b_count == 4:
                print(f"✅ Batch processing SUCCESS: Item {i} has 4 B-tags")
            else:
                print(f"❌ Batch processing FAILED: Item {i} has {b_count} B-tags instead of 4")
    
    print("\n" + "=" * 50)
    print("DIAGNOSIS:")
    
    if b_count == 4:
        print("✅ Alignment fix successful - ready for training")
        print("Expected training results:")
        print("   - Opinion F1: 0.25-0.45 (major improvement)")
        print("   - Triplet F1: 0.20-0.35 (working triplets)")
        return True
    else:
        print("❌ Alignment bug not fixed")
        print("Root cause: Label generation creates 4 B-tags")
        print("            Tensor processing loses 3 B-tags")
        print("Need to debug: __getitem__ method alignment logic")
        return False

if __name__ == "__main__":
    test_alignment_fix()