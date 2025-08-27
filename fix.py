#!/usr/bin/env python3
"""
QUICK TEST - Run this to verify your F1 fix works before full training
Save as quick_test.py and run: python quick_test.py
"""

import torch
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / 'src'
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

def quick_f1_test():
    """Test F1 computation in 30 seconds"""
    
    print("QUICK F1 TEST - Testing your fixes")
    print("="*40)
    
    try:
        # Import your classes
        from train import NovelABSAConfig, NovelGradientABSAModel, NovelABSATrainer, SimplifiedABSADataset
        from torch.utils.data import DataLoader
        
        print("✓ Imports successful")
        
        # Create minimal config
        config = NovelABSAConfig('dev')
        config.num_epochs = 1  # Just one epoch for testing
        config.batch_size = 2
        
        print("✓ Config created")
        
        # Create model
        model = NovelGradientABSAModel(config)
        print("✓ Model created")
        
        # Create tiny dataset
        dataset = SimplifiedABSADataset("dummy", config.model_name, config.max_length, config.dataset_name)
        # Limit to just 10 samples for quick test
        dataset.data = dataset.data[:10]
        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        print("✓ Dataset created (10 samples)")
        
        # Create trainer
        trainer = NovelABSATrainer(model, config, dataloader, dataloader, config.device)
        print("✓ Trainer created")
        
        # Test evaluation function
        print("\nTesting evaluation function...")
        eval_metrics = trainer.evaluate()
        
        print("\nQUICK TEST RESULTS:")
        print(f"   Aspect F1: {eval_metrics.get('aspect_f1', 'MISSING')}")
        print(f"   Opinion F1: {eval_metrics.get('opinion_f1', 'MISSING')}")
        print(f"   Sentiment F1: {eval_metrics.get('sentiment_f1', 'MISSING')}")
        print(f"   Triplet F1: {eval_metrics.get('triplet_f1', 'MISSING')}")
        
        # Check if fix worked
        aspect_f1 = eval_metrics.get('aspect_f1', 0)
        
        if isinstance(aspect_f1, (int, float)) and aspect_f1 >= 0:
            print("\n✓ SUCCESS: F1 computation working!")
            if aspect_f1 > 0:
                print("✓ Getting non-zero F1 scores - fix complete!")
            else:
                print("✓ F1 function works, but scores still 0 - may need training")
        else:
            print("\n✗ FAILED: F1 computation still broken")
            print(f"   Got: {aspect_f1} (type: {type(aspect_f1)})")
        
        return eval_metrics
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Check your imports and make sure _compute_metrics is added to NovelABSATrainer")
        return None
        
    except AttributeError as e:
        print(f"✗ Attribute error: {e}")
        if "_compute_metrics" in str(e):
            print("SOLUTION: You need to add the _compute_metrics function to your NovelABSATrainer class")
        return None
        
    except Exception as e:
        print(f"✗ Other error: {e}")
        return None

def test_one_training_step():
    """Test just one training step to see if everything works"""
    
    print("\nTEST ONE TRAINING STEP:")
    print("-" * 25)
    
    try:
        from train import NovelABSAConfig, NovelGradientABSAModel, NovelABSATrainer, SimplifiedABSADataset
        from torch.utils.data import DataLoader
        
        # Minimal setup
        config = NovelABSAConfig('dev')
        config.num_epochs = 1
        config.batch_size = 2
        
        model = NovelGradientABSAModel(config)
        dataset = SimplifiedABSADataset("dummy", config.model_name, config.max_length, config.dataset_name)
        dataset.data = dataset.data[:4]  # Just 4 samples
        
        train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        trainer = NovelABSATrainer(model, config, train_loader, val_loader, config.device)
        
        print("Testing one training epoch...")
        
        # Run one training epoch
        train_losses = trainer.train_epoch(0)
        print(f"✓ Training step successful - Loss: {train_losses['total_loss']:.4f}")
        
        # Test evaluation
        eval_metrics = trainer.evaluate()
        print(f"✓ Evaluation step successful")
        
        # Show results
        print(f"\nONE-STEP RESULTS:")
        print(f"   Train Loss: {train_losses['total_loss']:.4f}")
        print(f"   Aspect F1: {eval_metrics.get('aspect_f1', 'ERROR')}")
        print(f"   Opinion F1: {eval_metrics.get('opinion_f1', 'ERROR')}")
        print(f"   Triplet F1: {eval_metrics.get('triplet_f1', 'ERROR')}")
        
        # Check if everything looks good for full training
        if (eval_metrics.get('aspect_f1', -1) >= 0 and 
            train_losses['total_loss'] > 0 and 
            train_losses['total_loss'] < 100):
            print("\n✓ ALL SYSTEMS READY - You can run full training!")
            return True
        else:
            print("\n✗ Issues detected - check above output")
            return False
            
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False

if __name__ == "__main__":
    # Run both tests
    print("RUNNING QUICK DIAGNOSTIC TESTS")
    print("="*50)
    
    # Test 1: Evaluation function
    eval_result = quick_f1_test()
    
    # Test 2: One training step
    if eval_result is not None:
        training_ready = test_one_training_step()
        
        if training_ready:
            print("\n🎯 FINAL RECOMMENDATION:")
            print("✓ All fixes verified - run full training with:")
            print("   python train.py --config dev --num_epochs 3")
            print("   (Use dev config for faster testing)")
        else:
            print("\n🔧 ISSUES REMAIN:")
            print("   Fix the errors shown above before full training")
    
    print("\nTest completed.")