#!/usr/bin/env python
"""
Clean, unified training script for ABSA
Replaces the complex train.py with a working version
"""

import torch
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.config import ABSAConfig, create_development_config, create_research_config
from data.dataset import verify_datasets
from training.trainer import train_absa_model

def main():
    """Main training function"""
    print("🚀 ABSA Training System")
    print("=" * 60)
    
    # Parse command line arguments (simplified)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["dev", "research", "minimal"], default="dev")
    parser.add_argument("--dataset", type=str, default="laptop14")
    args = parser.parse_args()
    
    # Create configuration
    if args.config == "dev":
        config = create_development_config()
    elif args.config == "research":
        config = create_research_config()
    else:
        config = ABSAConfig()
    
    # Override dataset if specified
    if args.dataset:
        config.datasets = [args.dataset]
    
    print(f"📋 Configuration: {args.config}")
    print(f"📊 Datasets: {config.datasets}")
    print(f"🔧 Features:")
    print(f"   - Implicit detection: {config.use_implicit_detection}")
    print(f"   - Few-shot learning: {config.use_few_shot_learning}")
    print(f"   - Generative framework: {config.use_generative_framework}")
    print(f"   - Contrastive learning: {config.use_contrastive_learning}")
    
    # Verify datasets
    if not verify_datasets(config):
        print("❌ Dataset verification failed!")
        return
    
    # Train model
    try:
        results, model, trainer = train_absa_model(config)
        
        print("\n🎉 Training completed successfully!")
        print(f"   Best F1 Score: {results['best_f1']:.4f}")
        print(f"   Output directory: {results['output_dir']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
