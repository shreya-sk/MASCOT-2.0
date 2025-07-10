#!/usr/bin/env python3
"""
Fix Configuration Import Issues for GRADIENT
This script adds missing configuration functions to your config.py
"""

import os
from pathlib import Path

def fix_config_file():
    """Add missing configuration functions to config.py"""
    
    config_file = Path("src/utils/config.py")
    
    if not config_file.exists():
        print("❌ Config file not found at src/utils/config.py")
        return False
    
    # Read existing config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check if functions already exist
    if 'def create_development_config' in content:
        print("✅ Configuration functions already exist")
        return True
    
    # Add missing functions
    additional_config = '''

def create_development_config():
    """Create development configuration for GRADIENT"""
    config = ABSAConfig()
    
    # Development settings
    config.batch_size = 4
    config.num_epochs = 5
    config.learning_rate = 3e-5
    config.max_seq_length = 128
    
    # Enable key GRADIENT features
    config.use_implicit_detection = True
    config.use_few_shot_learning = True
    config.use_contrastive_learning = True
    config.use_domain_adversarial = True
    config.use_generative_framework = False  # Disable for faster training
    
    # Multi-domain datasets for adversarial training
    config.datasets = ['laptop14', 'rest14']
    config.experiment_name = "gradient_dev"
    
    print("✅ Development config created with GRADIENT features")
    return config


def create_research_config():
    """Create research configuration with all GRADIENT features"""
    config = ABSAConfig()
    
    # Research settings
    config.batch_size = 8
    config.num_epochs = 25
    config.learning_rate = 1e-5
    config.max_seq_length = 256
    
    # Enable ALL GRADIENT features
    config.use_implicit_detection = True
    config.use_few_shot_learning = True
    config.use_contrastive_learning = True
    config.use_domain_adversarial = True
    config.use_generative_framework = True
    
    # Full multi-domain training
    config.datasets = ['laptop14', 'rest14', 'rest15', 'rest16']
    config.experiment_name = "gradient_research"
    
    print("✅ Research config created with all GRADIENT features")
    return config


def create_domain_adversarial_config():
    """Create specialized config for domain adversarial training"""
    config = ABSAConfig()
    
    # Optimized for domain adversarial training
    config.batch_size = 16
    config.num_epochs = 15
    config.learning_rate = 2e-5
    
    # Focus on domain adversarial features
    config.use_implicit_detection = True
    config.use_domain_adversarial = True
    config.use_contrastive_learning = True
    config.use_few_shot_learning = False  # Focus on domain transfer
    config.use_generative_framework = False
    
    # Maximum domain diversity
    config.datasets = ['laptop14', 'rest14', 'rest15', 'rest16']
    config.experiment_name = "gradient_domain_adversarial"
    
    # Domain adversarial specific settings
    config.domain_loss_weight = 0.15
    config.orthogonal_loss_weight = 0.1
    config.alpha_schedule = 'progressive'
    
    print("✅ Domain adversarial config created")
    return config
'''
    
    # Append the new functions
    with open(config_file, 'a') as f:
        f.write(additional_config)
    
    print("✅ Added missing configuration functions to config.py")
    return True

def create_missing_init_files():
    """Ensure all __init__.py files exist"""
    
    directories = [
        "src",
        "src/utils",
        "src/data", 
        "src/models",
        "src/training"
    ]
    
    for dir_path in directories:
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"✅ Created {init_file}")

def main():
    """Fix all configuration issues"""
    print("🔧 Fixing GRADIENT Configuration Issues...")
    print("=" * 50)
    
    # Fix 1: Add missing config functions
    if fix_config_file():
        print("✅ Configuration functions fixed")
    else:
        print("❌ Failed to fix configuration")
        return
    
    # Fix 2: Create missing __init__.py files
    create_missing_init_files()
    
    # Fix 3: Test imports
    print("\n🧪 Testing imports...")
    try:
        import sys
        sys.path.insert(0, 'src')
        
        from utils.config import ABSAConfig, create_development_config, create_research_config
        print("✅ Config imports working")
        
        # Test config creation
        config = create_development_config()
        print(f"✅ Development config created: {config.experiment_name}")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All configuration issues fixed!")
    print("✅ Ready to run setup_and_test.py again")

if __name__ == "__main__":
    main()