# src/data/clean_dataset.py
"""
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent if current_dir.name == 'training' else current_dir.parent.parent if current_dir.name in ['models', 'data', 'utils'] else current_dir / 'src'
if src_dir.name != 'src':
    src_dir = src_dir / 'src'
sys.path.insert(0, str(src_dir))

Clean, simplified dataset handler
Replaces complex dataset implementations with working version
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch
import os

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import torch
import os

@dataclass
class ABSAConfig:
    """ABSA Configuration"""
    
    # Model settings
    model_name: str = "bert-base-uncased"
    hidden_size: int = 768
    num_classes: int = 3
    dropout: float = 0.1
    
    # Training settings
    batch_size: int = 8
    learning_rate: float = 3e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data settings
    max_seq_length: int = 128
    datasets: List[str] = field(default_factory=lambda: ['laptop14', 'rest14'])
    data_dir: str = "Datasets"
    
    # Feature toggles
    use_implicit_detection: bool = True
    use_few_shot_learning: bool = True
    use_generative_framework: bool = False
    use_contrastive_learning: bool = True
    
    # Few-shot settings
    few_shot_k: int = 5
    few_shot_episodes: int = 100
    
    # Training intervals (ADD THESE MISSING PARAMETERS)
    eval_interval: int = 100
    save_interval: int = 500
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42
    
    # Output settings
    output_dir: str = "outputs"
    experiment_name: str = "absa_experiment"


    
    def __post_init__(self):
        """Validate and adjust configuration"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate datasets
        self.datasets = self._validate_datasets()
    
    def _validate_datasets(self) -> List[str]:
        """Validate dataset availability"""
        valid_datasets = []
        
        for dataset in self.datasets:
            train_file = os.path.join(self.data_dir, "aste", dataset, "train.txt")
            if os.path.exists(train_file):
                valid_datasets.append(dataset)
                print(f"✅ Dataset found: {dataset}")
            else:
                print(f"❌ Dataset missing: {dataset} (looking for {train_file})")
        
        if not valid_datasets:
            print("⚠️ No valid datasets found, using default laptop14")
            return ['laptop14']
        
        return valid_datasets
    
    def get_dataset_path(self, dataset: str) -> str:
        """Get path for a specific dataset"""
        return os.path.join(self.data_dir, "aste", dataset)
    
    def get_experiment_dir(self) -> str:
        """Get experiment output directory"""
        exp_dir = os.path.join(self.output_dir, self.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            key: getattr(self, key) for key in self.__dataclass_fields__.keys()
        }
def create_development_config():
    """Create development configuration"""
    return ABSAConfig(
        batch_size=4,
        num_epochs=5,
        learning_rate=1e-5,
        max_seq_length=64,
        eval_interval=50,
        save_interval=200,
        datasets=['laptop14'],
        experiment_name="absa_dev",
        num_workers=0
    )

def create_research_config():
    """Create research configuration"""
    return ABSAConfig(
        batch_size=8,
        num_epochs=25,
        learning_rate=3e-5,
        use_implicit_detection=True,
        use_few_shot_learning=True,
        use_generative_framework=True,
        use_contrastive_learning=True,
        datasets=['laptop14', 'rest14', 'rest15', 'rest16'],
        experiment_name="absa_research"
    )

def create_minimal_config():
    """Create minimal configuration for quick testing"""
    return ABSAConfig(
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-4,
        max_seq_length=32,
        datasets=['laptop14'],
        experiment_name="absa_minimal"
    )
# src/utils/config.py - Add these domain adversarial configurations

# Add to ABSAConfig class:

# ============================================================================
# DOMAIN ADVERSARIAL TRAINING CONFIGURATION
# ============================================================================

# Enable/disable domain adversarial training
use_domain_adversarial: bool = True

# Domain adversarial training parameters
num_domains: int = 4  # restaurant, laptop, hotel, general
domain_loss_weight: float = 0.1
orthogonal_loss_weight: float = 0.1

# Gradient reversal parameters
alpha_schedule: str = 'progressive'  # 'progressive', 'fixed', 'cosine'
initial_alpha: float = 0.0
final_alpha: float = 1.0

# Domain mapping for datasets
domain_mapping: Dict[str, int] = field(default_factory=lambda: {
    'restaurant': 0, 'rest14': 0, 'rest15': 0, 'rest16': 0,
    'laptop': 1, 'laptop14': 1, 'laptop15': 1, 'laptop16': 1,
    'hotel': 2, 'hotel_reviews': 2,
    'general': 3
})

# Orthogonal constraint parameters
orthogonal_regularization: bool = True
orthogonal_lambda: float = 0.01

# Domain classifier architecture
domain_classifier_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
domain_classifier_dropout: float = 0.1

# ============================================================================
# UPDATED CONFIGURATION FACTORY FUNCTIONS
# ============================================================================

def create_development_config():
    """Development configuration with domain adversarial training"""
    config = ABSAConfig()
    
    # Basic settings
    config.model_name = "roberta-base"
    config.batch_size = 16
    config.num_epochs = 5
    config.learning_rate = 2e-5
    
    # Enable key features for development
    config.use_implicit_detection = True
    config.use_few_shot_learning = True
    config.use_contrastive_learning = True
    config.use_domain_adversarial = True  # NEW: Enable domain adversarial
    config.use_generative_framework = False  # Disable for faster training
    
    # Domain adversarial settings
    config.domain_loss_weight = 0.1
    config.orthogonal_loss_weight = 0.1
    config.alpha_schedule = 'progressive'
    
    # Datasets
    config.datasets = ['laptop14', 'rest14']  # Multi-domain for adversarial training
    
    print("✅ Development config created with domain adversarial training")
    return config


def create_research_config():
    """Research configuration with all features including domain adversarial"""
    config = ABSAConfig()
    
    # Research settings
    config.model_name = "roberta-large"
    config.batch_size = 8  # Smaller due to larger model + domain adversarial
    config.num_epochs = 10
    config.learning_rate = 1e-5
    
    # Enable ALL features
    config.use_implicit_detection = True
    config.use_few_shot_learning = True
    config.use_contrastive_learning = True
    config.use_domain_adversarial = True  # NEW: Full domain adversarial
    config.use_generative_framework = True
    
    # Advanced domain adversarial settings
    config.domain_loss_weight = 0.15
    config.orthogonal_loss_weight = 0.1
    config.alpha_schedule = 'cosine'
    config.orthogonal_lambda = 0.02
    
    # Multi-domain datasets for comprehensive evaluation
    config.datasets = ['laptop14', 'rest14', 'rest15', 'rest16']
    
    print("✅ Research config created with full domain adversarial training")
    return config


def create_domain_adversarial_config():
    """Specialized configuration focused on domain adversarial training"""
    config = ABSAConfig()
    
    # Optimized for domain adversarial training
    config.model_name = "roberta-base"
    config.batch_size = 24
    config.num_epochs = 8
    config.learning_rate = 3e-5
    
    # Focus on domain adversarial features
    config.use_implicit_detection = True
    config.use_few_shot_learning = False  # Disable to focus on domain adversarial
    config.use_contrastive_learning = True
    config.use_domain_adversarial = True
    config.use_generative_framework = False
    
    # Aggressive domain adversarial settings
    config.domain_loss_weight = 0.2
    config.orthogonal_loss_weight = 0.15
    config.alpha_schedule = 'progressive'
    config.orthogonal_lambda = 0.03
    
    # Maximum domain diversity
    config.datasets = ['laptop14', 'rest14', 'rest15', 'rest16', 'hotel_reviews']
    config.num_domains = 5  # Increased for more domains
    
    print("✅ Domain adversarial specialized config created")
    return config


# ============================================================================
# DOMAIN ADVERSARIAL TRAINING SPECIFIC HELPERS
# ============================================================================

def get_domain_adversarial_experiment_dir(config):
    """Get experiment directory for domain adversarial training"""
    base_dir = "outputs/domain_adversarial"
    
    # Create descriptive name
    features = []
    if config.use_implicit_detection:
        features.append("implicit")
    if config.use_domain_adversarial:
        features.append(f"da_{config.alpha_schedule}")
    if config.use_contrastive_learning:
        features.append("contrastive")
    
    feature_str = "_".join(features) if features else "basic"
    model_name = config.model_name.split('/')[-1]  # Get just model name
    
    experiment_name = f"{model_name}_{feature_str}_{len(config.datasets)}domains"
    return os.path.join(base_dir, experiment_name)


def validate_domain_adversarial_config(config):
    """Validate domain adversarial training configuration"""
    issues = []
    
    # Check if domain adversarial is enabled with multi-domain data
    if config.use_domain_adversarial:
        if len(config.datasets) < 2:
            issues.append("Domain adversarial training requires at least 2 domains")
        
        if config.domain_loss_weight <= 0:
            issues.append("Domain loss weight must be positive")
        
        if config.orthogonal_loss_weight < 0:
            issues.append("Orthogonal loss weight must be non-negative")
        
        if config.alpha_schedule not in ['progressive', 'fixed', 'cosine']:
            issues.append("Alpha schedule must be 'progressive', 'fixed', or 'cosine'")
        
        if config.num_domains < len(set(get_domain_id(ds) for ds in config.datasets)):
            issues.append("num_domains should be >= number of unique domains in datasets")
    
    # Memory considerations
    if config.use_domain_adversarial and config.batch_size > 32:
        issues.append("Large batch size with domain adversarial training may cause OOM")
    
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Domain adversarial configuration validated")
    return True


# ============================================================================
# INTEGRATION TEST FUNCTION
# ============================================================================

def test_domain_adversarial_integration():
    """Test domain adversarial training integration"""
    print("🧪 Testing Domain Adversarial Integration...")
    
    # Test development config
    dev_config = create_development_config()
    if not validate_domain_adversarial_config(dev_config):
        print("❌ Development config validation failed")
        return False
    
    # Test research config
    research_config = create_research_config()
    if not validate_domain_adversarial_config(research_config):
        print("❌ Research config validation failed")
        return False
    
    # Test specialized config
    da_config = create_domain_adversarial_config()
    if not validate_domain_adversarial_config(da_config):
        print("❌ Domain adversarial config validation failed")
        return False
    
    print("✅ All domain adversarial configurations validated successfully!")
    
    # Test model creation
    try:
        from models.unified_absa_model import create_unified_absa_model
        model = create_unified_absa_model(dev_config)
        
        if hasattr(model, 'domain_adversarial') and model.domain_adversarial is not None:
            print("✅ Model created successfully with domain adversarial components")
        else:
            print("❌ Model missing domain adversarial components")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    print("🎉 Domain Adversarial Integration Test PASSED!")
    return True


# Helper function to add to existing config class
def add_domain_adversarial_to_config(config):
    """Add domain adversarial settings to existing config"""
    config.use_domain_adversarial = True
    config.num_domains = 4
    config.domain_loss_weight = 0.1
    config.orthogonal_loss_weight = 0.1
    config.alpha_schedule = 'progressive'
    config.domain_mapping = {
        'restaurant': 0, 'rest14': 0, 'rest15': 0, 'rest16': 0,
        'laptop': 1, 'laptop14': 1, 'laptop15': 1, 'laptop16': 1,
        'hotel': 2, 'hotel_reviews': 2,
        'general': 3
    }
    config.orthogonal_regularization = True
    config.orthogonal_lambda = 0.01
    config.domain_classifier_hidden_sizes = [256, 128]
    config.domain_classifier_dropout = 0.1
    
    print("✅ Domain adversarial settings added to existing config")
    return config

# Quick fix for src/utils/config.py
# Add these lines to your ABSAConfig class or create a patch

def patch_config():
    """
    Quick patch to add missing domain adversarial attributes to existing config
    """
    
    import sys
    from pathlib import Path
    
    # Add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir / 'src' if (current_dir / 'src').exists() else current_dir.parent / 'src'
    sys.path.insert(0, str(src_dir))
    
    try:
        from utils.config import ABSAConfig
        
        # Add missing attributes to ABSAConfig class
        def add_domain_adversarial_attrs(self):
            """Add domain adversarial attributes if missing"""
            if not hasattr(self, 'num_domains'):
                self.num_domains = 4
            if not hasattr(self, 'use_domain_adversarial'):
                self.use_domain_adversarial = True
            if not hasattr(self, 'domain_loss_weight'):
                self.domain_loss_weight = 0.1
            if not hasattr(self, 'orthogonal_loss_weight'):
                self.orthogonal_loss_weight = 0.1
            if not hasattr(self, 'alpha_schedule'):
                self.alpha_schedule = 'progressive'
            if not hasattr(self, 'domain_mapping'):
                self.domain_mapping = {
                    'restaurant': 0, 'rest14': 0, 'rest15': 0, 'rest16': 0,
                    'laptop': 1, 'laptop14': 1, 'laptop15': 1, 'laptop16': 1,
                    'hotel': 2, 'hotel_reviews': 2,
                    'general': 3
                }
            return self
        
        # Monkey patch the method
        ABSAConfig.add_domain_adversarial_attrs = add_domain_adversarial_attrs
        
        print("✅ Config patched successfully")
        return True
        
    except Exception as e:
        print(f"❌ Config patch failed: {e}")
        return False

if __name__ == "__main__":
    patch_config()