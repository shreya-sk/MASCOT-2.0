# src/utils/config.py
"""
Clean, unified ABSA configuration with domain adversarial training
"""

import torch
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ABSAConfig:
    """Clean ABSA Configuration with Domain Adversarial Training"""
    
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
    
    # Training intervals
    eval_interval: int = 100
    save_interval: int = 500
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42
    
    # Output settings
    output_dir: str = "outputs"
    experiment_name: str = "absa_experiment"
    
    # Domain Adversarial Training Configuration
    use_domain_adversarial: bool = True
    num_domains: int = 4
    domain_loss_weight: float = 0.1
    orthogonal_loss_weight: float = 0.1
    alpha_schedule: str = 'progressive'  # 'progressive', 'fixed', 'cosine'
    
    domain_mapping: Dict[str, int] = field(default_factory=lambda: {
        'restaurant': 0, 'rest14': 0, 'rest15': 0, 'rest16': 0,
        'laptop': 1, 'laptop14': 1, 'laptop15': 1, 'laptop16': 1,
        'hotel': 2, 'hotel_reviews': 2,
        'general': 3
    })
    
    # Additional domain adversarial parameters
    orthogonal_regularization: bool = True
    orthogonal_lambda: float = 0.01
    domain_classifier_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    domain_classifier_dropout: float = 0.1
    
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
    """Development configuration with domain adversarial training"""
    config = ABSAConfig(
        # Basic settings
        model_name="roberta-base",
        batch_size=16,
        num_epochs=5,
        learning_rate=2e-5,
        max_seq_length=64,
        eval_interval=50,
        save_interval=200,
        
        # Enable key features for development
        use_implicit_detection=True,
        use_few_shot_learning=True,
        use_contrastive_learning=True,
        use_domain_adversarial=True,
        use_generative_framework=False,
        
        # Domain adversarial settings
        domain_loss_weight=0.1,
        orthogonal_loss_weight=0.1,
        alpha_schedule='progressive',
        
        # Datasets
        datasets=['laptop14', 'rest14'],
        experiment_name="absa_dev",
        num_workers=0
    )
    
    print("✅ Development config created with domain adversarial training")
    return config


def create_research_config():
    """Research configuration with all features including domain adversarial"""
    config = ABSAConfig(
        # Research settings
        model_name="roberta-large",
        batch_size=8,
        num_epochs=10,
        learning_rate=1e-5,
        
        # Enable ALL features
        use_implicit_detection=True,
        use_few_shot_learning=True,
        use_contrastive_learning=True,
        use_domain_adversarial=True,
        use_generative_framework=True,
        
        # Advanced domain adversarial settings
        domain_loss_weight=0.15,
        orthogonal_loss_weight=0.1,
        alpha_schedule='cosine',
        orthogonal_lambda=0.02,
        
        # Multi-domain datasets
        datasets=['laptop14', 'rest14', 'rest15', 'rest16'],
        experiment_name="absa_research"
    )
    
    print("✅ Research config created with full domain adversarial training")
    return config


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


def create_domain_adversarial_config():
    """Specialized configuration focused on domain adversarial training"""
    config = ABSAConfig(
        # Optimized for domain adversarial training
        model_name="roberta-base",
        batch_size=24,
        num_epochs=8,
        learning_rate=3e-5,
        
        # Focus on domain adversarial features
        use_implicit_detection=True,
        use_few_shot_learning=False,
        use_contrastive_learning=True,
        use_domain_adversarial=True,
        use_generative_framework=False,
        
        # Aggressive domain adversarial settings
        domain_loss_weight=0.2,
        orthogonal_loss_weight=0.15,
        alpha_schedule='progressive',
        orthogonal_lambda=0.03,
        
        # Maximum domain diversity
        datasets=['laptop14', 'rest14', 'rest15', 'rest16'],
        num_domains=4,
        
        experiment_name="absa_domain_adversarial"
    )
    
    print("✅ Domain adversarial specialized config created")
    return config


def validate_domain_adversarial_config(config):
    """Validate domain adversarial training configuration"""
    issues = []
    
    # Check if domain adversarial is enabled with multi-domain data
    if getattr(config, 'use_domain_adversarial', False):
        if len(config.datasets) < 2:
            issues.append("Domain adversarial training requires at least 2 domains")
        
        if getattr(config, 'domain_loss_weight', 0) <= 0:
            issues.append("Domain loss weight must be positive")
        
        if getattr(config, 'orthogonal_loss_weight', 0) < 0:
            issues.append("Orthogonal loss weight must be non-negative")
        
        alpha_schedule = getattr(config, 'alpha_schedule', 'progressive')
        if alpha_schedule not in ['progressive', 'fixed', 'cosine']:
            issues.append("Alpha schedule must be 'progressive', 'fixed', or 'cosine'")
    
    # Memory considerations
    if getattr(config, 'use_domain_adversarial', False) and config.batch_size > 32:
        issues.append("Large batch size with domain adversarial training may cause OOM")
    
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Domain adversarial configuration validated")
    return True


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
    return True


# For backwards compatibility
def get_domain_id(dataset_name: str) -> int:
    """Get domain ID for dataset name"""
    domain_mapping = {
        'restaurant': 0, 'rest14': 0, 'rest15': 0, 'rest16': 0,
        'laptop': 1, 'laptop14': 1, 'laptop15': 1, 'laptop16': 1,
        'hotel': 2, 'hotel_reviews': 2,
        'general': 3
    }
    return domain_mapping.get(dataset_name.lower(), 3)
