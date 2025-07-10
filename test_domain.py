# demo_domain_adversarial.py
"""
Demo script to test domain adversarial training integration
Run this to verify everything is working correctly
"""

import torch
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.config import create_domain_adversarial_config, test_domain_adversarial_integration
from models.unified_absa_model import create_unified_absa_model
from models.domain_adversarial import DomainAdversarialModule, get_domain_id
from training.domain_adversarial import DomainAdversarialABSATrainer


def demo_gradient_reversal():
    """Demo gradient reversal layer functionality"""
    print("🔄 Testing Gradient Reversal Layer...")
    
    from models.domain_adversarial import gradient_reversal_layer
    
    # Create test tensor
    x = torch.randn(4, 10, requires_grad=True)
    
    # Forward pass through gradient reversal
    reversed_x = gradient_reversal_layer(x, alpha=1.0)
    
    # Compute dummy loss
    loss = reversed_x.sum()
    loss.backward()
    
    # Check that gradients are reversed (negative)
    if x.grad is not None and torch.all(x.grad < 0):
        print("✅ Gradient reversal working correctly")
        return True
    else:
        print("❌ Gradient reversal not working")
        return False


def demo_domain_classifier():
    """Demo domain classifier functionality"""
    print("🎯 Testing Domain Classifier...")
    
    from models.domain_adversarial import DomainClassifier
    
    # Create domain classifier
    domain_classifier = DomainClassifier(hidden_size=768, num_domains=4)
    
    # Test input
    features = torch.randn(8, 768)  # batch_size=8, hidden_size=768
    
    # Forward pass
    domain_logits = domain_classifier(features, alpha=1.0)
    
    # Check output shape
    if domain_logits.shape == (8, 4):
        print("✅ Domain classifier output shape correct: (8, 4)")
        return True
    else:
        print(f"❌ Domain classifier output shape incorrect: {domain_logits.shape}")
        return False


def demo_orthogonal_constraint():
    """Demo orthogonal constraint functionality"""
    print("📐 Testing Orthogonal Constraint...")
    
    from models.domain_adversarial import OrthogonalConstraint
    
    # Create orthogonal constraint
    orthogonal_constraint = OrthogonalConstraint()
    
    # Test features from different domains
    features = torch.randn(12, 256)  # 12 samples, 256 features
    domain_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])  # 4 domains, 3 samples each
    
    # Compute orthogonal loss
    orth_loss = orthogonal_constraint(features, domain_ids)
    
    if orth_loss.item() >= 0:
        print(f"✅ Orthogonal constraint computed: {orth_loss.item():.4f}")
        return True
    else:
        print("❌ Orthogonal constraint computation failed")
        return False


def demo_complete_domain_adversarial_module():
    """Demo complete domain adversarial module"""
    print("🏗️ Testing Complete Domain Adversarial Module...")
    
    # Create config
    config = create_domain_adversarial_config()
    config.hidden_size = 768
    
    # Create domain adversarial module
    da_module = DomainAdversarialModule(
        hidden_size=config.hidden_size,
        num_domains=config.num_domains,
        orthogonal_weight=config.orthogonal_loss_weight
    )
    
    # Test input
    features = torch.randn(8, 20, 768)  # batch, seq_len, hidden_size
    domain_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])  # Mixed domains
    
    # Forward pass
    outputs = da_module(features, domain_ids, return_losses=True)
    
    # Check outputs
    required_keys = ['domain_logits', 'adapted_features', 'domain_loss', 'orthogonal_loss']
    missing_keys = [key for key in required_keys if key not in outputs]
    
    if not missing_keys:
        print("✅ Domain adversarial module outputs complete")
        print(f"   Domain logits shape: {outputs['domain_logits'].shape}")
        print(f"   Domain loss: {outputs['domain_loss'].item():.4f}")
        print(f"   Orthogonal loss: {outputs['orthogonal_loss'].item():.4f}")
        return True
    else:
        print(f"❌ Missing outputs: {missing_keys}")
        return False


def demo_unified_model_integration():
    """Demo unified model with domain adversarial integration"""
    print("🔗 Testing Unified Model Integration...")
    
    # Create config
    config = create_domain_adversarial_config()
    config.hidden_size = 768
    
    # Create model
    model = create_unified_absa_model(config)
    
    # Test input
    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass with dataset name for domain identification
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        dataset_name='laptop14'
    )
    
    # Check that domain adversarial outputs are present
    if 'domain_outputs' in outputs and outputs['domain_outputs']:
        domain_outputs = outputs['domain_outputs']
        print("✅ Domain adversarial integration successful")
        print(f"   Domain logits shape: {domain_outputs['domain_logits'].shape}")
        print(f"   Current alpha: {domain_outputs.get('alpha', 'N/A')}")
        return True
    else:
        print("❌ Domain adversarial integration failed")
        return False


def demo_domain_mapping():
    """Demo domain ID mapping functionality"""
    print("🗺️ Testing Domain Mapping...")
    
    test_datasets = ['laptop14', 'rest14', 'rest15', 'hotel_reviews', 'unknown_dataset']
    expected_ids = [1, 0, 0, 2, 3]  # Based on DOMAIN_MAPPING
    
    all_correct = True
    for dataset, expected_id in zip(test_datasets, expected_ids):
        actual_id = get_domain_id(dataset)
        if actual_id == expected_id:
            print(f"✅ {dataset} -> Domain {actual_id}")
        else:
            print(f"❌ {dataset} -> Domain {actual_id} (expected {expected_id})")
            all_correct = False
    
    return all_correct


def demo_loss_computation():
    """Demo loss computation with domain adversarial components"""
    print("💰 Testing Loss Computation...")
    
    # Create config and model
    config = create_domain_adversarial_config()
    config.hidden_size = 768
    model = create_unified_absa_model(config)
    
    # Create mock batch
    batch_size = 4
    seq_len = 16
    
    # Mock model outputs
    outputs = {
        'aspect_logits': torch.randn(batch_size, seq_len, 3),
        'opinion_logits': torch.randn(batch_size, seq_len, 3),
        'sentiment_logits': torch.randn(batch_size, seq_len, 3),
        'domain_outputs': {
            'domain_logits': torch.randn(batch_size, 4),
            'domain_loss': torch.tensor(0.5),
            'orthogonal_loss': torch.tensor(0.3)
        }
    }
    
    # Mock targets
    targets = {
        'aspect_labels': torch.randint(0, 3, (batch_size, seq_len)),
        'opinion_labels': torch.randint(0, 3, (batch_size, seq_len)),
        'sentiment_labels': torch.randint(0, 3, (batch_size, seq_len))
    }
    
    # Compute loss
    total_loss, loss_dict = model.compute_loss(outputs, targets, dataset_name='laptop14')
    
    # Check loss components
    expected_components = ['aspect_loss', 'opinion_loss', 'sentiment_loss', 'domain_loss', 'orthogonal_loss', 'total_loss']
    present_components = [comp for comp in expected_components if comp in loss_dict]
    
    if len(present_components) >= 4:  # At least main losses + some domain losses
        print("✅ Loss computation successful")
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"   Loss components: {list(loss_dict.keys())}")
        return True
    else:
        print(f"❌ Missing loss components. Present: {present_components}")
        return False


def run_all_demos():
    """Run all domain adversarial training demos"""
    print("🚀 Domain Adversarial Training Integration Demo")
    print("=" * 60)
    
    demos = [
        ("Configuration Integration", test_domain_adversarial_integration),
        ("Gradient Reversal Layer", demo_gradient_reversal),
        ("Domain Classifier", demo_domain_classifier),
        ("Orthogonal Constraint", demo_orthogonal_constraint),
        ("Complete DA Module", demo_complete_domain_adversarial_module),
        ("Domain Mapping", demo_domain_mapping),
        ("Unified Model Integration", demo_unified_model_integration),
        ("Loss Computation", demo_loss_computation)
    ]
    
    results = []
    for name, demo_func in demos:
        print(f"\n🧪 {name}")
        print("-" * 40)
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Demo failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n📊 Demo Results Summary")
    print("=" * 60)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("🎉 ALL DEMOS PASSED! Domain adversarial training is ready!")
        print("\n🚀 Next steps:")
        print("1. Run training with: python train.py --config domain_adversarial")
        print("2. Monitor domain confusion metrics during training")
        print("3. Evaluate cross-domain performance")
        return True
    else:
        print("⚠️ Some demos failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_demos()
    sys.exit(0 if success else 1)