# quick_fix.py
"""
Quick fix for the remaining issues in domain adversarial integration
"""

import sys
from pathlib import Path

def fix_config():
    """Fix missing domain adversarial attributes in config"""
    
    config_file = Path('src/utils/config.py')
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Read the config file
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check if domain adversarial attributes are already there
    if 'num_domains' in content:
        print("✅ Config already has domain adversarial attributes")
        return True
    
    # Find the ABSAConfig class definition
    if '@dataclass' in content and 'class ABSAConfig' in content:
        # Add domain adversarial attributes after existing attributes
        domain_attrs = '''
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
'''
        
        # Insert before the end of the class (before any methods)
        lines = content.split('\n')
        new_lines = []
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            # Insert after the last field definition and before any methods
            if (line.strip().startswith('experiment_name:') or 
                line.strip().startswith('datasets:') or
                line.strip().startswith('output_dir:')):
                # Look ahead to see if we're at the end of fields
                next_non_empty = None
                for j in range(i+1, len(lines)):
                    if lines[j].strip():
                        next_non_empty = lines[j].strip()
                        break
                
                if next_non_empty and (next_non_empty.startswith('def ') or 
                                     next_non_empty.startswith('class ') or
                                     next_non_empty == '' or
                                     not ':' in next_non_empty):
                    new_lines.extend(domain_attrs.split('\n'))
                    break
        
        # Write back the modified content
        new_content = '\n'.join(new_lines)
        
        with open(config_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Added domain adversarial attributes to config")
        return True
    
    else:
        print("❌ Could not find ABSAConfig class definition")
        return False


def fix_model():
    """Fix missing compute_loss method in model"""
    
    model_file = Path('src/models/unified_absa_model.py')
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return False
    
    # Read the model file
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Check if compute_loss method already exists
    if 'def compute_loss(' in content:
        print("✅ Model already has compute_loss method")
        return True
    
    # Add the compute_loss method
    compute_loss_method = '''
    def compute_loss(self, outputs, targets, dataset_name=None):
        """
        Compatibility method for compute_loss
        Calls the comprehensive loss function
        """
        return self.compute_comprehensive_loss(outputs, targets, dataset_name)
'''
    
    # Find where to insert (before the get_model_summary method or at the end of the class)
    lines = content.split('\n')
    new_lines = []
    inserted = False
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Insert before get_model_summary or save methods
        if (line.strip().startswith('def get_model_summary(') or
            line.strip().startswith('def save_complete_model(') or
            line.strip().startswith('def _extract_spans_with_boundaries(')):
            new_lines.extend(compute_loss_method.split('\n'))
            inserted = True
            break
    
    if not inserted:
        # If we couldn't find a good spot, add at the end of the class
        # Find the last method in the class
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip().startswith('def ') and '    def ' in lines[i]:
                # Find the end of this method
                indent_level = len(lines[i]) - len(lines[i].lstrip())
                for j in range(i+1, len(lines)):
                    if (lines[j].strip() and 
                        (len(lines[j]) - len(lines[j].lstrip())) <= indent_level and
                        not lines[j].startswith(' ' * (indent_level + 4))):
                        # Insert before this line
                        new_lines = lines[:j] + compute_loss_method.split('\n') + lines[j:]
                        inserted = True
                        break
                break
    
    if inserted:
        new_content = '\n'.join(new_lines)
        
        with open(model_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Added compute_loss method to model")
        return True
    else:
        print("❌ Could not find good insertion point for compute_loss method")
        return False


def main():
    """Main quick fix function"""
    
    print("🔧 Quick Fix for Domain Adversarial Integration Issues")
    print("=" * 60)
    
    # Fix 1: Config issues
    print("\n1. Fixing config issues...")
    config_fixed = fix_config()
    
    # Fix 2: Model method issues
    print("\n2. Fixing model method issues...")
    model_fixed = fix_model()
    
    # Summary
    print("\n📊 Quick Fix Summary:")
    print(f"   Config fixed: {'✅' if config_fixed else '❌'}")
    print(f"   Model fixed: {'✅' if model_fixed else '❌'}")
    
    if config_fixed and model_fixed:
        print("\n🎉 All issues fixed!")
        print("🧪 Now run: python test_domain.py")
        print("🚀 Then run: python train.py --config dev")
    else:
        print("\n⚠️ Some issues remain. Check the error messages above.")
        
        if not config_fixed:
            print("\n📋 Manual config fix:")
            print("Add these lines to your ABSAConfig class in src/utils/config.py:")
            print("   use_domain_adversarial: bool = True")
            print("   num_domains: int = 4")
            print("   domain_loss_weight: float = 0.1")
            print("   orthogonal_loss_weight: float = 0.1")
            print("   alpha_schedule: str = 'progressive'")
        
        if not model_fixed:
            print("\n📋 Manual model fix:")
            print("Add this method to your UnifiedABSAModel class:")
            print("   def compute_loss(self, outputs, targets, dataset_name=None):")
            print("       return self.compute_comprehensive_loss(outputs, targets, dataset_name)")


if __name__ == "__main__":
    main()