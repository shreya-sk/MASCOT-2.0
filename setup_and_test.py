#!/usr/bin/env python
# setup_and_test.py - Complete setup and testing script
import os
import sys
import torch 
import subprocess
from pathlib import Path

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[94m",      # Blue
        "success": "\033[92m",   # Green
        "warning": "\033[93m",   # Yellow
        "error": "\033[91m",     # Red
        "reset": "\033[0m"       # Reset
    }
    
    symbols = {
        "info": "ℹ",
        "success": "✓",
        "warning": "⚠",
        "error": "❌"
    }
    
    color = colors.get(status, colors["info"])
    symbol = symbols.get(status, "•")
    reset = colors["reset"]
    
    print(f"{color}{symbol} {message}{reset}")

def check_python_version():
    """Check Python version compatibility"""
    print_status("Checking Python version...", "info")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+", "error")
        return False

def check_gpu():
    """Check GPU availability and memory"""
    print_status("Checking GPU availability...", "info")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_status(f"GPU found: {gpu_name}", "success")
        print_status(f"GPU Memory: {gpu_memory:.1f} GB", "success")
        
        if gpu_memory < 6:
            print_status("GPU memory < 6GB, consider reducing batch size", "warning")
        
        return True
    else:
        print_status("No GPU found, will use CPU (slow)", "warning")
        return False

def install_dependencies():
    """Install required dependencies"""
    print_status("Installing dependencies...", "info")
    
    # Required packages
    packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "spacy>=3.6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    try:
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        # Install spaCy model
        print("Installing spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        
        print_status("All dependencies installed successfully", "success")
        return True
        
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install dependencies: {e}", "error")
        return False

def clean_project():
    """Clean up redundant files"""
    print_status("Cleaning up redundant files...", "info")
    
    files_to_remove = [
        "src/models/context_span_detector.py",
        "src/models/cross_attention.py", 
        "src/models/generative_absa.py",
        "test_generation.py",
        "src/inference/inference.py"
    ]
    
    removed_count = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print_status(f"Removed {file_path}", "success")
            removed_count += 1
    
    if removed_count > 0:
        print_status(f"Cleaned up {removed_count} redundant files", "success")
    else:
        print_status("No redundant files found", "info")

def check_datasets():
    """Check if datasets are available"""
    print_status("Checking datasets...", "info")
    
    datasets = ['laptop14', 'rest14', 'rest15', 'rest16']
    splits = ['train.txt', 'dev.txt', 'test.txt']
    
    found_datasets = []
    missing_datasets = []
    
    for dataset in datasets:
        dataset_dir = f"Datasets/aste/{dataset}"
        if os.path.exists(dataset_dir):
            all_splits_found = True
            for split in splits:
                file_path = os.path.join(dataset_dir, split)
                if not os.path.exists(file_path):
                    all_splits_found = False
                    print_status(f"Missing: {file_path}", "warning")
            
            if all_splits_found:
                found_datasets.append(dataset)
                # Count samples in train file
                train_file = os.path.join(dataset_dir, 'train.txt')
                with open(train_file, 'r') as f:
                    num_samples = len(f.readlines())
                print_status(f"Found {dataset}: {num_samples} training samples", "success")
            else:
                missing_datasets.append(dataset)
        else:
            missing_datasets.append(dataset)
            print_status(f"Missing dataset directory: {dataset_dir}", "error")
    
    if not found_datasets:
        print_status("No datasets found! Please check dataset paths.", "error")
        return False
    
    return found_datasets

def test_model_initialization():
    """Test if the model can be initialized"""
    print_status("Testing model initialization...", "info")
    
    try:
        from src.utils.config import LLMABSAConfig
        from src.models.absa import LLMABSA
        from transformers import AutoTokenizer
        
        # Create config
        config = LLMABSAConfig()
        print_status(f"Config created: {config.model_name}", "success")
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        print_status("Tokenizer loaded successfully", "success")
        
        # Create model
        model = LLMABSA(config)
        param_count = sum(p.numel() for p in model.parameters())
        print_status(f"Model created: {param_count:,} parameters", "success")
        
        # Test forward pass
        test_text = "The food was delicious but service was slow"
        inputs = tokenizer(test_text, return_tensors='pt', max_length=128, 
                          padding='max_length', truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print_status("Forward pass successful", "success")
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print_status(f"  {k}: {v.shape}", "info")
        
        return True
        
    except Exception as e:
        print_status(f"Model initialization failed: {e}", "error")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading pipeline"""
    print_status("Testing data loading...", "info")
    
    try:
        from src.data.utils import read_aste_data
        
        # Test data loading
        test_file = "Datasets/aste/rest15/train.txt"
        if os.path.exists(test_file):
            data = read_aste_data(test_file)
            print_status(f"Loaded {len(data)} samples from {test_file}", "success")
            
            # Check first sample
            if data:
                text, labels = data[0]
                print_status(f"Sample text: {text[:50]}...", "info")
                print_status(f"Sample labels: {len(labels)} triplets", "info")
            
            return True
        else:
            print_status(f"Test file not found: {test_file}", "error")
            return False
            
    except Exception as e:
        print_status(f"Data loading test failed: {e}", "error")
        return False

def test_preprocessing():
    """Test preprocessing pipeline"""
    print_status("Testing preprocessing...", "info")
    
    try:
        from src.utils.config import LLMABSAConfig
        from src.data.preprocessor import LLMABSAPreprocessor
        from src.data.utils import read_aste_data, SpanLabel
        from transformers import AutoTokenizer
        
        # Initialize components
        config = LLMABSAConfig()
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        preprocessor = LLMABSAPreprocessor(tokenizer, max_length=128, use_syntax=True)
        
        # Test data
        test_text = "The food was delicious but service was slow"
        test_labels = [
            SpanLabel(aspect_indices=[1], opinion_indices=[3], sentiment="POS"),
            SpanLabel(aspect_indices=[5], opinion_indices=[7], sentiment="NEG")
        ]
        
        # Preprocess
        result = preprocessor.preprocess(test_text, test_labels)
        
        print_status("Preprocessing successful", "success")
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                print_status(f"  {k}: {v.shape}", "info")
        
        return True
        
    except Exception as e:
        print_status(f"Preprocessing test failed: {e}", "error")
        import traceback
        traceback.print_exc()
        return False

def run_quick_training_test():
    """Run a quick training test"""
    print_status("Running quick training test...", "info")
    
    try:
        # Run training for 1 epoch on debug mode
        cmd = [
            sys.executable, "train.py",
            "--dataset", "rest15",
            "--debug",
            "--batch_size", "2",
            "--gradient_accumulation_steps", "2", 
            "--no_wandb"
        ]
        
        print("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print_status("Quick training test passed", "success")
            return True
        else:
            print_status("Quick training test failed", "error")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Training test timed out (>5 min)", "warning")
        return False
    except Exception as e:
        print_status(f"Training test error: {e}", "error")
        return False

def create_directories():
    """Create necessary directories"""
    print_status("Creating directories...", "info")
    
    dirs = ['checkpoints', 'logs', 'results', 'visualizations']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print_status(f"Created directory: {dir_name}", "success")

def main():
    """Main setup and testing function"""
    print("="*60)
    print("🚀 ABSA PROJECT SETUP AND TESTING")
    print("="*60)
    
    # Track test results
    tests_passed = 0
    total_tests = 0
    
    # Basic checks
    total_tests += 1
    if check_python_version():
        tests_passed += 1
    
    total_tests += 1
    if check_gpu():
        tests_passed += 1
    
    # Install dependencies
    total_tests += 1
    if install_dependencies():
        tests_passed += 1
    
    # Clean project
    clean_project()
    
    # Create directories
    create_directories()
    
    # Check datasets
    total_tests += 1
    datasets = check_datasets()
    if datasets:
        tests_passed += 1
    
    # Test components
    total_tests += 1
    if test_data_loading():
        tests_passed += 1
    
    total_tests += 1
    if test_preprocessing():
        tests_passed += 1
    
    total_tests += 1
    if test_model_initialization():
        tests_passed += 1
    
    # Quick training test (optional - comment out if too slow)
    # total_tests += 1
    # if run_quick_training_test():
    #     tests_passed += 1
    
    # Final summary
    print("\n" + "="*60)
    print("📊 SETUP SUMMARY")
    print("="*60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print_status("🎉 Setup completed successfully!", "success")
        print_status("You can now run training with:", "info")
        print("  python train.py --dataset rest15 --debug")
    elif tests_passed >= total_tests - 2:
        print_status("⚠ Setup mostly successful with minor issues", "warning")
        print_status("You can try running training, but watch for errors", "info")
    else:
        print_status("❌ Setup has significant issues", "error")
        print_status("Please fix the errors above before training", "error")
    
    print("\n📚 Next steps:")
    print("1. Run: python train.py --dataset rest15 --debug")
    print("2. If successful, run: python train.py --dataset rest15")
    print("3. Evaluate: python evaluate.py --model checkpoints/model.pt --dataset rest15")
    print("4. Predict: python predict.py --model checkpoints/model.pt --text 'Your text here'")

if __name__ == "__main__":
    main()