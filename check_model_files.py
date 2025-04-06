# check_model_files.py
import os
import glob
from transformers import AutoConfig
import json

def check_llama_files():
    """Check if Llama model files are properly set up"""
    print("Checking for Llama model files...")
    
    # Check main directory
    model_dir = "Llama-3.3-70B-Instruct"
    if not os.path.exists(model_dir):
        print(f"❌ Main directory '{model_dir}' not found!")
        return False
    
    print(f"✅ Found main directory: {model_dir}")
    
    # Check for safetensors files
    safetensors_pattern = os.path.join(model_dir, "*.safetensors")
    safetensors_files = glob.glob(safetensors_pattern)
    
    if safetensors_files:
        print(f"✅ Found {len(safetensors_files)} safetensors files")
        for i, file in enumerate(sorted(safetensors_files)[:5]):
            print(f"  - {os.path.basename(file)}")
        if len(safetensors_files) > 5:
            print(f"  - ...and {len(safetensors_files) - 5} more")
    else:
        print("❌ No safetensors files found!")
        
        # Check subdirectories
        for subdir in ["original", "model", "weights"]:
            subdir_path = os.path.join(model_dir, subdir)
            if os.path.exists(subdir_path):
                safetensors_pattern = os.path.join(subdir_path, "*.safetensors")
                safetensors_files = glob.glob(safetensors_pattern)
                if safetensors_files:
                    print(f"✅ Found {len(safetensors_files)} safetensors files in {subdir}")
                    print(f"  You should move these files to the main {model_dir} directory")
    
    # Check for config.json
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        print(f"✅ Found config.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"  Model architecture: {config.get('architectures', ['Unknown'])[0]}")
            print(f"  Model hidden size: {config.get('hidden_size', 'Unknown')}")
            print(f"  Model vocab size: {config.get('vocab_size', 'Unknown')}")
        except Exception as e:
            print(f"  Warning: Error reading config.json: {e}")
    else:
        print("❌ config.json not found!")
    
    # Check for tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]
    found_tokenizer_files = []
    
    for file in tokenizer_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            found_tokenizer_files.append(file)
    
    if found_tokenizer_files:
        print(f"✅ Found tokenizer files: {', '.join(found_tokenizer_files)}")
    else:
        print("❌ No tokenizer files found in the main directory!")
        
        # Check subdirectories for tokenizer files
        for subdir in ["original", "tokenizer"]:
            subdir_path = os.path.join(model_dir, subdir)
            if os.path.exists(subdir_path):
                found_in_subdir = []
                for file in tokenizer_files:
                    file_path = os.path.join(subdir_path, file)
                    if os.path.exists(file_path):
                        found_in_subdir.append(file)
                
                if found_in_subdir:
                    print(f"✅ Found tokenizer files in {subdir}: {', '.join(found_in_subdir)}")
                    print(f"  You may need to move these to the main {model_dir} directory")
    
    # Try loading model config with transformers
    try:
        print("\nAttempting to load model config with transformers...")
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        print(f"✅ Successfully loaded config with transformers")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
    except Exception as e:
        print(f"❌ Error loading config with transformers: {e}")
    
    print("\nBased on the checks above, you might need to:")
    print("1. Make sure all safetensors files are in the main directory")
    print("2. Make sure config.json is in the main directory")
    print("3. Make sure tokenizer files are in the main directory")
    print("4. Consider using a separate downloaded tokenizer if your local tokenizer is corrupted")

if __name__ == "__main__":
    check_llama_files()