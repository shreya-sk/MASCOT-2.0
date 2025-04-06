# simplified_llama.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def test_simple_llama():
    """Test a simplified approach to using local Llama with HF tokenizer"""
    print("Starting simplified Llama test...")
    
    # Step 1: Check for local model directory
    model_dir = "Llama-3.3-70B-Instruct"
    if not os.path.exists(model_dir):
        print(f"Error: {model_dir} directory not found!")
        return False
    
    # Step 2: Download just the tokenizer from HF
    print("Downloading tokenizer from Hugging Face...")
    try:
        # Try smaller Llama model for tokenizer (they share the same tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3-8B-Instruct",
            use_fast=True
        )
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Added padding token")
            
        print("Successfully loaded tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying fallback tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "hf-internal-testing/llama-tokenizer",
                use_fast=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("Successfully loaded fallback tokenizer")
        except Exception as e2:
            print(f"Error loading fallback tokenizer: {e2}")
            return False
    
    # Step 3: Set up quantization config
    print("\nSetting up model loading with quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Step 4: Try alternative loading approaches
    print("Attempting to load model. This may take some time...")
    try:
        # Standard approach first
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        print("Successfully loaded model with standard approach")
    except Exception as e:
        print(f"Error with standard loading: {e}")
        
        print("\nTrying alternative loading approach...")
        try:
            # Try with low memory usage
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=True,
                low_cpu_mem_usage=True
            )
            print("Successfully loaded model with low memory approach")
        except Exception as e2:
            print(f"Error with alternative loading: {e2}")
            print("\nPlease check your model files with check_model_files.py")
            return False
    
    # Step 5: Test the model with a simple input
    print("\nTesting with a simple input...")
    test_text = "This is a test sentence for Llama."
    
    # Tokenize
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        padding="max_length", 
        max_length=32,
        truncation=True
    )
    
    print(f"Tokenized input shape: {inputs['input_ids'].shape}")
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings
    print("Getting model embeddings...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states
        hidden_states = outputs.hidden_states[-1]
        print(f"Hidden states shape: {hidden_states.shape}")
    
    print("\n✅ Test successful! You can now use your local Llama model.")
    return True

if __name__ == "__main__":
    success = test_simple_llama()
    print(f"\nTest {'succeeded' if success else 'failed'}")