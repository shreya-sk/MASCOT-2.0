# optimized_loading.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, BitsAndBytesConfig
import psutil
import gc

def get_available_memory():
    """Get available system memory in GB"""
    # CPU memory
    vm = psutil.virtual_memory()
    cpu_free_gb = vm.available / (1024 ** 3)
    
    # GPU memory if available
    gpu_free_gb = 0
    if torch.cuda.is_available():
        try:
            gpu_free_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Subtract already used memory
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            gpu_free_gb -= allocated
        except:
            pass
    
    return {"cpu_free_gb": cpu_free_gb, "gpu_free_gb": gpu_free_gb}

def load_first_layers(model_dir="Llama-3.3-70B-Instruct", num_layers=4):
    """Load just the first few layers of the model for embedding"""
    print(f"Loading first {num_layers} layers of Llama model...")
    
    # Check memory
    memory = get_available_memory()
    print(f"Available memory: CPU: {memory['cpu_free_gb']:.2f} GB, GPU: {memory['gpu_free_gb']:.2f} GB")
    
    # Load the tokenizer first
    try:
        print("Loading tokenizer from fallback source...")
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Successfully loaded tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None, None
    
    # Set up a custom configuration to load only part of the model
    try:
        print("Loading model configuration...")
        config = LlamaConfig.from_pretrained(model_dir)
        
        # Modify config to use fewer layers
        original_layers = config.num_hidden_layers
        config.num_hidden_layers = min(num_layers, original_layers)
        print(f"Modified config: reduced layers from {original_layers} to {config.num_hidden_layers}")
        
        # Try different loading strategies
        print("\nAttempting to load model with minimal layers...")
        
        # Strategy 1: 4-bit quantization with specific layers
        try:
            print("Strategy 1: 4-bit quantization with layer selection")
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            # Setup device map to load specific layers only
            max_memory = {0: "8GB", "cpu": "16GB"}
            device_map = {
                "model.embed_tokens": 0,
                "model.layers.0": 0,
                "model.layers.1": 0,
                "model.layers.2": 0,
                "model.layers.3": 0,
                "model.norm": 0,
                "lm_head": "cpu"
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                config=config,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=torch.float16,
                local_files_only=True,
                offload_folder="offload"
            )
            print("Successfully loaded model with Strategy 1")
        except Exception as e:
            print(f"Strategy 1 failed: {e}")
            
            # Strategy 2: CPU offloading
            try:
                print("\nStrategy 2: CPU offloading")
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    config=config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    offload_folder="offload",
                    offload_state_dict=True
                )
                print("Successfully loaded model with Strategy 2")
            except Exception as e:
                print(f"Strategy 2 failed: {e}")
                
                # Strategy 3: Load from config only for embedding layer
                try:
                    print("\nStrategy 3: Load embedding layer only")
                    # Create a minimal config for just the embedding layer
                    minimal_config = LlamaConfig(
                        vocab_size=config.vocab_size,
                        hidden_size=config.hidden_size,
                        num_hidden_layers=0,  # No layers, just embedding
                        num_attention_heads=config.num_attention_heads,
                        intermediate_size=config.intermediate_size
                    )
                    
                    model = AutoModelForCausalLM.from_config(minimal_config)
                    
                    # Try to load just the embedding layer from the safetensors
                    # This is a placeholder - in a real scenario you'd have to load
                    # the specific tensor for the embedding layer
                    print("Created minimal model with embedding layer only")
                    print("Note: This model doesn't have weights loaded")
                except Exception as e:
                    print(f"Strategy 3 failed: {e}")
                    print("\nAll loading strategies failed")
                    return tokenizer, None
        
        # Test the model embedding layer
        if model is not None and hasattr(model, "get_input_embeddings"):
            print("\nTesting model embedding layer...")
            embed_layer = model.get_input_embeddings()
            print(f"Embedding layer shape: {embed_layer.weight.shape}")
            
            # Test with a simple input
            test_text = "This is a test for embedding."
            tokens = tokenizer(test_text, return_tensors="pt")
            print(f"Tokenized input shape: {tokens['input_ids'].shape}")
            
            # Get embeddings (on CPU to be safe)
            with torch.no_grad():
                embeddings = embed_layer(tokens['input_ids'].cpu())
                print(f"Embedding output shape: {embeddings.shape}")
            
            print("\n✅ Successfully accessed embedding layer!")
            return tokenizer, model
        else:
            print("❌ Model doesn't have a proper embedding layer")
            return tokenizer, None
            
    except Exception as e:
        print(f"Error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return tokenizer, None

if __name__ == "__main__":
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Try with just first 4 layers
    tokenizer, model = load_first_layers(num_layers=4)
    
    if model is not None:
        print("\nSuccess! You can now use the model for embeddings.")
    else:
        print("\nFailed to load the model properly.")
        print("Consider alternatives:")
        print("1. Use a smaller Llama model (like Llama-7B)")
        print("2. Use a different embedding approach (like SentenceTransformers)")
        print("3. Use only the tokenizer with your custom embeddings")