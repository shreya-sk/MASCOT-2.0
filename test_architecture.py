#!/usr/bin/env python
# test_architecture.py - Quick test script to verify the model architecture
import torch
from transformers import AutoTokenizer
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model components
from src.models.absa import GenerativeLLMABSA
from src.utils.config import LLMABSAConfig

def test_model_architecture():
    """Test if the model architecture works end-to-end"""
    print("Testing model architecture...")
    
    # Create a config with minimized dimensions for quick testing
    config = LLMABSAConfig()
    # Ensure hidden_size is divisible by 8 for attention heads
    config.hidden_size = 64  # Small hidden size for quick testing
    config.model_name = "bert-base-uncased"  # Use a well-known model for testing
    config.max_seq_length = 16  # Small sequence length
    config.generate_explanations = True
    
    # Explicitly set number of attention heads to be compatible with hidden size
    if not hasattr(config, 'num_attention_heads') or config.hidden_size % config.num_attention_heads != 0:
        # Find a number of heads that divides hidden_size
        for heads in [8, 4, 2, 1]:
            if config.hidden_size % heads == 0:
                config.num_attention_heads = heads
                break
    
    print(f"Using hidden_size={config.hidden_size} with num_attention_heads={config.num_attention_heads}")
    
    try:
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Initialize model
        print("Initializing model...")
        model = GenerativeLLMABSA(config)
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        
        print(f"Creating dummy inputs (batch_size={batch_size}, seq_len={seq_len})...")
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        # Forward pass
        print("Running forward pass...")
        outputs = model(input_ids, attention_mask, generate=True)
        
        # Check outputs
        print("\nOutput shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        print("\nTest successful! Model architecture is working end-to-end.")
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_architecture()