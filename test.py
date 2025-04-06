# test.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.models.embedding import LlamaEmbedding
from src.models.hierarchical_encoder import HierarchicalEncoder
from src.utils.config import LlamaABSAConfig
from src.models.hierarchical_encoder import HierarchicalEncoder


def test_embeddings():
    try:
        print("Loading tokenizer...")
        config = LlamaABSAConfig()
        
        # Try to import bitsandbytes
        try:
           
            print("bitsandbytes available for 8-bit quantization")
        except ImportError:
            print("bitsandbytes not available, using fp16")
            config.use_8bit = False
            
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Add special tokens
        special_tokens = {
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "additional_special_tokens": ["[AT]", "[OT]", "[AC]", "[SP]"]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        print("Loading embedding model...")
        embedder = LlamaEmbedding(config)
        
        print("Loading encoder...")
        encoder = HierarchicalEncoder(config)
        
        # Test input
        text = "The food was delicious but service was terrible"
        print(f"\nTokenizing test input: {text}")
        inputs = tokenizer(
            text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_seq_length
        )
        
        # Get embeddings
        print("\nGenerating embeddings...")
        embeddings = embedder(inputs['input_ids'], inputs['attention_mask'])
        print(f"Embedding shape: {embeddings.shape}")
        
        # Test hierarchical encoder
        print("\nTesting hierarchical encoder...")
        encoded = encoder(embeddings, inputs['attention_mask'])
        print(f"Encoded shape: {encoded.shape}")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embeddings()