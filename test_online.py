# test_simple.py
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

from src.utils.config import LlamaABSAConfig
from src.models.embedding import OnlineLlamaEmbedding  # Use whatever class is in your file
import torch

def test_embedding():
    print("Testing embedding...")
    
    # Create config
    config = LlamaABSAConfig()
    config.use_online_model = True
    config.hidden_size = 768
    
    # Print current directory and PYTHONPATH
    print(f"Current directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Create embedding
    print("Creating embedding...")
    embedding = OnlineLlamaEmbedding(config)
    
    # Test with dummy inputs
    dummy_ids = torch.zeros((1, 10), dtype=torch.long)
    dummy_mask = torch.ones((1, 10), dtype=torch.long)
    test_texts = ["This is a test sentence."]
    
    # Forward pass
    print("Running forward pass...")
    embeddings = embedding(dummy_ids, dummy_mask, texts=test_texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("Test successful!")

if __name__ == "__main__":
    test_embedding()