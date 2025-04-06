# test_access.py
from huggingface_hub import HfApi
import os

# Get token from environment
token = os.environ.get("HF_TOKEN")
print(f"Using token: {token[:5]}...{token[-5:] if token else 'None'}")

# Initialize API
api = HfApi(token=token)

# Try to access model info
try:
    #model_info = api.model_info("meta-llama/Llama-3-8B-Instruct")  # Try with a smaller model first
    model_info = api.model_info("meta-llama/Llama-3.2-3B-Instruct-evals")  # This is a public model")  # Then try the larger model
    print(f"✅ Successfully accessed model: {model_info.modelId}")
    print(f"Model tags: {model_info.tags}")
except Exception as e:
    print(f"❌ Error accessing model: {e}")