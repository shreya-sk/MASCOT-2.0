from transformers import BitsAndBytesConfig
import torch

def get_memory_optimized_config(total_gpu_memory):
    """Configure model loading based on available GPU memory"""
    if total_gpu_memory >= 80:  # For 80GB+ GPUs
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=False),
            "device_map": "auto"
        }
    elif total_gpu_memory >= 40:  # For 40GB GPUs
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": "auto"
        }
    else:  # For smaller GPUs
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": "auto",
            "max_memory": {0: "20GB"}
        }
