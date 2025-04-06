from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import torch.nn as nn
import torch
from huggingface_hub import InferenceApi

class LlamaEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        try:
            # Try loading with 8-bit quantization first
            model_config = AutoConfig.from_pretrained(
                config.model_name,
                rope_scaling={
                    "type": "linear",
                    "factor": 2.0
                }
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                config=model_config,
                token=config.auth_token,
                device_map="auto",
                torch_dtype=torch.float16  # Use fp16 instead of 8-bit
            )
            
        except ImportError as e:
            print(f"Warning: 8-bit quantization not available: {e}")
            print("Falling back to fp16...")
            
            # Fallback to basic fp16 without quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Project to desired dimension 
        self.projection = nn.Linear(
            self.model.config.hidden_size,
            config.hidden_size
        )
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
        hidden_states = outputs.hidden_states[-1]
        embeddings = self.projection(hidden_states)
            
        return self.dropout(embeddings)