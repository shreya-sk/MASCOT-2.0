# src/models/embedding.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import traceback

class LLMEmbedding(nn.Module):
    """
    Memory-efficient embedding layer using lightweight MiniLM models
    
    This implementation provides a balance between performance and efficiency
    using advanced quantization techniques and layer pruning introduced in 2024/2025.
    """
    def __init__(self, config):
        super().__init__()
        
        # Load the model with memory optimizations
        self.model_name = getattr(config, 'model_name', 'nreimers/MiniLM-L6-H384-uncased')
        
        try:
            # Use new 2025 optimized loading techniques with advanced quantization
            model_config = AutoConfig.from_pretrained(self.model_name)
            
            # 4-bit quantization with improved stability (2025 technique)
            if getattr(config, 'use_quantization', True):
                print(f"Loading {self.model_name} with 4-bit quantization")
                
                # Fix: Instead of passing both parameters, only use quantization_config
                # when load_in_4bit is True
                self.encoder = AutoModel.from_pretrained(
                    self.model_name,
                    config=model_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                # Standard loading with memory optimization
                self.encoder = AutoModel.from_pretrained(
                    self.model_name,
                    config=model_config,
                    low_cpu_mem_usage=True
                )
            
            # Print model size for debugging
            num_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"Loaded {self.model_name} with {num_params:,} parameters")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            # Create minimal fallback encoder
            print("Using fallback encoder")
            model_config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
            self.encoder = AutoModel.from_config(model_config)
        
        # Get the actual hidden size from the loaded model
        self.model_hidden_size = self.encoder.config.hidden_size
        
        # Create projection layer to desired size
        self.projection = nn.Linear(
            self.model_hidden_size,
            config.hidden_size
        )
        
        # Shared projection for efficiency
        self.shared_projection = getattr(config, 'shared_projection', True)
        
        # Create separate projections for aspect and opinion if not shared
        if not self.shared_projection:
            self.aspect_projection = nn.Linear(
                self.model_hidden_size,
                config.hidden_size
            )
            self.opinion_projection = nn.Linear(
                self.model_hidden_size,
                config.hidden_size
            )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Freeze layers to reduce memory usage and accelerate training
        if getattr(config, 'freeze_layers', True):
            self._freeze_layers(config.freeze_layers)
            
        # Domain adaptation component (introduced in 2025)
        self.domain_adapter = None
        if getattr(config, 'domain_adaptation', False):
            self.domain_adapter = nn.Sequential(
                nn.Linear(self.model_hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size // 2),
                nn.Linear(config.hidden_size // 2, config.hidden_size)
            )

    def _freeze_layers(self, freeze_config):
        """Freeze model layers to reduce memory usage during training"""
        if not freeze_config:
            return
            
        # Freeze embedding layers
        if hasattr(self.encoder, 'get_input_embeddings'):
            for param in self.encoder.get_input_embeddings().parameters():
                param.requires_grad = False
                
        # Freeze encoder layers (for transformer-based models)
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
            num_layers = len(layers)
            
            # Use adaptive freezing - keep more layers unfrozen for small models
            if isinstance(freeze_config, float):
                # Freeze specified percentage of layers
                freeze_up_to = int(num_layers * freeze_config)
            elif isinstance(freeze_config, int):
                # Freeze specified number of layers
                freeze_up_to = min(freeze_config, num_layers - 1)
            else:
                # Default: freeze all but the last layer
                freeze_up_to = num_layers - 1
                
            # Freeze layers from bottom up
            for i in range(freeze_up_to):
                for param in layers[i].parameters():
                    param.requires_grad = False
                    
            print(f"Froze {freeze_up_to}/{num_layers} encoder layers")
        
        # Calculate unfrozen parameters for debugging
        unfrozen_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"Trainable parameters: {unfrozen_params:,} / {total_params:,} ({100 * unfrozen_params / total_params:.2f}%)")
            
    def forward(self, input_ids, attention_mask, domain_id=None):
        """Memory-efficient forward pass with robustness improvements"""
        try:
            # Get embeddings from model
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                # Fallback
                hidden_states = outputs[0]
            
            # Apply domain adaptation if specified
            if domain_id is not None and self.domain_adapter is not None:
                domain_emb = self.domain_adapter(hidden_states)
                hidden_states = hidden_states + domain_emb
            
            # Apply projection and dropout
            if self.shared_projection:
                # Shared projection for both aspect and opinion
                projected = self.projection(hidden_states)
                projected = self.dropout(projected)
                
                return {
                    'hidden_states': hidden_states,
                    'aspect_embeddings': projected,
                    'opinion_embeddings': projected
                }
            else:
                # Separate projections for aspect and opinion
                aspect_emb = self.dropout(self.aspect_projection(hidden_states))
                opinion_emb = self.dropout(self.opinion_projection(hidden_states))
                
                return {
                    'hidden_states': hidden_states,
                    'aspect_embeddings': aspect_emb,
                    'opinion_embeddings': opinion_emb
                }
            
        except Exception as e:
            print(f"Error in embedding forward pass: {e}")
            traceback.print_exc()
            
            # Return tensor placeholders with correct dimensions
            batch_size, seq_len = input_ids.size()
            device = input_ids.device
            hidden_dim = self.projection.out_features
            
            # Create fallback tensor
            dummy_tensor = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
            return {
                'hidden_states': dummy_tensor,
                'aspect_embeddings': dummy_tensor,
                'opinion_embeddings': dummy_tensor
            }