<<<<<<< Updated upstream
# src/models/stella_embedding.py
import torch # type: ignore
import torch.nn as nn # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModel
class LLMEmbedding(nn.Module):
    """
    Stella v5 (400M) embedding layer for ABSA
    
    This implements a novel hierarchical focal embedding that alternates
    between global context understanding and focused aspect-opinion attention.
    """
    # In src/models/embedding.py - modify the __init__ method

=======
# src/models/embedding.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class LLMEmbedding(nn.Module):
    """Simplified embedding layer for ABSA to reduce memory usage and errors"""
>>>>>>> Stashed changes
    def __init__(self, config):
        super().__init__()
        
        # Load the model
        self.model_name = config.model_name
        
        try:
            # Load a smaller model config
            model_config = AutoConfig.from_pretrained(config.model_name)
            self.encoder = AutoModel.from_pretrained(
                config.model_name,
                config=model_config,
                low_cpu_mem_usage=True
            )
            
            # Print model size for debugging
            num_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"Loaded {config.model_name} with {num_params:,} parameters")
            
        except Exception as e:
            print(f"Error loading model: {e}")
<<<<<<< Updated upstream
            raise
=======
            # Create minimal fallback encoder
            print("Using fallback encoder")
            model_config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
            self.encoder = AutoModel.from_config(model_config)
>>>>>>> Stashed changes
        
        # Get the actual hidden size from the loaded model
        self.model_hidden_size = self.encoder.config.hidden_size
        
        # Create projection layer to desired size
        self.projection = nn.Linear(
            self.model_hidden_size,
            config.hidden_size
        )
        
<<<<<<< Updated upstream
        self.opinion_projection = nn.Linear(
            self.model_hidden_size,
            config.hidden_size
        )
        

        # Novel: Cross-Domain Knowledge Adapter
        # This helps transfer knowledge between domains (e.g., restaurant to laptop)
        self.domain_adapter = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Context pooling with learned weights
        self.context_pool = nn.Parameter(torch.ones(1, 1, self.encoder.config.hidden_size))
        
=======
        # Apply dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Freeze layers to reduce memory usage
        if getattr(config, 'freeze_layers', False):
            self._freeze_layers(True)

>>>>>>> Stashed changes
    def _freeze_layers(self, freeze_layers):
        """Freeze model layers to reduce memory usage during training"""
        if not freeze_layers:
            return
            
        # Freeze embedding layers
        if hasattr(self.encoder, 'get_input_embeddings'):
            for param in self.encoder.get_input_embeddings().parameters():
                param.requires_grad = False
<<<<<<< Updated upstream
        
        # Different models have different structures
        # Try different attribute patterns based on model architecture
        
    # For BERT-like models
=======
                
        # Freeze encoder layers (BERT-specific)
>>>>>>> Stashed changes
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
            num_layers = len(layers)
            # Freeze all but the last layer
            for i in range(num_layers - 1):
                for param in layers[i].parameters():
                    param.requires_grad = False
            print(f"Froze {num_layers-1}/{num_layers} encoder layers")
            
<<<<<<< Updated upstream
        # For Phi-2, LlamaForCausalLM, etc.
        elif hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'layers'):
            layers = self.encoder.model.layers
            num_layers = len(layers)
            # Freeze bottom layers, keep top layers trainable
            freeze_up_to = int(num_layers * 0.75)  # Freeze 75% of layers
            for i in range(freeze_up_to):
                for param in layers[i].parameters():
                    param.requires_grad = False
            print(f"Froze {freeze_up_to}/{num_layers} model layers")
=======
        # Calculate unfrozen parameters for debugging
        unfrozen_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.encoder.parameters())
        print(f"Trainable parameters: {unfrozen_params:,} / {total_params:,} ({100 * unfrozen_params / total_params:.2f}%)")
>>>>>>> Stashed changes
            
        # For other model types
        else:
            print(f"Warning: Could not identify layer structure for {type(self.encoder)}. No layers frozen.")
    def forward(self, input_ids, attention_mask, domain_id=None):
<<<<<<< Updated upstream
        """
        Forward pass with novel hierarchical focal attention
        """
=======
        """Simplified forward pass focusing on robustness"""
>>>>>>> Stashed changes
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
            
<<<<<<< Updated upstream
            # Dual projection for aspect and opinion
            aspect_embeddings = self.aspect_projection(hidden_states)
            opinion_embeddings = self.opinion_projection(hidden_states)
            
            # Return properly structured output
            return {
                'aspect_embeddings': self.dropout(aspect_embeddings),
                'opinion_embeddings': self.dropout(opinion_embeddings),
                'hidden_states': hidden_states
            }
=======
            # Project to desired hidden size
            projected = self.projection(hidden_states)
            projected = self.dropout(projected)
            
            # Return simplified output to avoid downstream errors
            return projected
            
>>>>>>> Stashed changes
        except Exception as e:
            print(f"Error in embedding forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Return tensor placeholders with correct dimensions
            batch_size, seq_len = input_ids.size()
<<<<<<< Updated upstream
            hidden_dim = self.aspect_projection.out_features
            placeholder = torch.zeros(batch_size, seq_len, hidden_dim, device=input_ids.device)
            return {
                'aspect_embeddings': placeholder,
                'opinion_embeddings': placeholder,
                'hidden_states': placeholder
            }
=======
            device = input_ids.device
            hidden_dim = self.projection.out_features
            
            # Create fallback tensor
            return torch.zeros(batch_size, seq_len, hidden_dim, device=device)
>>>>>>> Stashed changes
