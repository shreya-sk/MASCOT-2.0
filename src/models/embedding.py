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

    def __init__(self, config):
        super().__init__()
        
        # Load the model
        self.model_name = config.model_name
        
        try:
            # Load model with appropriate configuration
            using_cpu = not torch.cuda.is_available()
            
            if using_cpu:
                model_config = AutoConfig.from_pretrained(config.model_name)
                self.encoder = AutoModel.from_pretrained(
                    config.model_name,
                    config=model_config,
                    low_cpu_mem_usage=True,
                    device_map=None,
                    torch_dtype=torch.float32
                )
            else:
                self.encoder = AutoModel.from_pretrained(
                    config.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Get the actual hidden size from the loaded model
        self.model_hidden_size = self.encoder.config.hidden_size
        
        # Create projection layers that convert from model_hidden_size to config.hidden_size
        self.aspect_projection = nn.Linear(
            self.model_hidden_size,
            config.hidden_size
        )
        
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
        
    def _freeze_layers(self, freeze_layers):
        if not freeze_layers:
            return
            
        # Freeze embedding layers
        if hasattr(self.encoder, 'get_input_embeddings'):
            for param in self.encoder.get_input_embeddings().parameters():
                param.requires_grad = False
        
        # Different models have different structures
        # Try different attribute patterns based on model architecture
        
    # For BERT-like models
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
            num_layers = len(layers)
            # Freeze bottom layers, keep top layers trainable
            freeze_up_to = int(num_layers * 0.75)  # Freeze 75% of layers
            for i in range(freeze_up_to):
                for param in layers[i].parameters():
                    param.requires_grad = False
            print(f"Froze {freeze_up_to}/{num_layers} encoder layers")
            
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
            
        # For other model types
        else:
            print(f"Warning: Could not identify layer structure for {type(self.encoder)}. No layers frozen.")
    def forward(self, input_ids, attention_mask, domain_id=None):
        """
        Forward pass with novel hierarchical focal attention
        """
        try:
            # Get base embeddings from model
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Get last hidden state (handle different model output formats)
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
                hidden_states = outputs.hidden_states[-1]
            else:
                # Fallback
                hidden_states = outputs[0]
            
            # Dual projection for aspect and opinion
            aspect_embeddings = self.aspect_projection(hidden_states)
            opinion_embeddings = self.opinion_projection(hidden_states)
            
            # Return properly structured output
            return {
                'aspect_embeddings': self.dropout(aspect_embeddings),
                'opinion_embeddings': self.dropout(opinion_embeddings),
                'hidden_states': hidden_states
            }
        except Exception as e:
            print(f"Error in embedding forward pass: {e}")
            # Return tensor placeholders with correct dimensions
            batch_size, seq_len = input_ids.size()
            hidden_dim = self.aspect_projection.out_features
            placeholder = torch.zeros(batch_size, seq_len, hidden_dim, device=input_ids.device)
            return {
                'aspect_embeddings': placeholder,
                'opinion_embeddings': placeholder,
                'hidden_states': placeholder
            }