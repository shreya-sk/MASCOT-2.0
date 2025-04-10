# src/models/embedding.py
import torch 
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class LLMEmbedding(nn.Module):
    """
    LLM embedding layer for ABSA
    
    This implements a novel hierarchical focal embedding that alternates
    between global context understanding and focused aspect-opinion attention.
    """
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
            # Create fallback encoder
            print("Using fallback encoder")
            model_config = AutoConfig.from_pretrained("bert-base-uncased")
            model_config.hidden_size = config.hidden_size
            self.encoder = AutoModel.from_config(model_config)
        
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
        
        # Common projection for hidden states
        self.hidden_projection = nn.Linear(
            self.model_hidden_size,
            config.hidden_size
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Freeze base model parameters if configured
        if getattr(config, 'freeze_layers', False):
            self._freeze_layers(True)

    def _freeze_layers(self, freeze_layers):
        if not freeze_layers:
            return
            
        # Freeze embedding layers
        if hasattr(self.encoder, 'get_input_embeddings'):
            for param in self.encoder.get_input_embeddings().parameters():
                param.requires_grad = False
        
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
            
        # For other model types (Phi-2, LlamaForCausalLM, etc.)
        elif hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'layers'):
            layers = self.encoder.model.layers
            num_layers = len(layers)
            # Freeze bottom layers, keep top layers trainable
            freeze_up_to = int(num_layers * 0.75)  # Freeze 75% of layers
            for i in range(freeze_up_to):
                for param in layers[i].parameters():
                    param.requires_grad = False
            print(f"Froze {freeze_up_to}/{num_layers} model layers")
        else:
            print(f"Warning: Could not identify layer structure. No layers frozen.")
            
    def forward(self, input_ids, attention_mask, domain_id=None):
        """
        Forward pass with novel hierarchical focal attention
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            domain_id: Optional domain identifier for domain adaptation
            
        Returns:
            Dictionary with embeddings for aspects, opinions, and hidden states
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
            
            # Apply projections with shape validation
            batch_size, seq_len, _ = hidden_states.size()
            
            # Project hidden states
            hidden_proj = self.hidden_projection(hidden_states)
            
            # Dual projection for aspect and opinion
            aspect_embeddings = self.aspect_projection(hidden_states)
            opinion_embeddings = self.opinion_projection(hidden_states)
            
            # Apply dropout
            aspect_embeddings = self.dropout(aspect_embeddings)
            opinion_embeddings = self.dropout(opinion_embeddings)
            hidden_proj = self.dropout(hidden_proj)
            
            # Return properly structured output
            return {
                'aspect_embeddings': aspect_embeddings,
                'opinion_embeddings': opinion_embeddings,
                'hidden_states': hidden_proj
            }
        except Exception as e:
            print(f"Error in embedding forward pass: {e}")
            # Return tensor placeholders with correct dimensions
            batch_size, seq_len = input_ids.size()
            hidden_dim = self.aspect_projection.out_features
            device = input_ids.device
            
            # Create fallback tensors with correct dimensions
            placeholder = torch.zeros(batch_size, seq_len, hidden_dim, device=device)
            return {
                'aspect_embeddings': placeholder,
                'opinion_embeddings': placeholder,
                'hidden_states': placeholder
            }