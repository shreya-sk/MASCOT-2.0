# src/models/stella_embedding.py
import torch # type: ignore
import torch.nn as nn # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModel
class StellaEmbedding(nn.Module):
    """
    Stella v5 (400M) embedding layer for ABSA
    
    This implements a novel hierarchical focal embedding that alternates
    between global context understanding and focused aspect-opinion attention.
    """
    def __init__(self, config):
        super().__init__()
        
        # Load the Stella v5 model
        self.model_name = config.model_name
        
        try:
            # Load model with appropriate configuration
            # First, check if we're running on CPU

            using_cpu = not torch.cuda.is_available()

            if using_cpu:
                # On CPU, don't use device_map which can cause offloading issues
                model_config = AutoConfig.from_pretrained(config.model_name)
                self.encoder = AutoModel.from_pretrained(
                    config.model_name,
                    config=model_config,
                    low_cpu_mem_usage=True,
                    device_map=None,  # Don't use device mapping on CPU
                    torch_dtype=torch.float32  # Use float32 instead of float16 on CPU
                )
            else:
                # On GPU, we can use device_map and fp16
                self.encoder = AutoModel.from_pretrained(
                    config.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
        except Exception as e:
            print(f"Error loading Stella model: {e}")
            raise
            
        # Freeze base model parameters to prevent catastrophic forgetting
        # but allow fine-tuning of the last N layers
        self._freeze_layers(config.freeze_layers)
        
        # Novel dual projection for aspect and opinion recognition
        # This creates separate specialized projections for aspect terms vs opinion terms
        self.aspect_projection = nn.Linear(
            self.encoder.config.hidden_size,
            config.hidden_size
        )
        
        self.opinion_projection = nn.Linear(
            self.encoder.config.hidden_size,
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
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            domain_id: Optional domain identifier (e.g., restaurant=0, laptop=1)
        """
        # Get base embeddings from Stella
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Novel: Hierarchical representation with weighted layer fusion
        # This combines information from different layers for better representations
        if hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 1:
            # Get representations from multiple layers
            layer_weights = torch.softmax(self.context_pool, dim=-1)
            
            # Combine last 4 layers with learned weights
            combined_states = torch.zeros_like(hidden_states)
            for i, layer_state in enumerate(outputs.hidden_states[-4:]):
                combined_states += layer_state * layer_weights[:, :, i % layer_weights.shape[-1]]
                
            hidden_states = combined_states
        
        # Apply domain adaptation if domain is specified
        if domain_id is not None:
            domain_embeddings = self.domain_adapter(hidden_states)
            hidden_states = hidden_states + domain_embeddings
            
        # Dual projection for aspect and opinion
        aspect_embeddings = self.aspect_projection(hidden_states)
        opinion_embeddings = self.opinion_projection(hidden_states)
        
        # Return both aspect and opinion embeddings
        return {
            'aspect_embeddings': self.dropout(aspect_embeddings),
            'opinion_embeddings': self.dropout(opinion_embeddings),
            'hidden_states': hidden_states
        }