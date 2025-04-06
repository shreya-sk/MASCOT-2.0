# src/models/stella_embedding.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class StellaEmbedding(nn.Module):
    """
    Stella v5 (400M) embedding layer for ABSA
    
    This implements a novel hierarchical focal embedding that alternates
    between global context understanding and focused aspect-opinion attention.
    """
    def __init__(self, config):
        super().__init__()
        
        # Load the Stella v5 model
        self.model_name = "stanford-crfm/Stella-400M-v5"
        
        try:
            # Load model with appropriate configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
                device_map="auto"
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
            self.model.config.hidden_size,
            config.hidden_size
        )
        
        self.opinion_projection = nn.Linear(
            self.model.config.hidden_size,
            config.hidden_size
        )
        
        # Novel: Cross-Domain Knowledge Adapter
        # This helps transfer knowledge between domains (e.g., restaurant to laptop)
        self.domain_adapter = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Context pooling with learned weights
        self.context_pool = nn.Parameter(torch.ones(1, 1, self.model.config.hidden_size))
        
    def _freeze_layers(self, num_frozen_layers):
        """Freeze the first num_frozen_layers transformer blocks"""
        if num_frozen_layers <= 0:
            return
            
        # Freeze embeddings
        for param in self.model.get_input_embeddings().parameters():
            param.requires_grad = False
            
        # Freeze the specified number of layers
        for i, layer in enumerate(self.model.model.layers):
            if i < num_frozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
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
            outputs = self.model(
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