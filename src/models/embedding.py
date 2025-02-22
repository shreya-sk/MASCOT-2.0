# src/models/embeddings.py
from transformers import LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch

class LlamaEmbedding(nn.Module):
    """Enhanced Llama-based embedding layer for ABSA"""
    def __init__(self, config):
        super().__init__()
        self.llama = LlamaModel.from_pretrained(
            config.model_name,  # e.g. 'meta-llama/Llama-2-7b'
            device_map="auto",  # Handles model sharding
            torch_dtype=torch.float16  # Use fp16 for efficiency
        )
        
        # Freeze base Llama model parameters
        for param in self.llama.parameters():
            param.requires_grad = False
            
        # Add trainable projection layer to match downstream dimensions
        self.projection = nn.Linear(
            self.llama.config.hidden_size,
            config.hidden_size
        )
        
        # Optional span position embeddings
        self.span_embeddings = nn.Embedding(
            config.max_span_length,
            config.hidden_size
        )
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask, span_positions=None):
        """
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            span_positions: Optional span position ids
        """
        # Get Llama base embeddings
        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Project to desired dimension
        embeddings = self.projection(hidden_states)
        
        # Add span position embeddings if provided
        if span_positions is not None:
            span_embeddings = self.span_embeddings(span_positions)
            embeddings = embeddings + span_embeddings
            
        return self.dropout(embeddings)