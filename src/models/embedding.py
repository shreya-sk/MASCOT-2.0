# In src/models/embedding.py
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class ModernEmbedding(nn.Module):
    """Modern embedding layer for ABSA using pre-trained models"""
    def __init__(self, config):
        super().__init__()
        
        # Choose one of these high-quality models
        model_name = "intfloat/e5-large-v2"  # or "thenlper/gte-large" or "BAAI/bge-large-en-v1.5"
        print(f"Loading embedding model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Project to desired dimension if needed
        if hasattr(config, 'hidden_size') and config.hidden_size != self.embedding_dim:
            self.projection = nn.Linear(self.embedding_dim, config.hidden_size)
        else:
            self.projection = nn.Identity()
            
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.1))
        
    def forward(self, input_ids, attention_mask=None, texts=None):
        """
        Forward pass to get embeddings
        
        Args:
            input_ids: Token IDs (used if texts is None)
            attention_mask: Attention mask (not used with sentence-transformers)
            texts: Raw text input (preferred method)
        """
        if texts is None and hasattr(self, 'tokenizer'):
            # Convert input_ids back to text if needed
            texts = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                    for ids in input_ids]
        
        # Get embeddings 
        with torch.no_grad():
            if isinstance(texts, list):
                embeddings = self.model.encode(texts, convert_to_tensor=True)
            else:
                # Handle single text case
                embeddings = self.model.encode([texts], convert_to_tensor=True)
        
        # Project to desired dimension
        embeddings = self.projection(embeddings)
        
        # Expand dimensions to match expected shape [batch, seq_len, hidden]
        if len(embeddings.shape) == 2:
            # Create sequence length dimension (use same embedding for each token)
            embeddings = embeddings.unsqueeze(1).expand(-1, input_ids.size(1), -1)
            
        return self.dropout(embeddings)