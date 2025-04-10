import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplanationGenerator(nn.Module):
    """
    Novel generative component that creates natural language explanations
    for aspect-sentiment-opinion triplets
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        
        # Template embeddings for different sentiment types
        self.template_embeddings = nn.Parameter(
            torch.randn(3, config.hidden_size)  # POS, NEU, NEG
        )
        
        # Attention for selecting relevant context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            batch_first=True
        )
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=4,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(2)  # 2 decoder layers
        ])
        
        # Output projection to vocabulary
        # Use shared weights with embedding if possible
        if hasattr(config, 'vocab_size'):
            self.vocab_size = config.vocab_size
        else:
            self.vocab_size = 32000  # Default for most LLMs
            
        self.output_projection = nn.Linear(config.hidden_size, self.vocab_size)
        
    def forward(self, hidden_states, aspect_spans, opinion_spans, sentiments, attention_mask=None):
        """
        Generate explanations for the detected triplets
        
        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_dim]
            aspect_spans: List of aspect spans for each batch item
            opinion_spans: List of opinion spans for each batch item
            sentiments: Sentiment predictions for each batch item
            attention_mask: Attention mask for input sequence
            
        Returns:
            explanation_logits: Logits for generated tokens [batch_size, gen_len, vocab_size]
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Start with template tokens based on sentiment
        # Map sentiment indices to embeddings (0=POS, 1=NEU, 2=NEG)
        template_embeds = self.template_embeddings[sentiments]  # [batch_size, hidden_size]
        
        # Create initial decoder input
        decoder_input = template_embeds.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Maximum generation length (can be set in config)
        max_length = getattr(self.config, 'max_generation_length', 32)
        
        # Storage for output logits
        output_logits = []
        
        # Auto-regressive generation loop
        for step in range(max_length):
            # Context attention to select relevant information
            context_vector, _ = self.context_attention(
                decoder_input, 
                hidden_states, 
                hidden_states,
                key_padding_mask=None if attention_mask is None else ~attention_mask.bool()
            )
            
            # Pass through decoder layers
            decoder_output = context_vector
            for layer in self.decoder_layers:
                decoder_output = layer(
                    decoder_output,
                    hidden_states,
                    memory_key_padding_mask=None if attention_mask is None else ~attention_mask.bool()
                )
            
            # Project to vocabulary
            step_logits = self.output_projection(decoder_output[:, -1:, :])  # [batch_size, 1, vocab_size]
            output_logits.append(step_logits)
            
            # Get predicted tokens
            next_token_embeds = self._get_next_token_embedding(step_logits, hidden_states)
            
            # Concatenate to decoder input for next step
            decoder_input = torch.cat([decoder_input, next_token_embeds], dim=1)
            
        # Concatenate all step logits
        explanation_logits = torch.cat(output_logits, dim=1)  # [batch_size, max_length, vocab_size]
        
        return explanation_logits
    
    def _get_next_token_embedding(self, logits, hidden_states):
        """Get embedding for next token based on logits"""
        # For training, we use teacher forcing
        # For inference, we'd use the predicted token
        # This is a simple version that just uses the embedding of the most likely token
        next_token_ids = logits.argmax(dim=-1)  # [batch_size, 1]
        
        # In a full implementation, you would look up these tokens in an embedding table
        # For simplicity, we'll just use a projection of the hidden states
        token_projection = nn.Linear(self.vocab_size, self.hidden_size).to(hidden_states.device)
        token_embeds = token_projection(
            F.one_hot(next_token_ids, num_classes=self.vocab_size).float()
        )
        
        return token_embeds  # [batch_size, 1, hidden_size]
    
    def generate(self, hidden_states, triplets, attention_mask=None):
        """
        Generate explanation text (for inference)
        
        Args:
            hidden_states: Encoder hidden states
            triplets: Extracted triplets
            attention_mask: Attention mask
            
        Returns:
            explanations: Generated explanation text
        """
        # This would be a more complete implementation for inference
        # For now, we'll return template-based explanations
        batch_size = hidden_states.size(0)
        explanations = []
        
        for b in range(batch_size):
            batch_triplets = triplets[b] if isinstance(triplets, list) else triplets
            batch_explanations = []
            
            sentiment_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
            
            for triplet in batch_triplets:
                aspect = triplet.get('aspect', 'this aspect')
                opinion = triplet.get('opinion', 'this feature')
                sentiment = triplet.get('sentiment', 'NEU')
                
                # Convert sentiment label to text
                if isinstance(sentiment, int):
                    sentiment_text = sentiment_map.get(sentiment, 'neutral')
                elif sentiment in ['POS', 'NEG', 'NEU']:
                    sentiment_text = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}[sentiment]
                else:
                    sentiment_text = sentiment
                
                # Template-based explanation
                explanation = f"The aspect '{aspect}' is {sentiment_text} because of the opinion '{opinion}'."
                batch_explanations.append(explanation)
            
            explanations.append(batch_explanations)
        
        return explanations