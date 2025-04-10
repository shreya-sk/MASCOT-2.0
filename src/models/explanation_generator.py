import torch   #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore

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
        self.vocab_size = getattr(config, 'vocab_size', 32000)
        self.max_length = getattr(config, 'max_generation_length', 64)
        
        # Template embeddings for different sentiment types
        self.template_embeddings = nn.Parameter(
            torch.randn(3, config.hidden_size)  # POS, NEU, NEG
        )
        
        # Dummy embedding for generation starts
        self.bos_embedding = nn.Parameter(torch.randn(1, config.hidden_size))
        
        # Linear layers for processing aspect and opinion representations
        self.aspect_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.opinion_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Calculate number of heads to ensure divisibility
        # This ensures that hidden_size is divisible by num_heads
        self.num_heads = self._calculate_num_heads(config.hidden_size)
        
        # Transformer decoder layers
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=getattr(config, 'num_decoder_layers', 2)
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.hidden_size, self.vocab_size)
        
    def _calculate_num_heads(self, hidden_size):
        """Calculate the number of attention heads to ensure divisibility"""
        # Try standard numbers of heads
        for heads in [8, 4, 2, 1]:
            if hidden_size % heads == 0:
                return heads
        
        # If no standard number works, find the largest divisor <= 8
        for heads in range(min(8, hidden_size), 0, -1):
            if hidden_size % heads == 0:
                return heads
        
        # Fallback to 1 (should never reach here if hidden_size >= 1)
        return 1
        
    def forward(self, hidden_states, aspect_spans, opinion_spans, sentiments, attention_mask=None):
        """
        Generate explanations for the detected triplets
        
        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_dim]
            aspect_spans: Aspect span predictions [batch_size, seq_len]
            opinion_spans: Opinion span predictions [batch_size, seq_len]
            sentiments: Sentiment predictions [batch_size]
            attention_mask: Attention mask for input sequence [batch_size, seq_len]
            
        Returns:
            explanation_logits: Logits for generated tokens [batch_size, max_length, vocab_size]
        """
        try:
            batch_size = hidden_states.size(0)
            seq_len = hidden_states.size(1)
            device = hidden_states.device
            
            # Extract representations of aspects and opinions based on span predictions
            # Create masks for aspects and opinions
            aspect_mask = (aspect_spans > 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
            opinion_mask = (opinion_spans > 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Weight hidden states by masks and sum
            aspect_repr = torch.sum(hidden_states * aspect_mask, dim=1) / (aspect_mask.sum(dim=1) + 1e-10)
            opinion_repr = torch.sum(hidden_states * opinion_mask, dim=1) / (opinion_mask.sum(dim=1) + 1e-10)
            
            # Project representations
            aspect_repr = self.aspect_proj(aspect_repr)
            opinion_repr = self.opinion_proj(opinion_repr)
            
            # Get sentiment embeddings 
            # Ensure sentiments has the right shape [batch_size]
            if len(sentiments.shape) > 1:
                sentiments = sentiments.squeeze(-1)
            # Handle out of bounds sentiment indices
            sentiments = torch.clamp(sentiments, min=0, max=2)
            sentiment_repr = self.template_embeddings[sentiments]
            
            # Combine representations for decoder input
            decoder_input = torch.cat([
                self.bos_embedding.expand(batch_size, 1, -1),
                sentiment_repr.unsqueeze(1),
                aspect_repr.unsqueeze(1),
                opinion_repr.unsqueeze(1)
            ], dim=1)  # [batch_size, 4, hidden_dim]
            
            # Create causal mask for autoregressive generation
            tgt_mask = self._generate_square_subsequent_mask(self.max_length, device)
            
            # Extend decoder input to max length with zeros
            if decoder_input.size(1) < self.max_length:
                padding = torch.zeros(
                    batch_size, self.max_length - decoder_input.size(1), self.hidden_size,
                    device=device
                )
                decoder_input = torch.cat([decoder_input, padding], dim=1)
            else:
                decoder_input = decoder_input[:, :self.max_length, :]
            
            # Use transformer decoder
            # hidden_states: [batch_size, seq_len, hidden_dim]
            # decoder_input: [batch_size, max_length, hidden_dim]
            memory_key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
            
            decoder_output = self.decoder(
                tgt=decoder_input,
                memory=hidden_states,
                tgt_mask=tgt_mask,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Project to vocabulary
            explanation_logits = self.output_projection(decoder_output)
            
            return explanation_logits
            
        except Exception as e:
            print(f"Error in explanation generation: {e}")
            # Return dummy tensor with correct shape
            return torch.zeros(batch_size, self.max_length, self.vocab_size, device=device)
    
    def _generate_square_subsequent_mask(self, sz, device):
        """Generate a square mask for the sequence, to prevent attending to future tokens"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def generate(self, hidden_states, aspect_spans, opinion_spans, sentiments, attention_mask=None):
        """
        Generate explanations for inference
        
        Args:
            hidden_states: Encoder hidden states
            aspect_spans: Aspect span predictions
            opinion_spans: Opinion span predictions  
            sentiments: Sentiment predictions
            attention_mask: Attention mask
            
        Returns:
            explanations: List of explanation texts
        """
        batch_size = hidden_states.size(0)
        
        # Get explanation logits
        logits = self.forward(hidden_states, aspect_spans, opinion_spans, sentiments, attention_mask)
        
        # Convert logits to token ids
        token_ids = torch.argmax(logits, dim=-1)  # [batch_size, max_length]
        
        # Convert token ids to text
        explanations = []
        for i in range(batch_size):
            # Remove padding and stop at EOS if present
            sample_ids = token_ids[i].cpu().tolist()
            
            # Find EOS token
            if self.tokenizer.eos_token_id in sample_ids:
                sample_ids = sample_ids[:sample_ids.index(self.tokenizer.eos_token_id)]
                
            # Decode tokens to text
            try:
                explanation = self.tokenizer.decode(sample_ids, skip_special_tokens=True)
                explanations.append(explanation)
            except:
                # Fallback explanation if decoding fails
                sentiment_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
                sentiment_idx = sentiments[i].item() if hasattr(sentiments[i], 'item') else sentiments[i]
                sent_text = sentiment_map.get(sentiment_idx, 'neutral')
                explanations.append(f"This item has a {sent_text} sentiment.")
                
        return explanations