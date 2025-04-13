import torch   
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< Updated upstream
=======
import numpy as np
>>>>>>> Stashed changes

class ExplanationGenerator(nn.Module):
    """
    Enhanced generative component that creates natural language explanations
    for aspect-sentiment-opinion triplets using prompt-based generation
    """
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        
        # Calculate number of heads to ensure divisibility
        self.num_heads = self._calculate_num_heads(config.hidden_size)
        
        # Template embeddings for different sentiment types
        self.template_embeddings = nn.Parameter(
            torch.randn(3, config.hidden_size)  # POS, NEU, NEG
        )
        
<<<<<<< Updated upstream
        # Attention for selecting relevant context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
=======
        # Dummy embedding for generation starts
        self.bos_embedding = nn.Parameter(torch.randn(1, config.hidden_size))
        
        # Add prompt templates for controlled generation
        self.prompt_templates = {
            0: "This is positive because",  # POS
            1: "This is neutral because",   # NEU
            2: "This is negative because"   # NEG
        }
        
        # Create token embeddings for prompts
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Tokenize and cache prompt embeddings
        self.prompt_token_ids = {}
        for sentiment_id, prompt in self.prompt_templates.items():
            tokens = self.tokenizer(prompt, add_special_tokens=False)
            self.prompt_token_ids[sentiment_id] = tokens.input_ids
        
        # Linear layers for processing aspect and opinion representations
        self.aspect_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.opinion_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Enhanced context fusion mechanism
        self.context_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        # Transformer decoder layers
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=0.1,
>>>>>>> Stashed changes
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
        
<<<<<<< Updated upstream
=======
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
    
    def get_prompt_embedding(self, sentiment_id, device):
        """Get embeddings for prompt tokens corresponding to the sentiment"""
        # Default to neutral if sentiment_id is invalid
        sentiment_id = sentiment_id if sentiment_id in self.prompt_token_ids else 1
        
        # Get tokenized prompt
        prompt_ids = self.prompt_token_ids[sentiment_id]
        
        # Convert to tensor and get embeddings
        prompt_tensor = torch.tensor(prompt_ids, device=device)
        prompt_embeds = self.token_embedding(prompt_tensor).unsqueeze(0)  # [1, prompt_len, hidden_size]
        
        return prompt_embeds
        
>>>>>>> Stashed changes
    def forward(self, hidden_states, aspect_spans, opinion_spans, sentiments, attention_mask=None):
        """
        Generate explanations for the detected triplets with prompt guidance
        
        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_dim]
            aspect_spans: List of aspect spans for each batch item
            opinion_spans: List of opinion spans for each batch item
            sentiments: Sentiment predictions for each batch item
            attention_mask: Attention mask for input sequence
            
        Returns:
            explanation_logits: Logits for generated tokens [batch_size, gen_len, vocab_size]
        """
<<<<<<< Updated upstream
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
=======
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
            
            # Add more contextual representation (average of all tokens)
            context_repr = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            # Project representations
            aspect_repr = self.aspect_proj(aspect_repr)
            opinion_repr = self.opinion_proj(opinion_repr)
            
            # Enhanced context fusion
            triplet_repr = self.context_fusion(
                torch.cat([aspect_repr, opinion_repr, context_repr], dim=-1)
            )
            
            # Get sentiment embeddings 
            # Ensure sentiments has the right shape [batch_size]
            if len(sentiments.shape) > 1:
                sentiments = sentiments.squeeze(-1)
            
            # Handle out of bounds sentiment indices
            sentiments = torch.clamp(sentiments, min=0, max=2)
            sentiment_repr = self.template_embeddings[sentiments]
            
            # Prepare prompt-based decoder inputs
            decoder_inputs = []
            
            for i in range(batch_size):
                # Get sentiment ID for this example
                sentiment_id = sentiments[i].item()
                
                # Create prompt embeddings
                prompt_embeds = self.get_prompt_embedding(sentiment_id, device)
                
                # Create example-specific input
                example_input = torch.cat([
                    self.bos_embedding,                       # Start token
                    prompt_embeds,                            # Prompt ("This is positive because")
                    sentiment_repr[i].unsqueeze(0),           # Sentiment embedding
                    aspect_repr[i].unsqueeze(0),              # Aspect representation
                    opinion_repr[i].unsqueeze(0),             # Opinion representation
                    triplet_repr[i].unsqueeze(0)              # Fused triplet representation
                ], dim=0)
                
                decoder_inputs.append(example_input)
            
            # Stack decoder inputs with padding
            max_input_len = max(input.size(0) for input in decoder_inputs)
            padded_inputs = []
            
            for input in decoder_inputs:
                if input.size(0) < max_input_len:
                    padding = torch.zeros(
                        max_input_len - input.size(0), self.hidden_size, 
                        device=device
                    )
                    padded_input = torch.cat([input, padding], dim=0)
                else:
                    padded_input = input
                padded_inputs.append(padded_input)
            
            decoder_input = torch.stack(padded_inputs, dim=0)  # [batch_size, padded_len, hidden_dim]
            
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
>>>>>>> Stashed changes
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
            
<<<<<<< Updated upstream
            # Concatenate to decoder input for next step
            decoder_input = torch.cat([decoder_input, next_token_embeds], dim=1)
            
        # Concatenate all step logits
        explanation_logits = torch.cat(output_logits, dim=1)  # [batch_size, max_length, vocab_size]
        
        return explanation_logits
=======
        except Exception as e:
            print(f"Error in explanation generation: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy tensor with correct shape
            return torch.zeros(batch_size, self.max_length, self.vocab_size, device=device)
>>>>>>> Stashed changes
    
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
<<<<<<< Updated upstream
        Generate explanation text (for inference)
=======
        Generate explanations using beam search for better quality
>>>>>>> Stashed changes
        
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
<<<<<<< Updated upstream
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
=======
        device = hidden_states.device
        
        # Get explanation logits
        logits = self.forward(hidden_states, aspect_spans, opinion_spans, sentiments, attention_mask)
        
        # Improved generation with simple beam search
        explanations = []
        
        for i in range(batch_size):
            # Get sentiment ID for this example
            sentiment_id = sentiments[i].item()
            
            # Get sentiment text for readability
            sentiment_text = self.prompt_templates.get(sentiment_id, "This is neutral because").split()[2]
            
            # Extract aspect and opinion text
            aspect_indices = (aspect_spans[i] > 0).nonzero(as_tuple=True)[0].cpu().tolist()
            opinion_indices = (opinion_spans[i] > 0).nonzero(as_tuple=True)[0].cpu().tolist()
            
            # Default values if empty
            aspect_text = "this aspect"
            opinion_text = "this feature"
            
            # If we have valid indices, extract words
            if aspect_indices:
                # Convert input_ids to tokens
                if hasattr(self, 'input_ids'):
                    tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids[i])
                    aspect_text = ' '.join([tokens[idx] for idx in aspect_indices])
                else:
                    aspect_text = f"the aspect at positions {aspect_indices}"
                    
            if opinion_indices:
                # Convert input_ids to tokens
                if hasattr(self, 'input_ids'):
                    tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids[i])
                    opinion_text = ' '.join([tokens[idx] for idx in opinion_indices])
                else:
                    opinion_text = f"the opinion at positions {opinion_indices}"
                    
            # Use beam search
            beam_size = 3
            beams = [(0, [])]  # (score, tokens)
            
            sample_logits = logits[i]  # [max_length, vocab_size]
            
            for step in range(self.max_length):
                new_beams = []
                
                for beam_score, beam_tokens in beams:
                    if step < len(beam_tokens):
                        # This beam is already finished
                        new_beams.append((beam_score, beam_tokens))
                        continue
                        
                    # Get logits for current step
                    step_logits = sample_logits[step]  # [vocab_size]
                    
                    # Apply softmax to get probabilities
                    step_probs = F.softmax(step_logits, dim=-1)
                    
                    # Get top-k tokens
                    topk_probs, topk_indices = torch.topk(step_probs, beam_size)
                    
                    for prob, idx in zip(topk_probs, topk_indices):
                        # Skip padding and special tokens
                        if idx == self.tokenizer.pad_token_id or idx == self.tokenizer.eos_token_id:
                            continue
                            
                        # Calculate new score
                        new_score = beam_score - torch.log(prob).item()  # Negative log likelihood
                        
                        # Add token to beam
                        new_tokens = beam_tokens + [idx.item()]
                        
                        # Add to new beams
                        new_beams.append((new_score, new_tokens))
                        
                    # Add EOS token option as well
                    if self.tokenizer.eos_token_id is not None:
                        new_score = beam_score - torch.log(step_probs[self.tokenizer.eos_token_id]).item()
                        new_tokens = beam_tokens + [self.tokenizer.eos_token_id]
                        new_beams.append((new_score, new_tokens))
                
                # Sort and keep top beams
                beams = sorted(new_beams, key=lambda x: x[0])[:beam_size]
                
                # Stop if all beams have EOS
                if all(self.tokenizer.eos_token_id in beam[1] for beam in beams):
                    break
            
            # Get best beam
            _, best_tokens = beams[0]
            
            # Remove EOS if present
            if self.tokenizer.eos_token_id in best_tokens:
                best_tokens = best_tokens[:best_tokens.index(self.tokenizer.eos_token_id)]
                
            # Convert to text
            try:
                explanation = self.tokenizer.decode(best_tokens, skip_special_tokens=True)
            except:
                # Fallback to simple explanation if decoding fails
                explanation = f"The {aspect_text} has a {sentiment_text} sentiment because of the {opinion_text}."
                
            explanations.append(explanation)
                
        return explanations
    # Add to your explanation_generator.py
    def generate_structured_explanation(self, hidden_states, triplets, attention_mask=None):
        """Generate structured explanations by aspect"""
        # Group triplets by aspect
        aspect_groups = {}
        for t in triplets:
            aspect = t['aspect']
            if aspect not in aspect_groups:
                aspect_groups[aspect] = []
            aspect_groups[aspect].append(t)
        
        # Generate aspect-wise explanations
        explanations = []
        for aspect, aspect_triplets in aspect_groups.items():
            # Get majority sentiment
            sentiments = [t['sentiment'] for t in aspect_triplets]
            majority_sentiment = max(set(sentiments), key=sentiments.count)
            
            # Generate explanation using prompt template
            sentiment_text = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}[majority_sentiment]
            opinions = ", ".join([t['opinion'] for t in aspect_triplets])
            
            prompt = f"[ASPECT]: {aspect}\n[OPINION]: {opinions}\n[SENTIMENT]: {sentiment_text}\n"
            # Add to final output
            explanations.append(prompt)
>>>>>>> Stashed changes
        
        return explanations