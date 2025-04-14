# src/models/span_detector.py - Improved Rule-Based Detector
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class SpanDetector(nn.Module):
    """Enhanced span detector for restaurant reviews with better heuristics"""
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.hidden_dim = config.hidden_size
        self.dropout_rate = getattr(config, 'dropout', 0.1)
        
        # Get the actual input dimension - this is crucial
        # Use embedding_size if available, otherwise use hidden_size
        input_size = getattr(config, 'embedding_size', self.hidden_dim)
        
        # Simple recurrent layer (GRU is more memory efficient)
        self.rnn = nn.GRU(
            input_size=input_size,  # Use the actual embedding size
            hidden_size=self.hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0
        )
        
        print(f"SpanDetector RNN input size: {input_size}, output size: {self.hidden_dim}")
        
        # Improved aspect classifier with stronger bias towards multi-token spans
        self.aspect_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3)  # B-I-O tags
        )
        
        # Improved opinion classifier with stronger bias towards adjectives and descriptive terms
        self.opinion_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3)  # B-I-O tags
        )
        
        # Initialize with better biases for aspects and opinions
        self._initialize_with_better_bias()
        
        # Enhanced word lists for restaurant domain
        self.aspect_terms = {
            # Food items
            'food', 'meal', 'dish', 'pizza', 'pasta', 'sushi', 'burger', 'menu',
            'salad', 'appetizer', 'dessert', 'lunch', 'dinner', 'breakfast',
            'steak', 'chicken', 'fish', 'seafood', 'rice', 'noodle', 'soup',
            'sandwich', 'bread', 'roll', 'wine', 'beer', 'drink', 'coffee', 'tea',
            # Restaurant attributes
            'restaurant', 'place', 'spot', 'joint', 'establishment', 'bistro', 'cafe',
            'ambiance', 'atmosphere', 'decor', 'interior', 'setting', 'location',
            # Service attributes
            'service', 'staff', 'waiter', 'waitress', 'server', 'chef', 'host', 'hostess',
            # Price-related
            'price', 'value', 'cost', 'bill', 'check',
            # Experience attributes
            'experience', 'visit', 'time', 'reservation', 'wait',
            # Food attributes
            'portion', 'size', 'taste', 'flavor', 'texture', 'presentation'
        }
        
        self.opinion_terms = {
            # Positive opinions
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'delicious', 'outstanding',
            'tasty', 'fantastic', 'fabulous', 'perfect', 'fresh', 'flavorful', 'spectacular',
            'nice', 'lovely', 'enjoyable', 'pleasant', 'satisfying', 'impressive', 'exceptional',
            'friendly', 'attentive', 'prompt', 'fast', 'quick', 'efficient', 'professional',
            'worth', 'reasonable', 'fair', 'generous', 'large', 'authentic', 'cozy', 'clean',
            # Negative opinions
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'mediocre',
            'bland', 'tasteless', 'overcooked', 'undercooked', 'stale', 'cold', 'burnt',
            'unfriendly', 'rude', 'slow', 'inattentive', 'incompetent', 'inexperienced',
            'expensive', 'overpriced', 'pricey', 'steep', 'small', 'tiny', 'dirty', 'noisy',
            'crowded', 'unpleasant', 'uncomfortable', 'greasy', 'oily'
        }
        
        # Negation terms that can reverse sentiment
        self.negation_terms = {
            'not', 'no', 'never', 'none', 'neither', 'nor', "n't", 'without',
            'barely', 'hardly', 'rarely', 'seldom', 'cannot', "can't"
        }
        
        # Common stop words to avoid treating as aspects or opinions
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'while', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
            'as', 'from', 'up', 'down', 'off', 'over', 'under', 'again', 'once',
            'here', 'there', 'all', 'any', 'both', 'each', 'more', 'most', 'other',
            'some', 'such', 'that', 'this', 'these', 'those', 'only', 'very',
            'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
            'they', 'them', 'their', 'theirs', 'themselves'
        }
        
    def _initialize_with_better_bias(self):
        """Initialize with improved biases for better aspect and opinion detection"""
        # For aspect classifier, increase bias for B and I tags
        # Give strong bias to B tag (index 1) to encourage detection of new spans
        self.aspect_classifier[-1].bias.data[1] = 1.0
        # Give moderate bias to I tag (index 2) to encourage multi-token spans
        self.aspect_classifier[-1].bias.data[2] = 0.5
        # Slightly reduce bias for O tag (index 0)
        self.aspect_classifier[-1].bias.data[0] = -0.2
        
        # Similar for opinion classifier but with different values
        self.opinion_classifier[-1].bias.data[1] = 1.2  # Stronger bias for B tag
        self.opinion_classifier[-1].bias.data[2] = 0.7  # Stronger bias for I tag
        self.opinion_classifier[-1].bias.data[0] = -0.1  # Slight negative bias for O tag
    
    def forward(self, hidden_states, attention_mask=None, texts=None, input_ids=None, tokenizer=None):
        """
        Forward pass with enhanced rule-based improvement
        
        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            texts: Optional raw text for rule-based enhancement
            input_ids: Optional input token IDs
            tokenizer: Optional tokenizer for token-to-text conversion
            
        Returns:
            aspect_logits, opinion_logits, span_features, boundary_logits
        """
        try:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            device = hidden_states.device
            
            # Apply RNN for sequential context
            rnn_out, _ = self.rnn(hidden_states)
            
            # Get base logits from neural network
            aspect_logits = self.aspect_classifier(rnn_out)
            opinion_logits = self.opinion_classifier(rnn_out)
            
            # If both texts and tokenizer are provided, use enhanced rule-based detection
            if texts is not None and tokenizer is not None and input_ids is not None:
                # Apply rule-based enhancement
                for i, (text, ids) in enumerate(zip(texts, input_ids)):
                    if i >= batch_size:
                        break
                        
                    # Convert input_ids to tokens
                    tokens = [tokenizer.decode([token_id.item()]).lower().strip() for token_id in ids]
                    
                    # Apply enhanced rule-based improvement for this sample
                    aspect_logits[i], opinion_logits[i] = self._apply_enhanced_rules(
                        text, tokens, aspect_logits[i], opinion_logits[i]
                    )
            # If only texts are provided (fallback)
            elif texts is not None:
                # Apply simpler rule-based enhancement for this sample
                for i, text in enumerate(texts):
                    if i >= batch_size:
                        break
                        
                    # Apply simpler rule-based enhancement
                    aspect_logits[i], opinion_logits[i] = self._apply_rules(
                        text, aspect_logits[i], opinion_logits[i]
                    )
            
            # Apply attention mask
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                aspect_logits = aspect_logits * mask
                opinion_logits = opinion_logits * mask
                
                # Large negative for padding
                padding_mask = (1 - mask) * -10000.0
                aspect_logits = aspect_logits + padding_mask
                opinion_logits = opinion_logits + padding_mask
            
            # Create boundary logits (for span boundaries)
            boundary_logits = torch.zeros(batch_size, seq_len, 2, device=device)
            
            # Use RNN output as span features
            span_features = rnn_out
            
            return aspect_logits, opinion_logits, span_features, boundary_logits
        
        except Exception as e:
            print(f"Error in span detector: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback tensors
            device = hidden_states.device
            batch_size, seq_len = hidden_states.shape[:2]
            
            aspect_logits = torch.zeros(batch_size, seq_len, 3, device=device)
            opinion_logits = torch.zeros(batch_size, seq_len, 3, device=device)
            span_features = torch.zeros_like(hidden_states)
            boundary_logits = torch.zeros(batch_size, seq_len, 2, device=device)
            
            return aspect_logits, opinion_logits, span_features, boundary_logits
    
    def _apply_enhanced_rules(self, text, tokens, aspect_logits, opinion_logits):
        """Apply enhanced rules with token awareness for better span detection"""
        # Remove special tokens from consideration
        valid_tokens = []
        valid_indices = []
        for i, token in enumerate(tokens):
            # Skip special tokens, padding, etc.
            if token in ['[PAD]', '[CLS]', '[SEP]', '<s>', '</s>', '<pad>'] or not token.strip():
                continue
            valid_tokens.append(token)
            valid_indices.append(i)
            
        if not valid_tokens:
            return aspect_logits, opinion_logits
            
        # Process text to identify noun phrases as likely aspects
        # This is a simple heuristic that looks for noun patterns
        text_lower = text.lower()
        
        # Find noun phrases with simple regex patterns
        # Pattern: article/adjective + (optional adjective) + noun
        noun_phrases = re.findall(r'(the|a|an|this|that|my|our|their)?\s*(\w+\s+)?(\w+)', text_lower)
        
        # Process valid tokens to identify aspects and opinions
        for i, (token_idx, token) in enumerate(zip(valid_indices, valid_tokens)):
            # Skip special tokens and uninformative tokens
            if token in self.stop_words or not token.strip() or token in ['[UNK]', '.', ',', '?', '!']:
                aspect_logits[token_idx, 1] -= 2.0  # Strong negative bias for B tag
                opinion_logits[token_idx, 1] -= 2.0  # Strong negative bias for B tag
                continue
                
            # Check if token is part of a noun phrase (likely an aspect)
            is_in_noun_phrase = False
            for phrase in noun_phrases:
                phrase_text = ' '.join(p for p in phrase if p).strip()
                if token in phrase_text:
                    is_in_noun_phrase = True
                    break
            
            # Strong boost for aspect terms
            if token in self.aspect_terms or (i > 0 and token in ['served', 'included', 'offered']):
                aspect_logits[token_idx, 1] += 3.0  # Very strong bias for aspect B tag
                
                # If we have context (not at the end), check if next token should be part of span
                if i < len(valid_tokens) - 1:
                    next_token = valid_tokens[i+1]
                    next_idx = valid_indices[i+1]
                    if next_token not in self.stop_words and next_token not in ['.', ',', '?', '!']:
                        aspect_logits[next_idx, 2] += 2.0  # Bias for I tag in next token
            
            # Strong boost for opinion terms
            if token in self.opinion_terms:
                opinion_logits[token_idx, 1] += 3.0  # Very strong bias for opinion B tag
                
                # If previous token is a negation, also mark it as part of the opinion
                if i > 0 and valid_tokens[i-1] in self.negation_terms:
                    prev_idx = valid_indices[i-1]
                    opinion_logits[prev_idx, 1] += 2.0  # Mark previous token as B tag
                    opinion_logits[token_idx, 2] += 2.0  # Change current to I tag
            
            # Additional rules for token position
            if i == 0 and token not in self.aspect_terms and token not in ['i', 'we', 'it', 'they']:
                # Penalize first token as aspect unless it's a common aspect term
                aspect_logits[token_idx, 1] -= 1.0
            
            # If token is in a noun phrase but not recognized aspect term, give moderate boost
            if is_in_noun_phrase and token not in self.stop_words:
                aspect_logits[token_idx, 1] += 1.0
            
            # If token follows "is" or "was" and is an adjective-like, it's likely an opinion
            if i > 0 and valid_tokens[i-1] in ['is', 'was', 'are', 'were', 'feels', 'seemed', 'tastes']:
                opinion_logits[token_idx, 1] += 2.0
                
            # Penalties for punctuation and stop words
            if token in ['.', ',', '!', '?', ')', '(', ':', ';']:
                aspect_logits[token_idx, 1] -= 5.0  # Strong negative for aspect
                opinion_logits[token_idx, 1] -= 5.0  # Strong negative for opinion
        
        return aspect_logits, opinion_logits
    
    def _apply_rules(self, text, aspect_logits, opinion_logits):
        """Simplified rule application without token information"""
        # Text-based rules using the raw text
        tokens = text.lower().split()
        
        for i, token in enumerate(tokens):
            if i >= aspect_logits.size(0):
                break
                
            # Boost for aspect terms
            if token in self.aspect_terms:
                aspect_logits[i, 1] += 5.0
            
            # Boost for opinion terms
            if token in self.opinion_terms:
                opinion_logits[i, 1] += 5.0
                
            # Add penalties for stop words
            if token in self.stop_words:
                aspect_logits[i, 1] -= 2.0
                opinion_logits[i, 1] -= 1.0
                
            # Prevent punctuation from being aspects or opinions
            if token in ['.', ',', '!', '?', ')', '(', ':', ';']:
                aspect_logits[i, 1] -= 10.0
                opinion_logits[i, 1] -= 10.0
        
        return aspect_logits, opinion_logits