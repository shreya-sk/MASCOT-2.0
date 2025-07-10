# src/models/span_detector.py
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
        
        # FIXED: Better initialization for aspect classifier
        self.aspect_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3)  # B-I-O tags
        )
        
        # FIXED: Better initialization for opinion classifier
        self.opinion_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 3)  # B-I-O tags
        )
        
        # FIXED: Better initialization with balanced biases
        self._initialize_with_better_bias()
        
        # FIXED: Enhanced word lists for restaurant domain
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
            'portion', 'size', 'taste', 'flavor', 'texture', 'presentation',
            # Added common dishes and important aspects
            'appetizer', 'entree', 'main', 'course', 'starter', 'side', 'sauce',
            'parking', 'location', 'bathroom', 'restroom', 'menu', 'selection',
            'option', 'variety', 'quality', 'quantity', 'portion', 'temperature',
            'table', 'seating', 'chair', 'booth', 'patio', 'outdoor', 'indoor',
            'reservation', 'wait', 'time', 'hour', 'minute'
        }
        
        # FIXED: Improved opinion terms list
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
            'crowded', 'unpleasant', 'uncomfortable', 'greasy', 'oily',
            # Additional opinion terms
            'satisfactory', 'unsatisfactory', 'acceptable', 'unacceptable',
            'recommended', 'recommended', 'solid', 'decent', 'fine', 'okay', 'ok',
            'average', 'below-average', 'above-average', 'subpar', 'superior',
            'inferior', 'adequate', 'inadequate', 'sufficient', 'insufficient',
            'consistent', 'inconsistent', 'reliable', 'unreliable', 'memorable',
            'forgettable', 'ordinary', 'extraordinary', 'standard', 'substandard',
            'hot', 'warm', 'lukewarm', 'cool', 'spicy', 'mild', 'sweet', 'sour',
            'bitter', 'salty', 'rich', 'light', 'heavy', 'filling', 'hearty',
            'innovative', 'creative', 'traditional', 'authentic', 'inauthentic',
            'cramped', 'spacious', 'convenient', 'inconvenient', 'accommodating'
        }
        
        # Negation terms that can reverse sentiment
        self.negation_terms = {
            'not', 'no', 'never', 'none', 'neither', 'nor', "n't", 'without',
            'barely', 'hardly', 'rarely', 'seldom', 'cannot', "can't"
        }
        
        # FIXED: Improved stop words with common words that shouldn't be aspects/opinions
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
            'while', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
            'as', 'from', 'up', 'down', 'off', 'over', 'under', 'again', 'once',
            'here', 'there', 'all', 'any', 'both', 'each', 'more', 'most', 'other',
            'some', 'such', 'that', 'this', 'these', 'those', 'only', 'very',
            'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
            'they', 'them', 'their', 'theirs', 'themselves',
            'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'can', 'could', 'should', 'would', 'may', 'might', 'must',
            'how', 'what', 'when', 'where', 'why', 'who', 'whom',
            'get', 'got', 'gotten', 'getting', 'go', 'goes', 'going', 'went', 'gone',
            'so', 'just', 'now', 'then', 'always', 'often', 'sometimes', 'never',
            'also', 'too', 'either', 'neither', 'both', 'even', 'still',
            'actually', 'really', 'basically', 'literally', 'definitely',
            'well', 'anyway', 'however', 'though', 'although',
            'every', 'many', 'few', 'several', 'some', 'any', 'no', 'all'
        }
        
    def _initialize_with_better_bias(self):
        """Initialize with improved biases for better aspect and opinion detection"""
        # FIXED: More balanced initialization for aspects
        # For aspect classifier, moderate bias for B and I tags
        self.aspect_classifier[-1].bias.data[0] = 0.0   # O tag (no bias)
        self.aspect_classifier[-1].bias.data[1] = 0.7   # B tag (moderate positive bias - was 1.0)
        self.aspect_classifier[-1].bias.data[2] = 0.3   # I tag (slight positive bias - was 0.5)
        
        # FIXED: More balanced initialization for opinions
        # Similar for opinion classifier with more balanced values
        self.opinion_classifier[-1].bias.data[0] = 0.0  # O tag (no bias - was -0.1)
        self.opinion_classifier[-1].bias.data[1] = 0.8  # B tag (moderate positive bias - was 1.2)
        self.opinion_classifier[-1].bias.data[2] = 0.4  # I tag (slight positive bias - was 0.7)
    
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
            
            # FIXED: Apply post-processing to prevent common issues
            aspect_logits, opinion_logits = self._post_process_logits(
                aspect_logits, opinion_logits, attention_mask
            )
            
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
            
            # Create fallback with slight bias toward "O" tag (rather than all zeros)
            aspect_logits = torch.zeros(batch_size, seq_len, 3, device=device)
            aspect_logits[:, :, 0] = 0.7  # Bias toward O tag
            
            opinion_logits = torch.zeros(batch_size, seq_len, 3, device=device)
            opinion_logits[:, :, 0] = 0.7  # Bias toward O tag
            
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
        
        # FIXED: Improved noun phrase detection
        # Pattern: article/adjective + (optional adjective) + noun
        noun_phrases = re.findall(r'(the|a|an|this|that|my|our|their)?\s*(\w+\s+)?(\w+)', text_lower)
        
        # FIXED: Search for specific restaurant-related phrases
        aspect_phrases = [
            r'(the\s+)?food',
            r'(the\s+)?service',
            r'(the\s+)?atmosphere',
            r'(the\s+)?ambiance',
            r'(the\s+)?price',
            r'(the\s+)?value',
            r'(the\s+)?location',
            r'(the\s+)?menu',
            r'(the\s+)?portion',
            r'(the\s+)?staff',
            r'(the\s+)?taste',
            r'(the\s+)?flavor',
            r'(the\s+)?quality',
            r'(the\s+)?decor'
        ]
        
        detected_aspects = []
        for pattern in aspect_phrases:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                start, end = match.span()
                # Find tokens that fall into this span
                for i, token in enumerate(valid_tokens):
                    token_pos = text_lower.find(token, max(0, start-5), end+5)
                    if token_pos >= start and token_pos < end:
                        detected_aspects.append(valid_indices[i])
        
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
            
            # FIXED: More precise aspect term detection
            # Strong boost for aspect terms
            if token in self.aspect_terms or token_idx in detected_aspects:
                aspect_logits[token_idx, 1] += 2.5  # Strong but not overwhelming bias
                
                # Penalize this token as opinion
                opinion_logits[token_idx, 1] -= 1.0
                
                # If we have context (not at the end), check if next token should be part of span
                if i < len(valid_tokens) - 1:
                    next_token = valid_tokens[i+1]
                    next_idx = valid_indices[i+1]
                    if next_token not in self.stop_words and next_token not in ['.', ',', '?', '!']:
                        aspect_logits[next_idx, 2] += 1.5  # Moderate bias for I tag in next token
            
            # FIXED: More precise opinion term detection
            # Strong boost for opinion terms
            if token in self.opinion_terms:
                opinion_logits[token_idx, 1] += 2.5  # Strong but not overwhelming bias
                
                # Penalize this token as aspect
                aspect_logits[token_idx, 1] -= 1.0
                
                # If previous token is a negation, also mark it as part of the opinion
                if i > 0 and valid_tokens[i-1] in self.negation_terms:
                    prev_idx = valid_indices[i-1]
                    opinion_logits[prev_idx, 1] += 1.5  # Mark previous token as B tag
                    opinion_logits[token_idx, 2] += 2.0  # Change current to I tag
            
            # Additional rules for token position
            if i == 0 and token not in self.aspect_terms and token not in ['i', 'we', 'it', 'they']:
                # Penalize first token as aspect unless it's a common aspect term
                aspect_logits[token_idx, 1] -= 1.0
            
            # If token is in a noun phrase but not recognized aspect term, give moderate boost
            if is_in_noun_phrase and token not in self.stop_words:
                aspect_logits[token_idx, 1] += 0.8  # Moderate boost (was 1.0)
            
            # If token follows "is" or "was" and is an adjective-like, it's likely an opinion
            if i > 0 and valid_tokens[i-1] in ['is', 'was', 'are', 'were', 'feels', 'seemed', 'tastes']:
                opinion_logits[token_idx, 1] += 1.8  # Moderately strong boost (was 2.0)
                
            # Penalties for punctuation and stop words
            if token in ['.', ',', '!', '?', ')', '(', ':', ';']:
                aspect_logits[token_idx, 1] -= 5.0  # Strong negative for aspect
                opinion_logits[token_idx, 1] -= 5.0  # Strong negative for opinion
            
            # ADDED: Check for compound aspects (e.g., "chicken soup", "fried rice")
            if i > 0 and i < len(valid_tokens) - 1:
                compound = " ".join([valid_tokens[i-1], token])
                if compound.lower() in ["fried rice", "chicken soup", "ice cream", "egg roll", 
                                       "green tea", "spring roll", "hot dog", "french fries"]:
                    # Mark previous token as beginning
                    prev_idx = valid_indices[i-1]
                    aspect_logits[prev_idx, 1] += 3.0  # Strong bias for B tag
                    # Mark current token as inside
                    aspect_logits[token_idx, 2] += 3.0  # Strong bias for I tag
        
        return aspect_logits, opinion_logits
    
    def _apply_rules(self, text, aspect_logits, opinion_logits):
        """Simplified rule application without token information"""
        # Text-based rules using the raw text
        tokens = text.lower().split()
        
        # FIXED: More moderate boosts in the simple rule version
        for i, token in enumerate(tokens):
            if i >= aspect_logits.size(0):
                break
                
            # Boost for aspect terms
            if token in self.aspect_terms:
                aspect_logits[i, 1] += 2.5  # More moderate boost (was 5.0)
                # Add slight penalty for this being an opinion
                opinion_logits[i, 1] -= 1.0  # Prevent same token from being both
            
            # Boost for opinion terms
            if token in self.opinion_terms:
                opinion_logits[i, 1] += 2.5  # More moderate boost (was 5.0)
                # Add slight penalty for this being an aspect
                aspect_logits[i, 1] -= 1.0  # Prevent same token from being both
                
            # Add penalties for stop words
            if token in self.stop_words:
                aspect_logits[i, 1] -= 2.0
                opinion_logits[i, 1] -= 1.0
                
            # Prevent punctuation from being aspects or opinions
            if token in ['.', ',', '!', '?', ')', '(', ':', ';']:
                aspect_logits[i, 1] -= 10.0
                opinion_logits[i, 1] -= 10.0
                
            # Handle context - tokens after "is" or "are" likely opinions
            if i > 0 and tokens[i-1] in ["is", "are", "was", "were", "tastes", "seems", "looks"]:
                opinion_logits[i, 1] += 1.5
                
            # Boost for specific aspect patterns
            if i < len(tokens) - 1:
                two_gram = token + " " + tokens[i+1]
                if two_gram in ["chicken soup", "fried rice", "spring roll", "egg roll", 
                               "ice cream", "green tea", "sushi roll", "pad thai"]:
                    aspect_logits[i, 1] += 3.0  # Mark beginning
                    if i+1 < aspect_logits.size(0):
                        aspect_logits[i+1, 2] += 3.0  # Mark continuation
        
        return aspect_logits, opinion_logits
        
    def _post_process_logits(self, aspect_logits, opinion_logits, attention_mask=None):
        """
        Apply post-processing to prevent common issues like inconsistent BIO sequences
        and excessive detections
        """
        batch_size, seq_len, _ = aspect_logits.shape
        device = aspect_logits.device
        
        # Clone to avoid modifying the original tensors
        aspect_post = aspect_logits.clone()
        opinion_post = opinion_logits.clone()
        
        # Process each item in the batch
        for b in range(batch_size):
            # 1. Fix BIO inconsistencies (I tag without preceding B tag)
            # For aspects
            for i in range(1, seq_len):
                # If this is an I tag but previous token is not B or I
                if aspect_post[b, i, 2] > aspect_post[b, i, 0] and aspect_post[b, i, 2] > aspect_post[b, i, 1]:
                    # Check if previous token is not a B or I tag
                    prev_is_b = aspect_post[b, i-1, 1] > aspect_post[b, i-1, 0] and aspect_post[b, i-1, 1] > aspect_post[b, i-1, 2]
                    prev_is_i = aspect_post[b, i-1, 2] > aspect_post[b, i-1, 0] and aspect_post[b, i-1, 2] > aspect_post[b, i-1, 1]
                    
                    if not (prev_is_b or prev_is_i):
                        # Convert to B tag instead
                        aspect_post[b, i, 1] = aspect_post[b, i, 2] + 0.1  # Make B tag slightly stronger
                        aspect_post[b, i, 2] = aspect_post[b, i, 2] - 0.1  # Reduce I tag
            
            # For opinions
            for i in range(1, seq_len):
                # If this is an I tag but previous token is not B or I
                if opinion_post[b, i, 2] > opinion_post[b, i, 0] and opinion_post[b, i, 2] > opinion_post[b, i, 1]:
                    # Check if previous token is not a B or I tag
                    prev_is_b = opinion_post[b, i-1, 1] > opinion_post[b, i-1, 0] and opinion_post[b, i-1, 1] > opinion_post[b, i-1, 2]
                    prev_is_i = opinion_post[b, i-1, 2] > opinion_post[b, i-1, 0] and opinion_post[b, i-1, 2] > opinion_post[b, i-1, 1]
                    
                    if not (prev_is_b or prev_is_i):
                        # Convert to B tag instead
                        opinion_post[b, i, 1] = opinion_post[b, i, 2] + 0.1
                        opinion_post[b, i, 2] = opinion_post[b, i, 2] - 0.1
            
            # 2. Ensure we don't have too many detections (limit to most confident ones)
            if attention_mask is not None:
                # Count valid tokens
                valid_length = attention_mask[b].sum().item()
                # If more than half the tokens are predicted as aspects, reduce some
                aspect_preds = (aspect_post[b, :, 1] > aspect_post[b, :, 0]) & (aspect_post[b, :, 1] > aspect_post[b, :, 2])
                if aspect_preds.sum() > valid_length * 0.4:  # If >40% of tokens are aspects
                    # Find the least confident aspect predictions
                    aspect_b_scores = aspect_post[b, :, 1].clone()
                    aspect_b_scores[~aspect_preds] = float('inf')  # Ignore non-B tags
                    
                    # Get the threshold that would keep only top 30% of tokens
                    num_keep = int(valid_length * 0.3)
                    if num_keep > 0:
                        threshold = torch.topk(aspect_b_scores, num_keep, largest=False)[0][-1]
                        # Reduce confidence of aspects below threshold
                        reduce_mask = (aspect_b_scores < threshold) & (aspect_b_scores != float('inf'))
                        aspect_post[b, reduce_mask, 1] -= 1.0  # Reduce B tag confidence
                        aspect_post[b, reduce_mask, 0] += 0.5  # Increase O tag confidence
                
                # Similar for opinions
                opinion_preds = (opinion_post[b, :, 1] > opinion_post[b, :, 0]) & (opinion_post[b, :, 1] > opinion_post[b, :, 2])
                if opinion_preds.sum() > valid_length * 0.4:
                    opinion_b_scores = opinion_post[b, :, 1].clone()
                    opinion_b_scores[~opinion_preds] = float('inf')
                    
                    num_keep = int(valid_length * 0.3)
                    if num_keep > 0:
                        threshold = torch.topk(opinion_b_scores, num_keep, largest=False)[0][-1]
                        reduce_mask = (opinion_b_scores < threshold) & (opinion_b_scores != float('inf'))
                        opinion_post[b, reduce_mask, 1] -= 1.0
                        opinion_post[b, reduce_mask, 0] += 0.5
        
        return aspect_post, opinion_post