# src/models/classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AspectOpinionJointClassifier(nn.Module):
    """
    Improved aspect-opinion joint classifier with context-aware sentiment prediction
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1, num_classes=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Enhanced attention for aspects
        self.aspect_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Enhanced attention for opinions
        self.opinion_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
        # Feature fusion - adapt to the actual input dimensions
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # This handles combining the two vectors
            nn.GELU(),
            nn.Dropout(dropout)
        )
            
        # Sentiment classifier with BALANCED initialization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # FIXED: Initialize sentiment classifier with truly balanced class bias
        # Equal bias for all sentiment classes (no bias toward negative)
        self.classifier[-1].bias.data[0] = 0.1  # Positive sentiment (was 0.2)
        self.classifier[-1].bias.data[1] = 0.1  # Neutral sentiment (was 0.0)
        self.classifier[-1].bias.data[2] = 0.1  # Negative sentiment (was -0.1)
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Word lists for rule-based sentiment detection
        self.negative_words = set([
            'bad', 'terrible', 'poor', 'awful', 'mediocre', 'disappointing', 
            'horrible', 'disgusting', 'unpleasant', 'worst', 'cold', 'slow',
            'overpriced', 'expensive', 'rude', 'unfriendly', 'dirty', 'noisy',
            'bland', 'tasteless', 'burnt', 'bitter', 'stale', 'greasy', 'salty',
            'tough', 'dry', 'chewy', 'undercooked', 'overcooked', 'lukewarm',
            'small', 'tiny', 'skimpy', 'messy', 'chaotic', 'crowded', 'loud',
            'mistake', 'error', 'wrong', 'unhappy', 'upset', 'annoyed', 'angry',
            'waste', 'pricey', 'steep', 'outrageous', 'ridiculous', 'absurd',
            'avoid', 'never', 'skip', 'pass', 'problem', 'issue', 'complaint'
        ])
        
        self.positive_words = set([
            'good', 'great', 'excellent', 'amazing', 'fantastic', 'wonderful',
            'delicious', 'tasty', 'fresh', 'friendly', 'helpful', 'nice',
            'best', 'perfect', 'favorite', 'loved', 'enjoy', 'clean', 'quick',
            'outstanding', 'superb', 'exceptional', 'stellar', 'divine', 'heavenly',
            'delightful', 'sublime', 'impressive', 'extraordinary', 'phenomenal',
            'remarkable', 'splendid', 'marvelous', 'exquisite', 'delectable',
            'flavorful', 'savory', 'juicy', 'tender', 'succulent', 'authentic',
            'generous', 'huge', 'large', 'massive', 'ample', 'abundant', 'plentiful',
            'attentive', 'courteous', 'professional', 'knowledgeable', 'prompt',
            'efficient', 'speedy', 'fast', 'quick', 'reasonable', 'affordable',
            'worth', 'value', 'bargain', 'deal', 'recommend', 'return', 'satisfied'
        ])
        
        # Negation words that can flip sentiment
        self.negation_words = set([
            'not', 'no', 'never', 'none', 'neither', 'nor', "n't", 'without',
            'barely', 'hardly', 'rarely', 'seldom', 'cannot', "can't", "didn't",
            "doesn't", "don't", "wasn't", "weren't", "wouldn't", "couldn't",
            "shouldn't", "isn't", "aren't", "hasn't", "haven't", "won't"
        ])
        
        # Intensifiers that can strengthen sentiment
        self.intensifiers = set([
            'very', 'really', 'extremely', 'incredibly', 'absolutely', 'truly',
            'completely', 'totally', 'utterly', 'particularly', 'especially',
            'exceptionally', 'remarkably', 'notably', 'decidedly', 'genuinely',
            'thoroughly', 'entirely', 'fully', 'highly', 'greatly', 'immensely',
            'intensely', 'exceedingly', 'supremely', 'terribly', 'awfully',
            'insanely', 'super', 'so', 'too', 'quite', 'rather', 'pretty'
        ])
        
    def forward(self, hidden_states, aspect_logits=None, opinion_logits=None, 
                attention_mask=None, sentiment_labels=None, input_ids=None, tokenizer=None):
        """
        Forward pass through the sentiment classifier with enhanced error handling
        
        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_dim]
            aspect_logits: Aspect logits [batch_size, seq_len, 3]
            opinion_logits: Opinion logits [batch_size, seq_len, 3]
            attention_mask: Attention mask [batch_size, seq_len]
            sentiment_labels: Optional sentiment labels
            input_ids: Optional input token IDs for rule-based enhancement
            tokenizer: Optional tokenizer for decoding token IDs
            
        Returns:
            tuple: (sentiment_logits, confidence_scores)
        """
        try:
            # Verify input dimensions
            if hidden_states is None:
                raise ValueError("hidden_states is None")
                
            batch_size, seq_len, hidden_dim = hidden_states.shape
            device = hidden_states.device
            
            # Create aspect and opinion weights from logits or attention
            # For aspect weights
            if aspect_logits is not None:
                # Only consider B and I tags (indices 1 and 2)
                aspect_weights = torch.softmax(aspect_logits[:, :, 1:], dim=-1).sum(-1)  # [batch_size, seq_len]
            else:
                # Use attention if logits not available
                aspect_attn = self.aspect_attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
                if attention_mask is not None:
                    aspect_attn = aspect_attn.masked_fill(attention_mask == 0, -1e10)
                aspect_weights = F.softmax(aspect_attn, dim=-1)
            
            # For opinion weights
            if opinion_logits is not None:
                opinion_weights = torch.softmax(opinion_logits[:, :, 1:], dim=-1).sum(-1)
            else:
                opinion_attn = self.opinion_attention(hidden_states).squeeze(-1)
                if attention_mask is not None:
                    opinion_attn = opinion_attn.masked_fill(attention_mask == 0, -1e10)
                opinion_weights = F.softmax(opinion_attn, dim=-1)
            
            # Apply weights to get weighted representations
            aspect_weights = aspect_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            opinion_weights = opinion_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Handle cases where weights are all zero (apply uniform weights)
            if attention_mask is not None:
                # Use attention mask to create uniform weights for non-padding tokens
                mask_expanded = attention_mask.float().unsqueeze(-1)  # [batch_size, seq_len, 1]
                aspect_sum = aspect_weights.sum(dim=1, keepdim=True)
                opinion_sum = opinion_weights.sum(dim=1, keepdim=True)
                
                # If sum is close to zero, use uniform weights
                aspect_weights = torch.where(
                    aspect_sum > 1e-6, 
                    aspect_weights, 
                    mask_expanded / mask_expanded.sum(dim=1, keepdim=True).clamp(min=1e-6)
                )
                opinion_weights = torch.where(
                    opinion_sum > 1e-6, 
                    opinion_weights, 
                    mask_expanded / mask_expanded.sum(dim=1, keepdim=True).clamp(min=1e-6)
                )
            
            # Weight hidden states with attention weights
            aspect_repr = (hidden_states * aspect_weights).sum(dim=1)  # [batch_size, hidden_dim]
            opinion_repr = (hidden_states * opinion_weights).sum(dim=1)  # [batch_size, hidden_dim]
            
            # Check input dimensions
            if hidden_dim != self.input_dim:
                print(f"Warning: Hidden dim {hidden_dim} doesn't match expected input dim {self.input_dim}")
                if hidden_dim > self.input_dim:
                    # Truncate to expected size
                    aspect_repr = aspect_repr[:, :self.input_dim]
                    opinion_repr = opinion_repr[:, :self.input_dim]
                else:
                    # Pad to expected size
                    aspect_pad = torch.zeros(batch_size, self.input_dim - hidden_dim, device=device)
                    opinion_pad = torch.zeros(batch_size, self.input_dim - hidden_dim, device=device)
                    aspect_repr = torch.cat([aspect_repr, aspect_pad], dim=1)
                    opinion_repr = torch.cat([opinion_repr, opinion_pad], dim=1)
                    
            # Ensure no NaN or Inf values in representations
            aspect_repr = torch.nan_to_num(aspect_repr)
            opinion_repr = torch.nan_to_num(opinion_repr)
            
            # Combine aspect and opinion representations
            try:
                combined = torch.cat([aspect_repr, opinion_repr], dim=-1)
                
                # Apply fusion
                fused = self.fusion(combined)  # [batch_size, hidden_dim]
                
                # Basic sentiment classification
                sentiment_logits = self.classifier(fused)  # [batch_size, num_classes]
                
                # FIXED: Apply rule-based sentiment analysis if input_ids and tokenizer are available
                if input_ids is not None and tokenizer is not None:
                    sentiment_logits = self._apply_sentiment_rules(
                        input_ids, tokenizer, aspect_weights, opinion_weights, sentiment_logits
                    )
                
                # Compute confidence scores
                confidence = self.confidence_estimator(fused)  # [batch_size, 1]
                
                return sentiment_logits, confidence
            except Exception as e:
                print(f"Error in sentiment classification: {e}")
                # Create fallback sentiment logits (balanced)
                sentiment_logits = torch.ones(batch_size, self.num_classes, device=device) / self.num_classes
                confidence = torch.ones(batch_size, 1, device=device) * 0.5
                
                return sentiment_logits, confidence
                
        except Exception as e:
            print(f"Critical error in classifier forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback outputs with correct dimensions
            batch_size = hidden_states.size(0)
            device = hidden_states.device
            
            # Create fallback sentiment logits (balanced)
            sentiment_logits = torch.ones(batch_size, self.num_classes, device=device) / self.num_classes
            confidence = torch.ones(batch_size, 1, device=device) * 0.5
            
            return sentiment_logits, confidence
            
    def _apply_sentiment_rules(self, input_ids, tokenizer, aspect_weights, opinion_weights, sentiment_logits):
        """Apply enhanced rule-based sentiment analysis"""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Create a copy of sentiment_logits to modify
        modified_logits = sentiment_logits.clone()
        
        for b in range(batch_size):
            # Get the tokens for this example
            tokens = [tokenizer.decode([id_val.item()]).lower().strip() for id_val in input_ids[b]]
            
            # Find aspect words in this example (ADDED: also check aspect words)
            aspect_indices = torch.where(aspect_weights[b, :, 0] > 0.1)[0].cpu().tolist()
            
            # Find opinion words in this example
            opinion_indices = torch.where(opinion_weights[b, :, 0] > 0.1)[0].cpu().tolist()
            
            # Count positive and negative terms
            pos_count = 0
            neg_count = 0
            
            # Check each opinion token
            for idx in opinion_indices:
                if idx < len(tokens):
                    token = tokens[idx]
                    
                    # Check for negation words before this opinion
                    has_negation = False
                    for prev_idx in range(max(0, idx-3), idx):
                        if prev_idx < len(tokens) and tokens[prev_idx] in self.negation_words:
                            has_negation = True
                            break
                    
                    # Adjust sentiment counts based on token and negation
                    if token in self.positive_words:
                        if has_negation:
                            neg_count += 1
                        else:
                            pos_count += 1
                    
                    if token in self.negative_words:
                        if has_negation:
                            pos_count += 1
                        else:
                            neg_count += 1
                    
                    # Check for intensifiers before this opinion
                    for prev_idx in range(max(0, idx-2), idx):
                        if prev_idx < len(tokens) and tokens[prev_idx] in self.intensifiers:
                            # Add extra weight for intensified opinions
                            if token in self.positive_words and not has_negation:
                                pos_count += 0.5
                            elif token in self.negative_words and not has_negation:  
                                neg_count += 0.5
                            elif token in self.positive_words and has_negation:
                                neg_count += 0.5
                            elif token in self.negative_words and has_negation:
                                pos_count += 0.5
            
            # FIXED: Better rule-based sentiment adjustment
            # Determine the dominant sentiment based on counts
            if pos_count > neg_count:
                # Positive sentiment dominates
                # Boost positive (index 0), but don't completely override the model
                modified_logits[b, 0] += min(pos_count, 2.0)  # Cap the boost
                # Slightly reduce negative
                modified_logits[b, 2] -= min(pos_count * 0.5, 1.0)
            elif neg_count > pos_count:
                # Negative sentiment dominates 
                # Boost negative (index 2)
                modified_logits[b, 2] += min(neg_count, 2.0)
                # Slightly reduce positive
                modified_logits[b, 0] -= min(neg_count * 0.5, 1.0)
            else:
                # No clear dominance, slightly boost neutral (index 1)
                modified_logits[b, 1] += 0.5
            
            # ADDED: Detect specific sentiment patterns
            # Check for common expressions that strongly indicate sentiment
            text = " ".join(tokens).lower()
            
            # Strong positive indicators
            if any(phrase in text for phrase in ["highly recommend", "loved", "excellent", "favorite", "best"]):
                modified_logits[b, 0] += 1.5  # Strong positive boost
            
            # Strong negative indicators
            if any(phrase in text for phrase in ["terrible", "worst", "awful", "avoid", "never again"]):
                modified_logits[b, 2] += 1.5  # Strong negative boost
            
            # Neutral/mixed indicators
            if any(phrase in text for phrase in ["average", "okay", "mixed", "but", "however"]):
                modified_logits[b, 1] += 1.0  # Boost neutral sentiment
                
        return modified_logits

    def _decode_span(self, input_ids, span_indices, tokenizer):
        """Helper method to decode token spans to text"""
        try:
            if not span_indices:
                return ""
                
            # Get token IDs for this span
            span_token_ids = [input_ids[i].item() for i in span_indices if i < len(input_ids)]
            
            # Decode to text
            text = tokenizer.decode(span_token_ids, skip_special_tokens=True)
            return text.strip()
        except Exception as e:
            print(f"Error decoding span: {e}")
            return ""