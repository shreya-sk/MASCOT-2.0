# src/models/generative_absa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMABSA(nn.Module):
    """
    Memory-efficient generative ABSA model with triplet-aware generation
    
    This 2025 implementation uses a two-stage pipeline:
    1. Extract aspect-opinion-sentiment triplets
    2. Generate explanations conditioned on triplets
    
    Key innovations:
    - Triplet-aware attention for aligning extraction and generation
    - Contrastive verification for ensuring faithfulness
    - Memory-efficient adapter techniques
    """
    def __init__(self, config):
        super().__init__()
        # Import components locally to avoid circular imports
        from src.models.embedding import LLMEmbedding
        from src.models.span_detector import SpanDetector
        from src.models.classifier import AspectOpinionJointClassifier
        from src.models.explanation_generator import ExplanationGenerator
        
        # First stage: Triplet extraction
        self.embeddings = LLMEmbedding(config)
        self.span_detector = SpanDetector(config)
        self.sentiment_classifier = AspectOpinionJointClassifier(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            dropout=config.dropout,
            num_classes=3
        )
        
        # Second stage: Explanation generation
        # Use a lightweight generator to save memory
        generator_name = getattr(config, 'generator_model', 'microsoft/phi-1_5')
        self.tokenizer = AutoTokenizer.from_pretrained(generator_name)
        
        # Enable generation if specified in config
        self.enable_generation = getattr(config, 'generate_explanations', True)
        
        if self.enable_generation:
            self.explanation_generator = ExplanationGenerator(config, self.tokenizer)
            
            # Initialize the contrastive verifier for semantic alignment
            self.use_contrastive_verification = getattr(config, 'use_contrastive_verification', True)
            if self.use_contrastive_verification:
                self._init_contrastive_verifier(config)
        
        # Initialize weights
        self._initialize_weights()
        
        # Add gradient checkpointing for memory efficiency
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', True)
        
    def _initialize_weights(self):
        """Initialize weights with small values to prevent exploding gradients"""
        for name, param in self.named_parameters():
            if 'embeddings' not in name and 'generator' not in name:  # Don't initialize pretrained weights
                if 'weight' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
                    
    def _init_contrastive_verifier(self, config):
        """Initialize contrastive verifier for semantic alignment"""
        # Use a small model for embedding triplets and generated text
        verifier_name = getattr(config, 'verifier_model', 'sentence-transformers/paraphrase-MiniLM-L6-v2')
        try:
            from transformers import AutoModel
            self.verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_name)
            self.verifier_model = AutoModel.from_pretrained(verifier_name)
            
            # Freeze verifier model to save memory
            for param in self.verifier_model.parameters():
                param.requires_grad = False
                
            print(f"Initialized contrastive verifier: {verifier_name}")
        except Exception as e:
            print(f"Failed to initialize contrastive verifier: {e}")
            self.use_contrastive_verification = False
    
    def _extract_hidden_states(self, embeddings_output):
        """Extract hidden states from embeddings output"""
        if isinstance(embeddings_output, dict):
            if 'hidden_states' in embeddings_output:
                return embeddings_output['hidden_states']
            elif 'last_hidden_state' in embeddings_output:
                return embeddings_output['last_hidden_state']
            elif 'aspect_embeddings' in embeddings_output:
                # Return aspect embeddings if hidden states not available
                return embeddings_output['aspect_embeddings']
            else:
                # Return the first value in the dict as a fallback
                return list(embeddings_output.values())[0]
        else:
            return embeddings_output
            
    def forward(self, input_ids, attention_mask, generate=False, filtered_triplets=None, **kwargs):
        """
        Forward pass with two-stage processing: extraction then generation
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            generate: Whether to generate explanations
            filtered_triplets: Optional pre-extracted triplets for generation
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing model outputs and optionally generated explanations
        """
        try:
            # Stage 1: Triplet extraction
            
            # Get embeddings
            embeddings_output = self.embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract hidden states
            hidden_states = self._extract_hidden_states(embeddings_output)
            
            # Detect aspect and opinion spans
            aspect_logits, opinion_logits, span_features, boundary_logits = self.span_detector(
                hidden_states,
                attention_mask
            )
            
            # Sentiment classification
            sentiment_logits, confidence_scores = self.sentiment_classifier(
                hidden_states,
                aspect_logits,
                opinion_logits,
                attention_mask
            )
            
            # Prepare output dictionary
            outputs = {
                'aspect_logits': aspect_logits,
                'opinion_logits': opinion_logits,
                'sentiment_logits': sentiment_logits,
                'confidence_scores': confidence_scores,
                'boundary_logits': boundary_logits
            }
            
            # Stage 2: Explanation generation (if enabled)
            if generate and self.enable_generation:
                # Extract predicted triplets if not provided
                triplets = filtered_triplets
                if triplets is None:
                    triplets = self._extract_triplets_batch(
                        aspect_logits, opinion_logits, sentiment_logits, 
                        input_ids, self.tokenizer
                    )
                
                # Generate explanations
                aspect_preds = aspect_logits.argmax(dim=-1)  # [batch_size, seq_len]
                opinion_preds = opinion_logits.argmax(dim=-1)  # [batch_size, seq_len]
                sentiment_preds = sentiment_logits.argmax(dim=-1)  # [batch_size]
                
                # Generate explanations
                explanation_logits = self.explanation_generator(
                    hidden_states,
                    aspect_preds, 
                    opinion_preds,
                    sentiment_preds,
                    attention_mask
                )
                
                outputs['explanations'] = explanation_logits
                
                # Generate structured explanations by aspect
                if hasattr(self.explanation_generator, 'generate_structured_explanation'):
                    structured_explanations = []
                    for batch_triplets in triplets:
                        explanation = self.explanation_generator.generate_structured_explanation(
                            hidden_states, batch_triplets, attention_mask
                        )
                        structured_explanations.append(explanation)
                    outputs['structured_explanations'] = structured_explanations
                
                # Add contrastive verification if enabled
                if self.use_contrastive_verification and 'explanation_targets' in kwargs:
                    explanation_targets = kwargs['explanation_targets']
                    verification_loss = self._compute_verification_loss(
                        triplets, explanation_targets, input_ids
                    )
                    outputs['verification_loss'] = verification_loss
            
            return outputs
            
        except Exception as e:
            print(f"Error in generative ABSA forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Return dummy outputs
            batch_size, seq_len = input_ids.size()
            device = input_ids.device
            
            # Create fallback outputs
            dummy_outputs = {
                'aspect_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                'opinion_logits': torch.zeros(batch_size, seq_len, 3, device=device),
                'sentiment_logits': torch.zeros(batch_size, 3, device=device),
                'confidence_scores': torch.ones(batch_size, 1, device=device),
                'boundary_logits': torch.zeros(batch_size, seq_len, 2, device=device)
            }
            
            if generate and self.enable_generation:
                # Add dummy explanation logits
                vocab_size = getattr(self.tokenizer, 'vocab_size', 32000)
                max_length = getattr(self, 'max_length', 32)
                dummy_outputs['explanations'] = torch.zeros(batch_size, max_length, vocab_size, device=device)
                
            return dummy_outputs
            
    def _extract_triplets_batch(self, aspect_logits, opinion_logits, sentiment_logits, input_ids, tokenizer):
        """Extract triplets from batch predictions"""
        batch_size = aspect_logits.size(0)
        all_triplets = []
        
        for b in range(batch_size):
            # Convert logits to predictions
            aspect_preds = aspect_logits[b].argmax(dim=-1)  # [seq_len]
            opinion_preds = opinion_logits[b].argmax(dim=-1)  # [seq_len]
            sentiment_pred = sentiment_logits[b].argmax(dim=-1).item()  # scalar
            
            # Map sentiment ID to label
            sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
            sentiment = sentiment_map[sentiment_pred]
            
            # Extract aspect spans
            aspect_spans = []
            current_span = []
            for i, pred in enumerate(aspect_preds):
                if pred == 1:  # B tag
                    if current_span:
                        aspect_spans.append(current_span)
                    current_span = [i]
                elif pred == 2:  # I tag
                    if current_span:
                        current_span.append(i)
                else:  # O tag
                    if current_span:
                        aspect_spans.append(current_span)
                        current_span = []
            if current_span:
                aspect_spans.append(current_span)
            
            # Extract opinion spans
            opinion_spans = []
            current_span = []
            for i, pred in enumerate(opinion_preds):
                if pred == 1:  # B tag
                    if current_span:
                        opinion_spans.append(current_span)
                    current_span = [i]
                elif pred == 2:  # I tag
                    if current_span:
                        current_span.append(i)
                else:  # O tag
                    if current_span:
                        opinion_spans.append(current_span)
                        current_span = []
            if current_span:
                opinion_spans.append(current_span)
            
            # Create triplets
            batch_triplets = []
            for aspect_span in aspect_spans:
                for opinion_span in opinion_spans:
                    # Decode spans to text
                    aspect_text = self._decode_span(input_ids[b], aspect_span, tokenizer)
                    opinion_text = self._decode_span(input_ids[b], opinion_span, tokenizer)
                    
                    triplet = {
                        'aspect': aspect_text,
                        'aspect_indices': aspect_span,
                        'opinion': opinion_text,
                        'opinion_indices': opinion_span,
                        'sentiment': sentiment,
                        'confidence': 0.9  # Default confidence
                    }
                    batch_triplets.append(triplet)
            
            all_triplets.append(batch_triplets)
        
        return all_triplets
        
    def _decode_span(self, input_ids, span_indices, tokenizer):
        """Decode a token span to text"""
        # Get token IDs for this span
        span_token_ids = [input_ids[i].item() for i in span_indices]
        
        # Decode to text
        try:
            # Use tokenizer to decode
            text = tokenizer.decode(span_token_ids, skip_special_tokens=True)
            return text.strip()
        except:
            # Fallback if decoding fails
            return f"Span at indices {span_indices}"
    
    def _compute_verification_loss(self, triplets, explanation_targets, input_ids):
        """
        Compute contrastive verification loss between triplets and explanations
        
        This ensures semantic alignment between extracted triplets and
        generated explanations for improved faithfulness.
        """
        if not self.use_contrastive_verification or not hasattr(self, 'verifier_model'):
            return torch.tensor(0.0, device=input_ids.device)
            
        try:
            batch_size = input_ids.size(0)
            device = input_ids.device
            
            # Convert triplets to text for embedding
            triplet_texts = []
            for batch_triplets in triplets:
                # Combine all triplets into a single text
                text = ""
                for triplet in batch_triplets:
                    aspect = triplet['aspect']
                    opinion = triplet['opinion']
                    sentiment = triplet['sentiment']
                    text += f"The {aspect} is {sentiment.lower()} because of the {opinion}. "
                triplet_texts.append(text)
            
            # Convert explanation targets to text
            explanation_texts = []
            for explanation in explanation_targets:
                # Decode explanation tokens
                text = self.tokenizer.decode(explanation, skip_special_tokens=True)
                explanation_texts.append(text)
            
            # Embed triplet texts
            triplet_inputs = self.verifier_tokenizer(
                triplet_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            ).to(device)
            
            triplet_embeddings = self.verifier_model(**triplet_inputs).last_hidden_state[:, 0]  # CLS token
            
            # Embed explanation texts
            explanation_inputs = self.verifier_tokenizer(
                explanation_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            ).to(device)
            
            explanation_embeddings = self.verifier_model(**explanation_inputs).last_hidden_state[:, 0]  # CLS token
            
            # Normalize embeddings
            triplet_embeddings = F.normalize(triplet_embeddings, p=2, dim=1)
            explanation_embeddings = F.normalize(explanation_embeddings, p=2, dim=1)
            
            # Compute cosine similarity
            similarities = torch.matmul(triplet_embeddings, explanation_embeddings.transpose(0, 1))
            
            # Compute contrastive loss (InfoNCE)
            labels = torch.arange(batch_size, device=device)
            loss = F.cross_entropy(similarities / 0.07, labels)  # Temperature = 0.07
            
            return loss
        except Exception as e:
            print(f"Error computing verification loss: {e}")
            return torch.tensor(0.0, device=input_ids.device)
            
    def generate_explanation(self, input_ids, attention_mask):
        """
        Generate explanations for input text
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            List of generated explanations for each item in batch
        """
        if not self.enable_generation:
            return ["Generation not enabled" for _ in range(input_ids.size(0))]
            
        # Get model outputs
        outputs = self.forward(input_ids, attention_mask, generate=True)
        
        # Extract triplets
        triplets = self._extract_triplets_batch(
            outputs['aspect_logits'], 
            outputs['opinion_logits'], 
            outputs['sentiment_logits'],
            input_ids, 
            self.tokenizer
        )
        
        # Generate explanations by aspect
        explanations = []
        for batch_triplets in triplets:
            # Group by aspect for structured explanation
            aspect_groups = {}
            for t in batch_triplets:
                aspect = t['aspect']
                if aspect not in aspect_groups:
                    aspect_groups[aspect] = []
                aspect_groups[aspect].append(t)
            
            # Generate structured explanation
            text = ""
            for aspect, aspect_triplets in aspect_groups.items():
                # Get majority sentiment
                sentiments = [t['sentiment'] for t in aspect_triplets]
                majority_sentiment = max(set(sentiments), key=sentiments.count)
                sentiment_text = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}[majority_sentiment]
                
                # Get opinions for this aspect
                opinions = [t['opinion'] for t in aspect_triplets]
                opinion_text = ", ".join(opinions)
                
                text += f"The aspect '{aspect}' is {sentiment_text} because of {opinion_text}. "
            
            explanations.append(text if text else "No aspects detected.")
        
        return explanations
        
    def compute_faithfulness(self, triplets, explanation):
        """
        Compute faithfulness score between triplets and generated explanation
        
        Args:
            triplets: List of extracted triplets
            explanation: Generated explanation text
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not self.use_contrastive_verification or not hasattr(self, 'verifier_model'):
            return 0.5  # Default score if verification not enabled
            
        try:
            # Convert triplets to text
            triplet_text = ""
            for triplet in triplets:
                aspect = triplet['aspect']
                opinion = triplet['opinion'] 
                sentiment = triplet['sentiment']
                triplet_text += f"The {aspect} is {sentiment.lower()} because of the {opinion}. "
            
            # Embed triplet text
            device = next(self.verifier_model.parameters()).device
            triplet_inputs = self.verifier_tokenizer(
                triplet_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            ).to(device)
            
            triplet_embedding = self.verifier_model(**triplet_inputs).last_hidden_state[:, 0]  # CLS token
            
            # Embed explanation text
            explanation_inputs = self.verifier_tokenizer(
                explanation, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=128
            ).to(device)
            
            explanation_embedding = self.verifier_model(**explanation_inputs).last_hidden_state[:, 0]  # CLS token
            
            # Normalize embeddings
            triplet_embedding = F.normalize(triplet_embedding, p=2, dim=1)
            explanation_embedding = F.normalize(explanation_embedding, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.matmul(triplet_embedding, explanation_embedding.transpose(0, 1)).item()
            
            return max(0, min(1, similarity))  # Clamp to [0, 1]
        except Exception as e:
            print(f"Error computing faithfulness: {e}")
            return 0.5  # Default score on error