# src/training/metrics.py
import torch # type: ignore
import numpy as np 
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModel, AutoTokenizer

class TripletRecoveryMetric:
    """
    Novel metric introduced in 2025 for evaluating faithfulness of generated explanations
    
    This metric measures how accurately the original triplets can be recovered
    from the generated explanations, ensuring semantic alignment between
    extraction and generation.
    """
    def __init__(self, config):
        # Initialize embedding model for semantic similarity
        self.embed_model_name = getattr(config, 'metric_embed_model', 'sentence-transformers/paraphrase-MiniLM-L6-v2')
        
        # Load model for triplet recovery
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
            self.model = AutoModel.from_pretrained(self.embed_model_name)
            
            # Use GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            
            print(f"Loaded triplet recovery metric model: {self.embed_model_name}")
            self.model_available = True
        except Exception as e:
            print(f"Failed to load triplet recovery metric model: {e}")
            self.model_available = False
        
        # Thresholds for recovery
        self.similarity_threshold = 0.75
        
        # Initialize extractive ABSA model for triplet recovery
        self.absa_recovery_model = None
        absa_model_name = getattr(config, 'recovery_model', None)
        if absa_model_name:
            self._init_recovery_model(absa_model_name)
    
    def _init_recovery_model(self, model_name):
        """Initialize extractive ABSA model for triplet recovery"""
        try:
            from src.models.absa import LLMABSA
            
            # Simplified config for recovery model
            recovery_config = type('Config', (), {
                'model_name': model_name,
                'hidden_size': 256,
                'dropout': 0.1,
                'use_syntax': False
            })
            
            # Create recovery model
            self.absa_recovery_model = LLMABSA(recovery_config)
            self.absa_recovery_model = self.absa_recovery_model.to(self.device)
            self.absa_recovery_model.eval()
            
            # Load tokenizer
            self.recovery_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"Loaded ABSA recovery model: {model_name}")
        except Exception as e:
            print(f"Failed to load ABSA recovery model: {e}")
            self.absa_recovery_model = None
    
    def compute_embeddings(self, texts):
        """Compute embeddings for a list of texts"""
        if not self.model_available:
            return None
            
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Compute embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        # Normalize embeddings
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings_norm
    
    def extract_triplets_from_explanation(self, explanation):
        """Extract triplets from generated explanation using recovery model"""
        if self.absa_recovery_model is None or self.recovery_tokenizer is None:
            return []
            
        # Tokenize explanation
        inputs = self.recovery_tokenizer(
            explanation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.absa_recovery_model(**inputs)
            
            aspect_logits = outputs['aspect_logits'][0]
            opinion_logits = outputs['opinion_logits'][0]
            sentiment_logits = outputs['sentiment_logits'][0]
            
            # Get predictions
            aspect_preds = aspect_logits.argmax(dim=-1).cpu().numpy()
            opinion_preds = opinion_logits.argmax(dim=-1).cpu().numpy()
            sentiment_pred = sentiment_logits.argmax(dim=-1).item()
            
            # Extract spans
            aspect_spans = self._extract_spans(aspect_preds)
            opinion_spans = self._extract_spans(opinion_preds)
            
            # Map sentiment
            sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
            sentiment = sentiment_map.get(sentiment_pred, 'NEU')
            
            # Create triplets
            recovered_triplets = []
            for aspect_span in aspect_spans:
                for opinion_span in opinion_spans:
                    # Get token IDs for spans
                    aspect_ids = inputs['input_ids'][0, aspect_span].cpu().numpy()
                    opinion_ids = inputs['input_ids'][0, opinion_span].cpu().numpy()
                    
                    # Decode text
                    try:
                        aspect_text = self.recovery_tokenizer.decode(aspect_ids)
                        opinion_text = self.recovery_tokenizer.decode(opinion_ids)
                        
                        recovered_triplets.append({
                            'aspect': aspect_text,
                            'opinion': opinion_text,
                            'sentiment': sentiment
                        })
                    except:
                        continue
            
            return recovered_triplets
    
    def _extract_spans(self, predictions):
        """Extract spans from BIO predictions"""
        spans = []
        current_span = []
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # B tag
                if current_span:
                    spans.append(current_span)
                current_span = [i]
            elif pred == 2:  # I tag
                if current_span:
                    current_span.append(i)
            else:  # O tag
                if current_span:
                    spans.append(current_span)
                    current_span = []
        
        # Add last span if exists
        if current_span:
            spans.append(current_span)
        
        return spans
    
    def compute_triplet_recovery(self, original_triplets, generated_explanations):
        """
        Compute triplet recovery metric
        
        Args:
            original_triplets: List of original triplets extracted by model
            generated_explanations: List of generated explanations
            
        Returns:
            Dictionary of metrics (precision, recall, F1)
        """
        if not self.model_available:
            return {'recovery_precision': 0.0, 'recovery_recall': 0.0, 'recovery_f1': 0.0}
            
        # Convert triplets to text representations
        original_texts = []
        for triplets in original_triplets:
            batch_texts = []
            for triplet in triplets:
                aspect = triplet.get('aspect', '')
                opinion = triplet.get('opinion', '')
                sentiment = triplet.get('sentiment', 'NEU')
                text = f"{aspect} {opinion} {sentiment}"
                batch_texts.append(text)
            original_texts.append(batch_texts)
        
        # Extract triplets from explanations
        recovered_triplets = []
        for explanation in generated_explanations:
            if self.absa_recovery_model is not None:
                # Use recovery model
                triplets = self.extract_triplets_from_explanation(explanation)
            else:
                # Simple heuristic extraction
                triplets = self._extract_triplets_heuristic(explanation)
            recovered_triplets.append(triplets)
        
        # Convert recovered triplets to text
        recovered_texts = []
        for triplets in recovered_triplets:
            batch_texts = []
            for triplet in triplets:
                aspect = triplet.get('aspect', '')
                opinion = triplet.get('opinion', '')
                sentiment = triplet.get('sentiment', 'NEU')
                text = f"{aspect} {opinion} {sentiment}"
                batch_texts.append(text)
            recovered_texts.append(batch_texts)
        
        # Compute metrics
        precisions = []
        recalls = []
        f1_scores = []
        
        for orig_batch, rec_batch in zip(original_texts, recovered_texts):
            if not orig_batch or not rec_batch:
                # Skip empty batches
                continue
                
            # Compute semantic similarity matrix
            orig_embeddings = self.compute_embeddings(orig_batch)
            rec_embeddings = self.compute_embeddings(rec_batch)
            
            if orig_embeddings is None or rec_embeddings is None:
                continue
                
            # Compute cosine similarity
            similarity = torch.matmul(orig_embeddings, rec_embeddings.transpose(0, 1))
            
            # Calculate precision and recall based on similarity
            # A triplet is recovered if its similarity with any original triplet is above threshold
            recovered_count = (similarity.max(dim=0)[0] >= self.similarity_threshold).sum().item()
            matched_count = (similarity.max(dim=1)[0] >= self.similarity_threshold).sum().item()
            
            # Calculate metrics
            precision = recovered_count / len(rec_batch) if rec_batch else 0
            recall = matched_count / len(orig_batch) if orig_batch else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # Average metrics across batches
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        
        return {
            'recovery_precision': avg_precision,
            'recovery_recall': avg_recall,
            'recovery_f1': avg_f1
        }
    
    def _extract_triplets_heuristic(self, explanation):
        """Extract triplets from explanation using simple heuristics"""
        triplets = []
        
        # Common patterns in explanations
        patterns = [
            r"the (.*?) is (positive|negative|neutral) because of (.*?)[.,]",
            r"(.*?) is (positive|negative|neutral) due to (.*?)[.,]",
            r"(.*?) has a (positive|negative|neutral) sentiment because (.*?)[.,]"
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, explanation.lower())
            for match in matches:
                if len(match) >= 3:
                    aspect = match[0].strip()
                    sentiment = match[1].strip().upper()[:3]  # Convert to POS/NEU/NEG
                    opinion = match[2].strip()
                    
                    triplets.append({
                        'aspect': aspect,
                        'opinion': opinion,
                        'sentiment': sentiment
                    })
        
        return triplets

class ABSAMetrics:
    """
    Comprehensive evaluation metrics for ABSA tasks with triplet recovery
    
    This enhanced 2025 implementation provides standard metrics for aspect and opinion
    extraction, plus advanced metrics for triplet recovery and faithfulness.
    """
    def __init__(self, config=None):
        self.reset()
        
        # Initialize triplet recovery metric
        self.triplet_recovery = TripletRecoveryMetric(config) if config else None
        
        # Track original triplets and generated explanations
        self.original_triplets = []
        self.generated_explanations = []
        
    def reset(self):
        """Reset all metrics"""
        self.aspect_preds = []
        self.opinion_preds = []
        self.sentiment_preds = []
        self.aspect_labels = []
        self.opinion_labels = []
        self.sentiment_labels = []
        
        self.original_triplets = []
        self.generated_explanations = []
        
    def update(self, outputs, targets, generated=None, triplets=None):
        """
        Update metrics with batch predictions
        
        Args:
            outputs: Model output dictionary
            targets: Target dictionary
            generated: Optional generated explanations
            triplets: Optional extracted triplets
        """
        # Get predictions
        aspect_pred = outputs['aspect_logits'].argmax(dim=-1).cpu().numpy()
        opinion_pred = outputs['opinion_logits'].argmax(dim=-1).cpu().numpy()
        sentiment_pred = outputs['sentiment_logits'].argmax(dim=-1).cpu().numpy()
        
        # Get labels
        aspect_label = targets['aspect_labels'].cpu().numpy()
        opinion_label = targets['opinion_labels'].cpu().numpy()
        sentiment_label = targets['sentiment_labels'].cpu().numpy()
        
        # Update lists
        self.aspect_preds.extend(aspect_pred)
        self.opinion_preds.extend(opinion_pred)
        self.sentiment_preds.extend(sentiment_pred)
        self.aspect_labels.extend(aspect_label)
        self.opinion_labels.extend(opinion_label)
        self.sentiment_labels.extend(sentiment_label)
        
        # Track triplets and explanations for recovery metric
        if triplets is not None:
            self.original_triplets.extend(triplets)
        
        if generated is not None:
            self.generated_explanations.extend(generated)

    def compute(self):
        """Compute all metrics"""
        try:
            metrics = {}
            
            # Compute span detection metrics
            aspect_metrics = self._compute_span_metrics(
                self.aspect_preds, 
                self.aspect_labels,
                prefix='aspect'
            )
            
            opinion_metrics = self._compute_span_metrics(
                self.opinion_preds,
                self.opinion_labels,
                prefix='opinion'
            )
            
            # Compute sentiment classification metrics
            sentiment_metrics = self._compute_sentiment_metrics(
                self.sentiment_preds,
                self.sentiment_labels
            )
            
            metrics.update(aspect_metrics)
            metrics.update(opinion_metrics)
            metrics.update(sentiment_metrics)
            
            # Compute overall F1
            metrics['overall_f1'] = (
                metrics.get('aspect_f1', 0) +
                metrics.get('opinion_f1', 0) + 
                metrics.get('sentiment_f1', 0)
            ) / 3
            
            # Compute triplet recovery metrics if available
            if (self.triplet_recovery is not None and 
                self.original_triplets and self.generated_explanations):
                
                recovery_metrics = self.triplet_recovery.compute_triplet_recovery(
                    self.original_triplets,
                    self.generated_explanations
                )
                metrics.update(recovery_metrics)
            
            return metrics
        except Exception as e:
            print(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
            
            # Return default metrics
            return {
                'aspect_precision': 0.0,
                'aspect_recall': 0.0,
                'aspect_f1': 0.0,
                'opinion_precision': 0.0,
                'opinion_recall': 0.0,
                'opinion_f1': 0.0,
                'sentiment_precision': 0.0,
                'sentiment_recall': 0.0,
                'sentiment_f1': 0.0,
                'overall_f1': 0.0,
                'recovery_precision': 0.0,
                'recovery_recall': 0.0,
                'recovery_f1': 0.0
            }
    
    def _compute_span_metrics(self, preds, labels, prefix):
        """Compute precision, recall, F1 for span detection"""
        # Flatten predictions and labels
        flat_preds = []
        flat_labels = []
        
        for batch_preds, batch_labels in zip(preds, labels):
            # Handle multiple spans if needed
            if len(batch_labels.shape) > 1:
                batch_labels = batch_labels.max(axis=0)
            
            # Only consider valid tokens (not padding)
            valid_mask = batch_labels != -100
            
            # Extract valid predictions and labels
            valid_preds = batch_preds[valid_mask]
            valid_labels = batch_labels[valid_mask]
            
            # Add to flattened lists
            flat_preds.extend(valid_preds)
            flat_labels.extend(valid_labels)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels,
            flat_preds,
            average='macro',
            zero_division=0
        )
        
        return {
            f'{prefix}_precision': float(precision),
            f'{prefix}_recall': float(recall),
            f'{prefix}_f1': float(f1)
        }
    
    def _compute_sentiment_metrics(self, preds, labels):
        """Compute precision, recall, F1 for sentiment classification"""
        # Flatten predictions and labels
        flat_preds = []
        flat_labels = []
        
        for batch_preds, batch_labels in zip(preds, labels):
            # Handle multiple spans if needed
            if len(batch_labels.shape) > 1:
                batch_labels = batch_labels[0]  # Take first sentiment
            
            # Only consider valid labels (not padding)
            valid_mask = batch_labels != -100
            
            # Extract valid predictions and labels
            valid_preds = batch_preds[valid_mask]
            valid_labels = batch_labels[valid_mask]
            
            # Add to flattened lists
            flat_preds.extend(valid_preds)
            flat_labels.extend(valid_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(flat_labels, flat_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels,
            flat_preds,
            average='macro',
            zero_division=0
        )
        
        return {
            'sentiment_accuracy': float(accuracy),
            'sentiment_precision': float(precision),
            'sentiment_recall': float(recall),
            'sentiment_f1': float(f1)
        }

class FaithfulnessMetric:
    """
    Advanced metric for evaluating the faithfulness of generated explanations
    
    Inspired by factuality metrics in summarization like SummaC, this metric
    evaluates how faithful the generated explanation is to the original triplets.
    Introduced in 2025, it uses entailment and semantic similarity.
    """
    def __init__(self, config=None):
        # Initialize NLI model for entailment
        self.nli_model_name = 'facebook/bart-large-mnli'
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # Load NLI model and tokenizer
            self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
            
            # Move to GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.nli_model = self.nli_model.to(self.device)
            
            print(f"Loaded NLI model: {self.nli_model_name}")
            self.model_available = True
        except Exception as e:
            print(f"Failed to load NLI model: {e}")
            self.model_available = False
    
    def compute_faithfulness(self, triplets, explanation):
        """
        Compute faithfulness score for a single explanation
        
        Args:
            triplets: List of extracted triplets
            explanation: Generated explanation text
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not self.model_available or not triplets:
            return 0.5  # Default score
        
        # Convert triplets to statements
        statements = []
        for triplet in triplets:
            aspect = triplet.get('aspect', '')
            opinion = triplet.get('opinion', '')
            sentiment = triplet.get('sentiment', 'NEU')
            
            # Map sentiment to text
            sentiment_text = {
                'POS': 'positive',
                'NEG': 'negative',
                'NEU': 'neutral'
            }.get(sentiment, 'neutral')
            
            # Create statement
            statement = f"The {aspect} is {sentiment_text} because of the {opinion}."
            statements.append(statement)
        
        # Compute entailment scores
        entailment_scores = []
        
        for statement in statements:
            # Prepare inputs
            inputs = self.nli_tokenizer(
                premise=explanation,
                hypothesis=statement,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                logits = outputs.logits
                
                # Get entailment score (index 0: contradiction, 1: neutral, 2: entailment)
                probs = torch.nn.functional.softmax(logits, dim=1)
                entailment_score = probs[0, 2].item()  # Probability of entailment
                
                entailment_scores.append(entailment_score)
        
        # Average entailment scores
        faithfulness_score = np.mean(entailment_scores)
        
        return faithfulness_score
    
    def compute_batch_faithfulness(self, batch_triplets, batch_explanations):
        """
        Compute faithfulness scores for a batch of examples
        
        Args:
            batch_triplets: List of lists of triplets
            batch_explanations: List of explanation texts
            
        Returns:
            Dictionary with faithfulness metrics
        """
        if not self.model_available:
            return {'faithfulness': 0.5}
        
        # Compute faithfulness for each example
        faithfulness_scores = []
        
        for triplets, explanation in zip(batch_triplets, batch_explanations):
            score = self.compute_faithfulness(triplets, explanation)
            faithfulness_scores.append(score)
        
        # Compute statistics
        avg_faithfulness = np.mean(faithfulness_scores)
        median_faithfulness = np.median(faithfulness_scores)
        min_faithfulness = np.min(faithfulness_scores)
        max_faithfulness = np.max(faithfulness_scores)
        
        # Return metrics
        return {
            'faithfulness': avg_faithfulness,
            'faithfulness_median': median_faithfulness,
            'faithfulness_min': min_faithfulness,
            'faithfulness_max': max_faithfulness
        }

def compute_bert_score(references, candidates):
    """
    Compute BERTScore for evaluating generated explanations
    
    Args:
        references: List of reference texts
        candidates: List of candidate texts
        
    Returns:
        Dictionary with BERTScore metrics
    """
    try:
        from bert_score import score
        
        # Compute BERTScore
        P, R, F1 = score(candidates, references, lang="en", rescale_with_baseline=True)
        
        # Convert to numpy for averaging
        precision = P.numpy().mean()
        recall = R.numpy().mean()
        f1 = F1.numpy().mean()
        
        return {
            'bertscore_precision': float(precision),
            'bertscore_recall': float(recall),
            'bertscore_f1': float(f1)
        }
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }

def compute_triplet_recovery_score(triplets, explanation, model=None, tokenizer=None):
    """
    Compute triplet recovery score for a single explanation
    
    This is a standalone version of the triplet recovery metric
    that can be used without initializing the full class.
    
    Args:
        triplets: List of extracted triplets
        explanation: Generated explanation text
        model: Optional ABSA model for extraction
        tokenizer: Optional tokenizer for extraction
        
    Returns:
        Recovery score between 0 and 1
    """
    # Convert triplets to simplified form
    original_triplets = []
    for t in triplets:
        original_triplets.append({
            'aspect': t.get('aspect', '').lower(),
            'opinion': t.get('opinion', '').lower(),
            'sentiment': t.get('sentiment', 'NEU')
        })
    
    # Extract triplets from explanation using heuristics
    recovered_triplets = []
    
    # Common patterns in explanations
    patterns = [
        r"the (.*?) is (positive|negative|neutral) because of (.*?)[.,]",
        r"(.*?) is (positive|negative|neutral) due to (.*?)[.,]",
        r"(.*?) has a (positive|negative|neutral) sentiment because (.*?)[.,]"
    ]
    
    import re
    for pattern in patterns:
        matches = re.findall(pattern, explanation.lower())
        for match in matches:
            if len(match) >= 3:
                aspect = match[0].strip()
                sentiment = match[1].strip().upper()[:3]  # Convert to POS/NEU/NEG
                opinion = match[2].strip()
                
                recovered_triplets.append({
                    'aspect': aspect,
                    'opinion': opinion,
                    'sentiment': sentiment
                })
    
    # Count matches
    matched = 0
    for orig_t in original_triplets:
        for rec_t in recovered_triplets:
            # Check if triplets match approximately
            aspect_match = orig_t['aspect'] in rec_t['aspect'] or rec_t['aspect'] in orig_t['aspect']
            opinion_match = orig_t['opinion'] in rec_t['opinion'] or rec_t['opinion'] in orig_t['opinion']
            sentiment_match = orig_t['sentiment'] == rec_t['sentiment']
            
            if aspect_match and opinion_match and sentiment_match:
                matched += 1
                break
    
    # Compute precision and recall
    precision = matched / len(recovered_triplets) if recovered_triplets else 0
    recall = matched / len(original_triplets) if original_triplets else 0
    
    # Compute F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1