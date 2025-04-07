# src/inference/stella_predictor.py
import torch # type: ignore
from typing import List, Dict, Any, Union, Tuple
from transformers import AutoTokenizer
import numpy as np
import spacy # type: ignore
from spacy.tokens import Doc # type: ignore

from src.models.absa import StellaABSA
from src.data.preprocessor import StellaABSAPreprocessor
from src.utils.config import StellaABSAConfig

class StellaABSAPredictor:
    """Inference pipeline for Stella-based ABSA prediction"""
    
    def __init__(
        self,
        model_path: str,
        config: StellaABSAConfig = None,
        device: str = None,
        tokenizer_path: str = None
    ):
        # Initialize config if not provided
        if config is None:
            config = StellaABSAConfig()
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize tokenizer
        tokenizer_path = tokenizer_path or config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True
        )
        
        # Initialize preprocessor
        self.preprocessor = StellaABSAPreprocessor(
            tokenizer=self.tokenizer,
            max_length=config.max_seq_length,
            use_syntax=config.use_syntax
        )
        
        # Load model
        self.model = StellaABSA(config)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load spaCy for visualization
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("Warning: spaCy not available for visualization")
    
    def predict(self, text: str, domain_id: int = None) -> Dict[str, Any]:
        """
        Predict aspects, opinions and sentiments for input text
        
        Args:
            text: Input text
            domain_id: Optional domain identifier for domain adaptation
            
        Returns:
            Dictionary with predicted aspects, opinions, sentiments, and confidence scores
        """
        # Preprocess input
        inputs = self.preprocessor.preprocess_for_inference(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Add domain_id if provided
        if domain_id is not None and self.config.domain_adaptation:
            inputs['domain_id'] = torch.tensor([domain_id], device=self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process outputs
        predictions = self._post_process(outputs, text)
        
        return predictions
    
    def _post_process(self, outputs: Dict[str, torch.Tensor], text: str) -> Dict[str, Any]:
        """Convert model outputs to readable predictions"""
        # Get span predictions
        aspect_logits = outputs['aspect_logits'][0]
        opinion_logits = outputs['opinion_logits'][0]
        sentiment_logits = outputs['sentiment_logits']
        confidence_scores = outputs['confidence_scores'][0].item()
        
        # Extract spans and sentiments
        aspect_spans = self._extract_spans(aspect_logits, text)
        opinion_spans = self._extract_spans(opinion_logits, text)
        
        # Extract sentiments and align with spans
        sentiments, probs = self._extract_sentiments(sentiment_logits)
        
        # Only keep predictions above confidence threshold
        valid_indices = []
        for i, score in enumerate(confidence_scores):
            if i < len(sentiment_logits) and score >= self.config.confidence_threshold:
                valid_indices.append(i)
        
        # Filter predictions
        filtered_aspects = [aspect_spans[i] for i in valid_indices if i < len(aspect_spans)]
        filtered_opinions = [opinion_spans[i] for i in valid_indices if i < len(opinion_spans)]
        filtered_sentiments = [sentiments[i] for i in valid_indices if i < len(sentiments)]
        filtered_probs = [probs[i] for i in valid_indices if i < len(probs)]
        
        # If we have a mismatch in length, take min length
        min_len = min(len(filtered_aspects), len(filtered_opinions), len(filtered_sentiments))
        filtered_aspects = filtered_aspects[:min_len]
        filtered_opinions = filtered_opinions[:min_len]
        filtered_sentiments = filtered_sentiments[:min_len]
        filtered_probs = filtered_probs[:min_len]
        
        # Get aspect-opinion pairs with sentiment
        triplets = []
        for aspect, opinion, sentiment, prob in zip(
            filtered_aspects, filtered_opinions, filtered_sentiments, filtered_probs
        ):
            triplets.append({
                'aspect': aspect[0],
                'aspect_indices': aspect[1],
                'opinion': opinion[0],
                'opinion_indices': opinion[1],
                'sentiment': sentiment,
                'confidence': prob
            })
        
        # Get focal attention weights for visualization
        focal_weights = None
        if 'focal_weights' in outputs:
            focal_weights = outputs['focal_weights'][0].cpu().numpy()
        
        return {
            'triplets': triplets,
            'aspects': filtered_aspects,
            'opinions': filtered_opinions,
            'sentiments': filtered_sentiments,
            'confidence': confidence_scores,
            'focal_weights': focal_weights
        }
    
    def _extract_spans(self, logits: torch.Tensor, text: str) -> List[Tuple[str, List[int]]]:
        """Extract text spans from logits using BIO tags"""
        # Get predictions for each token
        preds = logits.argmax(dim=-1).cpu().numpy()
        
        # Extract spans using BIO tags
        spans = []
        current_span = []
        
        for i, pred in enumerate(preds):
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
        
        # Convert token indices to text spans
        tokens = text.split()
        text_spans = []
        
        for span in spans:
            valid_indices = [idx for idx in span if idx < len(tokens)]
            if valid_indices:
                span_text = ' '.join([tokens[idx] for idx in valid_indices])
                text_spans.append((span_text, valid_indices))
        
        return text_spans
    
    def _extract_sentiments(self, sentiment_logits: torch.Tensor) -> Tuple[List[str], List[float]]:
        """Extract sentiment labels and probabilities from logits"""
        # Get sentiment predictions
        probs = torch.softmax(sentiment_logits, dim=-1)
        sentiments = sentiment_logits.argmax(dim=-1).cpu().numpy()
        
        # Convert to labels
        sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
        sentiment_labels = [sentiment_map[s] for s in sentiments]
        
        # Get probabilities for predicted class
        prediction_probs = [probs[i, pred].item() for i, pred in enumerate(sentiments)]
        
        return sentiment_labels, prediction_probs
    
    def visualize(self, text: str, predictions: Dict[str, Any] = None) -> str:
        """
        Visualize ABSA predictions in HTML format
        
        Args:
            text: Input text
            predictions: Optional predictions (if None, will generate predictions)
            
        Returns:
            HTML string with visualized predictions
        """
        if self.nlp is None:
            return "spaCy not available for visualization"
        
        if predictions is None:
            predictions = self.predict(text)
        
        # Create spaCy doc
        doc = self.nlp(text)
        
        # Create entity spans for visualization
        spans = []
        
        for triplet in predictions['triplets']:
            aspect = triplet['aspect']
            opinion = triplet['opinion']
            sentiment = triplet['sentiment']
            
            aspect_indices = triplet['aspect_indices']
            opinion_indices = triplet['opinion_indices']
            
            # Find character spans for tokens
            if aspect_indices:
                start_char = doc[aspect_indices[0]].idx
                end_char = doc[aspect_indices[-1]].idx + len(doc[aspect_indices[-1]])
                spans.append((start_char, end_char, f"ASPECT-{sentiment}"))
            
            if opinion_indices:
                start_char = doc[opinion_indices[0]].idx
                end_char = doc[opinion_indices[-1]].idx + len(doc[opinion_indices[-1]])
                spans.append((start_char, end_char, f"OPINION-{sentiment}"))
        
        # Create HTML visualization
        html = self._create_html_visualization(text, spans, predictions)
        
        return html
    
    def _create_html_visualization(self, text: str, spans: List[Tuple], predictions: Dict[str, Any]) -> str:
        """Create HTML visualization for ABSA predictions"""
        # Sort spans by start position
        spans.sort()
        
        # Create HTML with highlighted spans
        html = "<div style='font-family: Arial; line-height: 1.5;'>"
        html += "<h3>ABSA Predictions</h3>"
        
        # Text with highlighted spans
        html += "<div style='margin-bottom: 20px; padding: 10px; border: 1px solid #ccc;'>"
        
        # Add text with highlighted spans
        last_end = 0
        for start, end, label in spans:
            # Add text before span
            html += text[last_end:start]
            
            # Add highlighted span
            color = "#8ef" if "ASPECT" in label else "#fe8"
            if "POS" in label:
                border_color = "#0c0"
            elif "NEG" in label:
                border_color = "#c00"
            else:
                border_color = "#cc0"
                
            html += f"<span style='background-color: {color}; border-bottom: 2px solid {border_color}; padding: 2px; border-radius: 3px;' title='{label}'>{text[start:end]}</span>"
            
            # Update last end
            last_end = end
        
        # Add remaining text
        html += text[last_end:]
        html += "</div>"
        
        # Add triplets table
        html += "<h4>Extracted Triplets</h4>"
        html += "<table style='border-collapse: collapse; width: 100%;'>"
        html += "<tr style='background-color: #f2f2f2;'><th style='border: 1px solid #ddd; padding: 8px;'>Aspect</th><th style='border: 1px solid #ddd; padding: 8px;'>Opinion</th><th style='border: 1px solid #ddd; padding: 8px;'>Sentiment</th><th style='border: 1px solid #ddd; padding: 8px;'>Confidence</th></tr>"
        
        for triplet in predictions['triplets']:
            sentiment_color = "#afa" if triplet['sentiment'] == 'POS' else "#faa" if triplet['sentiment'] == 'NEG' else "#ffa"
            html += f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{triplet['aspect']}</td><td style='border: 1px solid #ddd; padding: 8px;'>{triplet['opinion']}</td><td style='border: 1px solid #ddd; padding: 8px; background-color: {sentiment_color};'>{triplet['sentiment']}</td><td style='border: 1px solid #ddd; padding: 8px;'>{triplet['confidence']:.2f}</td></tr>"
        
        html += "</table>"
        html += "</div>"
        
        return html