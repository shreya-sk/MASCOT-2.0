# src/inference/stella_predictor.py
import torch # type: ignore
from typing import List, Dict, Any, Union, Tuple
from transformers import AutoTokenizer
import numpy as np

from src.models.absa import LLMABSA
from src.data.preprocessor import LLMABSAPreprocessor
from src.utils.config import LLMABSAConfig

class LLMABSAPredictor:
    """Inference pipeline for Stella-based ABSA prediction"""
    
    def __init__(
        self,
        model_path: str,
        config: LLMABSAConfig = None,
        device: str = None,
        tokenizer_path: str = None
    ):
        # Initialize config if not provided
        if config is None:
            config = LLMABSAConfig()
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
        self.preprocessor = LLMABSAPreprocessor(
            tokenizer=self.tokenizer,
            max_length=config.max_seq_length,
            use_syntax=config.use_syntax
        )
        
        # Load model
        self.model = LLMABSA(config)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
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
        confidence_scores = outputs.get('confidence_scores', torch.ones(1))
        
        # Extract spans and sentiments
        aspect_spans = self._extract_spans(aspect_logits, text)
        opinion_spans = self._extract_spans(opinion_logits, text)
        
        # Extract sentiments
        sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
        sentiments = sentiment_logits.argmax(dim=-1).cpu().numpy()
        sentiments = [sentiment_map[s] for s in sentiments]
        
        # Get confidence
        if confidence_scores.dim() > 0:
            conf_values = confidence_scores.cpu().numpy().flatten()
        else:
            conf_values = [confidence_scores.item()]
        
        # Create triplets
        triplets = []
        for i in range(min(len(aspect_spans), len(opinion_spans), len(sentiments))):
            if i < len(conf_values):
                confidence = conf_values[i]
            else:
                confidence = 0.5  # Default confidence
                
            triplets.append({
                'aspect': aspect_spans[i][0] if i < len(aspect_spans) else "",
                'aspect_indices': aspect_spans[i][1] if i < len(aspect_spans) else [],
                'opinion': opinion_spans[i][0] if i < len(opinion_spans) else "",
                'opinion_indices': opinion_spans[i][1] if i < len(opinion_spans) else [],
                'sentiment': sentiments[i] if i < len(sentiments) else "NEU",
                'confidence': float(confidence)
            })
        
        return {
            'triplets': triplets,
            'aspects': aspect_spans,
            'opinions': opinion_spans,
            'sentiments': sentiments,
            'confidence': conf_values.tolist() if isinstance(conf_values, np.ndarray) else conf_values
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