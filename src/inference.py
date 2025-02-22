# src/inference.py
# src/inference.py
import torch
from typing import List, Dict
from .models.model import LlamaABSA  # Explicit import
from .data.preprocessor import ABSAPreprocessor
from .utils.config import LlamaABSAConfig  # Add config import

class ABSAPredictor:
    """Inference pipeline for ABSA prediction"""
    
    def __init__(
        self,
        model_path: str,
        config: LlamaABSAConfig,  # Use specific config type
        device: str = "cuda"
    ):
        # Load model
        self.model = LlamaABSA(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        
        # Initialize preprocessor
        self.preprocessor = ABSAPreprocessor(config)
        self.device = device
        
    def predict(self, text: str) -> Dict[str, List]:
        """Predict aspects, opinions and sentiments for input text"""
        # Preprocess input
        inputs = self.preprocessor.preprocess(text)
        inputs = {k: v.unsqueeze(0).to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process outputs
        predictions = self._post_process(outputs, text)
        
        return predictions
    
    def _post_process(
        self,
        outputs: Dict[str, torch.Tensor],
        text: str
    ) -> Dict[str, List]:
        """Convert model outputs to readable predictions"""
        # Get span predictions
        aspect_spans = self._get_spans(
            outputs['aspect_logits'][0],
            text
        )
        opinion_spans = self._get_spans(
            outputs['opinion_logits'][0],
            text
        )
        
        # Get sentiment predictions
        sentiments = outputs['sentiment_logits'][0].argmax(dim=-1)
        sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
        sentiments = [sentiment_map[s.item()] for s in sentiments]
        
        return {
            'aspects': aspect_spans,
            'opinions': opinion_spans,
            'sentiments': sentiments
        }
    
    def _get_spans(
        self,
        logits: torch.Tensor,
        text: str
    ) -> List[str]:
        """Extract text spans from logits"""
        # Convert logits to predictions
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
                    
        # Convert token indices to text spans
        tokens = text.split()
        text_spans = [
            ' '.join([tokens[i] for i in span])
            for span in spans
        ]
        
        return text_spans

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = ABSAPredictor(
        model_path="checkpoints/best_model.pt",
        config=config
    )
    
    # Make prediction
    text = "The food was delicious but the service was terrible."
    predictions = predictor.predict(text)
    
    print("Aspects:", predictions['aspects'])
    print("Opinions:", predictions['opinions'])
    print("Sentiments:", predictions['sentiments'])