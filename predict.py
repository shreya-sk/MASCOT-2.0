import torch
from transformers import AutoTokenizer
import numpy as np
from train import NovelGradientABSAModel, NovelABSAConfig
from typing import List, Dict, Any

class ABSAInference:
    """Complete inference class for your trained ABSA model"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the saved model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        
        # Reconstruct config (or load from saved config)
        if config_path:
            self.config = torch.load(config_path)
        else:
            # Use the saved config from checkpoint
            self.config = checkpoint.get('config', None)
            if not self.config:
                # Fallback to default config
                self.config = self._create_default_config()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Initialize model
        self.model = NovelGradientABSAModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Label mappings for decoding
        self.aspect_labels = {0: 'O', 1: 'B-ASP', 2: 'I-ASP'}
        self.opinion_labels = {0: 'O', 1: 'B-OP', 2: 'I-OP'}
        self.sentiment_labels = {0: 'O', 1: 'POS', 2: 'NEG', 3: 'NEU'}
        
        print(f"✅ ABSA model loaded successfully on {self.device}")
    
    def _create_default_config(self):
        """Fallback config if not saved with model"""
        from types import SimpleNamespace
        return SimpleNamespace(
            model_name='bert-base-uncased',
            max_length=128,
            datasets=['laptop14', 'rest14', 'rest15', 'rest16']
        )
    
    def predict(self, text: str, return_confidence: bool = False) -> Dict[str, Any]:
        """
        Predict ABSA components for a single text
        
        Args:
            text: Input review/sentence
            return_confidence: Whether to return prediction confidences
            
        Returns:
            Dictionary with aspects, opinions, sentiments, and triplets
        """
        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False
            )
        
        # Extract logits and convert to predictions
        aspect_logits = outputs['aspect_logits'].cpu()
        opinion_logits = outputs['opinion_logits'].cpu()
        sentiment_logits = outputs['sentiment_logits'].cpu()
        
        # Get predicted labels
        aspect_preds = torch.argmax(aspect_logits, dim=-1).squeeze(0).numpy()
        opinion_preds = torch.argmax(opinion_logits, dim=-1).squeeze(0).numpy()
        sentiment_preds = torch.argmax(sentiment_logits, dim=-1).squeeze(0).numpy()
        
        # Get confidence scores if requested
        confidences = {}
        if return_confidence:
            aspect_conf = torch.softmax(aspect_logits, dim=-1).squeeze(0).numpy()
            opinion_conf = torch.softmax(opinion_logits, dim=-1).squeeze(0).numpy()
            sentiment_conf = torch.softmax(sentiment_logits, dim=-1).squeeze(0).numpy()
            
            confidences = {
                'aspect_confidence': aspect_conf,
                'opinion_confidence': opinion_conf,
                'sentiment_confidence': sentiment_conf
            }
        
        # Convert to readable format
        results = self._decode_predictions(
            text, aspect_preds, opinion_preds, sentiment_preds,
            attention_mask.cpu().squeeze(0).numpy()
        )
        
        if return_confidence:
            results['confidences'] = confidences
            
        return results
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict ABSA components for multiple texts"""
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def _decode_predictions(self, text: str, aspect_preds: np.ndarray, 
                          opinion_preds: np.ndarray, sentiment_preds: np.ndarray,
                          attention_mask: np.ndarray) -> Dict[str, Any]:
        """Decode model predictions into readable format"""
        
        # Tokenize text to align with predictions
        tokens = self.tokenizer.tokenize(text)
        
        # Extract valid predictions (remove padding and special tokens)
        valid_length = min(len(tokens), len(aspect_preds) - 2)  # Account for [CLS], [SEP]
        
        # Skip [CLS] token (index 0), take content tokens
        valid_aspect_preds = aspect_preds[1:valid_length + 1]
        valid_opinion_preds = opinion_preds[1:valid_length + 1]
        valid_sentiment_preds = sentiment_preds[1:valid_length + 1]
        
        # Extract spans
        aspect_spans = self._extract_spans(valid_aspect_preds, tokens, 'aspect')
        opinion_spans = self._extract_spans(valid_opinion_preds, tokens, 'opinion')
        sentiment_spans = self._extract_spans(valid_sentiment_preds, tokens, 'sentiment')
        
        # Form triplets
        triplets = self._form_triplets(aspect_spans, opinion_spans, sentiment_spans)
        
        return {
            'text': text,
            'aspects': aspect_spans,
            'opinions': opinion_spans, 
            'sentiments': sentiment_spans,
            'triplets': triplets,
            'tokens': tokens
        }
    
    def _extract_spans(self, predictions: np.ndarray, tokens: List[str], 
                      span_type: str) -> List[Dict[str, Any]]:
        """Extract named entity spans from BIO predictions"""
        spans = []
        current_span = None
        
        for i, pred in enumerate(predictions):
            if span_type in ['aspect', 'opinion']:
                # BIO tagging: 0=O, 1=B, 2=I
                if pred == 1:  # B-tag (beginning)
                    # Save previous span if exists
                    if current_span:
                        spans.append(current_span)
                    # Start new span
                    current_span = {
                        'start': i,
                        'end': i,
                        'tokens': [tokens[i]],
                        'text': tokens[i].replace('##', '')
                    }
                elif pred == 2 and current_span:  # I-tag (inside)
                    # Extend current span
                    current_span['end'] = i
                    current_span['tokens'].append(tokens[i])
                    current_span['text'] += tokens[i].replace('##', '')
                elif pred == 0:  # O-tag (outside)
                    # End current span
                    if current_span:
                        spans.append(current_span)
                        current_span = None
                        
            elif span_type == 'sentiment':
                # Individual token predictions: 1=POS, 2=NEG, 3=NEU
                if pred > 0:
                    sentiment_label = self.sentiment_labels[pred]
                    spans.append({
                        'start': i,
                        'end': i,
                        'tokens': [tokens[i]],
                        'text': tokens[i].replace('##', ''),
                        'sentiment': sentiment_label
                    })
        
        # Add final span if exists
        if current_span:
            spans.append(current_span)
            
        return spans
    
    def _form_triplets(self, aspects: List[Dict], opinions: List[Dict], 
                      sentiments: List[Dict]) -> List[Dict[str, str]]:
        """Form aspect-opinion-sentiment triplets"""
        triplets = []
        
        # Simple proximity-based triplet formation
        for aspect in aspects:
            for opinion in opinions:
                # Find closest sentiment to this opinion
                best_sentiment = None
                min_distance = float('inf')
                
                for sentiment in sentiments:
                    # Calculate distance between opinion and sentiment
                    opinion_center = (opinion['start'] + opinion['end']) / 2
                    sentiment_center = (sentiment['start'] + sentiment['end']) / 2
                    distance = abs(opinion_center - sentiment_center)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_sentiment = sentiment
                
                # Form triplet if sentiment found within reasonable distance
                if best_sentiment and min_distance <= 5:  # Within 5 tokens
                    triplets.append({
                        'aspect': aspect['text'],
                        'opinion': opinion['text'],
                        'sentiment': best_sentiment['sentiment'],
                        'confidence': 'high' if min_distance <= 2 else 'medium'
                    })
        
        return triplets

# Usage Examples
def main():
    """Example usage of the ABSA inference system"""
    
    # Initialize the inference system
    inferencer = ABSAInference(
        model_path='outputs/best_model.pt'  # Path to your saved model
    )
    
    # Single prediction
    test_text = "The food was delicious but the service was terrible."
    result = inferencer.predict(test_text, return_confidence=True)
    
    print("Input:", result['text'])
    print("Aspects:", [asp['text'] for asp in result['aspects']])
    print("Opinions:", [op['text'] for op in result['opinions']])
    print("Sentiments:", [(s['text'], s['sentiment']) for s in result['sentiments']])
    print("Triplets:", result['triplets'])
    
    # Batch prediction
    test_texts = [
        "Great laptop with amazing battery life.",
        "The screen quality is poor but keyboard is good.",
        "Excellent restaurant with friendly staff and delicious food."
    ]
    
    batch_results = inferencer.predict_batch(test_texts)
    
    print("\nBatch Results:")
    for i, result in enumerate(batch_results):
        print(f"\nText {i+1}: {result['text']}")
        print(f"Triplets: {result['triplets']}")

if __name__ == "__main__":
    main()