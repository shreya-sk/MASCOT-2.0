# predict.py
"""
Working prediction script for GRADIENT ABSA model
Compatible with your current codebase
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import your actual modules
from utils.config import Config
from models.unified_absa_model import UnifiedABSAModel

class GRADIENTPredictor:
    """
    Prediction class for your GRADIENT ABSA model
    Works with your actual trained models
    """
    
    def __init__(self, model_path: str, config_path: str = None, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load config
        if config_path and os.path.exists(config_path):
            self.config = Config.from_json(config_path)
        else:
            self.config = Config()  # Use default config
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            getattr(self.config, 'model_name', 'roberta-base')
        )
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✅ GRADIENT model loaded on {self.device}")
    
    def _load_model(self, model_path: str) -> UnifiedABSAModel:
        """Load trained GRADIENT model"""
        try:
            # Load model state
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model instance
            model = UnifiedABSAModel(self.config)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"Attempting alternative loading method...")
            
            # Alternative: try loading just the weights
            try:
                model = UnifiedABSAModel(self.config)
                model.to(self.device)
                # If model file is just weights, load directly
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                return model
            except Exception as e2:
                raise Exception(f"Failed to load model with both methods: {e}, {e2}")
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Predict ABSA triplets for input text
        
        Args:
            text: Input text to analyze
            return_confidence: Whether to include confidence scores
            
        Returns:
            Dictionary with triplets and metadata
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=getattr(self.config, 'max_length', 512),
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract predictions
        triplets = self._extract_triplets(outputs, text, inputs, return_confidence)
        
        # Create result
        result = {
            'text': text,
            'triplets': triplets,
            'model_info': {
                'model_type': 'GRADIENT',
                'has_gradient_reversal': getattr(self.config, 'use_gradient_reversal', True),
                'has_implicit_detection': getattr(self.config, 'use_implicit_detection', True)
            }
        }
        
        return result
    
    def _extract_triplets(self, outputs: Dict, text: str, inputs: Dict, return_confidence: bool) -> List[Dict]:
        """Extract triplets from model outputs"""
        triplets = []
        
        try:
            # Get predictions from model outputs
            aspect_logits = outputs.get('aspect_logits')
            opinion_logits = outputs.get('opinion_logits') 
            sentiment_logits = outputs.get('sentiment_logits')
            
            if aspect_logits is None or opinion_logits is None or sentiment_logits is None:
                print("⚠️ Model outputs missing required logits")
                return triplets
            
            # Convert to predictions
            aspect_preds = torch.argmax(aspect_logits, dim=-1)[0].cpu().numpy()
            opinion_preds = torch.argmax(opinion_logits, dim=-1)[0].cpu().numpy()
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1)[0].cpu().numpy()
            
            # Convert token predictions to text spans
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Extract aspect spans
            aspect_spans = self._extract_spans(aspect_preds, tokens, 'aspect')
            opinion_spans = self._extract_spans(opinion_preds, tokens, 'opinion')
            
            # Pair aspects with opinions and sentiments
            triplets = self._form_triplets(aspect_spans, opinion_spans, sentiment_preds, tokens, return_confidence)
            
        except Exception as e:
            print(f"⚠️ Error extracting triplets: {e}")
            # Return empty triplets on error
        
        return triplets
    
    def _extract_spans(self, predictions: List[int], tokens: List[str], span_type: str) -> List[Dict]:
        """Extract spans from BIO predictions"""
        spans = []
        current_span = None
        
        for i, (pred, token) in enumerate(zip(predictions, tokens)):
            # Skip special tokens
            if token in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]']:
                continue
            
            if pred == 1:  # B- (Beginning)
                if current_span:
                    spans.append(current_span)
                current_span = {
                    'start': i,
                    'end': i,
                    'tokens': [token],
                    'type': span_type
                }
            elif pred == 2 and current_span:  # I- (Inside)
                current_span['end'] = i
                current_span['tokens'].append(token)
            else:  # O (Outside)
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        # Add final span
        if current_span:
            spans.append(current_span)
        
        # Convert tokens back to text
        for span in spans:
            span['text'] = self._tokens_to_text(span['tokens'])
        
        return spans
    
    def _tokens_to_text(self, tokens: List[str]) -> str:
        """Convert tokens back to readable text"""
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text.strip()
    
    def _form_triplets(self, aspect_spans: List[Dict], opinion_spans: List[Dict], 
                      sentiment_preds: List[int], tokens: List[str], return_confidence: bool) -> List[Dict]:
        """Form triplets by pairing aspects with opinions and sentiments"""
        triplets = []
        
        # Simple pairing strategy: match closest aspects and opinions
        for aspect in aspect_spans:
            best_opinion = None
            best_distance = float('inf')
            
            # Find closest opinion to this aspect
            for opinion in opinion_spans:
                distance = abs(aspect['start'] - opinion['start'])
                if distance < best_distance:
                    best_distance = distance
                    best_opinion = opinion
            
            if best_opinion:
                # Get sentiment for this region
                region_start = min(aspect['start'], best_opinion['start'])
                region_end = max(aspect['end'], best_opinion['end'])
                
                # Get most common sentiment in region
                region_sentiments = sentiment_preds[region_start:region_end+1]
                if region_sentiments:
                    sentiment_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # O, POS, NEG, NEU
                    for s in region_sentiments:
                        if s in sentiment_counts:
                            sentiment_counts[s] += 1
                    
                    # Get most frequent non-O sentiment
                    sentiment_id = max([1, 2, 3], key=lambda x: sentiment_counts[x])
                    sentiment_map = {1: 'POS', 2: 'NEG', 3: 'NEU'}
                    sentiment = sentiment_map.get(sentiment_id, 'NEU')
                    
                    triplet = {
                        'aspect': aspect['text'],
                        'opinion': best_opinion['text'],
                        'sentiment': sentiment,
                        'aspect_span': (aspect['start'], aspect['end']),
                        'opinion_span': (best_opinion['start'], best_opinion['end'])
                    }
                    
                    if return_confidence:
                        triplet['confidence'] = 0.8  # Placeholder confidence
                    
                    triplets.append(triplet)
        
        return triplets
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict triplets for multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description="GRADIENT ABSA Prediction")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--text", help="Text to analyze")
    parser.add_argument("--file", help="File with texts (one per line)")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.error("Either --text or --file must be provided")
    
    # Initialize predictor
    print(f"🔄 Loading GRADIENT model from {args.model}")
    predictor = GRADIENTPredictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    results = []
    
    if args.text:
        # Single text prediction
        print(f"\n🔍 Analyzing: {args.text}")
        result = predictor.predict(args.text)
        results.append(result)
        
        # Print results
        print(f"\n📊 Results:")
        print(f"Text: {result['text']}")
        print(f"Triplets ({len(result['triplets'])}):")
        for i, triplet in enumerate(result['triplets'], 1):
            print(f"  {i}. Aspect: '{triplet['aspect']}' | Opinion: '{triplet['opinion']}' | Sentiment: {triplet['sentiment']}")
    
    elif args.file:
        # Batch prediction
        print(f"📁 Reading texts from {args.file}")
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"🔄 Processing {len(texts)} texts...")
        results = predictor.predict_batch(texts)
        
        # Print summary
        total_triplets = sum(len(r['triplets']) for r in results)
        print(f"\n📊 Processed {len(texts)} texts, found {total_triplets} triplets")
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {args.output}")
    
    print("\n✅ Prediction completed!")


if __name__ == "__main__":
    main()