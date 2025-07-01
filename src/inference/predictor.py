# src/inference/predictor.py
import torch
from typing import List, Dict, Any, Union, Tuple
from transformers import AutoTokenizer
import numpy as np

class LLMABSAPredictor:
    """Inference pipeline for ABSA prediction with improved triplet extraction and explanation generation"""
    
    def __init__(
        self,
        model_path: str,
        config=None,
        device: str = None,
        tokenizer_path: str = None
    ):
        # Import required classes here to avoid circular imports
        from src.models.absa import LLMABSA
        from src.utils.config import LLMABSAConfig
        from src.models.explanation_generator import ExplanationGenerator
        
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
        
        # Make sure tokenizer has proper tokens
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'
            
        # Initialize model
        self.model = LLMABSA.load(model_path, config=config, device=self.device)
        

        if hasattr(self.model, 'set_tokenizer'):
            self.model.set_tokenizer(self.tokenizer)
        else:
            self.model.tokenizer = self.tokenizer
        
        
        # Initialize explanation generator
        self.explanation_generator = ExplanationGenerator()
        
        # Confidence thresholds for filtering triplets
        self.min_confidence = 0.5
        self.high_confidence = 0.7
  
    def filter_triplets(self, triplets, confidence_threshold=0.6):
        """Filter low-confidence triplets to improve generation quality"""
        filtered = [t for t in triplets if t["confidence"] > confidence_threshold]
        # Always return at least one triplet (the highest confidence one)
        if not filtered and triplets:
            filtered = [max(triplets, key=lambda t: t["confidence"])]
        return filtered
    
    def compute_summary_confidence(self, triplets):
        """Compute overall confidence score for the summary"""
        if not triplets:
            return 0.0
        return sum(t['confidence'] for t in triplets) / len(triplets)
    
    def predict(self, text: str, domain_id: int = None, generate=True) -> Dict[str, Any]:
        """
        Predict aspects, opinions and sentiments for input text
        
        Args:
            text: Input text
            domain_id: Optional domain identifier for domain adaptation
            generate: Whether to generate explanations
            
        Returns:
            Dictionary with predicted aspects, opinions, sentiments, and confidence scores
        """
        try:
            # Preprocess input
            inputs = self.tokenize_text(text)
            
            # Add domain_id if provided
            if domain_id is not None and hasattr(self.config, 'domain_adaptation') and self.config.domain_adaptation:
                inputs['domain_id'] = torch.tensor([domain_id], device=self.device)
            
            # Extract triplets from model
            triplets = self.model.extract_triplets(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                tokenizer=self.tokenizer,
                texts=[text]
            )[0]  # Get first batch item
            
            # Apply post-processing to further improve triplet quality
            triplets = self.post_process_triplets(triplets, text)
            
            # Generate explanations if requested
            explanations = []
            if generate:
                explanation = self.explanation_generator.generate_explanation(triplets)
                explanations.append(explanation)
            
            # Create response
            result = {
                'text': text,
                'triplets': triplets,
                'summary_confidence': self.compute_summary_confidence(triplets)
            }
            
            # Add explanations if generated
            if explanations:
                result['explanations'] = explanations
                
            return result
        
        except Exception as e:
            # Handle errors gracefully
            import traceback
            traceback.print_exc()
            
            # Return a minimal result with error info
            return {
                'text': text,
                'triplets': [],
                'summary_confidence': 0.0,
                'error': str(e)
            }
    
    def post_process_triplets(self, triplets, text):
        """Apply additional post-processing to improve triplet quality"""
        # Sort by confidence
        triplets = sorted(triplets, key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Filter very low confidence triplets
        triplets = [t for t in triplets if t.get('confidence', 0.0) > self.min_confidence]
        
        # Clean up aspect and opinion text
        for triplet in triplets:
            # Clean up aspect text
            aspect = triplet.get('aspect', '').strip()
            # Remove unwanted prefixes/suffixes
            aspect = aspect.strip('.,;:!?()')
            # Remove tokenizer artifacts
            aspect = aspect.replace('##', '').replace(' ##', '')
            triplet['aspect'] = aspect
            
            # Clean up opinion text
            opinion = triplet.get('opinion', '').strip()
            # Remove unwanted prefixes/suffixes
            opinion = opinion.strip('.,;:!?()')
            # Remove tokenizer artifacts
            opinion = opinion.replace('##', '').replace(' ##', '')
            triplet['opinion'] = opinion
            
        # Handle common review patterns that might be missed by the model
        if not triplets and any(keyword in text.lower() for keyword in ['love', 'loved', 'best', 'excellent', 'amazing']):
            # Strong positive sentiment, create fallback triplet
            triplets.append({
                'aspect': 'overall experience',
                'aspect_indices': [0],
                'opinion': 'excellent',
                'opinion_indices': [0],
                'sentiment': 'POS',
                'confidence': 0.75
            })
        elif not triplets and any(keyword in text.lower() for keyword in ['hate', 'hated', 'worst', 'terrible', 'awful']):
            # Strong negative sentiment, create fallback triplet
            triplets.append({
                'aspect': 'overall experience',
                'aspect_indices': [0],
                'opinion': 'terrible',
                'opinion_indices': [0],
                'sentiment': 'NEG',
                'confidence': 0.75
            })
            
        # Limit to top 5 triplets maximum
        return triplets[:5]
    
    def tokenize_text(self, text):
        """Tokenize text for model input"""
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def batch_predict(self, texts, generate=True):
        """Predict on a batch of texts"""
        results = []
        for text in texts:
            result = self.predict(text, generate=generate)
            results.append(result)
        return results
    
    def visualize(self, text, result=None):
        """Create HTML visualization of the predictions"""
        if result is None:
            result = self.predict(text)
            
        triplets = result.get('triplets', [])
        explanations = result.get('explanations', [])
        
        # Create HTML visualization
        html = "<div style='font-family: Arial; padding: 20px;'>"
        html += f"<h2>ABSA Analysis: {text}</h2>"
        
        # Add text with highlighted spans
        html += "<div style='margin: 15px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
        
        # Process text to highlight aspects and opinions
        text_parts = []
        text_pos = 0
        
        # Sort triplets by position of aspect in text for better highlighting
        sorted_triplets = sorted(triplets, key=lambda t: text.lower().find(t.get('aspect', '').lower()) if t.get('aspect', '') else 999)
        
        for triplet in sorted_triplets:
            aspect = triplet.get('aspect', '')
            opinion = triplet.get('opinion', '')
            sentiment = triplet.get('sentiment', 'NEU')
            
            # Skip empty aspects or opinions
            if not aspect:
                continue
                
            # Find aspect in text (case insensitive)
            aspect_pos = text.lower().find(aspect.lower(), text_pos)
            if aspect_pos >= 0:
                # Add text before the aspect
                text_parts.append(text[text_pos:aspect_pos])
                
                # Add highlighted aspect
                aspect_color = '#8ef' if sentiment == 'POS' else '#fe8' if sentiment == 'NEU' else '#fa8'
                text_parts.append(f"<span style='background-color: {aspect_color}; border-radius: 3px; padding: 2px 4px;' title='Aspect: {aspect}'>{text[aspect_pos:aspect_pos+len(aspect)]}</span>")
                
                # Update text position
                text_pos = aspect_pos + len(aspect)
                
            # Find opinion in text (case insensitive)
            if opinion:
                opinion_pos = text.lower().find(opinion.lower(), text_pos)
                if opinion_pos >= 0:
                    # Add text before the opinion
                    text_parts.append(text[text_pos:opinion_pos])
                    
                    # Add highlighted opinion
                    opinion_color = '#afa' if sentiment == 'POS' else '#ffa' if sentiment == 'NEU' else '#faa'
                    text_parts.append(f"<span style='background-color: {opinion_color}; border-radius: 3px; padding: 2px 4px;' title='Opinion: {opinion}'>{text[opinion_pos:opinion_pos+len(opinion)]}</span>")
                    
                    # Update text position
                    text_pos = opinion_pos + len(opinion)
        
        # Add any remaining text
        if text_pos < len(text):
            text_parts.append(text[text_pos:])
            
        # Join all text parts
        html += ''.join(text_parts)
        html += "</div>"
        
        # Add extracted triplets
        html += "<div style='margin: 15px 0;'>"
        html += "<h3>Extracted Triplets</h3>"
        
        if triplets:
            html += "<table style='width: 100%; border-collapse: collapse;'>"
            html += "<tr style='background-color: #f2f2f2;'><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Aspect</th><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Opinion</th><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Sentiment</th><th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Confidence</th></tr>"
            
            for triplet in triplets:
                aspect = triplet.get('aspect', '')
                opinion = triplet.get('opinion', '')
                sentiment = triplet.get('sentiment', 'NEU')
                confidence = triplet.get('confidence', 0.0)
                
                # Set row color based on sentiment
                row_color = 'white'
                sentiment_cell_color = '#d9ead3' if sentiment == 'POS' else '#fff2cc' if sentiment == 'NEU' else '#f4cccc'
                
                html += f"<tr style='background-color: {row_color};'>"
                html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{aspect}</td>"
                html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{opinion}</td>"
                html += f"<td style='border: 1px solid #ddd; padding: 8px; background-color: {sentiment_cell_color};'>{sentiment}</td>"
                html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{confidence:.2f}</td>"
                html += "</tr>"
                
            html += "</table>"
        else:
            html += "<p>No triplets extracted.</p>"
            
        html += "</div>"
        
        # Add explanation
        if explanations:
            html += "<div style='margin: 15px 0;'>"
            html += "<h3>Generated Explanation</h3>"
            html += f"<p style='padding: 10px; background-color: #f9f9f9; border-radius: 5px;'>{explanations[0]}</p>"
            html += "</div>"
        
        html += "</div>"
        
        return html