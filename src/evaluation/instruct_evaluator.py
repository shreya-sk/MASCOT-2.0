"""
Evaluation for instruction-following ABSA
"""
import torch
import re
from typing import List, Dict

class InstructABSAEvaluator:
    """
    Evaluator that combines extraction and generation metrics
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def evaluate(self, model, dataloader, device):
        """
        Comprehensive evaluation of instruction-following ABSA
        """
        model.eval()
        
        extraction_metrics = {
            'aspect_f1': 0, 'opinion_f1': 0, 'sentiment_acc': 0
        }
        generation_metrics = {
            'bleu': 0, 'rouge_l': 0, 'triplet_match': 0
        }
        
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get model outputs
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task_type='triplet_extraction'
                )
                
                # Evaluate extraction performance
                extraction_scores = self._evaluate_extraction(
                    outputs['extraction_outputs'], batch
                )
                
                # Evaluate generation performance
                generation_scores = self._evaluate_generation(
                    outputs['generated_text'], batch
                )
                
                # Accumulate metrics
                for k, v in extraction_scores.items():
                    extraction_metrics[k] += v
                
                for k, v in generation_scores.items():
                    generation_metrics[k] += v
                
                total_samples += batch['input_ids'].size(0)
        
        # Average metrics
        for k in extraction_metrics:
            extraction_metrics[k] /= total_samples
        
        for k in generation_metrics:
            generation_metrics[k] /= total_samples
        
        # Combine metrics
        combined_metrics = {
            **extraction_metrics,
            **generation_metrics,
            'overall_f1': (extraction_metrics['aspect_f1'] + 
                          extraction_metrics['opinion_f1'] + 
                          generation_metrics['triplet_match']) / 3
        }
        
        return combined_metrics
    
    def _evaluate_extraction(self, outputs, batch):
        """
        Evaluate traditional extraction metrics
        """
        # Use your existing evaluation logic
        aspect_pred = outputs['aspect_logits'].argmax(dim=-1)
        opinion_pred = outputs['opinion_logits'].argmax(dim=-1)
        sentiment_pred = outputs['sentiment_logits'].argmax(dim=-1)
        
        # Simplified metrics calculation
        aspect_f1 = self._calculate_span_f1(aspect_pred, batch.get('aspect_labels'))
        opinion_f1 = self._calculate_span_f1(opinion_pred, batch.get('opinion_labels'))
        sentiment_acc = self._calculate_sentiment_acc(sentiment_pred, batch.get('sentiment_labels'))
        
        return {
            'aspect_f1': aspect_f1,
            'opinion_f1': opinion_f1,
            'sentiment_acc': sentiment_acc
        }
    
    def _evaluate_generation(self, generated_text, batch):
        """
        Evaluate generation quality
        """
        if isinstance(generated_text, list):
            generated_text = generated_text[0]
        
        # Parse generated triplets
        generated_triplets = self._parse_generated_triplets(generated_text)
        
        # Create gold triplets from batch (simplified)
        gold_triplets = self._create_gold_triplets(batch)
        
        # Calculate triplet matching score
        triplet_match = self._calculate_triplet_match(generated_triplets, gold_triplets)
        
        return {
            'triplet_match': triplet_match,
            'bleu': 0.0,  # Placeholder - implement BLEU if needed
            'rouge_l': 0.0  # Placeholder - implement ROUGE if needed
        }
    
    def _parse_generated_triplets(self, text):
        """
        Parse triplets from generated text
        """
        triplets = []
        
        # Look for triplet patterns
        pattern = r'<triplet><aspect>(.*?)</aspect><opinion>(.*?)</opinion><sentiment>(.*?)</sentiment></triplet>'
        matches = re.findall(pattern, text)
        
        for aspect, opinion, sentiment in matches:
            triplets.append({
                'aspect': aspect.strip(),
                'opinion': opinion.strip(), 
                'sentiment': sentiment.strip()
            })
        
        return triplets
    
    def _create_gold_triplets(self, batch):
        """
        Create gold standard triplets from batch labels
        """
        # Simplified - you'd want to make this more sophisticated
        return [{'aspect': 'gold_aspect', 'opinion': 'gold_opinion', 'sentiment': 'POS'}]
    
    def _calculate_triplet_match(self, generated, gold):
        """
        Calculate triplet matching score
        """
        if not generated or not gold:
            return 0.0
        
        matches = 0
        for gen_triplet in generated:
            for gold_triplet in gold:
                if (gen_triplet['sentiment'] == gold_triplet['sentiment']):
                    matches += 1
                    break
        
        return matches / len(gold) if gold else 0.0
    
    def _calculate_span_f1(self, predictions, labels):
        """
        Calculate F1 for span predictions (simplified)
        """
        if labels is None:
            return 0.0
        
        # Simplified F1 calculation
        tp = ((predictions > 0) & (labels > 0)).sum().item()
        fp = ((predictions > 0) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels > 0)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def _calculate_sentiment_acc(self, predictions, labels):
        """
        Calculate sentiment accuracy (simplified)
        """
        if labels is None:
            return 0.0
        
        if len(labels.shape) > 1:
            labels = labels[:, 0]  # Take first sentiment
        
        correct = (predictions == labels).sum().item()
        total = labels.numel()
        
        return correct / total if total > 0 else 0.0