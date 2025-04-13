# src/training/metrics.py
from typing import List, Dict, Tuple
import torch # type: ignore
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class ABSAMetrics:
    """Detailed evaluation metrics for ABSA tasks"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.aspect_preds = []
        self.opinion_preds = []
        self.sentiment_preds = []
        self.aspect_labels = []
        self.opinion_labels = []
        self.sentiment_labels = []
        
    def update(self, outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
        """Update metrics with batch predictions"""
        # Get predictions
        aspect_pred = outputs['aspect_logits'].argmax(dim=-1).cpu().numpy()
        opinion_pred = outputs['opinion_logits'].argmax(dim=-1).cpu().numpy()
        sentiment_pred = outputs['sentiment_logits'].argmax(dim=-1).cpu().numpy()
        
        # Get labels
        aspect_label = labels['aspect_labels'].cpu().numpy()
        opinion_label = labels['opinion_labels'].cpu().numpy()
        sentiment_label = labels['sentiment_labels'].cpu().numpy()
        
        # Update lists
        self.aspect_preds.extend(aspect_pred)
        self.opinion_preds.extend(opinion_pred)
        self.sentiment_preds.extend(sentiment_pred)
        self.aspect_labels.extend(aspect_label)
        self.opinion_labels.extend(opinion_label)
        self.sentiment_labels.extend(sentiment_label)

    def compute(self) -> Dict[str, float]:
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
            metrics['overall_f1'] = np.mean([
                metrics.get('aspect_f1', 0),
                metrics.get('opinion_f1', 0),
                metrics.get('sentiment_f1', 0)
            ])
        except Exception as e:
            print(f"Error computing metrics: {e}")
            # Return default metrics in case of error
            metrics = {
                'aspect_precision': 0.0,
                'aspect_recall': 0.0,
                'aspect_f1': 0.0,
                'opinion_precision': 0.0,
                'opinion_recall': 0.0,
                'opinion_f1': 0.0,
                'sentiment_precision': 0.0,
                'sentiment_recall': 0.0,
                'sentiment_f1': 0.0,
                'overall_f1': 0.0
            }
        
        return metrics
        
    def _compute_sentiment_metrics(self, preds, labels):
        try:
            # Convert to numpy arrays carefully
            preds_array = np.array([p.cpu().item() if isinstance(p, torch.Tensor) else p for p in preds])
            labels_array = np.array([l.cpu().item() if isinstance(l, torch.Tensor) else l for l in labels])
            
            # Ensure 1D arrays
            preds_array = preds_array.flatten()
            labels_array = labels_array.flatten()
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_array,
                preds_array,
                average='macro',
                zero_division=0
            )
        except Exception as e:
            print(f"Error computing sentiment metrics: {e}")
            precision = recall = f1 = 0.0
            
        return {
            'sentiment_precision': precision,
            'sentiment_recall': recall,
            'sentiment_f1': f1
        }
    

    def _compute_span_metrics(
        self,
        preds: List[np.ndarray],
        labels: List[np.ndarray],
        prefix: str
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for span detection"""
        try:
            # Convert lists to proper numpy arrays
            preds_array = np.array(preds)
            labels_array = np.array(labels)
            
            # Ensure we're dealing with 1D arrays
            if preds_array.ndim > 1:
                preds_array = preds_array.flatten()
            if labels_array.ndim > 1:
                labels_array = labels_array.flatten()
            
            # Ensure we have the same shape
            min_len = min(len(preds_array), len(labels_array))
            preds_array = preds_array[:min_len]
            labels_array = labels_array[:min_len]
            
            # Ensure we're using valid classes
            # Replace any invalid values with 0
            preds_array = np.nan_to_num(preds_array).astype(int)
            labels_array = np.nan_to_num(labels_array).astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_array,
                preds_array,
                average='macro',
                zero_division=0
            )
        except Exception as e:
            print(f"Error computing metrics: {e}")
            # Return default values in case of error
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        return {
            f'{prefix}_precision': float(precision),
            f'{prefix}_recall': float(recall),
            f'{prefix}_f1': float(f1)
        }
    # Add to metrics.py
    def compute_faithfulness_score(triplets, generated_summary):
        """Compute faithfulness score between triplets and generated summary"""
        # Create reference text from triplets
        reference = ""
        for t in triplets:
            aspect = t['aspect']
            opinion = t['opinion']
            sentiment = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}[t['sentiment']]
            reference += f"The {aspect} is {sentiment} because of the {opinion}. "
        
        # Use BERTScore to compute similarity
        from bert_score import score
        P, R, F1 = score([generated_summary], [reference], lang="en", return_hash=False)
        return F1.item()  # Return F1 as faithfulness score