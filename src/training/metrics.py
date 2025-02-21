# src/training/metrics.py
from typing import List, Dict, Tuple
import torch
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
            metrics['aspect_f1'],
            metrics['opinion_f1'],
            metrics['sentiment_f1']
        ])
        
        return metrics
    
    def _compute_span_metrics(
        self,
        preds: List[np.ndarray],
        labels: List[np.ndarray],
        prefix: str
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for span detection"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='macro'
        )
        
        return {
            f'{prefix}_precision': precision,
            f'{prefix}_recall': recall,
            f'{prefix}_f1': f1
        }
        
    def _compute_sentiment_metrics(
        self,
        preds: List[np.ndarray],
        labels: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute sentiment classification metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average='macro'
        )
        
        return {
            'sentiment_precision': precision,
            'sentiment_recall': recall,
            'sentiment_f1': f1
        }