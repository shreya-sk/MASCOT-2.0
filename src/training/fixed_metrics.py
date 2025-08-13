
# Fixed ABSA Metrics - Publication Ready
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict, Counter

def compute_absa_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """
    Compute comprehensive ABSA metrics for publication
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
    
    Returns:
        Dictionary of metrics
    """
    
    # 1. Triplet-level metrics (Primary)
    triplet_metrics = compute_triplet_extraction_metrics(predictions, targets)
    
    # 2. Component-level metrics
    aspect_metrics = compute_aspect_extraction_metrics(predictions, targets)
    opinion_metrics = compute_opinion_extraction_metrics(predictions, targets)
    sentiment_metrics = compute_sentiment_classification_metrics(predictions, targets)
    
    # 3. Combine all metrics
    all_metrics = {}
    all_metrics.update(triplet_metrics)
    all_metrics.update(aspect_metrics)
    all_metrics.update(opinion_metrics)
    all_metrics.update(sentiment_metrics)
    
    return all_metrics

def compute_triplet_extraction_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """Compute triplet extraction metrics (most important for ABSA)"""
    
    pred_triplets = []
    true_triplets = []
    
    for pred, target in zip(predictions, targets):
        # Extract triplets - adapt this to your data format
        pred_triplets.extend(extract_triplets_from_prediction(pred))
        true_triplets.extend(extract_triplets_from_target(target))
    
    # Exact matching
    pred_set = set(pred_triplets)
    true_set = set(true_triplets)
    exact_matches = len(pred_set.intersection(true_set))
    
    # Calculate metrics
    precision = exact_matches / len(pred_triplets) if pred_triplets else 0.0
    recall = exact_matches / len(true_triplets) if true_triplets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'triplet_precision': precision,
        'triplet_recall': recall,
        'triplet_f1': f1,
        'triplet_exact_matches': exact_matches,
        'triplet_total_predicted': len(pred_triplets),
        'triplet_total_gold': len(true_triplets)
    }

def extract_triplets_from_prediction(pred: Dict) -> List[Tuple]:
    """Extract triplets from prediction - ADAPT THIS TO YOUR FORMAT"""
    triplets = []
    
    # Example format - modify based on your actual prediction format
    if 'triplets' in pred:
        for triplet in pred['triplets']:
            triplets.append((
                triplet.get('aspect', ''),
                triplet.get('opinion', ''),
                triplet.get('sentiment', '')
            ))
    
    return triplets

def extract_triplets_from_target(target: Dict) -> List[Tuple]:
    """Extract triplets from target - ADAPT THIS TO YOUR FORMAT"""
    triplets = []
    
    # Example format - modify based on your actual target format
    if 'triplets' in target:
        for triplet in target['triplets']:
            triplets.append((
                triplet.get('aspect', ''),
                triplet.get('opinion', ''),
                triplet.get('sentiment', '')
            ))
    
    return triplets

def compute_aspect_extraction_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """Compute aspect extraction metrics"""
    
    pred_aspects = []
    true_aspects = []
    
    for pred, target in zip(predictions, targets):
        pred_aspects.extend(pred.get('aspects', []))
        true_aspects.extend(target.get('aspects', []))
    
    # Convert to binary classification format
    all_aspects = list(set(pred_aspects + true_aspects))
    
    if not all_aspects:
        return {'aspect_precision': 0.0, 'aspect_recall': 0.0, 'aspect_f1': 0.0}
    
    # Calculate metrics using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_aspects, pred_aspects, average='macro', zero_division=0
    )
    
    return {
        'aspect_precision': precision,
        'aspect_recall': recall,
        'aspect_f1': f1
    }

def compute_opinion_extraction_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """Compute opinion extraction metrics"""
    
    pred_opinions = []
    true_opinions = []
    
    for pred, target in zip(predictions, targets):
        pred_opinions.extend(pred.get('opinions', []))
        true_opinions.extend(target.get('opinions', []))
    
    if not pred_opinions and not true_opinions:
        return {'opinion_precision': 0.0, 'opinion_recall': 0.0, 'opinion_f1': 0.0}
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_opinions, pred_opinions, average='macro', zero_division=0
    )
    
    return {
        'opinion_precision': precision,
        'opinion_recall': recall,
        'opinion_f1': f1
    }

def compute_sentiment_classification_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """Compute sentiment classification metrics"""
    
    pred_sentiments = []
    true_sentiments = []
    
    for pred, target in zip(predictions, targets):
        pred_sentiments.extend(pred.get('sentiments', []))
        true_sentiments.extend(target.get('sentiments', []))
    
    if not pred_sentiments and not true_sentiments:
        return {'sentiment_accuracy': 0.0}
    
    # Calculate accuracy
    accuracy = accuracy_score(true_sentiments, pred_sentiments)
    
    return {
        'sentiment_accuracy': accuracy
    }

# Bootstrap confidence intervals
def compute_bootstrap_confidence(predictions: List[Dict], targets: List[Dict], 
                                n_bootstrap: int = 1000) -> Dict[str, float]:
    """Compute bootstrap confidence intervals"""
    
    bootstrap_scores = []
    n_samples = len(predictions)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_preds = [predictions[i] for i in indices]
        sample_targets = [targets[i] for i in indices]
        
        # Compute F1 for this sample
        sample_metrics = compute_triplet_extraction_metrics(sample_preds, sample_targets)
        bootstrap_scores.append(sample_metrics['triplet_f1'])
    
    # Calculate confidence intervals
    lower_ci = np.percentile(bootstrap_scores, 2.5)
    upper_ci = np.percentile(bootstrap_scores, 97.5)
    
    return {
        'bootstrap_f1_mean': np.mean(bootstrap_scores),
        'bootstrap_f1_std': np.std(bootstrap_scores),
        'bootstrap_f1_lower_ci': lower_ci,
        'bootstrap_f1_upper_ci': upper_ci
    }
