# src/evaluation/advanced_metrics.py
import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report
import scipy.stats as stats

class QuadrupleExtractionMetrics:
    """
    Evaluation metrics for aspect sentiment quadruple extraction
    Following ABSA-Bench framework standards
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        
    def update(self, pred_quadruples, target_quadruples):
        """
        Update with batch of quadruples
        
        Args:
            pred_quadruples: List of predicted (aspect, category, opinion, sentiment) tuples
            target_quadruples: List of target (aspect, category, opinion, sentiment) tuples
        """
        self.predictions.extend(pred_quadruples)
        self.targets.extend(target_quadruples)
        
    def compute_exact_match(self):
        """Compute exact match accuracy for complete quadruples"""
        exact_matches = 0
        total = len(self.targets)
        
        for pred, target in zip(self.predictions, self.targets):
            if set(pred) == set(target):
                exact_matches += 1
                
        return exact_matches / total if total > 0 else 0.0
    
    def compute_partial_match(self):
        """Compute partial match scores for different components"""
        aspect_matches = 0
        category_matches = 0
        opinion_matches = 0
        sentiment_matches = 0
        total = len(self.targets)
        
        for pred, target in zip(self.predictions, self.targets):
            pred_aspects = {q[0] for q in pred}
            target_aspects = {q[0] for q in target}
            aspect_matches += len(pred_aspects & target_aspects) / max(len(target_aspects), 1)
            
            # Similar for other components...
            
        return {
            'aspect_partial': aspect_matches / total,
            'category_partial': category_matches / total,
            'opinion_partial': opinion_matches / total,
            'sentiment_partial': sentiment_matches / total
        }

class ImplicitSentimentMetrics:
    """
    Specialized metrics for implicit sentiment detection
    Critical for current ABSA research
    """
    
    def __init__(self):
        self.implicit_predictions = []
        self.implicit_targets = []
        self.explicit_predictions = []
        self.explicit_targets = []
        
    def update(self, predictions, targets, implicit_mask):
        """
        Update with predictions separated by implicit/explicit
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            implicit_mask: Boolean mask indicating implicit sentiments
        """
        for pred, target, is_implicit in zip(predictions, targets, implicit_mask):
            if is_implicit:
                self.implicit_predictions.append(pred)
                self.implicit_targets.append(target)
            else:
                self.explicit_predictions.append(pred)
                self.explicit_targets.append(target)
                
    def compute_implicit_performance(self):
        """Compute performance specifically on implicit sentiments"""
        if not self.implicit_targets:
            return {'implicit_f1': 0.0, 'implicit_acc': 0.0}
            
        # Compute metrics for implicit cases
        implicit_acc = sum(p == t for p, t in zip(self.implicit_predictions, self.implicit_targets)) / len(self.implicit_targets)
        
        # F1 score for implicit detection
        from sklearn.metrics import f1_score
        implicit_f1 = f1_score(self.implicit_targets, self.implicit_predictions, average='macro')
        
        return {
            'implicit_f1': implicit_f1,
            'implicit_acc': implicit_acc,
            'implicit_count': len(self.implicit_targets)
        }

class CrossDomainConsistencyMetrics:
    """
    Metrics for evaluating cross-domain robustness
    Essential for publication-ready evaluation
    """
    
    def __init__(self):
        self.domain_results = {}
        
    def update_domain(self, domain_name, predictions, targets):
        """Update results for a specific domain"""
        if domain_name not in self.domain_results:
            self.domain_results[domain_name] = {'predictions': [], 'targets': []}
            
        self.domain_results[domain_name]['predictions'].extend(predictions)
        self.domain_results[domain_name]['targets'].extend(targets)
        
    def compute_consistency(self):
        """
        Compute cross-domain consistency measures
        
        Returns:
            Dict with consistency metrics across domains
        """
        if len(self.domain_results) < 2:
            return {'cross_domain_std': 0.0, 'cross_domain_variance': 0.0}
            
        # Compute F1 for each domain
        domain_f1s = []
        for domain, results in self.domain_results.items():
            from sklearn.metrics import f1_score
            f1 = f1_score(results['targets'], results['predictions'], average='macro')
            domain_f1s.append(f1)
            
        # Compute consistency metrics
        f1_std = np.std(domain_f1s)
        f1_variance = np.var(domain_f1s)
        f1_range = max(domain_f1s) - min(domain_f1s)
        
        return {
            'cross_domain_std': f1_std,
            'cross_domain_variance': f1_variance,
            'cross_domain_range': f1_range,
            'domain_f1s': dict(zip(self.domain_results.keys(), domain_f1s))
        }

class TemporalSentimentMetrics:
    """
    Metrics for temporal sentiment dynamics
    Novel research direction highlighted in the report
    """
    
    def __init__(self):
        self.temporal_predictions = []
        self.temporal_targets = []
        self.timestamps = []
        
    def update(self, predictions, targets, timestamps):
        """Update with temporal data"""
        self.temporal_predictions.extend(predictions)
        self.temporal_targets.extend(targets)
        self.timestamps.extend(timestamps)
        
    def compute_temporal_consistency(self):
        """Compute consistency of predictions over time"""
        if len(self.timestamps) < 2:
            return {'temporal_consistency': 1.0}
            
        # Sort by timestamp
        sorted_data = sorted(zip(self.timestamps, self.temporal_predictions, self.temporal_targets))
        
        # Compute temporal correlations
        sentiment_changes = []
        for i in range(1, len(sorted_data)):
            prev_sentiment = sorted_data[i-1][1]
            curr_sentiment = sorted_data[i][1]
            sentiment_changes.append(abs(curr_sentiment - prev_sentiment))
            
        temporal_stability = 1.0 - (np.mean(sentiment_changes) if sentiment_changes else 0.0)
        
        return {'temporal_consistency': temporal_stability}

class ComprehensiveABSAMetrics:
    """
    Unified evaluation framework combining all advanced metrics
    """
    
    def __init__(self, config):
        self.quadruple_metrics = QuadrupleExtractionMetrics()
        self.implicit_metrics = ImplicitSentimentMetrics()
        self.cross_domain_metrics = CrossDomainConsistencyMetrics()
        self.temporal_metrics = TemporalSentimentMetrics()
        
        # Enable/disable specific metrics based on config
        self.use_quadruple = getattr(config, 'evaluate_quadruples', True)
        self.use_implicit = getattr(config, 'evaluate_implicit', True)
        self.use_cross_domain = getattr(config, 'evaluate_cross_domain', True)
        self.use_temporal = getattr(config, 'evaluate_temporal', False)
        
    def compute_all_metrics(self):
        """Compute comprehensive evaluation metrics"""
        results = {}
        
        if self.use_quadruple:
            results.update(self.quadruple_metrics.compute_exact_match())
            results.update(self.quadruple_metrics.compute_partial_match())
            
        if self.use_implicit:
            results.update(self.implicit_metrics.compute_implicit_performance())
            
        if self.use_cross_domain:
            results.update(self.cross_domain_metrics.compute_consistency())
            
        if self.use_temporal:
            results.update(self.temporal_metrics.compute_temporal_consistency())
            
        return results
    
    def statistical_significance_test(self, baseline_results, current_results):
        """
        Perform statistical significance testing
        Essential for publication claims
        """
        # Paired t-test for significance
        baseline_scores = list(baseline_results.values())
        current_scores = list(current_results.values())
        
        if len(baseline_scores) == len(current_scores) and len(baseline_scores) > 1:
            t_stat, p_value = stats.ttest_rel(current_scores, baseline_scores)
            
            return {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': (np.mean(current_scores) - np.mean(baseline_scores)) / np.std(baseline_scores)
            }
        
        return {'significant': False, 'p_value': 1.0}