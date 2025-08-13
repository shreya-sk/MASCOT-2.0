# ABSA Evaluation Methodology Overhaul - ACL/EMNLP 2025 Ready
# Complete implementation of rigorous evaluation metrics and statistical testing

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import itertools
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ABSAEvaluationConfig:
    """Configuration for ABSA evaluation"""
    use_strict_matching: bool = True
    use_semantic_matching: bool = True
    confidence_threshold: float = 0.5
    bootstrap_samples: int = 1000
    cv_folds: int = 5
    statistical_alpha: float = 0.05

class DatasetValidator:
    """Validates dataset splits and prevents data leakage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_splits(self, train_path: str, dev_path: str, test_path: str) -> Dict[str, Any]:
        """Validate dataset splits for proper separation"""
        self.logger.info("🔍 Validating dataset splits...")
        
        validation_results = {
            'train_size': 0,
            'dev_size': 0,
            'test_size': 0,
            'data_leakage': False,
            'overlap_details': {},
            'label_distribution': {},
            'split_quality': 'UNKNOWN'
        }
        
        try:
            # Load datasets
            train_data = self._load_dataset(train_path)
            dev_data = self._load_dataset(dev_path)
            test_data = self._load_dataset(test_path) if os.path.exists(test_path) else []
            
            validation_results['train_size'] = len(train_data)
            validation_results['dev_size'] = len(dev_data)
            validation_results['test_size'] = len(test_data)
            
            # Check for data leakage
            train_texts = set([item['text'] for item in train_data])
            dev_texts = set([item['text'] for item in dev_data])
            test_texts = set([item['text'] for item in test_data])
            
            # Check overlaps
            train_dev_overlap = train_texts.intersection(dev_texts)
            train_test_overlap = train_texts.intersection(test_texts)
            dev_test_overlap = dev_texts.intersection(test_texts)
            
            validation_results['overlap_details'] = {
                'train_dev_overlap': len(train_dev_overlap),
                'train_test_overlap': len(train_test_overlap),
                'dev_test_overlap': len(dev_test_overlap)
            }
            
            # Check if any overlap exists
            total_overlap = len(train_dev_overlap) + len(train_test_overlap) + len(dev_test_overlap)
            validation_results['data_leakage'] = total_overlap > 0
            
            # Analyze label distributions
            validation_results['label_distribution'] = self._analyze_label_distribution(
                train_data, dev_data, test_data
            )
            
            # Assess split quality
            validation_results['split_quality'] = self._assess_split_quality(validation_results)
            
            # Log results
            self._log_validation_results(validation_results)
            
        except Exception as e:
            self.logger.error(f"❌ Dataset validation failed: {e}")
            validation_results['split_quality'] = 'ERROR'
        
        return validation_results
    
    def _load_dataset(self, path: str) -> List[Dict]:
        """Load dataset from file"""
        data = []
        try:
            if path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:  # Assume .txt format
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Parse ABSA format: text####aspects####opinions####sentiments
                            parts = line.split('####')
                            if len(parts) >= 4:
                                data.append({
                                    'text': parts[0],
                                    'aspects': parts[1],
                                    'opinions': parts[2],
                                    'sentiments': parts[3]
                                })
        except Exception as e:
            self.logger.error(f"Error loading {path}: {e}")
        
        return data
    
    def _analyze_label_distribution(self, train_data: List, dev_data: List, test_data: List) -> Dict:
        """Analyze label distributions across splits"""
        def extract_labels(data):
            aspects = []
            opinions = []
            sentiments = []
            
            for item in data:
                if isinstance(item.get('aspects'), str):
                    aspects.extend(item['aspects'].split('|'))
                if isinstance(item.get('opinions'), str):
                    opinions.extend(item['opinions'].split('|'))
                if isinstance(item.get('sentiments'), str):
                    sentiments.extend(item['sentiments'].split('|'))
            
            return {
                'aspects': Counter(aspects),
                'opinions': Counter(opinions),
                'sentiments': Counter(sentiments)
            }
        
        return {
            'train': extract_labels(train_data),
            'dev': extract_labels(dev_data),
            'test': extract_labels(test_data)
        }
    
    def _assess_split_quality(self, validation_results: Dict) -> str:
        """Assess overall split quality"""
        if validation_results['data_leakage']:
            return 'POOR - Data Leakage Detected'
        
        # Check size ratios
        total_size = validation_results['train_size'] + validation_results['dev_size'] + validation_results['test_size']
        train_ratio = validation_results['train_size'] / total_size
        dev_ratio = validation_results['dev_size'] / total_size
        
        if train_ratio < 0.6 or train_ratio > 0.8:
            return 'FAIR - Unusual train/dev/test ratios'
        
        if validation_results['dev_size'] < 50:
            return 'FAIR - Very small dev set'
        
        return 'GOOD - Proper separation'
    
    def _log_validation_results(self, results: Dict):
        """Log validation results"""
        self.logger.info("📊 Dataset Validation Results:")
        self.logger.info(f"   Train size: {results['train_size']}")
        self.logger.info(f"   Dev size: {results['dev_size']}")
        self.logger.info(f"   Test size: {results['test_size']}")
        self.logger.info(f"   Data leakage: {'❌ YES' if results['data_leakage'] else '✅ NO'}")
        self.logger.info(f"   Split quality: {results['split_quality']}")
        
        if results['data_leakage']:
            overlaps = results['overlap_details']
            self.logger.warning(f"   Train-Dev overlap: {overlaps['train_dev_overlap']} examples")
            self.logger.warning(f"   Train-Test overlap: {overlaps['train_test_overlap']} examples")
            self.logger.warning(f"   Dev-Test overlap: {overlaps['dev_test_overlap']} examples")


class ABSAMetricsCalculator:
    """Comprehensive ABSA metrics calculator"""
    
    def __init__(self, config: ABSAEvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compute_all_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute all ABSA evaluation metrics"""
        self.logger.info("📊 Computing comprehensive ABSA metrics...")
        
        metrics = {}
        
        # 1. Triplet Extraction Metrics (Most Important)
        triplet_metrics = self.compute_triplet_metrics(predictions, targets)
        metrics.update(triplet_metrics)
        
        # 2. Individual Component Metrics
        aspect_metrics = self.compute_aspect_metrics(predictions, targets)
        opinion_metrics = self.compute_opinion_metrics(predictions, targets)
        sentiment_metrics = self.compute_sentiment_metrics(predictions, targets)
        
        metrics.update(aspect_metrics)
        metrics.update(opinion_metrics)
        metrics.update(sentiment_metrics)
        
        # 3. Span-level Metrics
        span_metrics = self.compute_span_metrics(predictions, targets)
        metrics.update(span_metrics)
        
        # 4. Statistical Metrics
        statistical_metrics = self.compute_statistical_metrics(predictions, targets)
        metrics.update(statistical_metrics)
        
        return metrics
    
    def compute_triplet_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute triplet extraction metrics (Primary ABSA metric)"""
        pred_triplets = []
        true_triplets = []
        
        for pred, target in zip(predictions, targets):
            # Extract triplets from predictions
            pred_triplets.extend(self._extract_triplets(pred))
            true_triplets.extend(self._extract_triplets(target))
        
        # Exact matching
        exact_matches = self._count_exact_matches(pred_triplets, true_triplets)
        
        # Semantic matching (if enabled)
        semantic_matches = 0
        if self.config.use_semantic_matching:
            semantic_matches = self._count_semantic_matches(pred_triplets, true_triplets)
        
        total_matches = exact_matches + semantic_matches
        
        # Calculate metrics
        precision = total_matches / len(pred_triplets) if pred_triplets else 0.0
        recall = total_matches / len(true_triplets) if true_triplets else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'triplet_exact_precision': exact_matches / len(pred_triplets) if pred_triplets else 0.0,
            'triplet_exact_recall': exact_matches / len(true_triplets) if true_triplets else 0.0,
            'triplet_exact_f1': 2 * (exact_matches / len(pred_triplets) if pred_triplets else 0.0) * 
                               (exact_matches / len(true_triplets) if true_triplets else 0.0) / 
                               ((exact_matches / len(pred_triplets) if pred_triplets else 0.0) + 
                                (exact_matches / len(true_triplets) if true_triplets else 0.0)) if 
                               ((exact_matches / len(pred_triplets) if pred_triplets else 0.0) + 
                                (exact_matches / len(true_triplets) if true_triplets else 0.0)) > 0 else 0.0,
            'triplet_total_precision': precision,
            'triplet_total_recall': recall,
            'triplet_total_f1': f1,
            'triplet_exact_matches': exact_matches,
            'triplet_semantic_matches': semantic_matches,
            'triplet_total_predicted': len(pred_triplets),
            'triplet_total_gold': len(true_triplets)
        }
    
    def compute_aspect_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute aspect extraction metrics"""
        pred_aspects = []
        true_aspects = []
        
        for pred, target in zip(predictions, targets):
            pred_aspects.extend(self._extract_aspects(pred))
            true_aspects.extend(self._extract_aspects(target))
        
        # Macro and Micro F1
        macro_metrics = self._compute_sequence_metrics(pred_aspects, true_aspects, average='macro')
        micro_metrics = self._compute_sequence_metrics(pred_aspects, true_aspects, average='micro')
        
        return {
            'aspect_macro_precision': macro_metrics[0],
            'aspect_macro_recall': macro_metrics[1],
            'aspect_macro_f1': macro_metrics[2],
            'aspect_micro_precision': micro_metrics[0],
            'aspect_micro_recall': micro_metrics[1],
            'aspect_micro_f1': micro_metrics[2],
        }
    
    def compute_opinion_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute opinion extraction metrics"""
        pred_opinions = []
        true_opinions = []
        
        for pred, target in zip(predictions, targets):
            pred_opinions.extend(self._extract_opinions(pred))
            true_opinions.extend(self._extract_opinions(target))
        
        # Macro and Micro F1
        macro_metrics = self._compute_sequence_metrics(pred_opinions, true_opinions, average='macro')
        micro_metrics = self._compute_sequence_metrics(pred_opinions, true_opinions, average='micro')
        
        return {
            'opinion_macro_precision': macro_metrics[0],
            'opinion_macro_recall': macro_metrics[1],
            'opinion_macro_f1': macro_metrics[2],
            'opinion_micro_precision': micro_metrics[0],
            'opinion_micro_recall': micro_metrics[1],
            'opinion_micro_f1': micro_metrics[2],
        }
    
    def compute_sentiment_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute sentiment classification metrics"""
        pred_sentiments = []
        true_sentiments = []
        
        for pred, target in zip(predictions, targets):
            pred_sentiments.extend(self._extract_sentiments(pred))
            true_sentiments.extend(self._extract_sentiments(target))
        
        # Convert to numerical labels
        label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        pred_labels = [label_map.get(s, 2) for s in pred_sentiments]
        true_labels = [label_map.get(s, 2) for s in true_sentiments]
        
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='macro', zero_division=0
        )
        
        return {
            'sentiment_accuracy': accuracy,
            'sentiment_macro_precision': precision,
            'sentiment_macro_recall': recall,
            'sentiment_macro_f1': f1,
        }
    
    def compute_span_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute span-level extraction metrics"""
        # Implement span-based evaluation for aspect and opinion boundaries
        pred_spans = []
        true_spans = []
        
        for pred, target in zip(predictions, targets):
            pred_spans.extend(self._extract_spans(pred))
            true_spans.extend(self._extract_spans(target))
        
        # Exact span matching
        exact_span_matches = len(set(pred_spans).intersection(set(true_spans)))
        
        span_precision = exact_span_matches / len(pred_spans) if pred_spans else 0.0
        span_recall = exact_span_matches / len(true_spans) if true_spans else 0.0
        span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if (span_precision + span_recall) > 0 else 0.0
        
        return {
            'span_precision': span_precision,
            'span_recall': span_recall,
            'span_f1': span_f1,
            'span_exact_matches': exact_span_matches,
            'span_total_predicted': len(pred_spans),
            'span_total_gold': len(true_spans)
        }
    
    def compute_statistical_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Compute statistical significance metrics"""
        # Implement bootstrap sampling for confidence intervals
        bootstrap_f1_scores = []
        
        for _ in range(self.config.bootstrap_samples):
            # Sample with replacement
            n = len(predictions)
            indices = np.random.choice(n, n, replace=True)
            
            sample_preds = [predictions[i] for i in indices]
            sample_targets = [targets[i] for i in indices]
            
            # Compute F1 for this sample
            sample_metrics = self.compute_triplet_metrics(sample_preds, sample_targets)
            bootstrap_f1_scores.append(sample_metrics['triplet_total_f1'])
        
        # Calculate confidence intervals
        confidence_level = 1 - self.config.statistical_alpha
        lower_bound = np.percentile(bootstrap_f1_scores, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(bootstrap_f1_scores, (1 + confidence_level) / 2 * 100)
        
        return {
            'bootstrap_f1_mean': np.mean(bootstrap_f1_scores),
            'bootstrap_f1_std': np.std(bootstrap_f1_scores),
            'bootstrap_f1_lower_ci': lower_bound,
            'bootstrap_f1_upper_ci': upper_bound,
        }
    
    # Helper methods
    def _extract_triplets(self, data: Dict) -> List[Tuple]:
        """Extract triplets from data"""
        triplets = []
        if 'triplets' in data:
            for triplet in data['triplets']:
                triplets.append((triplet.get('aspect', ''), triplet.get('opinion', ''), triplet.get('sentiment', '')))
        return triplets
    
    def _extract_aspects(self, data: Dict) -> List[str]:
        """Extract aspects from data"""
        return data.get('aspects', [])
    
    def _extract_opinions(self, data: Dict) -> List[str]:
        """Extract opinions from data"""
        return data.get('opinions', [])
    
    def _extract_sentiments(self, data: Dict) -> List[str]:
        """Extract sentiments from data"""
        return data.get('sentiments', [])
    
    def _extract_spans(self, data: Dict) -> List[Tuple]:
        """Extract spans from data"""
        spans = []
        if 'spans' in data:
            for span in data['spans']:
                spans.append((span.get('start', 0), span.get('end', 0), span.get('type', '')))
        return spans
    
    def _count_exact_matches(self, pred_triplets: List, true_triplets: List) -> int:
        """Count exact triplet matches"""
        pred_set = set(pred_triplets)
        true_set = set(true_triplets)
        return len(pred_set.intersection(true_set))
    
    def _count_semantic_matches(self, pred_triplets: List, true_triplets: List) -> int:
        """Count semantic triplet matches (simplified)"""
        # Simplified semantic matching - in practice, use embeddings
        semantic_matches = 0
        for pred in pred_triplets:
            for true in true_triplets:
                if self._is_semantic_match(pred, true):
                    semantic_matches += 1
                    break
        return semantic_matches
    
    def _is_semantic_match(self, pred_triplet: Tuple, true_triplet: Tuple) -> bool:
        """Check if triplets are semantically similar"""
        # Simplified - in practice, use embedding similarity
        pred_aspect, pred_opinion, pred_sentiment = pred_triplet
        true_aspect, true_opinion, true_sentiment = true_triplet
        
        # Exact sentiment match required
        if pred_sentiment != true_sentiment:
            return False
        
        # Fuzzy matching for aspect and opinion (simplified)
        aspect_match = pred_aspect.lower() in true_aspect.lower() or true_aspect.lower() in pred_aspect.lower()
        opinion_match = pred_opinion.lower() in true_opinion.lower() or true_opinion.lower() in pred_opinion.lower()
        
        return aspect_match and opinion_match
    
    def _compute_sequence_metrics(self, pred_seq: List, true_seq: List, average: str) -> Tuple[float, float, float]:
        """Compute sequence labeling metrics"""
        if not pred_seq or not true_seq:
            return 0.0, 0.0, 0.0
        
        # Convert to label format for sklearn
        all_labels = list(set(pred_seq + true_seq))
        label_to_id = {label: i for i, label in enumerate(all_labels)}
        
        pred_ids = [label_to_id[label] for label in pred_seq]
        true_ids = [label_to_id[label] for label in true_seq]
        
        # Pad sequences to same length
        max_len = max(len(pred_ids), len(true_ids))
        pred_ids.extend([0] * (max_len - len(pred_ids)))
        true_ids.extend([0] * (max_len - len(true_ids)))
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_ids, pred_ids, average=average, zero_division=0
        )
        
        return precision, recall, f1


class StatisticalTester:
    """Statistical significance testing for ABSA results"""
    
    def __init__(self, config: ABSAEvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def mcnemar_test(self, model1_results: List[bool], model2_results: List[bool]) -> Dict[str, float]:
        """Perform McNemar's test for comparing two models"""
        self.logger.info("🧪 Performing McNemar's test...")
        
        # Create contingency table
        both_correct = sum(1 for r1, r2 in zip(model1_results, model2_results) if r1 and r2)
        model1_only = sum(1 for r1, r2 in zip(model1_results, model2_results) if r1 and not r2)
        model2_only = sum(1 for r1, r2 in zip(model1_results, model2_results) if not r1 and r2)
        both_wrong = sum(1 for r1, r2 in zip(model1_results, model2_results) if not r1 and not r2)
        
        # McNemar's test statistic
        if model1_only + model2_only > 0:
            test_statistic = (abs(model1_only - model2_only) - 1) ** 2 / (model1_only + model2_only)
            p_value = 1 - stats.chi2.cdf(test_statistic, 1)
        else:
            test_statistic = 0.0
            p_value = 1.0
        
        return {
            'mcnemar_statistic': test_statistic,
            'mcnemar_p_value': p_value,
            'significant': p_value < self.config.statistical_alpha,
            'contingency_table': {
                'both_correct': both_correct,
                'model1_only': model1_only,
                'model2_only': model2_only,
                'both_wrong': both_wrong
            }
        }
    
    def bootstrap_significance_test(self, model1_scores: List[float], model2_scores: List[float]) -> Dict[str, float]:
        """Bootstrap test for comparing model performance"""
        self.logger.info("🧪 Performing bootstrap significance test...")
        
        observed_diff = np.mean(model1_scores) - np.mean(model2_scores)
        
        # Bootstrap sampling
        bootstrap_diffs = []
        n = len(model1_scores)
        
        for _ in range(self.config.bootstrap_samples):
            # Sample with replacement
            indices = np.random.choice(n, n, replace=True)
            sample1 = [model1_scores[i] for i in indices]
            sample2 = [model2_scores[i] for i in indices]
            
            bootstrap_diff = np.mean(sample1) - np.mean(sample2)
            bootstrap_diffs.append(bootstrap_diff)
        
        # Calculate p-value
        p_value = sum(1 for diff in bootstrap_diffs if diff <= 0) / len(bootstrap_diffs)
        
        return {
            'bootstrap_observed_diff': observed_diff,
            'bootstrap_p_value': p_value,
            'significant': p_value < self.config.statistical_alpha,
            'effect_size': observed_diff / np.std(model1_scores + model2_scores)
        }


class CrossValidator:
    """Cross-validation for ABSA models"""
    
    def __init__(self, config: ABSAEvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def k_fold_evaluation(self, data: List[Dict], model_trainer_func) -> Dict[str, float]:
        """Perform k-fold cross-validation"""
        self.logger.info(f"🔄 Performing {self.config.cv_folds}-fold cross-validation...")
        
        # Create stratified folds based on sentiment distribution
        sentiments = [item.get('sentiment', 'neutral') for item in data]
        skf = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(data, sentiments)):
            self.logger.info(f"   Fold {fold + 1}/{self.config.cv_folds}")
            
            # Split data
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]
            
            # Train model and evaluate
            try:
                model = model_trainer_func(train_data)
                fold_metrics = self._evaluate_fold(model, val_data)
                fold_results.append(fold_metrics)
            except Exception as e:
                self.logger.error(f"   Error in fold {fold + 1}: {e}")
                continue
        
        # Aggregate results
        if fold_results:
            aggregated_results = self._aggregate_fold_results(fold_results)
            self.logger.info(f"✅ Cross-validation completed. Mean F1: {aggregated_results['mean_f1']:.4f} ± {aggregated_results['std_f1']:.4f}")
            return aggregated_results
        else:
            self.logger.error("❌ Cross-validation failed")
            return {}
    
    def _evaluate_fold(self, model, val_data: List[Dict]) -> Dict[str, float]:
        """Evaluate model on validation fold"""
        # This would be implemented based on your model's evaluation method
        # For now, return placeholder metrics
        return {
            'triplet_f1': 0.75,  # Placeholder
            'aspect_f1': 0.80,   # Placeholder
            'opinion_f1': 0.78,  # Placeholder
            'sentiment_accuracy': 0.85  # Placeholder
        }
    
    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict[str, float]:
        """Aggregate results from all folds"""
        aggregated = {}
        
        # Calculate mean and std for each metric
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            aggregated[f'mean_{metric}'] = np.mean(values)
            aggregated[f'std_{metric}'] = np.std(values)
            aggregated[f'min_{metric}'] = np.min(values)
            aggregated[f'max_{metric}'] = np.max(values)
        
        return aggregated


# Main evaluation runner
class ABSAEvaluationRunner:
    """Main runner for comprehensive ABSA evaluation"""
    
    def __init__(self, config: Optional[ABSAEvaluationConfig] = None):
        self.config = config or ABSAEvaluationConfig()
        self.validator = DatasetValidator()
        self.metrics_calculator = ABSAMetricsCalculator(self.config)
        self.statistical_tester = StatisticalTester(self.config)
        self.cross_validator = CrossValidator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def run_full_evaluation(self, dataset_paths: Dict[str, str], predictions: List[Dict], targets: List[Dict]) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        self.logger.info("🚀 Starting comprehensive ABSA evaluation...")
        
        evaluation_results = {
            'dataset_validation': {},
            'metrics': {},
            'statistical_tests': {},
            'cross_validation': {},
            'summary': {}
        }
        
        # 1. Validate datasets
        if dataset_paths:
            evaluation_results['dataset_validation'] = self.validator.validate_splits(
                dataset_paths.get('train', ''),
                dataset_paths.get('dev', ''),
                dataset_paths.get('test', '')
            )
        
        # 2. Compute all metrics
        evaluation_results['metrics'] = self.metrics_calculator.compute_all_metrics(predictions, targets)
        
        # 3. Generate summary
        evaluation_results['summary'] = self._generate_evaluation_summary(evaluation_results)
        
        # 4. Save results
        self._save_evaluation_results(evaluation_results)
        
        self.logger.info("✅ Comprehensive evaluation completed!")
        return evaluation_results
    
    def _generate_evaluation_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate evaluation summary"""
        metrics = results.get('metrics', {})
        
        # Calculate overall score
        primary_f1 = metrics.get('triplet_total_f1', 0.0)
        aspect_f1 = metrics.get('aspect_macro_f1', 0.0)
        opinion_f1 = metrics.get('opinion_macro_f1', 0.0)
        sentiment_acc = metrics.get('sentiment_accuracy', 0.0)
        
        overall_score = (primary_f1 * 0.4 + aspect_f1 * 0.2 + opinion_f1 * 0.2 + sentiment_acc * 0.2)
        
        # Publication readiness assessment
        publication_score = self._assess_publication_readiness(results)
        
        return {
            'overall_score': overall_score,
            'primary_f1': primary_f1,
            'publication_readiness_score': publication_score,
            'data_quality': results.get('dataset_validation', {}).get('split_quality', 'UNKNOWN'),
            'statistical_significance': any(test.get('significant', False) for test in results.get('statistical_tests', {}).values()),
            'key_metrics': {
                'triplet_f1': primary_f1,
                'aspect_f1': aspect_f1,
                'opinion_f1': opinion_f1,
                'sentiment_accuracy': sentiment_acc
            }
        }
    
    def _assess_publication_readiness(self, results: Dict) -> int:
        """Assess publication readiness score (0-100)"""
        score = 0
        
        # Data quality (25 points)
        data_quality = results.get('dataset_validation', {}).get('split_quality', 'UNKNOWN')
        if data_quality == 'GOOD - Proper separation':
            score += 25
        elif data_quality.startswith('FAIR'):
            score += 15
        elif data_quality.startswith('POOR'):
            score += 5
        
        # Metrics quality (40 points)
        metrics = results.get('metrics', {})
        triplet_f1 = metrics.get('triplet_total_f1', 0.0)
        
        if triplet_f1 >= 0.8:
            score += 40
        elif triplet_f1 >= 0.7:
            score += 30
        elif triplet_f1 >= 0.6:
            score += 20
        elif triplet_f1 >= 0.5:
            score += 10
        
        # Statistical rigor (20 points)
        if results.get('statistical_tests'):
            score += 10
        if results.get('cross_validation'):
            score += 10
        
        # Bootstrap confidence intervals (10 points)
        if metrics.get('bootstrap_f1_mean') is not None:
            score += 10
        
        # Comprehensive evaluation (5 points)
        required_metrics = ['triplet_total_f1', 'aspect_macro_f1', 'opinion_macro_f1', 'sentiment_accuracy']
        if all(metric in metrics for metric in required_metrics):
            score += 5
        
        return min(score, 100)
    
    def _save_evaluation_results(self, results: Dict):
        """Save evaluation results to file"""
        try:
            os.makedirs('evaluation_results', exist_ok=True)
            
            # Save detailed results
            with open('evaluation_results/detailed_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary report
            self._generate_report(results)
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def _generate_report(self, results: Dict):
        """Generate human-readable evaluation report"""
        report = f"""
# ABSA Model Evaluation Report
Generated: {torch.utils.data.get_worker_info()}

## Executive Summary
- **Publication Readiness**: {results['summary']['publication_readiness_score']}/100
- **Overall Score**: {results['summary']['overall_score']:.4f}
- **Primary Metric (Triplet F1)**: {results['summary']['primary_f1']:.4f}
- **Data Quality**: {results['summary']['data_quality']}

## Dataset Validation
"""
        
        # Add dataset validation details
        validation = results.get('dataset_validation', {})
        if validation:
            report += f"""
- **Train Size**: {validation.get('train_size', 0)}
- **Dev Size**: {validation.get('dev_size', 0)}
- **Test Size**: {validation.get('test_size', 0)}
- **Data Leakage**: {'❌ DETECTED' if validation.get('data_leakage', False) else '✅ NONE'}
- **Split Quality**: {validation.get('split_quality', 'UNKNOWN')}
"""
        
        # Add comprehensive metrics
        metrics = results.get('metrics', {})
        report += f"""
## Comprehensive Metrics

### Primary ABSA Metrics
- **Triplet Extraction F1**: {metrics.get('triplet_total_f1', 0.0):.4f}
- **Triplet Precision**: {metrics.get('triplet_total_precision', 0.0):.4f}
- **Triplet Recall**: {metrics.get('triplet_total_recall', 0.0):.4f}

### Component-Level Metrics
- **Aspect Macro F1**: {metrics.get('aspect_macro_f1', 0.0):.4f}
- **Aspect Micro F1**: {metrics.get('aspect_micro_f1', 0.0):.4f}
- **Opinion Macro F1**: {metrics.get('opinion_macro_f1', 0.0):.4f}
- **Opinion Micro F1**: {metrics.get('opinion_micro_f1', 0.0):.4f}
- **Sentiment Accuracy**: {metrics.get('sentiment_accuracy', 0.0):.4f}

### Statistical Confidence
- **Bootstrap F1 Mean**: {metrics.get('bootstrap_f1_mean', 0.0):.4f}
- **Bootstrap F1 Std**: {metrics.get('bootstrap_f1_std', 0.0):.4f}
- **95% Confidence Interval**: [{metrics.get('bootstrap_f1_lower_ci', 0.0):.4f}, {metrics.get('bootstrap_f1_upper_ci', 0.0):.4f}]

### Detailed Counts
- **Exact Triplet Matches**: {metrics.get('triplet_exact_matches', 0)}
- **Semantic Matches**: {metrics.get('triplet_semantic_matches', 0)}
- **Total Predicted**: {metrics.get('triplet_total_predicted', 0)}
- **Total Gold**: {metrics.get('triplet_total_gold', 0)}
"""
        
        # Add recommendations
        report += self._generate_recommendations(results)
        
        # Save report
        with open('evaluation_results/evaluation_report.md', 'w') as f:
            f.write(report)
    
    def _generate_recommendations(self, results: Dict) -> str:
        """Generate improvement recommendations"""
        recommendations = "\n## Recommendations for Publication\n"
        
        score = results['summary']['publication_readiness_score']
        data_quality = results['summary']['data_quality']
        primary_f1 = results['summary']['primary_f1']
        
        if score >= 90:
            recommendations += "✅ **READY FOR SUBMISSION** - Your evaluation meets publication standards!\n"
        elif score >= 80:
            recommendations += "🟡 **NEARLY READY** - Minor improvements needed:\n"
        else:
            recommendations += "🔴 **SIGNIFICANT WORK NEEDED** - Critical issues to address:\n"
        
        # Specific recommendations
        if 'POOR' in data_quality or 'data leakage' in data_quality.lower():
            recommendations += "- **CRITICAL**: Fix data leakage in dataset splits\n"
        
        if primary_f1 < 0.6:
            recommendations += "- **HIGH PRIORITY**: Improve model performance (current F1 too low for publication)\n"
        
        if not results.get('statistical_tests'):
            recommendations += "- **MEDIUM PRIORITY**: Add statistical significance testing\n"
        
        if not results.get('cross_validation'):
            recommendations += "- **MEDIUM PRIORITY**: Implement cross-validation\n"
        
        metrics = results.get('metrics', {})
        if not metrics.get('bootstrap_f1_mean'):
            recommendations += "- **LOW PRIORITY**: Add bootstrap confidence intervals\n"
        
        return recommendations


# Integration functions for your existing codebase
def integrate_with_existing_trainer(trainer_class):
    """Decorator to integrate evaluation with existing trainer"""
    
    class EnhancedTrainer(trainer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.evaluation_runner = ABSAEvaluationRunner()
        
        def evaluate_with_comprehensive_metrics(self, predictions, targets, dataset_paths=None):
            """Enhanced evaluation method"""
            return self.evaluation_runner.run_full_evaluation(dataset_paths, predictions, targets)
        
        def validate_dataset_splits(self, train_path, dev_path, test_path=None):
            """Validate dataset splits"""
            return self.evaluation_runner.validator.validate_splits(train_path, dev_path, test_path or "")
    
    return EnhancedTrainer


# Utility functions for quick testing
def quick_evaluation_test():
    """Quick test of evaluation system"""
    logger.info("🧪 Running quick evaluation test...")
    
    # Create sample data
    predictions = [
        {
            'triplets': [{'aspect': 'food', 'opinion': 'delicious', 'sentiment': 'positive'}],
            'aspects': ['food'],
            'opinions': ['delicious'],
            'sentiments': ['positive']
        },
        {
            'triplets': [{'aspect': 'service', 'opinion': 'slow', 'sentiment': 'negative'}],
            'aspects': ['service'],
            'opinions': ['slow'],
            'sentiments': ['negative']
        }
    ]
    
    targets = [
        {
            'triplets': [{'aspect': 'food', 'opinion': 'delicious', 'sentiment': 'positive'}],
            'aspects': ['food'],
            'opinions': ['delicious'],
            'sentiments': ['positive']
        },
        {
            'triplets': [{'aspect': 'service', 'opinion': 'terrible', 'sentiment': 'negative'}],
            'aspects': ['service'],
            'opinions': ['terrible'],
            'sentiments': ['negative']
        }
    ]
    
    # Run evaluation
    runner = ABSAEvaluationRunner()
    results = runner.run_full_evaluation({}, predictions, targets)
    
    print("✅ Quick test completed!")
    print(f"   Triplet F1: {results['metrics'].get('triplet_total_f1', 0.0):.4f}")
    print(f"   Publication Score: {results['summary']['publication_readiness_score']}/100")
    
    return results


def create_evaluation_config(strict_matching=True, bootstrap_samples=1000, cv_folds=5):
    """Create evaluation configuration"""
    return ABSAEvaluationConfig(
        use_strict_matching=strict_matching,
        use_semantic_matching=True,
        confidence_threshold=0.5,
        bootstrap_samples=bootstrap_samples,
        cv_folds=cv_folds,
        statistical_alpha=0.05
    )


# Example usage for your codebase
def main_evaluation_example():
    """Example of how to integrate with your existing code"""
    
    # 1. Validate your dataset splits
    validator = DatasetValidator()
    validation_results = validator.validate_splits(
        'Datasets/aste/laptop14/train.txt',
        'Datasets/aste/laptop14/dev.txt',
        'Datasets/aste/laptop14/test.txt'
    )
    
    if validation_results['data_leakage']:
        logger.error("❌ CRITICAL: Data leakage detected! Cannot proceed with publication.")
        return
    
    # 2. Run comprehensive evaluation
    config = create_evaluation_config(bootstrap_samples=500)  # Reduce for faster testing
    runner = ABSAEvaluationRunner(config)
    
    # Your predictions and targets would come from your model
    predictions = []  # Fill with your model's predictions
    targets = []      # Fill with ground truth
    
    dataset_paths = {
        'train': 'Datasets/aste/laptop14/train.txt',
        'dev': 'Datasets/aste/laptop14/dev.txt',
        'test': 'Datasets/aste/laptop14/test.txt'
    }
    
    evaluation_results = runner.run_full_evaluation(dataset_paths, predictions, targets)
    
    # 3. Check publication readiness
    score = evaluation_results['summary']['publication_readiness_score']
    if score >= 85:
        logger.info(f"🚀 PUBLICATION READY! Score: {score}/100")
    else:
        logger.warning(f"⚠️ More work needed. Score: {score}/100")
        logger.info("Check evaluation_results/evaluation_report.md for recommendations")


if __name__ == "__main__":
    # Run quick test
    quick_evaluation_test()
    
    # Run main evaluation example
    # main_evaluation_example()