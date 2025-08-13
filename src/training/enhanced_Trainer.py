
# Enhanced Trainer with Proper Evaluation
import torch
import logging
from typing import Dict, List, Any, Optional
from .fixed_metrics import compute_absa_metrics, compute_bootstrap_confidence

logger = logging.getLogger(__name__)

class EnhancedABSATrainer:
    """Enhanced trainer with proper evaluation metrics"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Track evaluation history
        self.evaluation_history = []
        self.best_score = 0.0
        self.best_epoch = 0
    
    def evaluate_model(self, dataloader, split_name: str = "validation") -> Dict[str, float]:
        """
        Proper evaluation with comprehensive metrics
        
        CRITICAL: This replaces the broken evaluation that gives perfect scores
        """
        self.logger.info(f"🔍 Evaluating on {split_name} set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Forward pass
                    outputs = self.model(batch)
                    
                    # Extract predictions and targets
                    batch_predictions = self._extract_predictions_from_outputs(outputs, batch)
                    batch_targets = self._extract_targets_from_batch(batch)
                    
                    all_predictions.extend(batch_predictions)
                    all_targets.extend(batch_targets)
                    
                    if batch_idx % 10 == 0:
                        self.logger.info(f"   Processed batch {batch_idx}/{len(dataloader)}")
                
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Compute comprehensive metrics
        metrics = compute_absa_metrics(all_predictions, all_targets)
        
        # Add bootstrap confidence intervals for important metrics
        if len(all_predictions) > 50:  # Only if we have enough samples
            bootstrap_metrics = compute_bootstrap_confidence(all_predictions, all_targets)
            metrics.update(bootstrap_metrics)
        
        # Log results
        self._log_evaluation_results(metrics, split_name)
        
        # Store in history
        self.evaluation_history.append({
            'split': split_name,
            'metrics': metrics,
            'epoch': getattr(self, 'current_epoch', -1)
        })
        
        return metrics
    
    def _extract_predictions_from_outputs(self, outputs: Dict, batch: Dict) -> List[Dict]:
        """
        Extract predictions from model outputs
        
        CRITICAL: Adapt this to your actual model output format
        """
        predictions = []
        
        batch_size = outputs.get('aspect_logits', torch.tensor([])).size(0)
        
        for i in range(batch_size):
            pred = {
                'triplets': [],
                'aspects': [],
                'opinions': [],
                'sentiments': []
            }
            
            # Extract triplets from model outputs
            # ADAPT THIS SECTION TO YOUR MODEL'S OUTPUT FORMAT
            try:
                if 'triplet_predictions' in outputs:
                    # Direct triplet predictions
                    triplet_preds = outputs['triplet_predictions'][i]
                    for triplet in triplet_preds:
                        pred['triplets'].append({
                            'aspect': triplet.get('aspect', ''),
                            'opinion': triplet.get('opinion', ''),
                            'sentiment': triplet.get('sentiment', '')
                        })
                else:
                    # Reconstruct from individual components
                    aspect_logits = outputs.get('aspect_logits', torch.tensor([]))[i] if 'aspect_logits' in outputs else None
                    opinion_logits = outputs.get('opinion_logits', torch.tensor([]))[i] if 'opinion_logits' in outputs else None
                    sentiment_logits = outputs.get('sentiment_logits', torch.tensor([]))[i] if 'sentiment_logits' in outputs else None
                    
                    # Convert logits to predictions
                    if aspect_logits is not None:
                        aspect_preds = torch.argmax(aspect_logits, dim=-1)
                        pred['aspects'] = self._decode_sequence_predictions(aspect_preds, batch['texts'][i])
                    
                    if opinion_logits is not None:
                        opinion_preds = torch.argmax(opinion_logits, dim=-1)
                        pred['opinions'] = self._decode_sequence_predictions(opinion_preds, batch['texts'][i])
                    
                    if sentiment_logits is not None:
                        sentiment_preds = torch.argmax(sentiment_logits, dim=-1)
                        pred['sentiments'] = self._decode_sentiment_predictions(sentiment_preds)
            
            except Exception as e:
                self.logger.warning(f"Error extracting predictions for sample {i}: {e}")
                # Return empty prediction to avoid crashes
                pass
            
            predictions.append(pred)
        
        return predictions
    
    def _extract_targets_from_batch(self, batch: Dict) -> List[Dict]:
        """
        Extract ground truth targets from batch
        
        CRITICAL: Adapt this to your actual data format
        """
        targets = []
        
        batch_size = len(batch.get('texts', []))
        
        for i in range(batch_size):
            target = {
                'triplets': [],
                'aspects': [],
                'opinions': [],
                'sentiments': []
            }
            
            try:
                # Extract from your batch format
                # ADAPT THIS SECTION TO YOUR DATA FORMAT
                
                if 'aspect_labels' in batch:
                    aspect_labels = batch['aspect_labels'][i]
                    target['aspects'] = self._decode_sequence_labels(aspect_labels, batch['texts'][i])
                
                if 'opinion_labels' in batch:
                    opinion_labels = batch['opinion_labels'][i]
                    target['opinions'] = self._decode_sequence_labels(opinion_labels, batch['texts'][i])
                
                if 'sentiment_labels' in batch:
                    sentiment_labels = batch['sentiment_labels'][i]
                    target['sentiments'] = self._decode_sentiment_labels(sentiment_labels)
                
                # If you have direct triplet labels, use them
                if 'triplet_labels' in batch:
                    triplet_labels = batch['triplet_labels'][i]
                    for triplet in triplet_labels:
                        target['triplets'].append({
                            'aspect': triplet.get('aspect', ''),
                            'opinion': triplet.get('opinion', ''),
                            'sentiment': triplet.get('sentiment', '')
                        })
            
            except Exception as e:
                self.logger.warning(f"Error extracting targets for sample {i}: {e}")
                # Return empty target to avoid crashes
                pass
            
            targets.append(target)
        
        return targets
    
    def _decode_sequence_predictions(self, predictions: torch.Tensor, text: str) -> List[str]:
        """Decode sequence predictions to text spans"""
        # Simplified implementation - adapt to your tokenization
        decoded = []
        
        # Convert BIO tags to spans
        current_span = []
        for i, pred in enumerate(predictions):
            if pred == 1:  # B- tag
                if current_span:
                    decoded.append(' '.join(current_span))
                current_span = [f"token_{i}"]  # Replace with actual tokens
            elif pred == 2:  # I- tag
                if current_span:
                    current_span.append(f"token_{i}")
            else:  # O tag
                if current_span:
                    decoded.append(' '.join(current_span))
                    current_span = []
        
        if current_span:
            decoded.append(' '.join(current_span))
        
        return decoded
    
    def _decode_sequence_labels(self, labels: torch.Tensor, text: str) -> List[str]:
        """Decode sequence labels to text spans"""
        return self._decode_sequence_predictions(labels, text)
    
    def _decode_sentiment_predictions(self, predictions: torch.Tensor) -> List[str]:
        """Decode sentiment predictions"""
        sentiment_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        return [sentiment_map.get(int(pred), 'neutral') for pred in predictions]
    
    def _decode_sentiment_labels(self, labels: torch.Tensor) -> List[str]:
        """Decode sentiment labels"""
        return self._decode_sentiment_predictions(labels)
    
    def _log_evaluation_results(self, metrics: Dict[str, float], split_name: str):
        """Log evaluation results"""
        self.logger.info(f"📊 {split_name.upper()} RESULTS:")
        
        # Primary metrics
        triplet_f1 = metrics.get('triplet_f1', 0.0)
        aspect_f1 = metrics.get('aspect_f1', 0.0)
        opinion_f1 = metrics.get('opinion_f1', 0.0)
        sentiment_acc = metrics.get('sentiment_accuracy', 0.0)
        
        self.logger.info(f"   🎯 Triplet F1: {triplet_f1:.4f}")
        self.logger.info(f"   📝 Aspect F1: {aspect_f1:.4f}")
        self.logger.info(f"   💭 Opinion F1: {opinion_f1:.4f}")
        self.logger.info(f"   😊 Sentiment Acc: {sentiment_acc:.4f}")
        
        # Bootstrap confidence intervals if available
        if 'bootstrap_f1_mean' in metrics:
            ci_lower = metrics.get('bootstrap_f1_lower_ci', 0.0)
            ci_upper = metrics.get('bootstrap_f1_upper_ci', 0.0)
            self.logger.info(f"   📈 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Detailed counts
        exact_matches = metrics.get('triplet_exact_matches', 0)
        total_pred = metrics.get('triplet_total_predicted', 0)
        total_gold = metrics.get('triplet_total_gold', 0)
        self.logger.info(f"   🔢 Matches: {exact_matches}/{total_pred} predicted, {total_gold} gold")
        
        # Warning for suspicious results
        if triplet_f1 > 0.95:
            self.logger.warning("⚠️ WARNING: Suspiciously high F1 score - check for data leakage!")
        
        if triplet_f1 == 0.0 and total_pred > 0 and total_gold > 0:
            self.logger.warning("⚠️ WARNING: Zero F1 score - check prediction extraction logic!")
    
    def get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get primary metric for model selection"""
        return metrics.get('triplet_f1', 0.0)
    
    def is_best_score(self, metrics: Dict[str, float]) -> bool:
        """Check if this is the best score so far"""
        current_score = self.get_primary_metric(metrics)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_epoch = getattr(self, 'current_epoch', -1)
            return True
        return False
    
    def save_evaluation_report(self, output_dir: str):
        """Save comprehensive evaluation report"""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'evaluation_history': self.evaluation_history,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'model_config': str(self.config),
            'evaluation_summary': self._generate_evaluation_summary()
        }
        
        with open(output_path / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Evaluation report saved to {output_path}/evaluation_report.json")
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary"""
        if not self.evaluation_history:
            return {}
        
        # Get best results
        best_eval = max(self.evaluation_history, 
                       key=lambda x: x['metrics'].get('triplet_f1', 0.0))
        
        return {
            'best_triplet_f1': best_eval['metrics'].get('triplet_f1', 0.0),
            'best_epoch': best_eval.get('epoch', -1),
            'total_evaluations': len(self.evaluation_history),
            'final_score': self.evaluation_history[-1]['metrics'].get('triplet_f1', 0.0) if self.evaluation_history else 0.0
        }
