# EVALUATION PIPELINE DEBUG - Copy into Python terminal
# This will show exactly why Opinion F1 is zero despite valid spans

import torch
import numpy as np
from train import SimplifiedABSADataset, NovelGradientABSAModel, NovelABSAConfig, NovelABSATrainer, collate_fn
from torch.utils.data import DataLoader

print("EVALUATION PIPELINE DEBUG")
print("=" * 40)

# Create small test setup
config = NovelABSAConfig('dev')
model = NovelGradientABSAModel(config)
dataset = SimplifiedABSADataset("dummy", config.model_name, config.max_length, config.dataset_name)
dataset.data = dataset.data[:3]  # Just 3 samples

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
trainer = NovelABSATrainer(model, config, loader, loader, config.device)

# Get one batch and test the full evaluation pipeline
batch = next(iter(loader))

# Move to device
for key in batch:
    if isinstance(batch[key], torch.Tensor):
        batch[key] = batch[key].to(config.device)

print("TESTING FULL EVALUATION PIPELINE:")

# Get model predictions
with torch.no_grad():
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        training=False
    )

# Extract predictions exactly like trainer.evaluate() does
aspect_preds = torch.argmax(outputs['aspect_logits'], dim=-1).cpu().numpy()
opinion_preds = torch.argmax(outputs['opinion_logits'], dim=-1).cpu().numpy()
sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1).cpu().numpy()

aspect_targets = batch['aspect_labels'].cpu().numpy()
opinion_targets = batch['opinion_labels'].cpu().numpy()
sentiment_targets = batch['sentiment_labels'].cpu().numpy()
attention_mask = batch['attention_mask'].cpu().numpy()

print("RAW PREDICTIONS AND TARGETS:")
print(f"  Batch size: {len(aspect_preds)}")

# Process exactly like trainer.evaluate()
all_predictions = []
all_targets = []

for i in range(len(aspect_preds)):
    valid_mask = (attention_mask[i] == 1) & (aspect_targets[i] != -100)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        pred = {
            'aspect_preds': aspect_preds[i][valid_indices],
            'opinion_preds': opinion_preds[i][valid_indices], 
            'sentiment_preds': sentiment_preds[i][valid_indices]
        }
        
        target = {
            'aspect_labels': aspect_targets[i][valid_indices],
            'opinion_labels': opinion_targets[i][valid_indices],
            'sentiment_labels': sentiment_targets[i][valid_indices]
        }
        
        print(f"\n  Sample {i}:")
        print(f"    Opinion predictions: {pred['opinion_preds']}")
        print(f"    Opinion targets:     {target['opinion_labels']}")
        print(f"    Opinion B pred count: {sum(pred['opinion_preds'] == 1)}")
        print(f"    Opinion B target count: {sum(target['opinion_labels'] == 1)}")
        
        # Test span extraction on this sample
        pred_opinion_spans = trainer._extract_spans(pred['opinion_preds'], 'opinion')
        target_opinion_spans = trainer._extract_spans(target['opinion_labels'], 'opinion')
        
        print(f"    Extracted pred opinion spans: {pred_opinion_spans}")
        print(f"    Extracted target opinion spans: {target_opinion_spans}")
        
        all_predictions.append(pred)
        all_targets.append(target)

# Now test the full metric computation
print("\nTESTING METRIC COMPUTATION:")
metrics = trainer._compute_metrics(all_predictions, all_targets)

print(f"  Final metrics:")
for key, value in metrics.items():
    print(f"    {key}: {value:.4f}")

print("\nDIAGNOSIS:")
if metrics['opinion_f1'] == 0.0:
    print("  Opinion F1 is still zero - investigating span-level evaluation")
    
    # Test span-level computation directly
    span_predictions = []
    span_targets = []
    
    for pred, target in zip(all_predictions, all_targets):
        pred_opinion_spans = trainer._extract_spans(pred['opinion_preds'], 'opinion')
        target_opinion_spans = trainer._extract_spans(target['opinion_labels'], 'opinion')
        
        span_predictions.append({'opinions': pred_opinion_spans})
        span_targets.append({'opinions': target_opinion_spans})
    
    # Check what goes into F1 computation
    all_pred_spans = []
    all_target_spans = []
    for pred, target in zip(span_predictions, span_targets):
        all_pred_spans.extend(pred['opinions'])
        all_target_spans.extend(target['opinions'])
    
    print(f"  All predicted opinion spans: {all_pred_spans}")
    print(f"  All target opinion spans: {all_target_spans}")
    
    pred_set = set(tuple(span) for span in all_pred_spans)
    target_set = set(tuple(span) for span in all_target_spans)
    
    print(f"  Pred set: {pred_set}")
    print(f"  Target set: {target_set}")
    print(f"  Intersection: {pred_set & target_set}")
    
    if len(pred_set & target_set) == 0:
        print("  🚨 NO MATCHES: Predictions and targets don't overlap")
        print("  Issue: Model predictions don't match ground truth spans")
    else:
        print("  ✅ Spans match - computation should work")
else:
    print("  ✅ Opinion F1 is working!")