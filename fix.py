# DEBUG SPAN EXTRACTION - Copy this into Python terminal
# This will show you exactly why Opinion F1 is zero

import torch
import numpy as np
from train import NovelABSATrainer, NovelABSAConfig, NovelGradientABSAModel

# Create trainer to test span extraction
config = NovelABSAConfig('dev')
model = NovelGradientABSAModel(config)

# Create dummy dataloaders to avoid None error
from train import SimplifiedABSADataset, collate_fn
from torch.utils.data import DataLoader

dummy_dataset = SimplifiedABSADataset("dummy", config.model_name, config.max_length, config.dataset_name)
dummy_dataset.data = dummy_dataset.data[:2]  # Minimal data
dummy_loader = DataLoader(dummy_dataset, batch_size=1, collate_fn=collate_fn)

trainer = NovelABSATrainer(model, config, dummy_loader, dummy_loader, config.device)

print("DEBUG: Span Extraction Analysis")
print("=" * 40)

# Test with sample labels that should produce spans
test_opinion_labels = np.array([0, 1, 0, 1, 2, 0, 1, 0, 0, 0])  # Should give 3 opinion spans
test_aspect_labels = np.array([0, 1, 2, 0, 0, 1, 0, 0, 0, 0])   # Should give 2 aspect spans

print("TEST LABELS:")
print(f"  Opinion labels: {test_opinion_labels}")
print(f"  Aspect labels:  {test_aspect_labels}")

# Test current span extraction
opinion_spans = trainer._extract_spans(test_opinion_labels, 'opinion')
aspect_spans = trainer._extract_spans(test_aspect_labels, 'aspect')

print(f"\nCURRENT SPAN EXTRACTION RESULTS:")
print(f"  Opinion spans extracted: {opinion_spans}")
print(f"  Aspect spans extracted:  {aspect_spans}")

# Check for invalid spans
invalid_opinion_spans = [span for span in opinion_spans if len(span) >= 2 and span[0] > span[1]]
invalid_aspect_spans = [span for span in aspect_spans if len(span) >= 2 and span[0] > span[1]]

if invalid_opinion_spans:
    print(f"  🚨 INVALID OPINION SPANS: {invalid_opinion_spans}")
    print("     These spans have start > end, causing zero F1!")
else:
    print(f"  ✅ All opinion spans valid")

if invalid_aspect_spans:
    print(f"  🚨 INVALID ASPECT SPANS: {invalid_aspect_spans}")
else:
    print(f"  ✅ All aspect spans valid")

print(f"\nEXPECTED RESULTS:")
print(f"  Opinion spans should be: [(1, 1), (3, 4), (6, 6)]")
print(f"  Aspect spans should be:  [(1, 2), (5, 5)]")

print(f"\nFIX NEEDED:")
if invalid_opinion_spans or len(opinion_spans) == 0:
    print("  Replace _extract_spans method with the fixed version from the artifact")
else:
    print("  Span extraction looks correct - issue might be elsewhere")