import random
from typing import List, Dict, Any

def augment_dataset(examples, max_augmentations=3):
    augmented = []
    for example in examples:
        text, spans = example
        augmented.append((text, spans))  # Keep original
        
        # Create augmentations
        for _ in range(random.randint(1, max_augmentations)):
            aug_text, aug_spans = create_augmentation(text, spans)
            augmented.append((aug_text, aug_spans))
    
    return augmented

def create_augmentation(text, spans):
    # Implement simple augmentation techniques
    words = text.split()
    spans_copy = spans.copy()
    
    # 1. Word substitution for non-aspect/opinion words
    for i, word in enumerate(words):
        # Check if word is part of any span
        is_in_span = False
        for span in spans:
            if i in span.aspect_indices or i in span.opinion_indices:
                is_in_span = True
                break
        
        # If not in a span, 20% chance to replace with synonym
        if not is_in_span and random.random() < 0.2:
            words[i] = get_synonym(word)
    
    # Update text
    new_text = ' '.join(words)
    return new_text, spans_copy

def get_synonym(word):
    # Simple synonym dictionary for common words
    synonyms = {
        'good': ['great', 'excellent', 'fantastic'],
        'bad': ['poor', 'terrible', 'awful'],
        'like': ['enjoy', 'love', 'appreciate'],
        'food': ['meal', 'dish', 'cuisine'],
        # Add more as needed
    }
    
    if word.lower() in synonyms:
        return random.choice(synonyms[word.lower()])
    return word