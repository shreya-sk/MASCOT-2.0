from typing import List, Tuple, Dict
from dataclasses import dataclass
import torch

@dataclass
class SpanLabel:
    """Class to hold span label information"""
    aspect_indices: List[int]
    opinion_indices: List[int] 
    sentiment: str

def read_aste_data(file_path: str) -> List[Tuple[str, List[SpanLabel]]]:
    """Read ASTE format data from file"""
    processed_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split text and labels
            text, labels = line.strip().split('####')
            
            # Parse span labels
            span_labels = []
            eval_labels = eval(labels)
            
            for aspect_indices, opinion_indices, sentiment in eval_labels:
                span_labels.append(
                    SpanLabel(
                        aspect_indices=aspect_indices,
                        opinion_indices=opinion_indices,
                        sentiment=sentiment
                    )
                )
            
            processed_data.append((text, span_labels))
            
    return processed_data

def convert_to_bio_labels(indices: List[int], seq_length: int) -> torch.Tensor:
    """Convert indices to BIO scheme labels"""
    labels = torch.zeros(seq_length)
    for i, idx in enumerate(indices):
        if idx < seq_length:
            labels[idx] = 1 if i == 0 else 2  # 1=B, 2=I
    return labels

SENTIMENT_MAP = {'POS': 0, 'NEU': 1, 'NEG': 2}