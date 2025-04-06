from typing import List, Dict, Any
#from data.utils import SENTIMENT_MAP, convert_to_bio_labels
from data.utils import SENTIMENT_MAP, convert_to_bio_labels
import torch
from transformers import PreTrainedTokenizer

class ABSAPreprocessor:
    """Preprocessor for ABSA data"""
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self, text: str, span_labels: List) -> Dict[str, torch.Tensor]:
        """Preprocess a single instance"""
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create span labels
        aspect_labels_list = []
        opinion_labels_list = []
        sentiment_labels_list = []
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for span_label in span_labels:
            # Convert to BIO scheme
            aspect_labels = convert_to_bio_labels(
                span_label.aspect_indices, 
                self.max_length
            )
            opinion_labels = convert_to_bio_labels(
                span_label.opinion_indices,
                self.max_length
            )
            
            # Convert sentiment to label id
            sentiment_label = torch.tensor(SENTIMENT_MAP[span_label.sentiment])
            
            aspect_labels_list.append(aspect_labels)
            opinion_labels_list.append(opinion_labels)
            sentiment_labels_list.append(sentiment_label)
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'aspect_labels': torch.stack(aspect_labels_list),
            'opinion_labels': torch.stack(opinion_labels_list),
            'sentiment_labels': torch.stack(sentiment_labels_list),
            'num_spans': len(span_labels)
        }