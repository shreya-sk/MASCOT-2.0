# src/data/dataset.py
"""
Fixed ABSA Dataset with proper path handling and enhanced preprocessing
Supports instruction-following and multi-domain training
"""
import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import json
import logging
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ABSAExample:
    """Data structure for ABSA examples"""
    sentence: str
    triplets: List[Dict[str, Any]]
    tokens: List[str]
    aspect_labels: List[str]
    opinion_labels: List[str]
    sentiment_labels: List[int]
    implicit_labels: Optional[List[int]] = None
    domain_id: Optional[int] = None
    line_num: Optional[int] = None


class ABSADataset(Dataset):
    """
    Enhanced ABSA dataset class supporting multiple datasets with instruction-following
    and proper error handling
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        preprocessor=None,
        split: str = 'train',
        dataset_name: str = 'rest15',
        max_length: int = 128,
        domain_id: Optional[int] = None,
        use_instruction_following: bool = True,
        instruction_template: str = "Extract aspect terms, opinion terms, and sentiment from the following review:"
    ):
        """
        Initialize ABSA dataset with proper path resolution
        
        Args:
            data_dir: Base directory containing dataset folders
            tokenizer: Tokenizer for text encoding
            preprocessor: Optional custom preprocessor
            split: Data split ('train', 'dev', 'test')
            dataset_name: Dataset name ('laptop14', 'rest14', 'rest15', 'rest16')
            max_length: Maximum sequence length
            domain_id: Optional domain identifier for domain adaptation
            use_instruction_following: Whether to use instruction templates
            instruction_template: Template for instruction-following
        """
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.split = split
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.domain_id = domain_id
        self.use_instruction_following = use_instruction_following
        self.instruction_template = instruction_template
        
        # Domain mapping
        self.domain_map = {
            'laptop14': 0, 'laptop': 0,
            'rest14': 1, 'rest15': 1, 'rest16': 1, 'restaurant': 1
        }
        
        # Sentiment mapping
        self.sentiment_map = {
            'positive': 2, 'pos': 2, 'POS': 2,
            'negative': 0, 'neg': 0, 'NEG': 0,
            'neutral': 1, 'neu': 1, 'NEU': 1
        }
        
        # Load dataset
        file_path = self._find_dataset_file(data_dir, dataset_name, split)
        
        # Load and parse data
        self.raw_data = self._load_aste_data(file_path)
        self.examples = self._prepare_examples()
        
        logger.info(f"✅ Loaded {len(self.examples)} samples from {dataset_name} {split}")
        
        # Dataset statistics
        self._print_dataset_stats()
    
    def _find_dataset_file(self, data_dir: str, dataset_name: str, split: str) -> str:
        """Find dataset file with robust path resolution"""
        
        # Common file extensions and naming patterns
        file_patterns = [
            f'{split}.txt',
            f'{split}.json',
            f'{dataset_name}_{split}.txt',
            f'{dataset_name}_{split}.json'
        ]
        
        # FIXED: Proper path construction
        possible_base_paths = []
        
        if os.path.isabs(data_dir):
            # Absolute path provided
            possible_base_paths = [data_dir]
        else:
            # Relative path - try multiple locations
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            
            possible_base_paths = [
                os.path.join(project_root, data_dir),
                os.path.join(project_root, 'Datasets', 'aste'),
                os.path.join(project_root, 'Dataset', 'aste'),
                os.path.join(project_root, 'data'),
                os.path.join(current_dir, '..', '..', 'Datasets', 'aste'),
                data_dir  # Try as-is
            ]
        
        # Try all combinations of base paths, dataset names, and file patterns
        for base_path in possible_base_paths:
            for file_pattern in file_patterns:
                # Try with dataset subdirectory
                candidate_paths = [
                    os.path.join(base_path, dataset_name, file_pattern),
                    os.path.join(base_path, dataset_name.lower(), file_pattern),
                    os.path.join(base_path, dataset_name.upper(), file_pattern),
                    os.path.join(base_path, file_pattern)  # Direct file
                ]
                
                for candidate_path in candidate_paths:
                    if os.path.exists(candidate_path):
                        logger.info(f"📂 Found dataset at: {candidate_path}")
                        return candidate_path
        
        # If not found, provide helpful error message
        searched_paths = []
        for base_path in possible_base_paths[:3]:  # Show first 3 for brevity
            searched_paths.append(os.path.join(base_path, dataset_name, f'{split}.txt'))
        
        raise FileNotFoundError(
            f"Dataset file not found for {dataset_name} {split}.\n"
            f"Searched paths include:\n" + 
            "\n".join(f"  - {path}" for path in searched_paths) +
            f"\n\nPlease ensure the dataset exists in the correct directory structure.\n"
            f"Expected structure: data_dir/{dataset_name}/{split}.txt"
        )
    
    def _load_aste_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load ASTE format data with robust parsing
        
        ASTE format: sentence####triplets
        Where triplets are: [([aspect_indices], [opinion_indices], sentiment)]
        """
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Split sentence and triplets
                        if '####' not in line:
                            logger.warning(f"Line {line_num} missing '####' separator, skipping")
                            continue
                        
                        parts = line.split('####')
                        if len(parts) != 2:
                            logger.warning(f"Line {line_num} has incorrect format, skipping")
                            continue
                        
                        sentence, triplets_str = parts
                        sentence = sentence.strip()
                        
                        # Parse triplets
                        triplets = []
                        if triplets_str.strip():
                            try:
                                # Safely evaluate the triplets string
                                triplets_raw = eval(triplets_str.strip())
                                if not isinstance(triplets_raw, list):
                                    triplets_raw = [triplets_raw]
                                
                                for triplet in triplets_raw:
                                    if len(triplet) >= 3:
                                        aspect_indices, opinion_indices, sentiment = triplet[:3]
                                        
                                        # Normalize sentiment
                                        if isinstance(sentiment, str):
                                            sentiment = sentiment.lower()
                                        
                                        triplets.append({
                                            'aspect_indices': aspect_indices,
                                            'opinion_indices': opinion_indices,
                                            'sentiment': sentiment
                                        })
                            
                            except Exception as e:
                                logger.warning(f"Failed to parse triplets on line {line_num}: {e}")
                                triplets = []
                        
                        # Create data entry
                        data_entry = {
                            'sentence': sentence,
                            'triplets': triplets,
                            'line_num': line_num
                        }
                        
                        data.append(data_entry)
                        
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
            
            logger.info(f"📊 Successfully loaded {len(data)} entries from {file_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {file_path}: {e}")
        
        return data
    
    def _prepare_examples(self) -> List[ABSAExample]:
        """
        Prepare examples for training with tokenization and label creation
        """
        examples = []
        
        for data_entry in self.raw_data:
            try:
                sentence = data_entry['sentence']
                triplets = data_entry['triplets']
                line_num = data_entry.get('line_num')
                
                # Tokenize sentence (split by space for now)
                tokens = sentence.split()
                
                # Create labels
                aspect_labels = ['O'] * len(tokens)
                opinion_labels = ['O'] * len(tokens)
                sentiment_labels = [1] * len(tokens)  # Default to neutral
                
                # Process triplets
                for triplet in triplets:
                    aspect_indices = triplet['aspect_indices']
                    opinion_indices = triplet['opinion_indices']
                    sentiment = triplet['sentiment']
                    
                    # Mark aspect terms with BIO tagging
                    for i, idx in enumerate(aspect_indices):
                        if 0 <= idx < len(tokens):
                            if i == 0:
                                aspect_labels[idx] = 'B-ASP'
                            else:
                                aspect_labels[idx] = 'I-ASP'
                    
                    # Mark opinion terms with BIO tagging
                    for i, idx in enumerate(opinion_indices):
                        if 0 <= idx < len(tokens):
                            if i == 0:
                                opinion_labels[idx] = 'B-OPN'
                            else:
                                opinion_labels[idx] = 'I-OPN'
                    
                    # Set sentiment for the triplet region
                    sentiment_id = self._normalize_sentiment(sentiment)
                    all_indices = set(aspect_indices + opinion_indices)
                    for idx in all_indices:
                        if 0 <= idx < len(tokens):
                            sentiment_labels[idx] = sentiment_id
                
                # Convert sentiment labels to numeric
                sentiment_numeric = []
                for label in sentiment_labels:
                    if isinstance(label, str):
                        sentiment_numeric.append(self._normalize_sentiment(label))
                    else:
                        sentiment_numeric.append(label)
                
                # Create example
                example = ABSAExample(
                    sentence=sentence,
                    triplets=triplets,
                    tokens=tokens,
                    aspect_labels=aspect_labels,
                    opinion_labels=opinion_labels,
                    sentiment_labels=sentiment_numeric,
                    domain_id=self.domain_map.get(self.dataset_name, 0),
                    line_num=line_num
                )
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error preparing example from line {data_entry.get('line_num', 'unknown')}: {e}")
                continue
        
        logger.info(f"📝 Prepared {len(examples)} examples")
        return examples
    
    def _normalize_sentiment(self, sentiment) -> int:
        """Normalize sentiment to numeric value"""
        if isinstance(sentiment, int):
            return max(0, min(2, sentiment))
        
        sentiment_str = str(sentiment).lower().strip()
        return self.sentiment_map.get(sentiment_str, 1)  # Default to neutral
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        if not self.examples:
            return
        
        # Count triplets
        total_triplets = sum(len(ex.triplets) for ex in self.examples)
        
        # Count sentiments
        sentiment_counts = {0: 0, 1: 0, 2: 0}  # neg, neu, pos
        for example in self.examples:
            for triplet in example.triplets:
                sentiment_id = self._normalize_sentiment(triplet['sentiment'])
                sentiment_counts[sentiment_id] = sentiment_counts.get(sentiment_id, 0) + 1
        
        # Count aspects and opinions
        total_aspects = sum(ex.aspect_labels.count('B-ASP') for ex in self.examples)
        total_opinions = sum(ex.opinion_labels.count('B-OPN') for ex in self.examples)
        
        logger.info(f"📈 Dataset Statistics for {self.dataset_name} {self.split}:")
        logger.info(f"  Examples: {len(self.examples)}")
        logger.info(f"  Triplets: {total_triplets}")
        logger.info(f"  Aspects: {total_aspects}")
        logger.info(f"  Opinions: {total_opinions}")
        logger.info(f"  Sentiments - Negative: {sentiment_counts[0]}, Neutral: {sentiment_counts[1]}, Positive: {sentiment_counts[2]}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item with proper tokenization and encoding
        """
        example = self.examples[idx]
        
        # Prepare input text
        if self.use_instruction_following:
            input_text = f"{self.instruction_template} {example.sentence}"
        else:
            input_text = example.sentence
        
        # Tokenize with proper handling
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Extract tensors and remove batch dimension
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding.get('offset_mapping', None)
        if offset_mapping is not None:
            offset_mapping = offset_mapping.squeeze(0)
        
        # Create token-level labels aligned with tokenized input
        aspect_labels_aligned, opinion_labels_aligned, sentiment_labels_aligned = self._align_labels_with_tokens(
            example, encoding, offset_mapping
        )
        
        # Prepare output
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'aspect_labels': torch.tensor(aspect_labels_aligned, dtype=torch.long),
            'opinion_labels': torch.tensor(opinion_labels_aligned, dtype=torch.long),
            'sentiment_labels': torch.tensor(sentiment_labels_aligned, dtype=torch.long),
            'sentence': example.sentence,
            'triplets': example.triplets,
            'example_idx': idx
        }
        
        # Add domain information if available
        if example.domain_id is not None:
            item['domain_id'] = torch.tensor(example.domain_id, dtype=torch.long)
        
        # Add offset mapping for debugging
        if offset_mapping is not None:
            item['offset_mapping'] = offset_mapping
        
        return item
    
    def _align_labels_with_tokens(self, 
                                 example: ABSAExample, 
                                 encoding, 
                                 offset_mapping: Optional[torch.Tensor]) -> Tuple[List[int], List[int], List[int]]:
        """
        Align BIO labels with tokenized input
        """
        seq_len = len(encoding['input_ids'][0])
        
        # Initialize with padding labels
        aspect_labels = [-100] * seq_len
        opinion_labels = [-100] * seq_len
        sentiment_labels = [-100] * seq_len
        
        if offset_mapping is None or not self.use_instruction_following:
            # Simple alignment for non-instruction-following mode
            # Map original token positions to tokenized positions
            tokens = example.tokens
            
            # Find start of actual sentence in tokenized input
            sentence_start = 0
            if self.use_instruction_following:
                # Find where the sentence starts after instruction
                sentence_tokens = self.tokenizer.encode(example.sentence, add_special_tokens=False)
                input_tokens = encoding['input_ids'][0].tolist()
                
                # Simple search for sentence start
                for i in range(len(input_tokens) - len(sentence_tokens) + 1):
                    if input_tokens[i:i+len(sentence_tokens)] == sentence_tokens:
                        sentence_start = i
                        break
            
            # Map labels
            for i, token in enumerate(tokens):
                if sentence_start + i < seq_len:
                    # Convert BIO labels to numeric
                    aspect_label = self._bio_to_numeric(example.aspect_labels[i], 'aspect')
                    opinion_label = self._bio_to_numeric(example.opinion_labels[i], 'opinion')
                    
                    aspect_labels[sentence_start + i] = aspect_label
                    opinion_labels[sentence_start + i] = opinion_label
                    sentiment_labels[sentence_start + i] = example.sentiment_labels[i]
        
        else:
            # Advanced alignment using offset mapping
            # This is more complex but more accurate
            original_tokens = example.tokens
            token_char_spans = []
            
            # Calculate character spans for original tokens
            current_pos = 0
            original_text = example.sentence
            for token in original_tokens:
                token_start = original_text.find(token, current_pos)
                if token_start != -1:
                    token_end = token_start + len(token)
                    token_char_spans.append((token_start, token_end))
                    current_pos = token_end
                else:
                    # Token not found, use approximate position
                    token_char_spans.append((current_pos, current_pos + len(token)))
                    current_pos += len(token) + 1
            
            # Find instruction offset
            instruction_offset = 0
            if self.use_instruction_following:
                instruction_offset = len(self.instruction_template) + 1  # +1 for space
            
            # Map tokenized positions to original tokens
            for token_idx, (start_char, end_char) in enumerate(offset_mapping):
                if start_char == 0 and end_char == 0:  # Special tokens
                    continue
                
                # Adjust for instruction offset
                adjusted_start = start_char - instruction_offset
                adjusted_end = end_char - instruction_offset
                
                if adjusted_start < 0:  # Still in instruction part
                    continue
                
                # Find corresponding original token
                for orig_idx, (orig_start, orig_end) in enumerate(token_char_spans):
                    if (adjusted_start >= orig_start and adjusted_start < orig_end) or \
                       (adjusted_end > orig_start and adjusted_end <= orig_end):
                        
                        # Map labels
                        aspect_label = self._bio_to_numeric(example.aspect_labels[orig_idx], 'aspect')
                        opinion_label = self._bio_to_numeric(example.opinion_labels[orig_idx], 'opinion')
                        
                        aspect_labels[token_idx] = aspect_label
                        opinion_labels[token_idx] = opinion_label
                        sentiment_labels[token_idx] = example.sentiment_labels[orig_idx]
                        break
        
        return aspect_labels, opinion_labels, sentiment_labels
    
    def _bio_to_numeric(self, bio_label: str, label_type: str) -> int:
        """Convert BIO labels to numeric format"""
        if bio_label == 'O':
            return 0
        elif bio_label.startswith('B-'):
            return 1
        elif bio_label.startswith('I-'):
            return 1
        else:
            return 0
    
    def get_example(self, idx: int) -> ABSAExample:
        """Get raw example without tokenization"""
        return self.examples[idx]
    
    def get_domain_examples(self, domain_id: int) -> List[ABSAExample]:
        """Get all examples from a specific domain"""
        return [ex for ex in self.examples if ex.domain_id == domain_id]
    
    def get_few_shot_examples(self, k: int = 5, random_seed: int = 42) -> List[ABSAExample]:
        """Get k examples for few-shot learning"""
        import random
        random.seed(random_seed)
        
        if k >= len(self.examples):
            return self.examples.copy()
        
        return random.sample(self.examples, k)
    
    def create_few_shot_episodes(self, 
                                support_size: int = 5, 
                                query_size: int = 10,
                                num_episodes: int = 100) -> List[Dict[str, List[ABSAExample]]]:
        """Create few-shot learning episodes"""
        import random
        
        episodes = []
        for _ in range(num_episodes):
            # Sample support and query sets
            available_examples = self.examples.copy()
            
            # Sample support set
            support_examples = random.sample(available_examples, 
                                           min(support_size, len(available_examples)))
            
            # Remove support examples from available pool
            remaining_examples = [ex for ex in available_examples if ex not in support_examples]
            
            # Sample query set
            query_examples = random.sample(remaining_examples, 
                                         min(query_size, len(remaining_examples)))
            
            episodes.append({
                'support': support_examples,
                'query': query_examples
            })
        
        return episodes


class MultiDomainABSADataset(Dataset):
    """
    Multi-domain ABSA dataset for domain adaptation experiments
    """
    
    def __init__(self,
                 data_dir: str,
                 tokenizer,
                 datasets: List[str] = ['laptop14', 'rest14', 'rest15', 'rest16'],
                 split: str = 'train',
                 max_length: int = 128,
                 balance_domains: bool = True,
                 use_instruction_following: bool = True):
        """
        Initialize multi-domain dataset
        
        Args:
            data_dir: Base directory containing dataset folders
            tokenizer: Tokenizer for text encoding
            datasets: List of dataset names to include
            split: Data split ('train', 'dev', 'test')
            max_length: Maximum sequence length
            balance_domains: Whether to balance examples across domains
            use_instruction_following: Whether to use instruction templates
        """
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.split = split
        self.max_length = max_length
        self.balance_domains = balance_domains
        self.use_instruction_following = use_instruction_following
        
        # Load individual datasets
        self.domain_datasets = {}
        self.all_examples = []
        
        for i, dataset_name in enumerate(datasets):
            try:
                domain_dataset = ABSADataset(
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    split=split,
                    dataset_name=dataset_name,
                    max_length=max_length,
                    domain_id=i,
                    use_instruction_following=use_instruction_following
                )
                
                self.domain_datasets[dataset_name] = domain_dataset
                
                # Add domain ID to all examples
                domain_examples = []
                for example in domain_dataset.examples:
                    example.domain_id = i
                    domain_examples.append(example)
                
                self.all_examples.extend(domain_examples)
                
                logger.info(f"✅ Loaded {len(domain_examples)} examples from {dataset_name}")
                
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        # Balance domains if requested
        if self.balance_domains:
            self.all_examples = self._balance_domains()
        
        logger.info(f"📊 Multi-domain dataset created with {len(self.all_examples)} total examples")
        self._print_domain_stats()
    
    def _balance_domains(self) -> List[ABSAExample]:
        """Balance examples across domains"""
        domain_examples = {}
        for example in self.all_examples:
            domain_id = example.domain_id
            if domain_id not in domain_examples:
                domain_examples[domain_id] = []
            domain_examples[domain_id].append(example)
        
        # Find minimum domain size
        min_size = min(len(examples) for examples in domain_examples.values())
        
        # Sample equally from each domain
        balanced_examples = []
        import random
        for domain_id, examples in domain_examples.items():
            sampled = random.sample(examples, min_size)
            balanced_examples.extend(sampled)
        
        logger.info(f"🔄 Balanced domains: {min_size} examples per domain")
        return balanced_examples
    
    def _print_domain_stats(self):
        """Print statistics per domain"""
        domain_counts = {}
        for example in self.all_examples:
            domain_id = example.domain_id
            domain_counts[domain_id] = domain_counts.get(domain_id, 0) + 1
        
        logger.info("📈 Domain distribution:")
        for i, dataset_name in enumerate(self.datasets):
            count = domain_counts.get(i, 0)
            logger.info(f"  {dataset_name}: {count} examples")
    
    def __len__(self) -> int:
        return len(self.all_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with domain information"""
        example = self.all_examples[idx]
        
        # Use the original dataset's __getitem__ method
        dataset_name = self.datasets[example.domain_id]
        original_dataset = self.domain_datasets[dataset_name]
        
        # Find the example in the original dataset
        original_idx = None
        for i, orig_example in enumerate(original_dataset.examples):
            if orig_example.sentence == example.sentence and orig_example.line_num == example.line_num:
                original_idx = i
                break
        
        if original_idx is not None:
            item = original_dataset[original_idx]
        else:
            # Fallback: create item manually
            item = self._create_item_from_example(example)
        
        # Ensure domain_id is set
        item['domain_id'] = torch.tensor(example.domain_id, dtype=torch.long)
        
        return item
    
    def _create_item_from_example(self, example: ABSAExample) -> Dict[str, Any]:
        """Create item from example (fallback method)"""
        # Prepare input text
        if self.use_instruction_following:
            input_text = f"Extract aspect terms, opinion terms, and sentiment from the following review: {example.sentence}"
        else:
            input_text = example.sentence
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Simple label alignment (fallback)
        seq_len = len(encoding['input_ids'][0])
        aspect_labels = [-100] * seq_len
        opinion_labels = [-100] * seq_len
        sentiment_labels = [-100] * seq_len
        
        # Fill first few positions with actual labels
        for i, (asp_label, opn_label, sent_label) in enumerate(
            zip(example.aspect_labels, example.opinion_labels, example.sentiment_labels)
        ):
            if i < seq_len - 2:  # Leave space for special tokens
                aspect_labels[i + 1] = 1 if asp_label != 'O' else 0
                opinion_labels[i + 1] = 1 if opn_label != 'O' else 0
                sentiment_labels[i + 1] = sent_label
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'aspect_labels': torch.tensor(aspect_labels, dtype=torch.long),
            'opinion_labels': torch.tensor(opinion_labels, dtype=torch.long),
            'sentiment_labels': torch.tensor(sentiment_labels, dtype=torch.long),
            'sentence': example.sentence,
            'triplets': example.triplets,
            'domain_id': torch.tensor(example.domain_id, dtype=torch.long)
        }
    
    def get_domain_dataset(self, dataset_name: str) -> Optional[ABSADataset]:
        """Get individual domain dataset"""
        return self.domain_datasets.get(dataset_name)
    
    def get_domain_examples(self, domain_id: int) -> List[ABSAExample]:
        """Get all examples from a specific domain"""
        return [ex for ex in self.all_examples if ex.domain_id == domain_id]


class ImplicitABSADataset(ABSADataset):
    """
    Enhanced ABSA dataset with implicit sentiment detection capabilities
    """
    
    def __init__(self, *args, **kwargs):
        # Extract implicit-specific parameters
        self.implicit_detection_threshold = kwargs.pop('implicit_detection_threshold', 0.3)
        self.add_implicit_labels = kwargs.pop('add_implicit_labels', True)
        
        super().__init__(*args, **kwargs)
        
        if self.add_implicit_labels:
            self._add_implicit_labels()
    
    def _add_implicit_labels(self):
        """Add implicit sentiment labels based on heuristics"""
        logger.info("🔍 Adding implicit sentiment labels...")
        
        # Simple heuristics for implicit detection
        implicit_indicators = {
            'negation': ['not', 'never', 'no', 'nothing', 'none'],
            'intensifiers': ['very', 'extremely', 'really', 'quite'],
            'comparatives': ['better', 'worse', 'more', 'less'],
            'temporal': ['used to', 'before', 'previously'],
            'conditional': ['if', 'unless', 'should', 'would']
        }
        
        for example in self.examples:
            implicit_labels = [0] * len(example.tokens)  # 0 = explicit, 1 = implicit
            
            tokens_lower = [token.lower() for token in example.tokens]
            
            # Check for implicit indicators
            for i, token in enumerate(tokens_lower):
                for category, indicators in implicit_indicators.items():
                    if token in indicators:
                        # Mark surrounding tokens as potentially implicit
                        start = max(0, i - 2)
                        end = min(len(tokens_lower), i + 3)
                        for j in range(start, end):
                            if (example.aspect_labels[j] != 'O' or 
                                example.opinion_labels[j] != 'O'):
                                implicit_labels[j] = 1
            
            example.implicit_labels = implicit_labels
        
        logger.info("✅ Added implicit labels to all examples")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with implicit labels"""
        item = super().__getitem__(idx)
        
        example = self.examples[idx]
        if hasattr(example, 'implicit_labels') and example.implicit_labels:
            # Align implicit labels with tokenized input
            seq_len = len(item['input_ids'])
            implicit_labels_aligned = [-100] * seq_len
            
            # Simple alignment (can be improved)
            for i, label in enumerate(example.implicit_labels):
                if i < seq_len - 2:  # Leave space for special tokens
                    implicit_labels_aligned[i + 1] = label
            
            item['implicit_labels'] = torch.tensor(implicit_labels_aligned, dtype=torch.long)
        
        return item


def create_absa_datasets(data_dir: str,
                        tokenizer,
                        dataset_names: List[str] = ['rest15'],
                        splits: List[str] = ['train', 'dev', 'test'],
                        max_length: int = 128,
                        use_instruction_following: bool = True,
                        multi_domain: bool = False,
                        add_implicit_labels: bool = False) -> Dict[str, Dataset]:
    """
    Factory function to create ABSA datasets
    
    Args:
        data_dir: Base directory containing dataset folders
        tokenizer: Tokenizer for text encoding
        dataset_names: List of dataset names to load
        splits: List of data splits to load
        max_length: Maximum sequence length
        use_instruction_following: Whether to use instruction templates
        multi_domain: Whether to create multi-domain datasets
        add_implicit_labels: Whether to add implicit sentiment labels
    
    Returns:
        Dictionary mapping split names to datasets
    """
    datasets = {}
    
    if multi_domain and len(dataset_names) > 1:
        # Create multi-domain datasets
        for split in splits:
            try:
                if add_implicit_labels:
                    logger.warning("Implicit labels not supported for multi-domain datasets yet")
                
                dataset = MultiDomainABSADataset(
                    data_dir=data_dir,
                    tokenizer=tokenizer,
                    datasets=dataset_names,
                    split=split,
                    max_length=max_length,
                    use_instruction_following=use_instruction_following
                )
                datasets[split] = dataset
                
            except Exception as e:
                logger.error(f"Failed to create multi-domain dataset for {split}: {e}")
    
    else:
        # Create single-domain datasets
        for split in splits:
            dataset_name = dataset_names[0] if dataset_names else 'rest15'
            
            try:
                if add_implicit_labels:
                    dataset = ImplicitABSADataset(
                        data_dir=data_dir,
                        tokenizer=tokenizer,
                        split=split,
                        dataset_name=dataset_name,
                        max_length=max_length,
                        use_instruction_following=use_instruction_following,
                        add_implicit_labels=True
                    )
                else:
                    dataset = ABSADataset(
                        data_dir=data_dir,
                        tokenizer=tokenizer,
                        split=split,
                        dataset_name=dataset_name,
                        max_length=max_length,
                        use_instruction_following=use_instruction_following
                    )
                
                datasets[split] = dataset
                
            except Exception as e:
                logger.error(f"Failed to create dataset for {dataset_name} {split}: {e}")
    
    return datasets


# Utility functions for dataset manipulation
def combine_datasets(*datasets: ABSADataset) -> ABSADataset:
    """Combine multiple ABSA datasets into one"""
    if not datasets:
        raise ValueError("At least one dataset must be provided")
    
    # Use first dataset as template
    combined = datasets[0]
    
    # Combine examples from all datasets
    all_examples = []
    for dataset in datasets:
        all_examples.extend(dataset.examples)
    
    combined.examples = all_examples
    combined._print_dataset_stats()
    
    return combined


def split_dataset(dataset: ABSADataset, 
                 train_ratio: float = 0.8, 
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_seed: int = 42) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """Split dataset into train/val/test"""
    import random
    random.seed(random_seed)
    
    examples = dataset.examples.copy()
    random.shuffle(examples)
    
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Create new datasets
    train_dataset = ABSADataset.__new__(ABSADataset)
    val_dataset = ABSADataset.__new__(ABSADataset)
    test_dataset = ABSADataset.__new__(ABSADataset)
    
    # Copy attributes
    for attr in ['tokenizer', 'max_length', 'use_instruction_following', 'dataset_name']:
        if hasattr(dataset, attr):
            setattr(train_dataset, attr, getattr(dataset, attr))
            setattr(val_dataset, attr, getattr(dataset, attr))
            setattr(test_dataset, attr, getattr(dataset, attr))
    
    # Assign examples
    train_dataset.examples = examples[:train_end]
    val_dataset.examples = examples[train_end:val_end]
    test_dataset.examples = examples[val_end:]
    
    # Set splits
    train_dataset.split = 'train'
    val_dataset.split = 'val'
    test_dataset.split = 'test'
    
    return train_dataset, val_dataset, test_dataset