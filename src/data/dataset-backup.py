# src/data/dataset_with_implicit.py
"""
Enhanced ABSA Dataset with Complete Implicit Label Generation
Extends your existing dataset to include comprehensive implicit sentiment labels
"""

import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import json
import logging
from dataclasses import dataclass
import re

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ImplicitABSAExample:
    """Enhanced data structure for ABSA examples with implicit labels"""
    sentence: str
    triplets: List[Dict[str, Any]]
    tokens: List[str]
    aspect_labels: List[str]
    opinion_labels: List[str]
    sentiment_labels: List[int]
    
    # NEW: Implicit detection labels
    implicit_aspect_labels: List[int] = None  # 0=explicit, 1=implicit, 2=none
    implicit_opinion_labels: List[int] = None  # 0=explicit, 1=implicit, 2=none
    confidence_labels: List[float] = None  # Confidence scores for implicit detection
    grid_labels: List[int] = None  # Grid tagging matrix labels
    sentiment_combination_labels: List[int] = None  # Aspect-sentiment combination labels
    
    # Metadata
    domain_id: Optional[int] = None
    line_num: Optional[int] = None
    implicit_patterns: Optional[List[str]] = None  # Detected implicit patterns


class ABSADatasetWithImplicit(Dataset):
    """
    Enhanced ABSA dataset with comprehensive implicit label generation
    Extends your existing ABSADataset with state-of-the-art implicit detection
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
        instruction_template: str = "Extract aspect terms, opinion terms, and sentiment from the following review:",
        
        # NEW: Implicit detection parameters
        add_implicit_labels: bool = True,
        implicit_detection_method: str = 'advanced',  # 'simple', 'advanced', 'pattern_based'
        implicit_confidence_threshold: float = 0.7,
        use_pattern_detection: bool = True,
        generate_grid_labels: bool = True,
        generate_confidence_labels: bool = True
    ):
        """
        Initialize enhanced ABSA dataset with implicit label generation
        
        Args:
            All existing ABSADataset parameters plus:
            add_implicit_labels: Whether to generate implicit labels
            implicit_detection_method: Method for implicit detection
            implicit_confidence_threshold: Confidence threshold for implicit labels
            use_pattern_detection: Whether to use pattern-based detection
            generate_grid_labels: Whether to generate grid tagging labels
            generate_confidence_labels: Whether to generate confidence labels
        """
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.split = split
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.domain_id = domain_id
        self.use_instruction_following = use_instruction_following
        self.instruction_template = instruction_template
        
        # Implicit detection parameters
        self.add_implicit_labels = add_implicit_labels
        self.implicit_detection_method = implicit_detection_method
        self.implicit_confidence_threshold = implicit_confidence_threshold
        self.use_pattern_detection = use_pattern_detection
        self.generate_grid_labels = generate_grid_labels
        self.generate_confidence_labels = generate_confidence_labels
        
        # Domain and sentiment mappings
        self.domain_map = {
            'laptop14': 0, 'laptop': 0,
            'rest14': 1, 'rest15': 1, 'rest16': 1, 'restaurant': 1
        }
        
        self.sentiment_map = {'POS': 0, 'NEU': 1, 'NEG': 2}
        self.reverse_sentiment_map = {0: 'POS', 1: 'NEU', 2: 'NEG'}
        
        # Load and process data
        self.data_dir = data_dir
        self.examples = self._load_and_process_data()
        
        # Generate implicit labels if requested
        if self.add_implicit_labels:
            self._generate_comprehensive_implicit_labels()
        
        logger.info(f"✅ Loaded {len(self.examples)} examples with implicit labels for {dataset_name} ({split})")
    
    def _load_and_process_data(self) -> List[ImplicitABSAExample]:
        """Load and process data with enhanced structure"""
        examples = []
        
        # Construct file path
        dataset_path = os.path.join(self.data_dir, 'aste', self.dataset_name)
        file_path = os.path.join(dataset_path, f'{self.split}.txt')
        
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            return examples
        
        logger.info(f"Loading data from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the line (assuming standard ASTE format)
                example = self._parse_line(line, line_num)
                if example:
                    examples.append(example)
            except Exception as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue
        
        return examples
    
    def _parse_line(self, line: str, line_num: int) -> Optional[ImplicitABSAExample]:
        """Parse a single line into ImplicitABSAExample"""
        parts = line.split('\t')
        if len(parts) < 2:
            return None
        
        sentence = parts[0].strip()
        triplets_str = parts[1].strip() if len(parts) > 1 else ""
        
        # Parse triplets
        triplets = self._parse_triplets(triplets_str)
        
        # Tokenize sentence
        tokens = self.tokenizer.tokenize(sentence)
        
        # Generate labels
        aspect_labels, opinion_labels, sentiment_labels = self._generate_basic_labels(
            tokens, triplets, sentence
        )
        
        return ImplicitABSAExample(
            sentence=sentence,
            triplets=triplets,
            tokens=tokens,
            aspect_labels=aspect_labels,
            opinion_labels=opinion_labels,
            sentiment_labels=sentiment_labels,
            domain_id=self.domain_map.get(self.dataset_name, 0),
            line_num=line_num
        )
    
    def _parse_triplets(self, triplets_str: str) -> List[Dict[str, Any]]:
        """Parse triplet string into structured format"""
        triplets = []
        
        if not triplets_str or triplets_str == "[]":
            return triplets
        
        # Remove brackets and split by triplet
        triplets_str = triplets_str.strip('[]')
        triplet_matches = re.findall(r'\([^)]+\)', triplets_str)
        
        for match in triplet_matches:
            # Parse individual triplet
            match = match.strip('()')
            parts = [part.strip().strip("'\"") for part in match.split(',')]
            
            if len(parts) >= 3:
                triplets.append({
                    'aspect': parts[0],
                    'opinion': parts[1],
                    'sentiment': parts[2]
                })
        
        return triplets
    
    def _generate_basic_labels(self, tokens: List[str], triplets: List[Dict], 
                             sentence: str) -> Tuple[List[str], List[str], List[int]]:
        """Generate basic BIO labels for aspects, opinions, and sentiments"""
        # Initialize with 'O' labels
        aspect_labels = ['O'] * len(tokens)
        opinion_labels = ['O'] * len(tokens)
        sentiment_labels = [1] * len(tokens)  # Default to neutral
        
        # Map triplets to token positions
        for triplet in triplets:
            aspect_term = triplet['aspect']
            opinion_term = triplet['opinion']
            sentiment = triplet['sentiment']
            
            # Find aspect positions
            aspect_positions = self._find_term_positions(tokens, aspect_term)
            if aspect_positions:
                self._assign_bio_labels(aspect_labels, aspect_positions, 'ASP')
            
            # Find opinion positions
            opinion_positions = self._find_term_positions(tokens, opinion_term)
            if opinion_positions:
                self._assign_bio_labels(opinion_labels, opinion_positions, 'OPN')
                
                # Assign sentiment labels
                sentiment_id = self.sentiment_map.get(sentiment, 1)
                for pos in opinion_positions:
                    sentiment_labels[pos] = sentiment_id
        
        return aspect_labels, opinion_labels, sentiment_labels
    
    def _find_term_positions(self, tokens: List[str], term: str) -> List[int]:
        """Find token positions for a given term"""
        if not term or term.lower() == 'null':
            return []
        
        # Tokenize the term
        term_tokens = self.tokenizer.tokenize(term.lower())
        if not term_tokens:
            return []
        
        # Find matching positions in token sequence
        positions = []
        tokens_lower = [token.lower() for token in tokens]
        
        for i in range(len(tokens_lower) - len(term_tokens) + 1):
            if tokens_lower[i:i+len(term_tokens)] == term_tokens:
                positions.extend(range(i, i + len(term_tokens)))
                break
        
        return positions
    
    def _assign_bio_labels(self, labels: List[str], positions: List[int], label_type: str):
        """Assign BIO labels to specified positions"""
        for i, pos in enumerate(positions):
            if i == 0:
                labels[pos] = f'B-{label_type}'
            else:
                labels[pos] = f'I-{label_type}'
    
    def _generate_comprehensive_implicit_labels(self):
        """
        Generate comprehensive implicit labels using advanced methods
        This is the MAJOR NEW COMPONENT for implicit detection
        """
        logger.info(f"🔍 Generating comprehensive implicit labels using '{self.implicit_detection_method}' method...")
        
        # Initialize implicit pattern detectors
        self.implicit_patterns = self._initialize_implicit_patterns()
        
        for example in self.examples:
            # Generate implicit aspect labels
            example.implicit_aspect_labels = self._generate_implicit_aspect_labels(example)
            
            # Generate implicit opinion labels
            example.implicit_opinion_labels = self._generate_implicit_opinion_labels(example)
            
            # Generate confidence labels
            if self.generate_confidence_labels:
                example.confidence_labels = self._generate_confidence_labels(example)
            
            # Generate grid tagging labels
            if self.generate_grid_labels:
                example.grid_labels = self._generate_grid_labels(example)
            
            # Generate sentiment combination labels
            example.sentiment_combination_labels = self._generate_sentiment_combination_labels(example)
            
            # Detect implicit patterns
            if self.use_pattern_detection:
                example.implicit_patterns = self._detect_implicit_patterns(example)
        
        logger.info("✅ Comprehensive implicit labels generated for all examples")
    
    def _initialize_implicit_patterns(self) -> Dict[str, List[str]]:
        """Initialize comprehensive implicit pattern dictionaries"""
        return {
            # Negation patterns (high implicit signal)
            'negation': [
                'not', 'never', 'no', 'nothing', 'none', 'neither', 'nor',
                'hardly', 'barely', 'scarcely', 'rarely', 'seldom',
                'without', 'lack', 'lacking', 'missing', 'absent'
            ],
            
            # Intensifier patterns (moderate implicit signal)
            'intensifiers': [
                'very', 'extremely', 'really', 'quite', 'rather', 'pretty',
                'incredibly', 'amazingly', 'surprisingly', 'remarkably',
                'exceptionally', 'particularly', 'especially', 'totally',
                'absolutely', 'completely', 'utterly', 'thoroughly'
            ],
            
            # Comparative patterns (strong implicit signal)
            'comparative': [
                'better', 'worse', 'superior', 'inferior', 'prefer', 'rather',
                'instead', 'alternative', 'compared', 'versus', 'vs',
                'more', 'less', 'most', 'least', 'best', 'worst',
                'greater', 'smaller', 'higher', 'lower', 'stronger', 'weaker'
            ],
            
            # Temporal patterns (moderate implicit signal)
            'temporal': [
                'used to', 'before', 'previously', 'formerly', 'once',
                'now', 'currently', 'nowadays', 'recently', 'lately',
                'anymore', 'still', 'yet', 'already', 'since', 'until'
            ],
            
            # Conditional patterns (high implicit signal)
            'conditional': [
                'if', 'unless', 'should', 'would', 'could', 'might',
                'may', 'perhaps', 'maybe', 'possibly', 'probably',
                'hopefully', 'suppose', 'imagine', 'wish', 'expect'
            ],
            
            # Evaluative patterns (strong implicit signal)
            'evaluative': [
                'worth', 'worthwhile', 'deserve', 'deserving', 'merit',
                'recommend', 'suggest', 'advise', 'encourage', 'discourage',
                'approve', 'disapprove', 'endorse', 'support', 'oppose'
            ],
            
            # Emotional patterns (moderate implicit signal)
            'emotional': [
                'feel', 'feeling', 'felt', 'emotion', 'emotional',
                'mood', 'impression', 'sense', 'experience', 'reaction',
                'response', 'attitude', 'opinion', 'view', 'perspective'
            ],
            
            # Uncertainty patterns (moderate implicit signal)
            'uncertainty': [
                'seem', 'seems', 'appear', 'appears', 'look', 'looks',
                'sound', 'sounds', 'feel like', 'seems like',
                'apparently', 'supposedly', 'allegedly', 'presumably'
            ]
        }
    
    def _generate_implicit_aspect_labels(self, example: ImplicitABSAExample) -> List[int]:
        """
        Generate implicit aspect labels using advanced pattern detection
        Labels: 0=explicit, 1=implicit, 2=none
        """
        tokens = example.tokens
        explicit_aspects = example.aspect_labels
        implicit_labels = [2] * len(tokens)  # Initialize as 'none'
        
        # Method selection
        if self.implicit_detection_method == 'simple':
            return self._simple_implicit_aspect_detection(tokens, explicit_aspects)
        elif self.implicit_detection_method == 'advanced':
            return self._advanced_implicit_aspect_detection(tokens, explicit_aspects, example)
        elif self.implicit_detection_method == 'pattern_based':
            return self._pattern_based_implicit_aspect_detection(tokens, explicit_aspects, example)
        
        return implicit_labels
    
    def _advanced_implicit_aspect_detection(self, tokens: List[str], 
                                          explicit_aspects: List[str],
                                          example: ImplicitABSAExample) -> List[int]:
        """Advanced implicit aspect detection using multiple signals"""
        implicit_labels = [2] * len(tokens)  # Start with 'none'
        tokens_lower = [token.lower() for token in tokens]
        
        # 1. Pattern-based detection
        for i, token in enumerate(tokens_lower):
            # Check for implicit indicators around explicit aspects
            for pattern_type, patterns in self.implicit_patterns.items():
                if token in patterns:
                    # Look for nearby context that could indicate implicit aspects
                    context_range = range(max(0, i-3), min(len(tokens), i+4))
                    
                    for j in context_range:
                        if explicit_aspects[j] != 'O':  # Near explicit aspect
                            # Mark surrounding tokens as potentially implicit
                            implicit_range = range(max(0, j-2), min(len(tokens), j+3))
                            for k in implicit_range:
                                if explicit_aspects[k] == 'O' and implicit_labels[k] == 2:
                                    # Check if this could be an implicit aspect
                                    if self._is_potential_implicit_aspect(tokens[k], pattern_type):
                                        implicit_labels[k] = 1
        
        # 2. Contextual dependency detection
        for i, token in enumerate(tokens_lower):
            if explicit_aspects[i] == 'O':  # Not explicitly labeled
                # Look for contextual clues
                if self._has_aspect_context(tokens_lower, i):
                    implicit_labels[i] = 1
        
        # 3. Pronoun and reference resolution
        pronouns = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
        for i, token in enumerate(tokens_lower):
            if token in pronouns and explicit_aspects[i] == 'O':
                # Could be referring to an implicit aspect
                if self._has_referential_context(tokens_lower, i):
                    implicit_labels[i] = 1
        
        # 4. Mark explicit aspects
        for i, label in enumerate(explicit_aspects):
            if label != 'O':
                implicit_labels[i] = 0  # Explicit
        
        return implicit_labels
    
    def _is_potential_implicit_aspect(self, token: str, pattern_type: str) -> bool:
        """Check if a token could be an implicit aspect given the pattern context"""
        token_lower = token.lower()
        
        # Common aspect categories that could be implicit
        implicit_aspect_indicators = {
            'service_related': ['staff', 'server', 'waiter', 'waitress', 'employee', 'management'],
            'food_related': ['dish', 'meal', 'cuisine', 'flavor', 'taste', 'portion', 'ingredient'],
            'ambiance_related': ['atmosphere', 'environment', 'setting', 'mood', 'vibe', 'decor'],
            'price_related': ['cost', 'price', 'value', 'expense', 'budget', 'affordable'],
            'quality_related': ['quality', 'standard', 'level', 'grade', 'condition', 'state']
        }
        
        # Check if token matches any implicit aspect category
        for category, indicators in implicit_aspect_indicators.items():
            if any(indicator in token_lower for indicator in indicators):
                return True
        
        # Pattern-specific checks
        if pattern_type in ['comparative', 'evaluative', 'conditional']:
            # These patterns often indicate implicit aspects
            return len(token_lower) > 2 and token_lower.isalpha()
        
        return False
    
    def _has_aspect_context(self, tokens_lower: List[str], position: int) -> bool:
        """Check if a position has contextual clues for implicit aspects"""
        # Look for aspect-related words in context window
        context_window = 3
        start = max(0, position - context_window)
        end = min(len(tokens_lower), position + context_window + 1)
        
        context_tokens = tokens_lower[start:end]
        
        # Aspect context indicators
        aspect_contexts = [
            'about', 'regarding', 'concerning', 'with', 'for', 'of',
            'quality', 'service', 'food', 'place', 'restaurant', 'hotel'
        ]
        
        return any(context in context_tokens for context in aspect_contexts)
    
    def _has_referential_context(self, tokens_lower: List[str], position: int) -> bool:
        """Check if a pronoun has referential context indicating aspect reference"""
        # Look backwards for potential antecedents
        lookback_window = 5
        start = max(0, position - lookback_window)
        
        previous_tokens = tokens_lower[start:position]
        
        # Referential indicators
        referential_indicators = [
            'aspect', 'feature', 'element', 'part', 'component',
            'food', 'service', 'place', 'atmosphere', 'price'
        ]
        
        return any(indicator in previous_tokens for indicator in referential_indicators)
    
    def _generate_implicit_opinion_labels(self, example: ImplicitABSAExample) -> List[int]:
        """
        Generate implicit opinion labels using pattern-based detection
        Labels: 0=explicit, 1=implicit, 2=none
        """
        tokens = example.tokens
        explicit_opinions = example.opinion_labels
        implicit_labels = [2] * len(tokens)  # Initialize as 'none'
        tokens_lower = [token.lower() for token in tokens]
        
        # 1. Pattern-based implicit opinion detection
        for i, token in enumerate(tokens_lower):
            for pattern_type, patterns in self.implicit_patterns.items():
                if token in patterns:
                    # Check if this pattern indicates implicit opinion
                    if self._is_implicit_opinion_pattern(token, pattern_type, tokens_lower, i):
                        implicit_labels[i] = 1
        
        # 2. Contextual implicit opinion detection
        for i, token in enumerate(tokens_lower):
            if explicit_opinions[i] == 'O':  # Not explicitly labeled
                if self._is_contextual_implicit_opinion(token, tokens_lower, i):
                    implicit_labels[i] = 1
        
        # 3. Mark explicit opinions
        for i, label in enumerate(explicit_opinions):
            if label != 'O':
                implicit_labels[i] = 0  # Explicit
        
        return implicit_labels
    
    def _is_implicit_opinion_pattern(self, token: str, pattern_type: str, 
                                   tokens_lower: List[str], position: int) -> bool:
        """Check if a pattern indicates implicit opinion"""
        # High-confidence implicit opinion patterns
        if pattern_type in ['negation', 'conditional', 'comparative']:
            return True
        
        # Context-dependent patterns
        if pattern_type in ['intensifiers', 'evaluative']:
            # Check surrounding context for opinion indicators
            context_window = 2
            start = max(0, position - context_window)
            end = min(len(tokens_lower), position + context_window + 1)
            
            context = tokens_lower[start:end]
            opinion_contexts = ['good', 'bad', 'great', 'terrible', 'nice', 'awful', 'amazing']
            
            return any(opinion in context for opinion in opinion_contexts)
        
        return False
    
    def _is_contextual_implicit_opinion(self, token: str, tokens_lower: List[str], position: int) -> bool:
        """Check if a token represents contextual implicit opinion"""
        # Implicit opinion indicators
        implicit_opinion_words = [
            'worth', 'deserve', 'recommend', 'suggest', 'avoid',
            'appreciate', 'enjoy', 'love', 'hate', 'regret',
            'disappointed', 'satisfied', 'pleased', 'upset'
        ]
        
        return token in implicit_opinion_words
    
    def _generate_confidence_labels(self, example: ImplicitABSAExample) -> List[float]:
        """Generate confidence labels for implicit detection"""
        tokens = example.tokens
        confidence_scores = [0.5] * len(tokens)  # Default medium confidence
        
        # High confidence for explicit labels
        for i, (asp_label, opn_label) in enumerate(zip(example.aspect_labels, example.opinion_labels)):
            if asp_label != 'O' or opn_label != 'O':
                confidence_scores[i] = 0.9
        
        # Adjust confidence based on implicit patterns
        if hasattr(example, 'implicit_aspect_labels'):
            for i, implicit_label in enumerate(example.implicit_aspect_labels):
                if implicit_label == 1:  # Implicit
                    confidence_scores[i] = self.implicit_confidence_threshold
        
        if hasattr(example, 'implicit_opinion_labels'):
            for i, implicit_label in enumerate(example.implicit_opinion_labels):
                if implicit_label == 1:  # Implicit
                    confidence_scores[i] = max(confidence_scores[i], self.implicit_confidence_threshold)
        
        return confidence_scores
    
    def _generate_grid_labels(self, example: ImplicitABSAExample) -> List[int]:
        """Generate grid tagging matrix labels (GM-GTM approach)"""
        tokens = example.tokens
        # Grid labels: 0=O, 1=B-ASP, 2=I-ASP, 3=implicit
        grid_labels = [0] * len(tokens)
        
        # Convert BIO labels to grid labels
        for i, label in enumerate(example.aspect_labels):
            if label == 'B-ASP':
                grid_labels[i] = 1
            elif label == 'I-ASP':
                grid_labels[i] = 2
        
        # Add implicit labels
        if hasattr(example, 'implicit_aspect_labels'):
            for i, implicit_label in enumerate(example.implicit_aspect_labels):
                if implicit_label == 1 and grid_labels[i] == 0:  # Implicit and not explicit
                    grid_labels[i] = 3
        
        return grid_labels
    
    def _generate_sentiment_combination_labels(self, example: ImplicitABSAExample) -> List[int]:
        """Generate aspect-sentiment combination labels"""
        # Use sentiment labels as combination labels for now
        # Can be enhanced with more sophisticated combination logic
        return example.sentiment_labels.copy()
    
    def _detect_implicit_patterns(self, example: ImplicitABSAExample) -> List[str]:
        """Detect implicit patterns in the example"""
        tokens_lower = [token.lower() for token in example.tokens]
        detected_patterns = []
        
        for pattern_type, patterns in self.implicit_patterns.items():
            for pattern in patterns:
                if pattern in tokens_lower:
                    detected_patterns.append(f"{pattern_type}:{pattern}")
        
        return detected_patterns
    
    def _simple_implicit_aspect_detection(self, tokens: List[str], explicit_aspects: List[str]) -> List[int]:
        """Simple implicit aspect detection (fallback method)"""
        implicit_labels = [2] * len(tokens)  # Initialize as 'none'
        tokens_lower = [token.lower() for token in tokens]
        
        # Simple pattern matching
        simple_implicit_indicators = ['not', 'never', 'no', 'better', 'worse', 'more', 'less']
        
        for i, token in enumerate(tokens_lower):
            if token in simple_implicit_indicators:
                # Look for nearby context
                context_range = range(max(0, i-2), min(len(tokens), i+3))
                for j in context_range:
                    if explicit_aspects[j] == 'O':
                        implicit_labels[j] = 1
        
        # Mark explicit aspects
        for i, label in enumerate(explicit_aspects):
            if label != 'O':
                implicit_labels[i] = 0
        
        return implicit_labels
    
    def _pattern_based_implicit_aspect_detection(self, tokens: List[str], 
                                               explicit_aspects: List[str],
                                               example: ImplicitABSAExample) -> List[int]:
        """Pattern-based implicit aspect detection"""
        # Use advanced method as base for pattern-based approach
        return self._advanced_implicit_aspect_detection(tokens, explicit_aspects, example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with comprehensive implicit labels"""
        example = self.examples[idx]
        
        # Prepare text
        if self.use_instruction_following:
            text = f"{self.instruction_template} {example.sentence}"
        else:
            text = example.sentence
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with tokenized input
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Align labels (simplified alignment - can be improved)
        seq_len = len(input_ids)
        
        # Initialize aligned labels
        aligned_aspect_labels = [-100] * seq_len
        aligned_opinion_labels = [-100] * seq_len
        aligned_sentiment_labels = [-100] * seq_len
        aligned_implicit_aspect_labels = [-100] * seq_len
        aligned_implicit_opinion_labels = [-100] * seq_len
        aligned_confidence_labels = [-100.0] * seq_len
        aligned_grid_labels = [-100] * seq_len
        
        # Simple alignment (offset by 1 for [CLS] token)
        offset = 1 if self.use_instruction_following else 1
        
        for i, (asp, opn, sent) in enumerate(zip(example.aspect_labels, 
                                                example.opinion_labels,
                                                example.sentiment_labels)):
            if i + offset < seq_len - 1:  # Leave space for [SEP]
                # Convert BIO labels to integers
                aligned_aspect_labels[i + offset] = self._bio_to_int(asp)
                aligned_opinion_labels[i + offset] = self._bio_to_int(opn)
                aligned_sentiment_labels[i + offset] = sent
        
        # Align implicit labels
        if example.implicit_aspect_labels:
            for i, label in enumerate(example.implicit_aspect_labels):
                if i + offset < seq_len - 1:
                    aligned_implicit_aspect_labels[i + offset] = label
        
        if example.implicit_opinion_labels:
            for i, label in enumerate(example.implicit_opinion_labels):
                if i + offset < seq_len - 1:
                    aligned_implicit_opinion_labels[i + offset] = label
        
        if example.confidence_labels:
            for i, label in enumerate(example.confidence_labels):
                if i + offset < seq_len - 1:
                    aligned_confidence_labels[i + offset] = label
        
        if example.grid_labels:
            for i, label in enumerate(example.grid_labels):
                if i + offset < seq_len - 1:
                    aligned_grid_labels[i + offset] = label
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'aspect_labels': torch.tensor(aligned_aspect_labels, dtype=torch.long),
            'opinion_labels': torch.tensor(aligned_opinion_labels, dtype=torch.long),
            'sentiment_labels': torch.tensor(aligned_sentiment_labels, dtype=torch.long),
            
            # NEW: Implicit detection labels
            'implicit_aspect_labels': torch.tensor(aligned_implicit_aspect_labels, dtype=torch.long),
            'implicit_opinion_labels': torch.tensor(aligned_implicit_opinion_labels, dtype=torch.long),
            'confidence_labels': torch.tensor(aligned_confidence_labels, dtype=torch.float),
            'grid_labels': torch.tensor(aligned_grid_labels, dtype=torch.long),
            'sentiment_combination_labels': torch.tensor(aligned_sentiment_labels, dtype=torch.long),
            
            # Additional metadata
            'domain_id': example.domain_id,
            'example_id': idx,
            'sentence': example.sentence,
            'implicit_patterns': example.implicit_patterns if example.implicit_patterns else []
        }
    
    def _bio_to_int(self, bio_label: str) -> int:
        """Convert BIO label to integer"""
        if bio_label == 'O':
            return 0
        elif bio_label.startswith('B-'):
            return 1
        elif bio_label.startswith('I-'):
            return 2
        else:
            return 0
    
    def get_implicit_statistics(self) -> Dict[str, Any]:
        """Get statistics about implicit labels in the dataset"""
        stats = {
            'total_examples': len(self.examples),
            'examples_with_implicit_aspects': 0,
            'examples_with_implicit_opinions': 0,
            'total_implicit_aspects': 0,
            'total_implicit_opinions': 0,
            'pattern_distribution': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        for example in self.examples:
            # Count implicit aspects
            if example.implicit_aspect_labels:
                implicit_aspect_count = sum(1 for label in example.implicit_aspect_labels if label == 1)
                if implicit_aspect_count > 0:
                    stats['examples_with_implicit_aspects'] += 1
                    stats['total_implicit_aspects'] += implicit_aspect_count
            
            # Count implicit opinions
            if example.implicit_opinion_labels:
                implicit_opinion_count = sum(1 for label in example.implicit_opinion_labels if label == 1)
                if implicit_opinion_count > 0:
                    stats['examples_with_implicit_opinions'] += 1
                    stats['total_implicit_opinions'] += implicit_opinion_count
            
            # Pattern distribution
            if example.implicit_patterns:
                for pattern in example.implicit_patterns:
                    pattern_type = pattern.split(':')[0]
                    stats['pattern_distribution'][pattern_type] = stats['pattern_distribution'].get(pattern_type, 0) + 1
            
            # Confidence distribution
            if example.confidence_labels:
                for conf in example.confidence_labels:
                    if conf > 0.8:
                        stats['confidence_distribution']['high'] += 1
                    elif conf > 0.5:
                        stats['confidence_distribution']['medium'] += 1
                    else:
                        stats['confidence_distribution']['low'] += 1
        
        return stats
    
    def print_implicit_statistics(self):
        """Print comprehensive statistics about implicit detection"""
        stats = self.get_implicit_statistics()
        
        print("\n" + "="*60)
        print("🔍 IMPLICIT DETECTION STATISTICS")
        print("="*60)
        print(f"📊 Dataset: {self.dataset_name} ({self.split})")
        print(f"   Total Examples: {stats['total_examples']}")
        print(f"   Detection Method: {self.implicit_detection_method}")
        
        print(f"\n🎯 Implicit Aspects:")
        print(f"   Examples with implicit aspects: {stats['examples_with_implicit_aspects']} ({stats['examples_with_implicit_aspects']/stats['total_examples']*100:.1f}%)")
        print(f"   Total implicit aspects: {stats['total_implicit_aspects']}")
        
        print(f"\n💭 Implicit Opinions:")
        print(f"   Examples with implicit opinions: {stats['examples_with_implicit_opinions']} ({stats['examples_with_implicit_opinions']/stats['total_examples']*100:.1f}%)")
        print(f"   Total implicit opinions: {stats['total_implicit_opinions']}")
        
        print(f"\n🔍 Pattern Distribution:")
        for pattern_type, count in sorted(stats['pattern_distribution'].items()):
            print(f"   {pattern_type}: {count}")
        
        print(f"\n📈 Confidence Distribution:")
        total_conf = sum(stats['confidence_distribution'].values())
        if total_conf > 0:
            for level, count in stats['confidence_distribution'].items():
                print(f"   {level}: {count} ({count/total_conf*100:.1f}%)")
        
        print("="*60)


def create_dataset_with_implicit(data_dir: str, tokenizer, split: str = 'train', 
                                dataset_name: str = 'rest15', **kwargs) -> ABSADatasetWithImplicit:
    """
    Factory function to create dataset with implicit labels
    
    Args:
        data_dir: Data directory path
        tokenizer: Tokenizer
        split: Data split
        dataset_name: Dataset name
        **kwargs: Additional parameters
        
    Returns:
        Dataset with comprehensive implicit labels
    """
    dataset = ABSADatasetWithImplicit(
        data_dir=data_dir,
        tokenizer=tokenizer,
        split=split,
        dataset_name=dataset_name,
        add_implicit_labels=True,
        implicit_detection_method='advanced',
        **kwargs
    )
    
    # Print statistics
    dataset.print_implicit_statistics()
    
    return dataset