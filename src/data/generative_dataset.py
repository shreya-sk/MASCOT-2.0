# src/data/generative_dataset.py
"""
Generative ABSA Dataset
Converts your existing dataset to support generative training with text targets
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
import json
import re
from dataclasses import dataclass

from .dataset import dataset  # Your existing dataset


@dataclass
class GenerativeABSAExample:
    """Data structure for generative ABSA examples"""
    input_text: str
    target_text: str
    task_type: str
    original_triplets: List[Dict[str, str]]
    prompt: str
    few_shot_examples: Optional[List[Dict]] = None


class SequenceConverter:
    """Convert between structured ABSA data and text sequences"""
    
    def __init__(self):
        self.sentiment_map = {
            'POS': 'positive',
            'NEG': 'negative', 
            'NEU': 'neutral',
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        
        self.reverse_sentiment_map = {
            'positive': 'POS',
            'negative': 'NEG',
            'neutral': 'NEU'
        }
    
    def triplets_to_text(self, triplets: List[Dict[str, str]], format_type: str = 'natural') -> str:
        """Convert triplets to text representation"""
        if not triplets:
            return "No aspects found."
        
        if format_type == 'natural':
            # Natural language format
            parts = []
            for triplet in triplets:
                aspect = triplet.get('aspect', triplet.get('aspect_term', ''))
                opinion = triplet.get('opinion', triplet.get('opinion_term', ''))
                sentiment = triplet.get('sentiment', triplet.get('polarity', ''))
                
                # Normalize sentiment
                sentiment = self.sentiment_map.get(sentiment, sentiment)
                
                if aspect and opinion and sentiment:
                    parts.append(f"The aspect '{aspect}' has {sentiment} sentiment due to '{opinion}'")
                elif aspect and sentiment:
                    parts.append(f"The aspect '{aspect}' has {sentiment} sentiment")
            
            return ". ".join(parts) + "."
            
        elif format_type == 'structured':
            # Structured format: (aspect, opinion, sentiment)
            triplet_strs = []
            for triplet in triplets:
                aspect = triplet.get('aspect', triplet.get('aspect_term', ''))
                opinion = triplet.get('opinion', triplet.get('opinion_term', ''))
                sentiment = triplet.get('sentiment', triplet.get('polarity', ''))
                
                sentiment = self.sentiment_map.get(sentiment, sentiment)
                
                if aspect and opinion and sentiment:
                    triplet_strs.append(f"({aspect}, {opinion}, {sentiment})")
                elif aspect and sentiment:
                    triplet_strs.append(f"({aspect}, , {sentiment})")
            
            return "Triplets: " + "; ".join(triplet_strs)
            
        elif format_type == 'json':
            # JSON format
            formatted_triplets = []
            for triplet in triplets:
                formatted_triplet = {}
                if 'aspect' in triplet or 'aspect_term' in triplet:
                    formatted_triplet['aspect'] = triplet.get('aspect', triplet.get('aspect_term', ''))
                if 'opinion' in triplet or 'opinion_term' in triplet:
                    formatted_triplet['opinion'] = triplet.get('opinion', triplet.get('opinion_term', ''))
                if 'sentiment' in triplet or 'polarity' in triplet:
                    sentiment = triplet.get('sentiment', triplet.get('polarity', ''))
                    formatted_triplet['sentiment'] = self.sentiment_map.get(sentiment, sentiment)
                
                formatted_triplets.append(formatted_triplet)
            
            return json.dumps(formatted_triplets, indent=2)
        
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def aspects_to_text(self, aspects: List[str]) -> str:
        """Convert aspect list to text"""
        if not aspects:
            return "No aspects found."
        return "Aspects: " + ", ".join(aspects)
    
    def opinions_to_text(self, opinions: List[str]) -> str:
        """Convert opinion list to text"""
        if not opinions:
            return "No opinions found."
        return "Opinions: " + ", ".join(opinions)
    
    def sentiment_to_text(self, sentiment: str, aspect: str = "") -> str:
        """Convert sentiment to text"""
        sentiment = self.sentiment_map.get(sentiment, sentiment)
        
        if aspect:
            return f"The sentiment of '{aspect}' is {sentiment}."
        else:
            return f"Sentiment: {sentiment}"
    
    def text_to_triplets(self, text: str) -> List[Dict[str, str]]:
        """Parse text back to triplets (for evaluation)"""
        triplets = []
        
        # Try structured format first: (aspect, opinion, sentiment)
        triplet_pattern = re.compile(r'\(([^,]+),\s*([^,]*),\s*([^)]+)\)')
        matches = triplet_pattern.findall(text)
        
        for match in matches:
            aspect = match[0].strip()
            opinion = match[1].strip() if match[1].strip() else None
            sentiment = match[2].strip()
            
            # Normalize sentiment
            sentiment = sentiment.lower()
            if sentiment in ['pos', 'positive']:
                sentiment = 'positive'
            elif sentiment in ['neg', 'negative']:
                sentiment = 'negative'
            elif sentiment in ['neu', 'neutral']:
                sentiment = 'neutral'
            
            triplet = {'aspect': aspect, 'sentiment': sentiment}
            if opinion:
                triplet['opinion'] = opinion
            
            triplets.append(triplet)
        
        # Try JSON format
        if not triplets:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    triplets = parsed
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Try natural language parsing (basic)
        if not triplets:
            # Look for patterns like "aspect 'X' has Y sentiment"
            aspect_pattern = re.compile(r"aspect '([^']+)' has (\w+) sentiment")
            matches = aspect_pattern.findall(text)
            
            for match in matches:
                aspect = match[0].strip()
                sentiment = match[1].strip().lower()
                
                triplets.append({
                    'aspect': aspect,
                    'sentiment': sentiment
                })
        
        return triplets


class GenerativeABSADataset(Dataset):
    """
    Dataset that converts ABSA data to generative format with text targets
    Builds on your existing ABSADataset
    """
    
    def __init__(self,
                 data_dir: str,
                 tokenizer,
                 split: str = 'train',
                 dataset_name: str = 'rest15',
                 max_input_length: int = 512,
                 max_target_length: int = 128,
                 task_types: List[str] = None,
                 output_format: str = 'structured',  # 'natural', 'structured', 'json'
                 include_explanations: bool = True,
                 few_shot_examples: Optional[List[Dict]] = None):
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.output_format = output_format
        self.include_explanations = include_explanations
        self.few_shot_examples = few_shot_examples
        
        # Default task types
        if task_types is None:
            task_types = ['triplet_generation', 'aspect_extraction', 'opinion_extraction', 'explanation_generation']
        self.task_types = task_types
        
        # Initialize sequence converter
        self.converter = SequenceConverter()
        
        # Load base dataset (your existing dataset)
        try:
            self.base_dataset = ABSADataset(
                data_dir=data_dir,
                tokenizer=tokenizer,
                split=split,
                dataset_name=dataset_name
            )
            print(f"✅ Loaded base dataset: {len(self.base_dataset)} examples")
        except Exception as e:
            print(f"⚠️ Could not load existing dataset: {e}")
            # Create minimal dataset structure for testing
            self.base_dataset = self._create_minimal_dataset(data_dir, split, dataset_name)
        
        # Convert to generative examples
        self.examples = self._convert_to_generative()
        
        print(f"✅ Created generative dataset: {len(self.examples)} examples")
        print(f"   Task types: {self.task_types}")
        print(f"   Output format: {self.output_format}")
    
    def _create_minimal_dataset(self, data_dir: str, split: str, dataset_name: str):
        """Create minimal dataset structure for testing"""
        print("🔧 Creating minimal dataset structure...")
        
        # Try to read raw data files
        file_path = f"{data_dir}/aste/{dataset_name}/{split}.txt"
        
        examples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse ASTE format: sentence####triplets
                    if '####' in line:
                        sentence, triplets_str = line.split('####', 1)
                        
                        # Parse triplets
                        triplets = []
                        if triplets_str.strip():
                            # Format: (aspect, opinion, sentiment)|(aspect, opinion, sentiment)
                            triplet_parts = triplets_str.split('|')
                            for triplet_part in triplet_parts:
                                triplet_part = triplet_part.strip()
                                if triplet_part.startswith('(') and triplet_part.endswith(')'):
                                    # Remove parentheses and split
                                    content = triplet_part[1:-1]
                                    parts = [p.strip() for p in content.split(',')]
                                    if len(parts) >= 3:
                                        triplets.append({
                                            'aspect': parts[0],
                                            'opinion': parts[1],
                                            'sentiment': parts[2]
                                        })
                        
                        examples.append({
                            'sentence': sentence.strip(),
                            'triplets': triplets,
                            'line_num': line_num
                        })
        
        except FileNotFoundError:
            print(f"⚠️ Dataset file not found: {file_path}")
            # Create dummy examples for testing
            examples = [
                {
                    'sentence': "The food was delicious but the service was terrible.",
                    'triplets': [
                        {'aspect': 'food', 'opinion': 'delicious', 'sentiment': 'positive'},
                        {'aspect': 'service', 'opinion': 'terrible', 'sentiment': 'negative'}
                    ],
                    'line_num': 0
                },
                {
                    'sentence': "Great atmosphere and friendly staff.",
                    'triplets': [
                        {'aspect': 'atmosphere', 'opinion': 'great', 'sentiment': 'positive'},
                        {'aspect': 'staff', 'opinion': 'friendly', 'sentiment': 'positive'}
                    ],
                    'line_num': 1
                }
            ]
        
        # Create simple dataset object
        class SimpleDataset:
            def __init__(self, examples):
                self.examples = examples
            
            def __len__(self):
                return len(self.examples)
            
            def __getitem__(self, idx):
                return self.examples[idx]
        
        return SimpleDataset(examples)
    
    def _convert_to_generative(self) -> List[GenerativeABSAExample]:
        """Convert base dataset to generative format"""
        generative_examples = []
        
        for base_example in self.base_dataset:
            # Extract data from base example
            if isinstance(base_example, dict):
                sentence = base_example.get('sentence', '')
                triplets = base_example.get('triplets', [])
            else:
                # Handle other formats if needed
                sentence = str(base_example)
                triplets = []
            
            # Create examples for each task type
            for task_type in self.task_types:
                generative_example = self._create_generative_example(
                    sentence, triplets, task_type
                )
                if generative_example:
                    generative_examples.append(generative_example)
        
        return generative_examples
    
    def _create_generative_example(self, 
                                  sentence: str, 
                                  triplets: List[Dict[str, str]], 
                                  task_type: str) -> Optional[GenerativeABSAExample]:
        """Create a single generative example for specific task"""
        
        if task_type == 'triplet_generation':
            prompt = f"Extract aspect-opinion-sentiment triplets from the following text: {sentence}"
            target = self.converter.triplets_to_text(triplets, self.output_format)
            
        elif task_type == 'aspect_extraction':
            aspects = [t.get('aspect', '') for t in triplets if t.get('aspect')]
            prompt = f"Extract aspect terms from the following text: {sentence}"
            target = self.converter.aspects_to_text(aspects)
            
        elif task_type == 'opinion_extraction':
            opinions = [t.get('opinion', '') for t in triplets if t.get('opinion')]
            prompt = f"Extract opinion terms from the following text: {sentence}"
            target = self.converter.opinions_to_text(opinions)
            
        elif task_type == 'sentiment_analysis':
            if triplets:
                # Pick first triplet for sentiment analysis
                triplet = triplets[0]
                aspect = triplet.get('aspect', '')
                sentiment = triplet.get('sentiment', '')
                prompt = f"Analyze the sentiment of '{aspect}' in the following text: {sentence}"
                target = self.converter.sentiment_to_text(sentiment, aspect)
            else:
                return None
                
        elif task_type == 'explanation_generation':
            if triplets and self.include_explanations:
                prompt = f"Explain the sentiment analysis for the following text: {sentence}"
                
                # Generate explanation based on triplets
                explanations = []
                for triplet in triplets:
                    aspect = triplet.get('aspect', '')
                    opinion = triplet.get('opinion', '')
                    sentiment = triplet.get('sentiment', '')
                    
                    if aspect and opinion and sentiment:
                        sentiment = self.converter.sentiment_map.get(sentiment, sentiment)
                        explanations.append(f"The aspect '{aspect}' has {sentiment} sentiment because of the opinion term '{opinion}'")
                
                target = "Explanation: " + ". ".join(explanations) + "."
            else:
                return None
                
        elif task_type == 'unified_generation':
            prompt = f"Analyze the following review and extract all aspects, opinions, sentiments, and provide explanations: {sentence}"
            
            # Create unified target
            triplet_text = self.converter.triplets_to_text(triplets, 'structured')
            aspects = [t.get('aspect', '') for t in triplets if t.get('aspect')]
            opinions = [t.get('opinion', '') for t in triplets if t.get('opinion')]
            
            target_parts = [
                self.converter.aspects_to_text(aspects),
                self.converter.opinions_to_text(opinions),
                triplet_text
            ]
            
            if self.include_explanations and triplets:
                explanations = []
                for triplet in triplets:
                    aspect = triplet.get('aspect', '')
                    opinion = triplet.get('opinion', '')
                    sentiment = triplet.get('sentiment', '')
                    
                    if aspect and opinion and sentiment:
                        sentiment = self.converter.sentiment_map.get(sentiment, sentiment)
                        explanations.append(f"'{aspect}' is {sentiment} due to '{opinion}'")
                
                if explanations:
                    target_parts.append("Explanation: " + "; ".join(explanations))
            
            target = " | ".join(target_parts)
            
        else:
            return None
        
        return GenerativeABSAExample(
            input_text=sentence,
            target_text=target,
            task_type=task_type,
            original_triplets=triplets,
            prompt=prompt,
            few_shot_examples=self.few_shot_examples
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        # Add few-shot examples to prompt if available
        input_text = example.prompt
        if example.few_shot_examples:
            few_shot_part = self._format_few_shot_examples(example.few_shot_examples, example.task_type)
            input_text = few_shot_part + "\n\n" + input_text
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            example.target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(),
            'task_type': example.task_type,
            'original_text': example.input_text,
            'target_text': example.target_text,
            'original_triplets': example.original_triplets
        }
    
    def _format_few_shot_examples(self, examples: List[Dict], task_type: str) -> str:
        """Format few-shot examples for prompting"""
        formatted_examples = []
        
        for i, example in enumerate(examples):
            input_text = example['input']
            output_text = example['output']
            
            if task_type == 'triplet_generation':
                formatted_examples.append(f"Example {i+1}:")
                formatted_examples.append(f"Extract aspect-opinion-sentiment triplets from: {input_text}")
                formatted_examples.append(output_text)
            elif task_type == 'aspect_extraction':
                formatted_examples.append(f"Example {i+1}:")
                formatted_examples.append(f"Extract aspect terms from: {input_text}")
                formatted_examples.append(output_text)
            # Add more task types as needed
            
            formatted_examples.append("")  # Empty line between examples
        
        return "\n".join(formatted_examples)
    
    def get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of task types in dataset"""
        task_counts = {}
        for example in self.examples:
            task_type = example.task_type
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        return task_counts
    
    def filter_by_task(self, task_type: str) -> 'GenerativeABSADataset':
        """Create new dataset with only specific task type"""
        filtered_examples = [ex for ex in self.examples if ex.task_type == task_type]
        
        # Create new dataset with filtered examples
        new_dataset = GenerativeABSADataset.__new__(GenerativeABSADataset)
        new_dataset.tokenizer = self.tokenizer
        new_dataset.max_input_length = self.max_input_length
        new_dataset.max_target_length = self.max_target_length
        new_dataset.output_format = self.output_format
        new_dataset.include_explanations = self.include_explanations
        new_dataset.few_shot_examples = self.few_shot_examples
        new_dataset.task_types = [task_type]
        new_dataset.converter = self.converter
        new_dataset.examples = filtered_examples
        
        return new_dataset


def create_generative_dataloaders(config, tokenizer) -> Dict[str, torch.utils.data.DataLoader]:
    """Create data loaders for generative training"""
    
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'dev', 'test']:
        try:
            dataset = GenerativeABSADataset(
                data_dir=config.data_dir,
                tokenizer=tokenizer,
                split=split,
                dataset_name=config.dataset_name,
                max_input_length=getattr(config, 'max_input_length', 512),
                max_target_length=getattr(config, 'max_target_length', 128),
                task_types=getattr(config, 'task_types', ['triplet_generation']),
                output_format=getattr(config, 'output_format', 'structured'),
                include_explanations=getattr(config, 'include_explanations', True)
            )
            
            datasets[split] = dataset
            
            # Create dataloader
            batch_size = config.batch_size if split == 'train' else config.batch_size * 2
            shuffle = split == 'train'
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=getattr(config, 'num_workers', 4),
                pin_memory=True
            )
            
            dataloaders[split] = dataloader
            
            print(f"✅ Created {split} dataloader: {len(dataset)} examples, {len(dataloader)} batches")
            
            # Print task distribution
            task_dist = dataset.get_task_distribution()
            print(f"   Task distribution: {task_dist}")
            
        except Exception as e:
            print(f"⚠️ Could not create {split} dataset: {e}")
    
    return dataloaders


# Few-shot examples for common tasks
DEFAULT_FEW_SHOT_EXAMPLES = {
    'triplet_generation': [
        {
            'input': "The pizza was delicious but the service was slow.",
            'output': "Triplets: (pizza, delicious, positive); (service, slow, negative)"
        },
        {
            'input': "Great atmosphere and friendly staff.",
            'output': "Triplets: (atmosphere, great, positive); (staff, friendly, positive)"
        }
    ],
    'aspect_extraction': [
        {
            'input': "The pizza was delicious but the service was slow.",
            'output': "Aspects: pizza, service"
        },
        {
            'input': "Great atmosphere and friendly staff.",
            'output': "Aspects: atmosphere, staff"
        }
    ],
    'explanation_generation': [
        {
            'input': "The pizza was delicious but the service was slow.",
            'output': "Explanation: The pizza has positive sentiment due to being delicious. The service has negative sentiment due to being slow."
        }
    ]
}