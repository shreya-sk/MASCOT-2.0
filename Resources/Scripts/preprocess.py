
import string
import random
import warnings
import pandas as pd
import numpy as np
from collections import Counter
import os
from typing import Tuple, List, Dict, Union
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    logging
)
from sklearn.utils import resample
import nlpaug.augmenter.word as naw

# Configure warnings and logging
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

class ModernAugmenter:
    """Modern data augmentation with multiple strategies"""
    def __init__(self, model_name: str = 'roberta-base', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
            
            # Initialize contextual augmenter
            self.aug_contextual = naw.ContextualWordEmbsAug(
                model_path=model_name,
                action="substitute",
                device=self.device
            )
            
        except Exception as e:
            print(f"Error initializing augmenter: {e}")
            raise
    
    def get_contextual_synonyms(self, word: str, context: str, top_k: int = 5) -> List[str]:
        """Generate contextually appropriate synonyms"""
        try:
            masked_text = context.replace(word, self.tokenizer.mask_token)
            inputs = self.tokenizer(
                masked_text, 
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs).logits
                
            mask_idx = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1]
            probs = torch.softmax(outputs[0, mask_idx], dim=-1)
            top_k_weights, top_k_indices = torch.topk(probs, min(top_k + 5, probs.size(-1)))
            
            synonyms = []
            for idx in top_k_indices[0]:
                token = self.tokenizer.decode([idx.item()]).strip()
                if token != word and token.isalpha():
                    synonyms.append(token)
                    
            return synonyms[:top_k]
        
        except Exception as e:
            print(f"Error generating synonyms: {e}")
            return []
    
    def contextual_augment(self, text: str, aspect: str) -> str:
        """Contextual augmentation preserving aspect"""
        try:
            augmented_texts = self.aug_contextual.augment(text)
            if isinstance(augmented_texts, list) and augmented_texts:
                aug_text = augmented_texts[0]
                # Ensure aspect is preserved
                if aspect.lower() not in aug_text.lower():
                    aug_text = aug_text.replace(
                        aug_text.split()[0],
                        aspect
                    )
                return aug_text
        except Exception as e:
            print(f"Contextual augmentation failed: {e}")
        return text

def standardize_sentiment(polarity: str) -> str:
    """Standardize sentiment labels"""
    polarity = polarity.lower().strip(string.whitespace)
    sentiment_set = {"positive", "negative", "neutral"}
    return polarity if polarity in sentiment_set else "other"

def preprocess_xml_data(file_path: str) -> pd.DataFrame:
    """Process XML data with error handling"""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        data = []
        sentence_id = 1
        
        for sentence in root.findall('sentence'):
            text = sentence.find('text').text.strip()
            aspect_categories = sentence.find('aspectCategories')
            
            if aspect_categories is not None:
                for aspect_category in aspect_categories.findall('aspectCategory'):
                    data.append({
                        'id': sentence_id,
                        'Sentence': text,
                        'Aspect Term': aspect_category.get('category'),
                        'polarity': aspect_category.get('polarity')
                    })
            
            sentence_id += 1
            
        df = pd.DataFrame(data)
        print(f"Processed {len(df)} rows from {file_path}")
        return df
        
    except Exception as e:
        print(f"Error processing XML file {file_path}: {e}")
        raise

def preprocess_col_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe columns"""
    df = df.copy()
    df["polarity"] = df["polarity"].apply(standardize_sentiment)
    df['Sentence'] = df['Sentence'].str.lower()
    
    polarity_mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
        'other': 3
    }
    
    df['polarity_numeric'] = df['polarity'].map(polarity_mapping)
    return df.drop(['polarity'], axis=1)

def apply_augmentation_strategies(df: pd.DataFrame, indices: set, augmenter: ModernAugmenter) -> pd.DataFrame:
    """Apply multiple augmentation strategies"""
    augmented_data = []
    
    for idx in indices:
        row = df.loc[idx]
        aspect = row['Aspect Term']
        sentence = row['Sentence']
        aspect_number = row['Aspect_Number']
        
        if not aspect or aspect.lower() not in sentence.lower():
            continue
            
        # Base row data to preserve
        base_data = {
            'id': row['id'],
            'Aspect Term': aspect,
            'polarity_numeric': row['polarity_numeric'],
            'Aspect_Number': aspect_number
        }
        
        # Strategy 1: Contextual Augmentation
        aug_sentence = augmenter.contextual_augment(sentence, aspect)
        if aug_sentence != sentence:
            aug_data = base_data.copy()
            aug_data['Sentence'] = aug_sentence
            augmented_data.append(aug_data)
        
        # Strategy 2: Synonym Replacement
        synonyms = augmenter.get_contextual_synonyms(aspect, sentence)
        for synonym in synonyms[:2]:
            aug_data = base_data.copy()
            aug_data['Sentence'] = sentence.replace(aspect, synonym)
            aug_data['Aspect Term'] = synonym
            augmented_data.append(aug_data)
    
    return pd.DataFrame(augmented_data)

def extract_and_process_multiAspects(df: pd.DataFrame) -> pd.DataFrame:
    """Process multiple aspects and add aspect numbers"""
    # Extract multi-aspect sentences
    id_counts = df['id'].value_counts()
    multi_aspect_ids = id_counts[id_counts > 1].index
    df_multi = df[df['id'].isin(multi_aspect_ids)].copy()
    
    # Sort by id and add Aspect_Number
    df_multi = df_multi.sort_values('id')
    df_multi = df_multi.groupby('id').apply(
        lambda x: x.assign(Aspect_Number=range(len(x)))
    ).reset_index(drop=True)
    
    return df_multi

def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Balance classes using stratified resampling"""
    print("Starting class balancing...")
    
    # Get class distribution
    class_counts = df['polarity_numeric'].value_counts()
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[1]
    min_class_size = class_counts.min()
    
    print(f"Initial class distribution: {class_counts}")
    
    # Separate majority and minority classes
    df_majority = df[df['polarity_numeric'] == majority_class]
    df_minority = df[df['polarity_numeric'] == minority_class]
    
    # Downsample majority class
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=min_class_size,
        random_state=42
    )
    
    # Combine balanced dataset
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final class distribution: {df_balanced['polarity_numeric'].value_counts()}")
    
    return df_balanced

def augment(df: pd.DataFrame) -> pd.DataFrame:
    """Main augmentation pipeline"""
    try:
        print("Initializing augmentation...")
        augmenter = ModernAugmenter()
        
        # Select subset for augmentation
        unique_ids = df['id'].unique()
        selected_ids = random.sample(list(unique_ids), len(unique_ids) // 2)
        subset_df = df[df['id'].isin(selected_ids)].copy()
        subset_df.reset_index(drop=True, inplace=True)
        
        print(f"Selected {len(selected_ids)} sentences for augmentation")
        
        # Generate augmented examples
        augmented_df = apply_augmentation_strategies(
            subset_df, 
            set(subset_df.index),
            augmenter
        )
        
        print(f"Generated {len(augmented_df)} augmented samples")
        
        # Combine and clean
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Sentence', 'Aspect Term'])
        
        print(f"Final dataset size after augmentation: {len(combined_df)}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error in augmentation: {str(e)}")
        raise

def check_data_quality(df: pd.DataFrame, stage: str = "") -> None:
    """Check data quality and print statistics"""
    print(f"\nData quality check - {stage}")
    print("-" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Unique sentences: {df['Sentence'].nunique()}")
    print(f"Unique aspects: {df['Aspect Term'].nunique()}")
    print(f"Class distribution:\n{df['polarity_numeric'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print("-" * 50)

def preprocess_data(
    train_data: str,
    test_data: str,
    val_data: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main preprocessing pipeline"""
    try:
        # Process training data
        print("Processing training data...")
        train_df = preprocess_xml_data(train_data)
        train_df = preprocess_col_data(train_df)
        
        # Filter aspects and polarities
        train_df = train_df[
            (train_df['polarity_numeric'].isin([0, 2])) &
            (train_df['Aspect Term'] != 'miscellaneous')
        ]
        
        # Process multi-aspects
        print("Processing multi-aspects...")
        train_df = extract_and_process_multiAspects(train_df)
        
        # Augment data
        print("Augmenting data...")
        train_df = augment(train_df)
        
        # Balance classes
        print("Balancing classes...")
        train_df = balance_classes(train_df)
        
        # Process validation and test sets
        print("Processing validation and test data...")
        val_df = process_dataset(val_data)
        test_df = process_dataset(test_data)
        
        return train_df, val_df, test_df
        
    except Exception as e:
        print(f"Error in preprocessing pipeline: {str(e)}")
        raise

def process_dataset(data_path: str) -> pd.DataFrame:
    """Process validation and test datasets"""
    df = preprocess_xml_data(data_path)
    df = preprocess_col_data(df)
    df = df[
        (df['polarity_numeric'].isin([0, 2])) &
        (df['Aspect Term'] != 'miscellaneous')
    ]
    df = extract_and_process_multiAspects(df)
    return df

def main(train_data: str, test_data: str, val_data: str) -> None:
    """Main execution function with quality checks"""
    print("Starting preprocessing pipeline...")
    
    try:
        train_df, val_df, test_df = preprocess_data(train_data, test_data, val_data)
        
        # Quality checks
        check_data_quality(train_df, "Training Data")
        check_data_quality(val_df, "Validation Data")
        check_data_quality(test_df, "Test Data")
        
        # Save processed data
        print("\nSaving processed data...")
        os.makedirs('model/dataframe', exist_ok=True)
        
        train_df.to_pickle('model/dataframe/train-aug.pkl')
        val_df.to_pickle('model/dataframe/val-aug.pkl')
        test_df.to_pickle('model/dataframe/test-aug.pkl')
        
        print("\nPreprocessing completed successfully!")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    train_data = "Datasets/MAMS-ACSA/raw/train.xml"
    test_data = "Datasets/MAMS-ACSA/raw/test.xml"
    val_data = "Datasets/MAMS-ACSA/raw/val.xml"
    main(train_data, test_data, val_data)

warnings.filterwarnings("default")

logging.set_verbosity_warning()
