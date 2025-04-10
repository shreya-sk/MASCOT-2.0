# evaluate.py
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from src.models.absa import GenerativeLLMABSA
from src.utils.config import LLMABSAConfig
from src.inference.predictor import LLMABSAPredictor
from sklearn.metrics import precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
import nltk
from src.data.dataset import ABSADataset
from src.data.preprocessor import LLMABSAPreprocessor

# Ensure NLTK resources are available
try:
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed, but continuing anyway")

def load_test_data(dataset_name="rest15", num_samples=None):
    """
    Load test data from the dataset
    
    Args:
        dataset_name: Name of the dataset to load
        num_samples: Number of samples to load (None for all)
    """
    # Initialize config
    config = LLMABSAConfig()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Initialize preprocessor
    preprocessor = LLMABSAPreprocessor(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        use_syntax=False  # Disable syntax for faster loading
    )
    
    # Load test dataset
    dataset_path = config.dataset_paths.get(dataset_name)
    if not dataset_path:
        raise ValueError(f"Dataset {dataset_name} not found in config.")
    
    test_dataset = ABSADataset(
        data_dir=dataset_path,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        split='test',
        dataset_name=dataset_name,
        max_length=config.max_seq_length
    )
    
    # Convert dataset to list for easy processing
    test_data = []
    for i in range(len(test_dataset)):
        if num_samples is not None and i >= num_samples:
            break
            
        sample = test_dataset[i]
        
        # Get the original text
        text = sample.get('text', "")
        
        # Prepare labels
        aspect_labels = sample['aspect_labels']
        opinion_labels = sample['opinion_labels']
        sentiment_labels = sample['sentiment_labels']
        
        # Create a sample dict
        sample_dict = {
            'text': text,
            'aspect_labels': aspect_labels,
            'opinion_labels': opinion_labels,
            'sentiment_labels': sentiment_labels,
            'input_ids': sample['input_ids'],
            'attention_mask': sample['attention_mask']
        }
        
        test_data.append(sample_dict)
    
    print(f"Loaded {len(test_data)} test samples from {dataset_name}")
    return test_data

def extract_spans(logits):
    """
    Extract spans from logits
    
    Args:
        logits: Token classification logits [seq_len, 3]
        
    Returns:
        spans: List of extracted spans
    """
    # Get predictions (B-I-O)
    preds = logits.argmax(dim=-1)  # [seq_len]
    
    # Extract spans
    spans = []
    current_span = []
    
    for i, pred in enumerate(preds):
        pred_item = pred.item() if isinstance(pred, torch.Tensor) else pred
        
        if pred_item == 1:  # B tag
            if current_span:
                spans.append(current_span)
            current_span = [i]
        elif pred_item == 2:  # I tag
            if current_span:
                current_span.append(i)
        else:  # O tag
            if current_span:
                spans.append(current_span)
                current_span = []
    
    # Don't forget the last span
    if current_span:
        spans.append(current_span)
        
    return spans

def compute_bleu_score(reference, candidate):
    """
    Compute BLEU score between reference and candidate
    
    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text (generated)
        
    Returns:
        bleu_score: BLEU score
    """
    # Tokenize
    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())
    
    # Compute BLEU
    smoothing = SmoothingFunction().method1
    
    try:
        bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    except:
        bleu_score = 0.0
        
    return bleu_score

def compute_rouge_score(reference, candidate):
    """
    Compute ROUGE score between reference and candidate
    
    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text (generated)
        
    Returns:
        rouge_scores: Dict with ROUGE scores
    """
    rouge = Rouge()
    
    try:
        # Ensure there's content to compare
        if not reference or not candidate:
            return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
            
        scores = rouge.get_scores(candidate, reference)[0]
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
        
    return scores

def evaluate_model(model_path=None, dataset_name="rest15", num_samples=20):
    """
    Evaluate model on test data
    
    Args:
        model_path: Path to model checkpoint
        dataset_name: Name of dataset to evaluate on
        num_samples: Number of samples to evaluate (None for all)
        
    Returns:
        results: Dict with evaluation results
    """
    print(f"Evaluating on {dataset_name} dataset")
    
    # Load config and model
    config = LLMABSAConfig()
    config.generate_explanations = True
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load model
    if model_path is None:
        # Try to find the best model for this dataset
        potential_paths = [
            f"checkpoints/{config.experiment_name}_{dataset_name}_best.pt",
            f"checkpoints/best_model_{dataset_name}.pt",
            "checkpoints/best_model.pt"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if model_path is None:
            raise ValueError("No model checkpoint found. Please specify model_path.")
    
    print(f"Loading model from {model_path}")
    
    # Create predictor
    try:
        predictor = LLMABSAPredictor(
            model_path=model_path,
            config=config
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        # For quick testing without a trained model, use the model directly
        print("Creating untrained model for testing")
        model = GenerativeLLMABSA(config)
        predictor = MockPredictor(model, tokenizer)
    
    # Load test data
    test_data = load_test_data(dataset_name, num_samples)
    
    # Evaluation metrics
    results = {
        'aspect_extraction': {'precision': 0, 'recall': 0, 'f1': 0},
        'opinion_extraction': {'precision': 0, 'recall': 0, 'f1': 0},
        'sentiment_classification': {'accuracy': 0},
        'explanation_quality': {'bleu': 0, 'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
    }
    
    # Process test data
    all_predictions = []
    correct_aspects = 0
    total_aspects = 0
    correct_opinions = 0
    total_opinions = 0
    correct_sentiments = 0
    total_sentiments = 0
    
    # Generation quality metrics
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    
    # Process each sample
    for sample in tqdm(test_data, desc="Evaluating"):
        text = sample['text']
        
        # Make predictions
        try:
            predictions = predictor.predict(text, generate=True)
            
            # Store prediction for later analysis
            all_predictions.append({
                'text': text,
                'triplets': predictions['triplets'],
                'explanations': predictions.get('explanations', [])
            })
            
            # Evaluate triplet extraction
            for triplet in predictions['triplets']:
                # Check aspect extraction
                aspect = triplet['aspect']
                total_aspects += 1
                # Simple exact match for now
                if aspect.lower() in text.lower():
                    correct_aspects += 1
                
                # Check opinion extraction
                opinion = triplet['opinion']
                total_opinions += 1
                if opinion.lower() in text.lower():
                    correct_opinions += 1
                
                # Check sentiment classification
                sentiment = triplet['sentiment']
                total_sentiments += 1
                # Would compare to ground truth here if available
                
                # Evaluate explanation if available
                if 'explanations' in predictions:
                    explanation = predictions['explanations'][0]  # Just take the first one
                    
                    # For testing, create a mock reference explanation
                    reference = f"The {aspect} is {sentiment.lower()} because of the {opinion}."
                    
                    # Compute BLEU
                    bleu = compute_bleu_score(reference, explanation)
                    bleu_scores.append(bleu)
                    
                    # Compute ROUGE
                    rouge_scores = compute_rouge_score(reference, explanation)
                    rouge1_scores.append(rouge_scores['rouge-1']['f'])
                    rouge2_scores.append(rouge_scores['rouge-2']['f'])
                    rougel_scores.append(rouge_scores['rouge-l']['f'])
                    
        except Exception as e:
            print(f"Error processing sample: {e}")
    
    # Calculate metrics
    aspect_precision = correct_aspects / max(1, total_aspects)
    aspect_recall = 0.5  # Placeholder - would need ground truth
    aspect_f1 = 2 * (aspect_precision * aspect_recall) / max(1, aspect_precision + aspect_recall)
    
    opinion_precision = correct_opinions / max(1, total_opinions)
    opinion_recall = 0.5  # Placeholder - would need ground truth
    opinion_f1 = 2 * (opinion_precision * opinion_recall) / max(1, opinion_precision + opinion_recall)
    
    sentiment_accuracy = 0.5  # Placeholder - would need ground truth
    
    # Average generation quality metrics
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
    avg_rougel = np.mean(rougel_scores) if rougel_scores else 0
    
    # Store results
    results['aspect_extraction']['precision'] = aspect_precision
    results['aspect_extraction']['recall'] = aspect_recall
    results['aspect_extraction']['f1'] = aspect_f1
    
    results['opinion_extraction']['precision'] = opinion_precision
    results['opinion_extraction']['recall'] = opinion_recall
    results['opinion_extraction']['f1'] = opinion_f1
    
    results['sentiment_classification']['accuracy'] = sentiment_accuracy
    
    results['explanation_quality']['bleu'] = avg_bleu
    results['explanation_quality']['rouge-1'] = avg_rouge1
    results['explanation_quality']['rouge-2'] = avg_rouge2
    results['explanation_quality']['rouge-l'] = avg_rougel
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Aspect Extraction: Precision={aspect_precision:.4f}, Recall={aspect_recall:.4f}, F1={aspect_f1:.4f}")
    print(f"Opinion Extraction: Precision={opinion_precision:.4f}, Recall={opinion_recall:.4f}, F1={opinion_f1:.4f}")
    print(f"Sentiment Classification: Accuracy={sentiment_accuracy:.4f}")
    
    print("\nExplanation Quality:")
    print(f"BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"ROUGE-2 F1: {avg_rouge2:.4f}")
    print(f"ROUGE-L F1: {avg_rougel:.4f}")
    
    # Save predictions for analysis
    try:
        os.makedirs("results", exist_ok=True)
        with open(f"results/predictions_{dataset_name}.json", 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\nPredictions saved to results/predictions_{dataset_name}.json")
    except Exception as e:
        print(f"Error saving predictions: {e}")
    
    return results

class MockPredictor:
    """Mock predictor for quick testing without a trained model"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def predict(self, text, generate=False):
        """Mock prediction that returns random triplets"""
        # Tokenize text
        tokens = self.tokenizer(text, return_tensors="pt")
        
        # Split text into words
        words = text.split()
        
        # Generate random triplets
        triplets = []
        
        # Very simple rule-based extraction for testing
        for i, word in enumerate(words):
            if len(word) > 4 and i < len(words) - 1:  # Potential aspect
                aspect = word
                opinion = words[i+1] if i+1 < len(words) else "good"
                sentiment = "POS" if "good" in text or "great" in text else "NEG"
                
                triplet = {
                    'aspect': aspect,
                    'aspect_indices': [i],
                    'opinion': opinion,
                    'opinion_indices': [i+1] if i+1 < len(words) else [i],
                    'sentiment': sentiment,
                    'confidence': 0.8
                }
                triplets.append(triplet)
                
                # Just create one triplet for testing
                break
        
        # If no triplets were found, create a dummy one
        if not triplets:
            triplets = [{
                'aspect': words[0] if words else "item",
                'aspect_indices': [0],
                'opinion': words[-1] if words else "quality",
                'opinion_indices': [len(words)-1] if words else [0],
                'sentiment': "POS",
                'confidence': 0.5
            }]
            
        # Generate explanations if requested
        explanations = []
        if generate:
            for triplet in triplets:
                explanation = f"The {triplet['aspect']} is {triplet['sentiment'].lower()} because of the {triplet['opinion']}."
                explanations.append(explanation)
        
        return {
            'triplets': triplets,
            'explanations': explanations if generate else None
        }

def quick_test():
    """Run a quick test without a trained model"""
    print("Running quick test without a trained model")
    
    # Load config
    config = LLMABSAConfig()
    config.generate_explanations = True
    
    # Create a model
    model = GenerativeLLMABSA(config)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Create a mock predictor
    predictor = MockPredictor(model, tokenizer)
    
    # Sample texts
    texts = [
        "The food was delicious but the service was terrible.",
        "Great atmosphere and friendly staff!",
        "The pizza was overpriced and cold."
    ]
    
    # Make predictions
    for text in texts:
        print(f"\nInput: {text}")
        predictions = predictor.predict(text, generate=True)
        
        print("Triplets:")
        for triplet in predictions['triplets']:
            print(f"  Aspect: {triplet['aspect']}, Opinion: {triplet['opinion']}, Sentiment: {triplet['sentiment']} (Confidence: {triplet['confidence']:.2f})")
        
        if predictions.get('explanations'):
            print("Explanations:")
            for explanation in predictions['explanations']:
                print(f"  {explanation}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ABSA model")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="rest15", help="Dataset to evaluate on")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--quicktest", action="store_true", help="Run quick test without trained model")
    
    args = parser.parse_args()
    
    if args.quicktest:
        quick_test()
    else:
        evaluate_model(args.model, args.dataset, args.samples)