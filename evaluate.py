#!/usr/bin/env python
# evaluate.py
import argparse
import torch
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils.config import LLMABSAConfig
from src.inference.predictor import LLMABSAPredictor

def calculate_span_metrics(pred_spans, gold_spans):
    """Calculate precision, recall, F1 for span extraction"""
    # Convert gold spans to lowercase text for matching
    gold_spans_text = [span.lower() for span in gold_spans if span]
    pred_spans_text = [span.lower() for span in pred_spans if span]
    
    # Calculate matches (true positives)
    tp = sum(1 for span in pred_spans_text if span in gold_spans_text)
    
    # Calculate metrics
    precision = tp / len(pred_spans_text) if pred_spans_text else 0.0
    recall = tp / len(gold_spans_text) if gold_spans_text else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return precision, recall, f1

def calculate_explanation_metrics(generated_explanations, reference_explanations):
    """Calculate BLEU and ROUGE scores for generated explanations"""
    if not generated_explanations or not reference_explanations:
        return {
            'bleu': 0.0,
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0
        }
    
    # Calculate BLEU score (unigram to 4-gram)
    bleu_scores = []
    for gen, ref in zip(generated_explanations, reference_explanations):
        gen_tokens = gen.lower().split()
        ref_tokens = [ref.lower().split()]
        bleu = sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu)
    
    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    
    try:
        for gen, ref in zip(generated_explanations, reference_explanations):
            scores = rouge.get_scores(gen, ref)[0]
            rouge_scores['rouge-1'] += scores['rouge-1']['f']
            rouge_scores['rouge-2'] += scores['rouge-2']['f']
            rouge_scores['rouge-l'] += scores['rouge-l']['f']
    except Exception as e:
        print(f"Warning: Rouge calculation failed: {e}")
    
    # Average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    avg_rouge_1 = rouge_scores['rouge-1'] / len(generated_explanations) if generated_explanations else 0.0
    avg_rouge_2 = rouge_scores['rouge-2'] / len(generated_explanations) if generated_explanations else 0.0
    avg_rouge_l = rouge_scores['rouge-l'] / len(generated_explanations) if generated_explanations else 0.0
    
    return {
        'bleu': avg_bleu,
        'rouge-1': avg_rouge_1,
        'rouge-2': avg_rouge_2,
        'rouge-l': avg_rouge_l
    }

def evaluate_absa(model_path, dataset_name, mode='all', batch_size=16):
    """Evaluate ABSA model on test set"""
    print(f"Evaluating on {dataset_name} dataset")
    
    # Load configuration
    config = LLMABSAConfig()
    config.batch_size = batch_size
    print(f"Effective batch size: {config.batch_size}")
    
    # Initialize predictor
    predictor = LLMABSAPredictor(
        model_path=model_path,
        config=config
    )
    
    # Load test data
    dataset_path = os.path.join('Datasets', 'aste', dataset_name, 'test.txt')
    if not os.path.exists(dataset_path):
        print(f"Test data not found at {dataset_path}")
        return
    
    # Import utils to read data
    from src.data.utils import read_aste_data
    
    # Read test data
    test_data = read_aste_data(dataset_path)
    print(f"Loaded {len(test_data)} test samples from {dataset_name}")
    
    # Set evaluation mode based on input
    evaluate_extraction = mode in ['all', 'extraction']
    evaluate_generation = mode in ['all', 'generation']
    
    # Lists to store evaluation results
    all_aspect_pred = []
    all_aspect_gold = []
    all_opinion_pred = []
    all_opinion_gold = []
    all_sentiment_pred = []
    all_sentiment_gold = []
    
    all_generated_explanations = []
    all_reference_explanations = []  # For simulation, use template-based references
    
    all_predictions = []
    
    # Process test data in batches
    for i, (text, spans) in enumerate(tqdm(test_data, desc="Evaluating")):
        # Get predictor result
        result = predictor.predict(text, generate=evaluate_generation)
        all_predictions.append(result)
        
        # Extract triplets
        triplets = result.get('triplets', [])
        
        # Extract gold standard data
        gold_aspects = []
        gold_opinions = []
        gold_sentiments = []
        
        for span_label in spans:
            # Extract span texts from indices
            aspect_indices = span_label.aspect_indices
            opinion_indices = span_label.opinion_indices
            sentiment = span_label.sentiment
            
            # Convert indices to text spans (simple approach by splitting text)
            tokens = text.split()
            aspect_text = ' '.join([tokens[idx] for idx in aspect_indices if idx < len(tokens)])
            opinion_text = ' '.join([tokens[idx] for idx in opinion_indices if idx < len(tokens)])
            
            gold_aspects.append(aspect_text)
            gold_opinions.append(opinion_text)
            gold_sentiments.append(sentiment)
        
        # Extract predicted data
        pred_aspects = [t.get('aspect', '') for t in triplets]
        pred_opinions = [t.get('opinion', '') for t in triplets]
        pred_sentiments = [t.get('sentiment', 'NEU') for t in triplets]
        
        # Add to evaluation lists
        all_aspect_pred.extend(pred_aspects)
        all_aspect_gold.extend(gold_aspects)
        all_opinion_pred.extend(pred_opinions)
        all_opinion_gold.extend(gold_opinions)
        all_sentiment_pred.extend(pred_sentiments)
        all_sentiment_gold.extend(gold_sentiments)
        
        # For explanation evaluation, use template-based approach
        if evaluate_generation and 'explanations' in result:
            explanation = result['explanations'][0] if result['explanations'] else ""
            all_generated_explanations.append(explanation)
            
            # Generate reference explanation using gold standard data
            # Create a simple reference explanation from gold data
            reference = ""
            if gold_aspects and gold_opinions and gold_sentiments:
                reference_parts = []
                for aspect, opinion, sentiment in zip(gold_aspects[:min(3, len(gold_aspects))], 
                                                     gold_opinions[:min(3, len(gold_opinions))], 
                                                     gold_sentiments[:min(3, len(gold_sentiments))]):
                    sentiment_text = "positive" if sentiment == "POS" else "negative" if sentiment == "NEG" else "neutral"
                    reference_parts.append(f"The {aspect} is {sentiment_text} because of its {opinion}.")
                reference = " ".join(reference_parts)
            else:
                reference = "No aspects detected."
                
            all_reference_explanations.append(reference)
    
    # Calculate metrics for extraction
    if evaluate_extraction:
        aspect_precision, aspect_recall, aspect_f1 = calculate_span_metrics(all_aspect_pred, all_aspect_gold)
        opinion_precision, opinion_recall, opinion_f1 = calculate_span_metrics(all_opinion_pred, all_opinion_gold)
        
        # Calculate sentiment accuracy (only for matching spans)
        matching_sentiments = 0
        total_sentiments = 0
        
        # Convert sentiment list for easier matching
        sentiment_gold_map = {}
        for aspect, sentiment in zip(all_aspect_gold, all_sentiment_gold):
            if aspect:
                sentiment_gold_map[aspect.lower()] = sentiment
        
        # Count matching sentiments
        for aspect, sentiment in zip(all_aspect_pred, all_sentiment_pred):
            if aspect and aspect.lower() in sentiment_gold_map:
                total_sentiments += 1
                if sentiment == sentiment_gold_map[aspect.lower()]:
                    matching_sentiments += 1
        
        sentiment_accuracy = matching_sentiments / total_sentiments if total_sentiments > 0 else 0.0
        
        print("Evaluation Results:")
        print(f"Aspect Extraction: Precision={aspect_precision:.4f}, Recall={aspect_recall:.4f}, F1={aspect_f1:.4f}")
        print(f"Opinion Extraction: Precision={opinion_precision:.4f}, Recall={opinion_recall:.4f}, F1={opinion_f1:.4f}")
        print(f"Sentiment Classification: Accuracy={sentiment_accuracy:.4f}")
    
    # Calculate metrics for generation
    if evaluate_generation and all_generated_explanations:
        explanation_metrics = calculate_explanation_metrics(all_generated_explanations, all_reference_explanations)
        
        print("Explanation Quality:")
        print(f"BLEU Score: {explanation_metrics['bleu']:.4f}")
        print(f"ROUGE-1 F1: {explanation_metrics['rouge-1']:.4f}")
        print(f"ROUGE-2 F1: {explanation_metrics['rouge-2']:.4f}")
        print(f"ROUGE-L F1: {explanation_metrics['rouge-l']:.4f}")
    
    # Save predictions to file
    os.makedirs('results', exist_ok=True)
    output_path = f"results/predictions_{dataset_name}.json"
    
    # Convert predictions to JSON-serializable format
    json_predictions = []
    for i, (text, _) in enumerate(test_data):
        if i < len(all_predictions):
            prediction = all_predictions[i]
            # Convert tensor values to Python types if needed
            for triplet in prediction.get('triplets', []):
                # Convert aspect and opinion indices to lists if they're tensors
                if 'aspect_indices' in triplet and hasattr(triplet['aspect_indices'], 'tolist'):
                    triplet['aspect_indices'] = triplet['aspect_indices'].tolist()
                if 'opinion_indices' in triplet and hasattr(triplet['opinion_indices'], 'tolist'):
                    triplet['opinion_indices'] = triplet['opinion_indices'].tolist()
                # Convert confidence to float if it's a tensor
                if 'confidence' in triplet and hasattr(triplet['confidence'], 'item'):
                    triplet['confidence'] = triplet['confidence'].item()
            json_predictions.append(prediction)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Predictions saved to {output_path}")
    
    # Return metrics
    results = {}
    if evaluate_extraction:
        results.update({
            'aspect_precision': aspect_precision,
            'aspect_recall': aspect_recall,
            'aspect_f1': aspect_f1,
            'opinion_precision': opinion_precision,
            'opinion_recall': opinion_recall,
            'opinion_f1': opinion_f1,
            'sentiment_accuracy': sentiment_accuracy
        })
    
    if evaluate_generation and all_generated_explanations:
        results.update(explanation_metrics)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate ABSA model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='rest15', help='Dataset to evaluate on')
    parser.add_argument('--mode', type=str, choices=['all', 'extraction', 'generation'], default='all', help='Evaluation mode')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default=None, help='Path to save evaluation results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations for predictions')
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_absa(args.model, args.dataset, args.mode, args.batch_size)
    
    # Save results if output path is specified
    if args.output and results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {args.output}")
    
    # Generate visualizations if requested
    if args.visualize:
        visualization_dir = os.path.join('results', 'visualizations', args.dataset)
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Load predictions
        predictions_path = f"results/predictions_{args.dataset}.json"
        if os.path.exists(predictions_path):
            with open(predictions_path, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            
            # Initialize predictor
            predictor = LLMABSAPredictor(
                model_path=args.model,
                config=LLMABSAConfig()
            )
            
            # Generate visualizations
            print(f"Generating visualizations for {len(predictions)} examples...")
            for i, pred in enumerate(predictions):
                text = pred.get('text', '')
                html = predictor.visualize(text, pred)
                
                # Save HTML to file
                with open(os.path.join(visualization_dir, f"example_{i}.html"), 'w', encoding='utf-8') as f:
                    f.write(html)
            
            print(f"Visualizations saved to {visualization_dir}")

if __name__ == '__main__':
    # Set up for ROUGE score calculation
    try:
        import nltk
        nltk.download('punkt', quiet=True)
    except:
        print("Warning: NLTK punkt not installed. BLEU scores may be affected.")
    
    try:
        from rouge import Rouge
    except ImportError:
        print("Warning: Rouge package not installed. ROUGE scores will not be calculated.")
        
    main()