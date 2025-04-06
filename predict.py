#!/usr/bin/env python
# predict_stella.py
import argparse
import torch
import json
from transformers import AutoTokenizer

from src.utils.stella_config import StellaABSAConfig
from src.inference.stella_predictor import StellaABSAPredictor

def main():
    parser = argparse.ArgumentParser(description='Run Stella ABSA predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File with texts to analyze (one per line)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--visualize', action='store_true', help='Create HTML visualization')
    parser.add_argument('--domain_id', type=int, default=None, help='Domain ID for domain adaptation')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    
    # Check that either text or file is provided
    if not args.text and not args.file:
        parser.error("Either --text or --file must be provided")
    
    # Set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config and initialize predictor
    config = StellaABSAConfig()
    
    print(f"Loading model from {args.model}...")
    predictor = StellaABSAPredictor(
        model_path=args.model,
        config=config,
        device=device
    )
    
    # Process input
    results = []
    
    if args.text:
        # Process single text
        print(f"\nAnalyzing: {args.text}")
        predictions = predictor.predict(args.text, domain_id=args.domain_id)
        
        # Print results
        print("\nPredictions:")
        for triplet in predictions['triplets']:
            print(f"  Aspect: {triplet['aspect']}, Opinion: {triplet['opinion']}, Sentiment: {triplet['sentiment']} (Confidence: {triplet['confidence']:.2f})")
        
        # Visualize if requested
        if args.visualize:
            html = predictor.visualize(args.text, predictions)
            
            # Save HTML to file
            viz_file = "visualization.html"
            with open(viz_file, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"\nVisualization saved to {viz_file}")
        
        # Store results
        results.append({
            "text": args.text,
            "predictions": predictions['triplets']
        })
    
    elif args.file:
        # Process file with multiple texts
        print(f"\nAnalyzing texts from {args.file}...")
        
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        for i, text in enumerate(texts):
            print(f"\nText {i+1}/{len(texts)}: {text[:50]}...")
            
            try:
                predictions = predictor.predict(text, domain_id=args.domain_id)
                
                # Print results
                print("Predictions:")
                for triplet in predictions['triplets']:
                    print(f"  Aspect: {triplet['aspect']}, Opinion: {triplet['opinion']}, Sentiment: {triplet['sentiment']} (Confidence: {triplet['confidence']:.2f})")
                
                # Visualize if requested
                if args.visualize:
                    html = predictor.visualize(text, predictions)
                    
                    # Save HTML to file
                    viz_file = f"visualization_{i+1}.html"
                    with open(viz_file, "w", encoding="utf-8") as f:
                        f.write(html)
                    print(f"Visualization saved to {viz_file}")
                
                # Store results
                results.append({
                    "text": text,
                    "predictions": predictions['triplets']
                })
            
            except Exception as e:
                print(f"Error processing text: {e}")
                results.append({
                    "text": text,
                    "error": str(e)
                })
    
    # Save results if output file is specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()