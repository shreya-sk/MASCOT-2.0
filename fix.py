# inference_cli.py
import argparse
from predict import ABSAInference
from train import NovelGradientABSAModel, NovelABSAConfig

def main():
    parser = argparse.ArgumentParser(description='ABSA Inference')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--text', required=True, help='Text to analyze')
    parser.add_argument('--confidence', action='store_true', help='Show confidence scores')
    
    args = parser.parse_args()
    
    inferencer = ABSAInference(args.model)
    result = inferencer.predict(args.text, return_confidence=args.confidence)
    
    print(f"Text: {result['text']}")
    print(f"Triplets: {result['triplets']}")

if __name__ == "__main__":
    main()