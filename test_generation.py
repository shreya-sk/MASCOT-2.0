# test_generation.py
import torch
from transformers import AutoTokenizer
from src.models.absa import LLMABSA
from src.utils.config import LLMABSAConfig
from src.inference.predictor import LLMABSAPredictor

def test_generation():
    # Load config and model
    config = LLMABSAConfig()
    config.generate_explanations = True
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load model
    model_path = "checkpoints/best_model.pt"  # Use your best model checkpoint
    predictor = LLMABSAPredictor(
        model_path=model_path,
        config=config
    )
    
    # Sample texts
    texts = [
        "The food was delicious but the service was terrible.",
        "Great atmosphere and friendly staff!",
        "The pizza was overpriced and cold."
    ]
    
    # Make predictions with explanations
    for text in texts:
        print(f"\nInput: {text}")
        predictions = predictor.predict(text, generate=True)
        
        print("Aspect-Opinion-Sentiment Triplets:")
        for triplet in predictions['triplets']:
            print(f"  Aspect: {triplet['aspect']}, Opinion: {triplet['opinion']}, Sentiment: {triplet['sentiment']} (Confidence: {triplet['confidence']:.2f})")
        
        if 'explanations' in predictions:
            print("\nExplanations:")
            for expl in predictions['explanations']:
                print(f"  {expl}")
        else:
            print("\nNo explanations generated.")

if __name__ == "__main__":
    test_generation()