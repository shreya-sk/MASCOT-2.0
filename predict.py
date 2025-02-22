# predict.py (in root directory)
import torch
from src.utils.config import LlamaABSAConfig
from src.inference import ABSAPredictor
from transformers import AutoTokenizer

def main():
    # Load config
    config = LlamaABSAConfig()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Initialize predictor
    predictor = ABSAPredictor(
        model_path="checkpoints/best_model.pt",
        config=config
    )
    
    # Example predictions
    texts = [
        "The food was delicious but the service was terrible.",
        "Great atmosphere and friendly staff!",
        "The pizza was overpriced and cold."
    ]
    
    # Make predictions
    for text in texts:
        predictions = predictor.predict(text)
        
        print(f"\nInput: {text}")
        print("Aspects:", predictions['aspects'])
        print("Opinions:", predictions['opinions'])
        print("Sentiments:", predictions['sentiments'])
        
        # Optional: Print attention visualization
        if hasattr(predictor.model, 'get_attention_weights'):
            attention = predictor.model.get_attention_weights(text)
            print("Attention weights:", attention)

if __name__ == "__main__":
    main()