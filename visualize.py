# visualize.py
import torch
from transformers import AutoTokenizer
from src.utils.config import ABSAConfig
from src.utils.visualization import AttentionVisualizer
from src.models import LlamaABSA

def main():
    # Load config
    config = ABSAConfig()
    
    # Initialize tokenizer and visualizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    visualizer = AttentionVisualizer(tokenizer)
    
    # Load model
    model = LlamaABSA(config)
    model.load_state_dict(torch.load("checkpoints/best_model.pt"))
    model.eval()
    
    # Example text
    text = "The food was delicious but the service was terrible."
    
    # Get attention weights
    tokens = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**tokens)
        attention_weights = outputs['attention_weights']
    
    # Visualize attention
    visualizer.plot_attention(
        attention_weights[0],  # Take first head
        tokens=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]),
        save_path='visualizations/attention_example.png'
    )
    
    # Visualize spans
    visualizer.visualize_spans(
        text,
        aspect_spans=[('food', 1, 2), ('service', 6, 7)],
        opinion_spans=[('delicious', 3, 4), ('terrible', 8, 9)],
        save_path='visualizations/spans_example.png'
    )

if __name__ == "__main__":
    main()